//! 生 interleaved f32 デバイスフレーム用の SPSC・RT 安全リング（rtrb バック）。
//!
//! producer（RT コールバック）は slice を**非ブロッキング**に push する。満杯時は
//! overflow カウンタ（[`AtomicU64`]）を増やして該当分をドロップし、**RT スレッドは
//! 絶対にブロックしない**（不変条件: RT 経路は非ブロッキング DROP）。
//! consumer は pop（取り込みスレッド側）。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rtrb::{Consumer, Producer, RingBuffer};

/// 生フレームリングを作る。`capacity_samples` は **f32 サンプル単位**の容量。
///
/// 返り値の producer は RT コールバックスレッドへ、consumer は取り込みスレッドへ
/// 渡す（SPSC）。`overflow` カウンタは両者で共有され、ドロップ済みサンプル数を数える。
pub fn raw_ring(capacity_samples: usize) -> (RawProducer, RawConsumer) {
    let cap = capacity_samples.max(1);
    let (prod, cons) = RingBuffer::<f32>::new(cap);
    let overflow = Arc::new(AtomicU64::new(0));
    (
        RawProducer {
            inner: prod,
            overflow: overflow.clone(),
        },
        RawConsumer {
            inner: cons,
            overflow,
        },
    )
}

/// RT コールバック側のハンドル。非ブロッキング push のみを行う。
pub struct RawProducer {
    inner: Producer<f32>,
    overflow: Arc<AtomicU64>,
}

impl RawProducer {
    /// interleaved サンプル slice を非ブロッキングに push する。
    ///
    /// 書ける分だけ書き、入り切らなかった残りはドロップして overflow カウンタに
    /// 加算する。返り値は実際に書き込めたサンプル数。**この関数は決してブロック
    /// しない**（RT 安全）。
    pub fn push_slice(&mut self, samples: &[f32]) -> usize {
        if samples.is_empty() {
            return 0;
        }
        let free = self.inner.slots();
        let writable = free.min(samples.len());

        if writable > 0 {
            // write_chunk_uninit でアロケート無しのバルクコピー。
            if let Ok(mut chunk) = self.inner.write_chunk_uninit(writable) {
                let (a, b) = chunk.as_mut_slices();
                let (head, tail) = samples.split_at(a.len().min(samples.len()));
                for (dst, &src) in a.iter_mut().zip(head.iter()) {
                    dst.write(src);
                }
                let tail = &tail[..b.len().min(tail.len())];
                for (dst, &src) in b.iter_mut().zip(tail.iter()) {
                    dst.write(src);
                }
                // SAFETY: writable 個の MaybeUninit を確かに初期化した。
                unsafe { chunk.commit_all() };
            }
        }

        let dropped = samples.len() - writable;
        if dropped > 0 {
            self.overflow.fetch_add(dropped as u64, Ordering::Relaxed);
        }
        writable
    }

    /// これまでにドロップした累計サンプル数。
    pub fn overflow_count(&self) -> u64 {
        self.overflow.load(Ordering::Relaxed)
    }
}

/// 取り込みスレッド側のハンドル。pop する。
pub struct RawConsumer {
    inner: Consumer<f32>,
    overflow: Arc<AtomicU64>,
}

impl RawConsumer {
    /// 利用可能なサンプルを最大 `dst.len()` 個まで `dst` へ取り出す。返り値は取り出し数。
    pub fn pop_slice(&mut self, dst: &mut [f32]) -> usize {
        let avail = self.inner.slots();
        let n = avail.min(dst.len());
        if n == 0 {
            return 0;
        }
        if let Ok(chunk) = self.inner.read_chunk(n) {
            let (a, b) = chunk.as_slices();
            let (alen, blen) = (a.len(), b.len());
            dst[..alen].copy_from_slice(a);
            dst[alen..alen + blen].copy_from_slice(b);
            chunk.commit_all();
            alen + blen
        } else {
            0
        }
    }

    /// 1 サンプル取り出す（無ければ `None`）。
    pub fn pop(&mut self) -> Option<f32> {
        self.inner.pop().ok()
    }

    /// 取り出し可能なサンプル数。
    pub fn available(&self) -> usize {
        self.inner.slots()
    }

    /// これまでに producer 側がドロップした累計サンプル数。
    pub fn overflow_count(&self) -> u64 {
        self.overflow.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_pop_roundtrip() {
        let (mut p, mut c) = raw_ring(16);
        let n = p.push_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(n, 4);
        let mut out = [0.0f32; 4];
        let got = c.pop_slice(&mut out);
        assert_eq!(got, 4);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(p.overflow_count(), 0);
    }

    #[test]
    fn overflow_counts_dropped_and_never_blocks() {
        let (mut p, mut c) = raw_ring(4);
        // 容量 4 に 6 サンプル push → 4 書けて 2 ドロップ。
        let written = p.push_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(written, 4);
        assert_eq!(p.overflow_count(), 2);
        assert_eq!(c.overflow_count(), 2);

        let mut out = [0.0f32; 8];
        let got = c.pop_slice(&mut out);
        assert_eq!(got, 4);
        assert_eq!(&out[..4], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn wraps_around() {
        let (mut p, mut c) = raw_ring(4);
        p.push_slice(&[1.0, 2.0, 3.0]);
        let mut out = [0.0f32; 2];
        c.pop_slice(&mut out); // 2 消費 → read 索引前進
        assert_eq!(out, [1.0, 2.0]);
        // 残 1 + 新 3 = 4 で満杯。折り返し書き込みを検証。
        let w = p.push_slice(&[4.0, 5.0, 6.0]);
        assert_eq!(w, 3);
        let mut out2 = [0.0f32; 4];
        let got = c.pop_slice(&mut out2);
        assert_eq!(got, 4);
        assert_eq!(out2, [3.0, 4.0, 5.0, 6.0]);
    }
}
