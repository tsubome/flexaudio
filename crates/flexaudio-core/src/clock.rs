//! デバイス PTS / 単調クロックを共通の `i64` ナノ秒へ正規化するヘルパ。
//!
//! ループバック系はデバイスクロック由来の PTS を持ち、マイク（cpal）系は
//! 相対的・不透明なタイムスタンプしか持たない。コアは両者を「open 時に
//! 取得した原点オフセット」で共通の単調クロックへ写像する。クロスストリーム
//! 厳密同期はベストエフォート（§0.7）。

use std::time::Instant;

/// プロセス単調クロックの現在値をナノ秒で返す。
///
/// [`Instant`] 基準の単調値。絶対的な壁時計時刻ではなく、
/// 同一プロセス内での差分・順序付けにのみ意味がある。
pub fn monotonic_now_ns() -> i64 {
    monotonic_base().elapsed().as_nanos() as i64
}

/// プロセス起動時に一度だけ確定する単調クロックの原点。
fn monotonic_base() -> Instant {
    use std::sync::OnceLock;
    static BASE: OnceLock<Instant> = OnceLock::new();
    *BASE.get_or_init(Instant::now)
}

/// デバイス PTS を共通単調クロックへ正規化する。
///
/// 初回サンプルで「デバイス PTS の原点」と「単調クロックの原点」を記録し、
/// 以降は `device_pts - device_origin + monotonic_origin` を返すことで、
/// デバイスクロックの歩度を保ちつつ単調クロック軸へ平行移動する。
///
/// マイクのようにデバイス PTS を持たない経路では、各サンプルの到着時刻
/// （[`monotonic_now_ns`]）を device_pts として渡せばよい。
#[derive(Debug, Clone)]
pub struct ClockNormalizer {
    /// 初回サンプルで記録するデバイス PTS の原点（ns）。
    device_origin_ns: Option<i64>,
    /// 初回サンプルで記録する単調クロックの原点（ns）。
    monotonic_origin_ns: i64,
}

impl ClockNormalizer {
    /// 新しい正規化器を作る。原点はまだ未確定（最初の [`normalize`](Self::normalize) で確定）。
    pub fn new() -> Self {
        Self {
            device_origin_ns: None,
            monotonic_origin_ns: 0,
        }
    }

    /// まだ原点が確定していない（最初のサンプル未到着）か。
    pub fn is_unset(&self) -> bool {
        self.device_origin_ns.is_none()
    }

    /// デバイス PTS（ns）を正規化済み単調 PTS（ns）へ写像する。
    ///
    /// 初回呼び出しで原点を確定し、その時点の [`monotonic_now_ns`] を
    /// 単調原点として採用する。以降はデバイスクロックの差分を保つ。
    pub fn normalize(&mut self, device_pts_ns: i64) -> i64 {
        match self.device_origin_ns {
            Some(origin) => self.monotonic_origin_ns + device_pts_ns.wrapping_sub(origin),
            None => {
                self.device_origin_ns = Some(device_pts_ns);
                self.monotonic_origin_ns = monotonic_now_ns();
                self.monotonic_origin_ns
            }
        }
    }
}

impl Default for ClockNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_is_non_decreasing() {
        let a = monotonic_now_ns();
        let b = monotonic_now_ns();
        assert!(b >= a);
    }

    #[test]
    fn first_sample_sets_origin_and_preserves_deltas() {
        let mut n = ClockNormalizer::new();
        assert!(n.is_unset());
        // 初回は device 原点を確定し、単調原点を返す。
        let t0 = n.normalize(1_000_000); // device pts 1ms
        assert!(!n.is_unset());
        // 以降はデバイス差分を保つ: +5ms → 正規化値も +5ms。
        let t1 = n.normalize(6_000_000);
        assert_eq!(t1 - t0, 5_000_000);
        // 後退してもデルタが負として保たれる（呼び出し側がギャップ検知に使う）。
        let t2 = n.normalize(4_000_000);
        assert_eq!(t2 - t0, 3_000_000);
    }
}
