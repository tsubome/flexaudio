//! 任意のデバイスフレーム（任意 SR / 任意 ch / interleaved f32）を
//! 固定契約 48000 Hz / ステレオ 2ch / 20ms = 960 frames のチャンクへ正規化する。
//!
//! パイプライン: 入力 interleaved → チャンネル mix（→stereo）→ SR 変換（rubato）
//! → 出力 stereo accumulator → 960 frame 境界で切り出し。
//!
//! - チャンネル mix: mono→stereo（L=R 複製）、2ch はそのまま、>2ch は当面
//!   フロント 2ch を使う（BS.775 ダウンミックスは TODO）。
//! - SR 変換: `in_sample_rate == 48000` ならパススルー。それ以外は rubato の
//!   `Async`（`FixedAsync::Input`）で 48000 へ変換し、端数はリサンプラ内部が保持。
//!   **960 frame 境界の切り出しはリサンプル後に行う。**
//! - PTS: 出力チャンクの先頭サンプルに対応する device_pts を割り当てる。
//!   入力サンプルオフセット → 出力サンプルオフセットを比で追跡する。
//! - seq は付与しない（ストリーム層が付与）。

use rubato::{
    Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use rubato::audioadapter_buffers::direct::InterleavedSlice;

use crate::types::{CHANNELS, SAMPLE_RATE};

/// 1 チャンクのフレーム数（20ms @ 48kHz）。
pub const CHUNK_FRAMES: usize = 960;

const OUT_CH: usize = CHANNELS as usize; // 2

/// 入力デバイスフレームを 48k/stereo/20ms チャンクへ正規化するステートフルな変換器。
///
/// `push` で interleaved サンプルを蓄積し、`pop_chunk` で完成済みの
/// 960 frame チャンクを 1 つずつ取り出す。
pub struct Normalizer {
    in_sample_rate: u32,
    in_channels: usize,

    /// 48000 入力ならパススルー（リサンプラ無し）。
    resampler: Option<ResamplerState>,

    /// 完成待ちの stereo 出力（interleaved L,R,...）。`pop_chunk` がここから切る。
    out_buf: Vec<f32>,

    /// `out_buf` 先頭（まだ pop していない最古サンプル）に対応する出力フレーム索引。
    /// 出力サンプル→device_pts の写像に使う。
    out_frame_origin: u64,

    /// PTS アンカー: ある出力フレーム索引に device_pts(ns) を結び付ける。
    /// 入力 push 時に「その push の先頭サンプルが将来現れる出力フレーム位置」を
    /// 近似計算してアンカーを更新する。
    pts_anchor: Option<PtsAnchor>,

    /// これまでに生成した累計出力フレーム数（アンカー計算用）。
    total_out_frames: u64,
    /// これまでに投入した累計入力フレーム数（アンカー計算用）。
    total_in_frames: u64,
}

#[derive(Clone, Copy)]
struct PtsAnchor {
    /// 出力フレーム索引。
    out_frame: u64,
    /// その出力フレームに対応する device_pts(ns)。
    pts_ns: i64,
}

struct ResamplerState {
    inner: Async<f32>,
    /// rubato が要求する 1 回分の入力フレーム数（`FixedAsync::Input` で固定）。
    chunk_in_frames: usize,
    /// 1 回の `process` が生成しうる最大出力フレーム数。
    max_out_frames: usize,
    /// stereo 化済みで未処理の入力（interleaved L,R,...）。
    in_accum: Vec<f32>,
    /// rubato への入出力スクラッチ（再利用してアロケートを避ける）。
    out_scratch: Vec<f32>,
}

impl Normalizer {
    /// 入力 SR / 入力チャンネル数を指定して正規化器を作る。
    ///
    /// `in_channels` が 1 の場合は mono→stereo 複製、2 はそのまま、
    /// 3 以上はフロント 2ch を採用する。`in_sample_rate == 48000` なら
    /// SR 変換はパススルー。
    pub fn new(in_sample_rate: u32, in_channels: u16) -> Self {
        let in_channels = in_channels.max(1) as usize;

        let resampler = if in_sample_rate == SAMPLE_RATE {
            None
        } else {
            Some(ResamplerState::new(in_sample_rate, SAMPLE_RATE))
        };

        Self {
            in_sample_rate,
            in_channels,
            resampler,
            out_buf: Vec::with_capacity(CHUNK_FRAMES * OUT_CH * 4),
            out_frame_origin: 0,
            pts_anchor: None,
            total_out_frames: 0,
            total_in_frames: 0,
        }
    }

    /// 入力サンプルレート（Hz）。
    pub fn in_sample_rate(&self) -> u32 {
        self.in_sample_rate
    }

    /// SR 変換がパススルー（in == 48000）か。
    pub fn is_passthrough(&self) -> bool {
        self.resampler.is_none()
    }

    /// interleaved 入力サンプルを蓄積する。
    ///
    /// `interleaved` の長さは `in_channels` の倍数であること。`device_pts_ns` は
    /// この push の先頭フレームに対応するデバイス由来 PTS。
    pub fn push(&mut self, interleaved: &[f32], device_pts_ns: i64) {
        if interleaved.is_empty() {
            return;
        }
        let in_frames = interleaved.len() / self.in_channels;
        if in_frames == 0 {
            return;
        }

        // この push の先頭フレームが将来現れる出力フレーム位置を比で近似し、
        // PTS アンカーを更新する。（リサンプラ内部の保持端数があるため近似。）
        self.update_pts_anchor(device_pts_ns);

        // チャンネル mix → stereo interleaved。
        let mut stereo = Vec::with_capacity(in_frames * OUT_CH);
        Self::mix_to_stereo(interleaved, self.in_channels, in_frames, &mut stereo);
        self.total_in_frames += in_frames as u64;

        match &mut self.resampler {
            None => {
                // パススルー: そのまま出力 accumulator へ。
                self.total_out_frames += in_frames as u64;
                self.out_buf.extend_from_slice(&stereo);
            }
            Some(rs) => {
                rs.in_accum.extend_from_slice(&stereo);
                rs.drain_into(&mut self.out_buf, &mut self.total_out_frames);
            }
        }
    }

    /// 完成済みの 960 frame ステレオチャンクを 1 つ取り出す。
    ///
    /// 返り値は `(interleaved 1920 サンプル, 先頭サンプルの device_pts(ns))`。
    /// まだ 1 チャンク分溜まっていなければ `None`。
    pub fn pop_chunk(&mut self) -> Option<(Vec<f32>, i64)> {
        let need = CHUNK_FRAMES * OUT_CH;
        if self.out_buf.len() < need {
            return None;
        }

        let pts = self.pts_for_out_frame(self.out_frame_origin);

        let chunk: Vec<f32> = self.out_buf.drain(..need).collect();
        self.out_frame_origin += CHUNK_FRAMES as u64;

        Some((chunk, pts))
    }

    /// 現在 `out_buf` に溜まっている未取り出し出力フレーム数。
    pub fn buffered_out_frames(&self) -> usize {
        self.out_buf.len() / OUT_CH
    }

    // --- 内部ヘルパ ---

    /// 任意 ch interleaved を stereo interleaved へ mix して `dst` に push する。
    fn mix_to_stereo(src: &[f32], in_ch: usize, in_frames: usize, dst: &mut Vec<f32>) {
        match in_ch {
            1 => {
                // mono → stereo（L=R 複製）
                for &s in &src[..in_frames] {
                    dst.push(s);
                    dst.push(s);
                }
            }
            2 => {
                // 2ch はそのまま（必要分のみ）
                dst.extend_from_slice(&src[..in_frames * 2]);
            }
            _ => {
                // >2ch: 当面フロント 2ch を採用。
                // TODO(BS.775): 5.1 等の正式ダウンミックス係数を適用する。
                for f in 0..in_frames {
                    let base = f * in_ch;
                    dst.push(src[base]);
                    dst.push(src[base + 1]);
                }
            }
        }
    }

    /// 入力サンプルオフセット → 出力サンプルオフセットを SR 比で換算し、
    /// この push 先頭に対応する出力フレーム位置へ PTS アンカーを張る。
    ///
    /// アンカーは常に「最新の確かな (出力フレーム, device_pts) 対応」へ更新する。
    /// 出力フレーム位置はこれまでの累計入力フレーム数を SR 比で写像した近似値
    /// （リサンプラ内部の保持端数があるため厳密ではない）。
    fn update_pts_anchor(&mut self, device_pts_ns: i64) {
        let ratio = SAMPLE_RATE as f64 / self.in_sample_rate as f64;
        let projected_out_frame = (self.total_in_frames as f64 * ratio) as u64;
        self.pts_anchor = Some(PtsAnchor {
            out_frame: projected_out_frame,
            pts_ns: device_pts_ns,
        });
    }

    /// 出力フレーム索引 `out_frame` に対応する device_pts(ns) を、
    /// アンカーから SR 比で外挿して求める。
    fn pts_for_out_frame(&self, out_frame: u64) -> i64 {
        match self.pts_anchor {
            None => crate::clock::monotonic_now_ns(),
            Some(anchor) => {
                // 出力フレーム差 → ns 差（48kHz 基準）。
                let frame_delta = out_frame as i64 - anchor.out_frame as i64;
                let ns_per_out_frame = 1_000_000_000_i64 / SAMPLE_RATE as i64;
                anchor.pts_ns + frame_delta * ns_per_out_frame
            }
        }
    }
}

impl ResamplerState {
    fn new(in_sr: u32, out_sr: u32) -> Self {
        let ratio = out_sr as f64 / in_sr as f64;
        // 入力固定チャンク: 20ms 相当の入力フレーム（端数は rubato が内部保持）。
        let chunk_in_frames = (in_sr as usize / 50).max(64);

        let params = SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: WindowFunction::BlackmanHarris2,
        };

        let inner = Async::<f32>::new_sinc(
            ratio,
            1.0, // 比は固定（可変リサンプルは不要）
            &params,
            chunk_in_frames,
            OUT_CH,
            FixedAsync::Input,
        )
        .expect("rubato Async sinc resampler construction");

        let max_out_frames = inner.output_frames_max();

        Self {
            inner,
            chunk_in_frames,
            max_out_frames,
            in_accum: Vec::with_capacity(chunk_in_frames * OUT_CH * 4),
            out_scratch: vec![0.0; max_out_frames * OUT_CH],
        }
    }

    /// `in_accum` に溜まった分を chunk_in_frames 単位で可能な限りリサンプルし、
    /// 生成した stereo interleaved を `out_buf` へ追記する。
    fn drain_into(&mut self, out_buf: &mut Vec<f32>, total_out_frames: &mut u64) {
        let step = self.chunk_in_frames * OUT_CH;

        while self.in_accum.len() >= step {
            // 入力アダプタ: in_accum 先頭 chunk_in_frames フレーム（interleaved）。
            let in_adapter = InterleavedSlice::new(
                &self.in_accum[..step],
                OUT_CH,
                self.chunk_in_frames,
            )
            .expect("interleaved input adapter");

            let mut out_adapter = InterleavedSlice::new_mut(
                &mut self.out_scratch[..],
                OUT_CH,
                self.max_out_frames,
            )
            .expect("interleaved output adapter");

            let indexing = Indexing {
                input_offset: 0,
                output_offset: 0,
                partial_len: None,
                active_channels_mask: None,
            };

            let (_in_used, out_written) = self
                .inner
                .process_into_buffer(&in_adapter, &mut out_adapter, Some(&indexing))
                .expect("rubato process_into_buffer");

            // 生成 stereo を出力 accumulator へ。
            let n_samples = out_written * OUT_CH;
            out_buf.extend_from_slice(&self.out_scratch[..n_samples]);
            *total_out_frames += out_written as u64;

            // 消費した入力を取り除く（FixedAsync::Input なので chunk_in_frames 固定消費）。
            self.in_accum.drain(..step);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn mono_48k_to_stereo_duplicates_channels() {
        let mut n = Normalizer::new(48_000, 1);
        assert!(n.is_passthrough());
        // 960 フレーム分の mono 入力（パススルーなので 1 チャンクちょうど）。
        let mono: Vec<f32> = (0..CHUNK_FRAMES).map(|i| (i as f32) * 0.001).collect();
        n.push(&mono, 0);
        let (chunk, _pts) = n.pop_chunk().expect("one chunk");
        assert_eq!(chunk.len(), CHUNK_FRAMES * 2);
        // L == R がフレーム毎に成立。
        for f in 0..CHUNK_FRAMES {
            assert_eq!(chunk[f * 2], chunk[f * 2 + 1], "L==R at frame {f}");
            assert_eq!(chunk[f * 2], mono[f]);
        }
    }

    #[test]
    fn passthrough_preserves_frame_count() {
        let mut n = Normalizer::new(48_000, 2);
        assert!(n.is_passthrough());
        // 2 チャンク分 + 端数。
        let frames = CHUNK_FRAMES * 2 + 100;
        let stereo: Vec<f32> = (0..frames * 2).map(|i| (i as f32) * 1e-4).collect();
        n.push(&stereo, 0);

        let mut got_frames = 0usize;
        while let Some((c, _)) = n.pop_chunk() {
            assert_eq!(c.len(), CHUNK_FRAMES * 2);
            got_frames += CHUNK_FRAMES;
        }
        // ちょうど 2 チャンク取り出せ、端数 100 frame は残る。
        assert_eq!(got_frames, CHUNK_FRAMES * 2);
        assert_eq!(n.buffered_out_frames(), 100);
    }

    #[test]
    fn stereo_44100_to_48000_yields_about_50_chunks_per_second() {
        let mut n = Normalizer::new(44_100, 2);
        assert!(!n.is_passthrough());

        // 1 秒分の 44100Hz ステレオ サイン波。
        let in_frames = 44_100;
        let freq = 440.0_f32;
        let mut interleaved = Vec::with_capacity(in_frames * 2);
        for i in 0..in_frames {
            let s = (2.0 * PI * freq * (i as f32) / 44_100.0).sin() * 0.5;
            interleaved.push(s); // L
            interleaved.push(s); // R
        }

        // 細切れ push（実機の小バッファ到着を模す）でも panic しないこと。
        let mut pts = 0i64;
        for block in interleaved.chunks(441 * 2) {
            n.push(block, pts);
            pts += (block.len() as i64 / 2) * 1_000_000_000 / 44_100;
        }

        let mut chunks = 0usize;
        while let Some((c, _pts)) = n.pop_chunk() {
            assert_eq!(c.len(), CHUNK_FRAMES * 2);
            chunks += 1;
        }
        // 48000Hz / 960 = 50 チャンク/秒。リサンプラの遅延と端数保持のため
        // 約 50（48〜50 を許容）。
        assert!(
            (47..=50).contains(&chunks),
            "expected ~50 chunks, got {chunks}"
        );
    }

    #[test]
    fn pts_increases_monotonically_across_chunks() {
        let mut n = Normalizer::new(48_000, 2);
        let frames = CHUNK_FRAMES * 3;
        let stereo = vec![0.0f32; frames * 2];
        // device_pts 100ms 起点。
        n.push(&stereo, 100_000_000);

        let mut last = i64::MIN;
        let mut count = 0;
        while let Some((_, pts)) = n.pop_chunk() {
            assert!(pts >= last, "pts must be non-decreasing");
            last = pts;
            count += 1;
        }
        assert_eq!(count, 3);
    }
}
