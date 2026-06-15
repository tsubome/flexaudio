//! 固定契約の型定義: 出力フォーマット定数 / [`AudioChunk`] / [`ChunkFlags`] /
//! [`SourceKind`] / [`StreamConfig`] / [`Event`] / [`Error`]。
//!
//! 出力の固定契約（逸脱不可）:
//! interleaved `f32` / 48000 Hz / ステレオ 2ch / 20ms = 960 frames per chunk。

use bitflags::bitflags;

/// 出力サンプルレート（Hz）。全ストリームはこのレートへ正規化される。
pub const SAMPLE_RATE: u32 = 48_000;

/// 出力チャンネル数。常にステレオ（2ch interleaved）。
pub const CHANNELS: u16 = 2;

bitflags! {
    /// 1 つの [`AudioChunk`] に付随する状態フラグ。
    ///
    /// FFI 越しに安定したビット幅で渡すため `u32` 背景表現。
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct ChunkFlags: u32 {
        /// このチャンク直前にストリームの不連続（ドロップ / ギャップ）があった。
        const DISCONTINUITY = 0b0000_0001;
        /// デバイス喪失などからの自動復帰後の最初のチャンク。
        const RECOVERED = 0b0000_0010;
        /// 無音生成チャンク（ギャップ補填等で合成された無音）。
        const SILENCE = 0b0000_0100;
    }
}

/// 正規化済み 20ms オーディオチャンク。
///
/// `data` は interleaved `f32`（L,R,L,R,...）で長さは `frames * CHANNELS as usize`。
/// 固定契約では `frames == 960`（20ms @ 48kHz ステレオ → 1920 サンプル）。
#[derive(Debug, Clone, PartialEq)]
pub struct AudioChunk {
    /// interleaved `f32` サンプル。長さ = `frames * 2`。
    pub data: Vec<f32>,
    /// チャンク内のフレーム数（1 フレーム = ステレオ 1 サンプル対）。
    pub frames: usize,
    /// 先頭サンプルの正規化済み単調プレゼンテーションタイムスタンプ（ns）。
    pub pts_ns: i64,
    /// ストリーム層が単調増加で付与するシーケンス番号。
    pub seq: u64,
    /// このチャンクの状態フラグ。
    pub flags: ChunkFlags,
    /// このチャンクが届くまでに（直前に）ドロップされたチャンク数。
    pub dropped_before: u32,
}

/// キャプチャするオーディオソースの種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceKind {
    /// マイク入力（録音デバイス）。
    Mic,
    /// システム出力全体のループバック（既定スピーカーのミックス）。
    SystemLoopback,
    /// 特定プロセスの出力ループバック。
    ProcessLoopback,
}

/// 1 ストリームを開くための構成。
///
/// [`Default`] は固定契約に沿った値（`chunk_ms = 20`,
/// `ring_capacity_chunks = 50`, `exclude_self = false`, `kind = Mic`）を返す。
#[derive(Debug, Clone, PartialEq)]
pub struct StreamConfig {
    /// 対象デバイス ID。`None` なら既定デバイス。
    pub device_id: Option<String>,
    /// ソース種別。
    pub kind: SourceKind,
    /// チャンク長（ミリ秒）。固定契約は 20。
    pub chunk_ms: u32,
    /// チャンクリングの容量（チャンク数）。満杯時は DROP_OLDEST。
    pub ring_capacity_chunks: usize,
    /// [`SourceKind::ProcessLoopback`] の対象 PID。
    pub target_pid: Option<u32>,
    /// 自プロセスの再生音を除外するか（フィードバックループ防止）。
    pub exclude_self: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            device_id: None,
            kind: SourceKind::Mic,
            chunk_ms: 20,
            ring_capacity_chunks: 50,
            target_pid: None,
            exclude_self: false,
        }
    }
}

/// ストリーム実行中に消費側へ通知される非同期イベント。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// チャンクリング満杯により `count` 個のチャンクがドロップされた。
    ChunkDropped {
        /// 直近の通知以降にドロップされた累計（または増分）数。
        count: u64,
    },
    /// データ到着が途絶し、ストリームが失速したと判定された。
    StreamStalled,
    /// 失速後にデータ到着が復帰した。
    StreamRecovered,
    /// 必要な権限が拒否された。
    PermissionDenied,
    /// キャプチャデバイスが失われた（切断など）。
    DeviceLost,
    /// その他のバックエンドエラー（説明文付き）。
    Error(String),
}

/// flexaudio-core の操作で発生しうるエラー。
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// 引数が無効。
    #[error("invalid argument: {0}")]
    InvalidArg(String),
    /// 現在の状態では実行できない操作。
    #[error("invalid state: {0}")]
    InvalidState(String),
    /// 指定デバイスが見つからない。
    #[error("device not found")]
    DeviceNotFound,
    /// 権限が拒否された。
    #[error("permission denied")]
    PermissionDenied,
    /// 実行中の OS バージョンが当該機能を満たさない。
    #[error("unsupported OS version")]
    UnsupportedOsVersion,
    /// デバイスが実行中に失われた。
    #[error("device lost")]
    DeviceLost,
    /// バックエンド固有のエラー（説明文付き）。
    #[error("backend error: {0}")]
    Backend(String),
    /// この環境ではサポートされない操作。
    #[error("unsupported")]
    Unsupported,
}

/// flexaudio-core 全体で用いる結果型。
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_stream_config_matches_contract() {
        let c = StreamConfig::default();
        assert_eq!(c.chunk_ms, 20);
        assert_eq!(c.ring_capacity_chunks, 50);
        assert!(!c.exclude_self);
        assert_eq!(c.kind, SourceKind::Mic);
        assert_eq!(c.device_id, None);
        assert_eq!(c.target_pid, None);
    }

    #[test]
    fn chunk_flags_are_distinct_bits() {
        let all = ChunkFlags::DISCONTINUITY | ChunkFlags::RECOVERED | ChunkFlags::SILENCE;
        assert_eq!(all.bits(), 0b111);
        assert!(all.contains(ChunkFlags::SILENCE));
        assert_eq!(ChunkFlags::default(), ChunkFlags::empty());
    }
}
