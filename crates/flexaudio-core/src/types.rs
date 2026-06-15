//! 固定契約の型定義: 内部正規形の定数 / [`AudioChunk`] / [`ChunkFlags`] /
//! [`SourceKind`] / [`OutputFormat`] / [`StreamConfig`] / [`Event`] / [`Error`]。
//!
//! 内部正規形（逸脱不可）:
//! interleaved `f32` / 48000 Hz / ステレオ 2ch / 20ms = 960 frames per chunk。
//!
//! **出力フォーマット**は [`OutputFormat`] で指定でき（既定 `{48000, 2}`）、
//! Normalizer 第 2 段が内部正規形からそのレート/チャンネルへ再変換する。
//! 出力チャンクは常に**時間ベース 20ms 固定**なので、レートに応じて
//! [`AudioChunk::frames`] が変わる（48k=960 / 16k=320 / 8k=160）。
//! 既定 `{48000, 2}` のときは第 2 段が無変換パススルーになり、内部正規形が
//! そのまま出力される（回帰ゼロ）。

use bitflags::bitflags;

/// 内部正規形のサンプルレート（Hz）。全ストリームは一旦このレートへ正規化される。
pub const SAMPLE_RATE: u32 = 48_000;

/// 内部正規形のチャンネル数。常にステレオ（2ch interleaved）。
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
/// `data` は interleaved `f32`（出力チャンネル順）で長さは厳密に
/// `frames * output.channels as usize`。チャンクは**時間ベース 20ms 固定**なので、
/// 出力レートに応じて `frames` が変わる（48k=960 / 16k=320 / 8k=160）。既定の
/// 出力 `{48000, 2}` では `frames == 960`（1920 サンプル）で従来どおり。
#[derive(Debug, Clone, PartialEq)]
pub struct AudioChunk {
    /// interleaved `f32` サンプル。長さ = `frames * output.channels`。
    pub data: Vec<f32>,
    /// チャンク内のフレーム数（1 フレーム = 全出力チャンネル 1 サンプル組）。
    pub frames: usize,
    /// 先頭サンプルの正規化済み単調プレゼンテーションタイムスタンプ（ns）。
    pub pts_ns: i64,
    /// ストリーム層が単調増加で付与するシーケンス番号。
    pub seq: u64,
    /// このチャンクの状態フラグ。
    pub flags: ChunkFlags,
    /// このチャンクが届くまでに（直前に）ドロップされたチャンク数。
    pub dropped_before: u32,
    /// このチャンクの最終 `data`（出力フォーマット）における全サンプル絶対値の最大。
    /// 線形振幅（通常 `0.0..=1.0`）。
    pub peak: f32,
    /// このチャンクの最終 `data`（出力フォーマット）における二乗平均平方根（線形）。
    pub rms: f32,
}

/// `devices()`（統一デバイス列挙）が 1 デバイスにつき返す情報。
///
/// 全 OS バックエンド共通の正規形。マイク入力（[`SourceKind::Mic`]）と
/// システム音声出力（[`SourceKind::SystemLoopback`]）を 1 つのリストへ統合して返す。
///
/// # 安定 ID（M-5）
/// `id` は「再接続で index が変わる」問題を避けるため、取得できる範囲で**最も安定な
/// キー**を採る:
/// - cpal（マイク, 全 OS）: cpal は永続 ID を持たないため **デバイス名**を id にする。
/// - PipeWire（Linux）: 永続的な **`node.name`** を id にする（`node.description` は
///   表示名として `name` に使う）。
///
/// 同一マシン・同一構成で繰り返し列挙すれば同じ `id` が得られる安定性を意図する
/// （別マシンや OS をまたいだ大域一意性は保証しない）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// 安定 ID。[`StreamConfig::device_id`] に渡せるキー（cpal=デバイス名 /
    /// PipeWire=`node.name`）。
    pub id: String,
    /// 人間向け表示名（PipeWire は `node.description` 優先、無ければ `node.name`）。
    pub name: String,
    /// このデバイスをキャプチャするときのソース種別。
    pub source_kind: SourceKind,
    /// デバイスのネイティブ（既定）サンプルレート（Hz）。不明時は妥当な既定値。
    pub sample_rate: u32,
    /// デバイスのネイティブ（既定）チャンネル数。不明時は妥当な既定値。
    pub channels: u16,
    /// ループバック（システム出力の monitor）なら `true`、録音デバイス（マイク）なら
    /// `false`。
    pub is_loopback: bool,
    /// OS の既定デバイス（既定入力 / 既定出力 sink）なら `true`。
    pub is_default: bool,
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

/// 出力チャンクのフォーマット（サンプルレートとチャンネル数）。
///
/// Normalizer 第 2 段が内部正規形 48k/stereo からこのフォーマットへ再変換する。
/// 既定は内部正規形と同じ `{sample_rate: 48000, channels: 2}` で、その場合
/// 第 2 段は無変換パススルーになる（回帰ゼロ）。
///
/// - `sample_rate`: ダウン/アップサンプル先（rubato でアンチエイリアス込み）。
/// - `channels`: 1（mono）または 2（stereo）。stereo→mono は L/R 平均、
///   mono→stereo は複製。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputFormat {
    /// 出力サンプルレート（Hz）。
    pub sample_rate: u32,
    /// 出力チャンネル数（1 = mono / 2 = stereo）。
    pub channels: u16,
}

impl OutputFormat {
    /// MVP で扱える出力レートの下限/上限（極端な値・0 を弾く防御範囲）。
    const MIN_RATE: u32 = 4_000;
    const MAX_RATE: u32 = 384_000;

    /// 構成が妥当か検証する。非対応なら [`Error::UnsupportedFormat`]。
    ///
    /// - `channels` は 1 または 2 のみ（MVP）。
    /// - `sample_rate` は `MIN_RATE..=MAX_RATE`。
    pub fn validate(&self) -> Result<()> {
        if self.channels != 1 && self.channels != 2 {
            return Err(Error::UnsupportedFormat(format!(
                "output channels must be 1 or 2, got {}",
                self.channels
            )));
        }
        if self.sample_rate < Self::MIN_RATE || self.sample_rate > Self::MAX_RATE {
            return Err(Error::UnsupportedFormat(format!(
                "output sample_rate {} Hz out of supported range {}..={}",
                self.sample_rate,
                Self::MIN_RATE,
                Self::MAX_RATE
            )));
        }
        Ok(())
    }

    /// 出力レートにおける 20ms チャンクのフレーム数（`sample_rate * 20 / 1000`）。
    /// 48k=960 / 16k=320 / 8k=160。
    pub fn chunk_frames(&self) -> usize {
        (self.sample_rate as usize * 20) / 1000
    }
}

impl Default for OutputFormat {
    fn default() -> Self {
        // 既定は内部正規形と同一 → 第 2 段パススルー（回帰ゼロ）。
        Self {
            sample_rate: SAMPLE_RATE,
            channels: CHANNELS,
        }
    }
}

/// 1 ストリームを開くための構成。
///
/// [`Default`] は固定契約に沿った値（`chunk_ms = 20`,
/// `ring_capacity_chunks = 50`, `exclude_self = false`, `kind = Mic`,
/// `output = {48000, 2}`）を返す。
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
    /// 出力チャンクのフォーマット。既定 `{48000, 2}`（パススルー）。
    pub output: OutputFormat,
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
            output: OutputFormat::default(),
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
    /// 要求された出力フォーマット（レート/チャンネル）が非対応。
    #[error("unsupported output format: {0}")]
    UnsupportedFormat(String),
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
        // 既定の出力は内部正規形と同一（第 2 段パススルー）。
        assert_eq!(c.output.sample_rate, SAMPLE_RATE);
        assert_eq!(c.output.channels, CHANNELS);
        assert_eq!(c.output, OutputFormat::default());
    }

    #[test]
    fn output_format_chunk_frames_are_time_based() {
        assert_eq!(
            OutputFormat {
                sample_rate: 48_000,
                channels: 2
            }
            .chunk_frames(),
            960
        );
        assert_eq!(
            OutputFormat {
                sample_rate: 16_000,
                channels: 1
            }
            .chunk_frames(),
            320
        );
        assert_eq!(
            OutputFormat {
                sample_rate: 8_000,
                channels: 2
            }
            .chunk_frames(),
            160
        );
    }

    #[test]
    fn output_format_validation_rejects_bad_configs() {
        // ch=0 / ch=3 は非対応。
        assert!(OutputFormat {
            sample_rate: 48_000,
            channels: 0
        }
        .validate()
        .is_err());
        assert!(OutputFormat {
            sample_rate: 48_000,
            channels: 3
        }
        .validate()
        .is_err());
        // 極端なレートは非対応。
        assert!(OutputFormat {
            sample_rate: 100,
            channels: 1
        }
        .validate()
        .is_err());
        assert!(OutputFormat {
            sample_rate: 1_000_000,
            channels: 2
        }
        .validate()
        .is_err());
        // 妥当な構成は OK。
        assert!(OutputFormat {
            sample_rate: 16_000,
            channels: 1
        }
        .validate()
        .is_ok());
        assert!(OutputFormat::default().validate().is_ok());
    }

    #[test]
    fn device_info_builds_and_clones() {
        let mic = DeviceInfo {
            id: "alsa_input.pci-0000_00_1f.3".into(),
            name: "内蔵マイク".into(),
            source_kind: SourceKind::Mic,
            sample_rate: 48_000,
            channels: 2,
            is_loopback: false,
            is_default: true,
        };
        // Clone / PartialEq が機能すること（列挙結果の比較・複製に使う）。
        assert_eq!(mic, mic.clone());
        assert!(!mic.is_loopback);
        assert!(mic.is_default);
        assert_eq!(mic.source_kind, SourceKind::Mic);

        let sys = DeviceInfo {
            source_kind: SourceKind::SystemLoopback,
            is_loopback: true,
            is_default: false,
            ..mic.clone()
        };
        assert!(sys.is_loopback);
        assert_ne!(mic, sys);
    }

    #[test]
    fn chunk_flags_are_distinct_bits() {
        let all = ChunkFlags::DISCONTINUITY | ChunkFlags::RECOVERED | ChunkFlags::SILENCE;
        assert_eq!(all.bits(), 0b111);
        assert!(all.contains(ChunkFlags::SILENCE));
        assert_eq!(ChunkFlags::default(), ChunkFlags::empty());
    }
}
