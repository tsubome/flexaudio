//! flexaudio-os-windows — Windows バックエンド: WASAPI ループバック / プロセス
//! ループバック（windows-rs 0.54, Win11+）。
//!
//! 2 つの [`CaptureBackend`](flexaudio_core::backend::CaptureBackend) を提供する:
//!
//! - [`WasapiSystemBackend`] — 既定 render endpoint の **古典 loopback**
//!   （`AUDCLNT_STREAMFLAGS_LOOPBACK`）でシステム音声出力全体（スピーカーへ流れている
//!   ミックス）を録る。Linux の [`PwSystemBackend`](../flexaudio_os_linux) 相当。
//! - [`WasapiProcessBackend`] — `ActivateAudioInterfaceAsync` + プロセスループバック
//!   （`AUDIOCLIENT_ACTIVATION_PARAMS`）で**特定 PID（そのプロセスツリー）**の音声を
//!   録る。`exclude_self` で「対象ツリーを除く全システム音」へ反転する。
//!
//! # `!Send` 回避（cpal / PipeWire backend と同型）
//!
//! WASAPI の `IAudioClient` 等の COM インターフェイスは `!Send`。一方コア契約
//! [`CaptureBackend`] は `Send` を要求する。そこで「**専用スレッド 1 本の上で COM
//! 初期化〜キャプチャ〜破棄まで完結**」させ、バックエンド構造体が保持するのは `Send`
//! なものだけ（停止フラグ [`AtomicBool`] / [`JoinHandle`] / キャッシュ済み
//! フォーマット）にする。COM インターフェイスは決してスレッド境界を跨がない。
//!
//! # 非 Windows
//!
//! `#![cfg(target_os = "windows")]` により非 Windows では空コンパイルになり、
//! `windows` 依存も `Cargo.toml` の `target.'cfg(...windows)'` セクションでのみ引かれる。

#![cfg(target_os = "windows")]

mod common;
mod process;
mod system;

pub use process::WasapiProcessBackend;
pub use system::WasapiSystemBackend;
