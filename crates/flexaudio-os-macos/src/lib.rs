//! flexaudio-os-macos — macOS バックエンド: Core Audio Process Taps
//! (objc2-core-audio, macOS 14.4+)。
//!
//! システム音声出力全体（[`MacSystemBackend`]）と特定プロセス（[`MacProcessBackend`]）を
//! **CoreAudio Process Tap** で録る。Windows の WASAPI loopback / Linux の PipeWire
//! monitor に相当する。
//!
//! # アーキテクチャ
//! tap チェーン（`CATapDescription` → process tap → private aggregate device →
//! IOProc block → start）は [`tap`] モジュールに集約し、両バックエンドは INCLUDE/EXCLUDE
//! の `TapKind` を切り替えて同一チェーンを回す。`!Send` な ObjC オブジェクト
//! （`Retained<CATapDescription>` / `RcBlock` / `TapChain`）はバックエンドの専用スレッド内に
//! 閉じ込め、`Send` な本体（停止フラグ・`JoinHandle`・フォーマット）だけがスレッドを跨ぐ
//! （cpal / Windows / Linux バックエンドと同型）。
//!
//! # 権限（TCC）
//! システム/プロセス音声キャプチャは TCC の `kTCCServiceAudioCapture` を要求する
//! （Info.plist の `NSAudioCaptureUsageDescription`）。本クレートは **private TCC SPI を
//! 一切使わず**（オフライン契約）、権限の可否は初回キャプチャ時の OS プロンプトに委ねる。
//! tap 作成が未承認で弾かれた場合は [`map_os_status`](common::map_os_status) が
//! 権限拒否系 OSStatus を [`Error::PermissionDenied`](flexaudio_core::types::Error) へ寄せる。
//!
//! # 非 macOS
//! このクレートは macOS 専用。`#![cfg(target_os = "macos")]` により非 macOS では
//! 空コンパイルになり、objc2 系依存も `Cargo.toml` の `target.'cfg(...macos)'` セクション
//! でのみ引かれる（Linux/Windows ビルドは無傷）。

#![cfg(target_os = "macos")]

mod common;
mod process;
mod system;
mod tap;
mod version;

pub use process::MacProcessBackend;
pub use system::MacSystemBackend;
