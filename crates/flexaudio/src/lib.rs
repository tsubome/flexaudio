//! flexaudio — 統合 facade: コア + OS バックエンド + mic を cfg で束ねる。
//!
//! [`Stream`] が 1 ソースのキャプチャパイプライン（backend → RawRing → 加工スレッド
//! → Normalizer → ChunkRing → poll + ウォッチドッグ復帰）を駆動する。

pub use flexaudio_core as core;

pub mod mock;
pub mod stream;

pub use mock::MockBackend;
pub use stream::Stream;
