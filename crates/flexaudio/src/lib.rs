//! flexaudio — 統合 facade: コア + OS バックエンド + mic を cfg で束ねる。
//!
//! [`Stream`] が 1 ソースのキャプチャパイプライン（backend → RawRing → 加工スレッド
//! → Normalizer → ChunkRing → poll + ウォッチドッグ復帰）を駆動する。

pub use flexaudio_core as core;

pub mod device_watcher;
pub mod mock;
pub mod stream;

pub use device_watcher::DeviceWatcher;
pub use flexaudio_core::types::DeviceEvent;
pub use mock::MockBackend;
pub use stream::Stream;

use flexaudio_core::types::{DeviceInfo, Result};

/// 全ソース統合のデバイス列挙（§0.8 `devices()`）。
///
/// 利用可能なオーディオデバイスを **1 つのリスト**にまとめて返す:
/// - **マイク入力**（[`core::SourceKind::Mic`], `is_loopback = false`）—
///   [`flexaudio_mic::list_devices`] 経由（cpal, 全 OS）。
/// - **システム音声出力**（[`core::SourceKind::SystemLoopback`],
///   `is_loopback = true`）— OS 別バックエンド経由（Linux: PipeWire の Audio/Sink。
///   Linux では PipeWire が Audio/Source（マイク）も列挙するので、cpal 分と重複し得る）。
///
/// 各 [`DeviceInfo`] の `id` は取得できる範囲で最も安定なキー（cpal=デバイス名 /
/// PipeWire=`node.name`。M-5）。`is_default` は OS の既定デバイスに付く。
///
/// # OS 分岐
/// - **Linux**: cpal（マイク）+ PipeWire（sink + source）を結合。PipeWire セッションが
///   無ければ PipeWire 分は空になり、cpal 分のみ返る。
/// - **その他 OS**: 現状は cpal（マイク）のみ（システム出力バックエンドは未配線）。
///
/// デバイスが無い／列挙に失敗した環境でも **panic せず**、取得できた範囲のリスト
/// （しばしば空）を返す。
pub fn devices() -> Result<Vec<DeviceInfo>> {
    // マイク入力（cpal）は全 OS 共通。
    let mut all = flexaudio_mic::list_devices()?;

    // システム出力 sink（+ PipeWire 側マイク）は OS 別。
    #[cfg(target_os = "linux")]
    {
        let linux = flexaudio_os_linux::list_devices()?;
        all.extend(linux);
    }

    Ok(all)
}

/// デバイスの着脱・既定変更（ホットプラグ）を監視する [`DeviceWatcher`] を開始する。
///
/// 返ったウォッチャの [`DeviceWatcher::poll_event`] を周期的に呼ぶと、デバイスの
/// 接続・切断・既定変更が [`DeviceEvent`] として **pull 型**で取り出せる。capture
/// stream 単位の [`core::Event`] とは別系統で、デバイス単位の事象を扱う。
///
/// # OS 分岐 / 縮退
/// - **Linux**: PipeWire レジストリを永続監視する（`flexaudio-os-linux`）。
///   PipeWire デーモン不在・接続失敗時は **[`NoopWatcher`](device_watcher) へ縮退**
///   して `Ok` を返す（着脱が来ないだけ。`devices()` がデーモン不在を空リストに
///   握るのと一貫）。
/// - **その他 OS**: 常に no-op（着脱は配信されない）。
///
/// したがって本関数は実用上 `Ok` を返す（PipeWire 不在でも panic せず縮退）。
pub fn watch_devices() -> Result<DeviceWatcher> {
    device_watcher::watch_devices()
}
