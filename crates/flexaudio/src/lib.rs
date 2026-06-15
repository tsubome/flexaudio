//! flexaudio — 統合 facade: コア + OS バックエンド + mic を cfg で束ねる。
//!
//! [`Stream`] が 1 ソースのキャプチャパイプライン（backend → RawRing → 加工スレッド
//! → Normalizer → ChunkRing → poll + ウォッチドッグ復帰）を駆動する。

pub use flexaudio_core as core;

pub mod device_watcher;
pub mod mock;
pub mod stream;

pub use device_watcher::DeviceWatcher;
pub use mock::MockBackend;
pub use stream::Stream;

// facade トップから「`open()` と一緒に使う型」を一通り見えるようにする
// （`flexaudio::open` の利用側・napi バインディングが `flexaudio::core` を
// 経由せず `flexaudio::{StreamConfig, SourceKind, ...}` で揃えられるように）。
// 型の定義そのものは変えず、`flexaudio-core` の再エクスポートを facade へ橋渡しするだけ。
pub use flexaudio_core::backend::CaptureBackend;
pub use flexaudio_core::types::{
    AudioChunk, ChunkFlags, DeviceEvent, DeviceInfo, Error, Event, OutputFormat, Result,
    SourceKind, StreamConfig,
};

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

/// [`StreamConfig`] から **ソース種別 / OS に応じてバックエンドを選び**、
/// [`Stream`] を構築して返す高レベル入口（まだ start しない・二段方式）。
///
/// バックエンド選択を **この 1 箇所へ一元化**するための facade。利用側
/// （CLI・napi バインディング等）は backend を自前で構築せず、`StreamConfig`
/// を渡すだけでよい。低レベル入口 [`Stream::open`]（呼び元が
/// `Box<dyn CaptureBackend>` を渡す）は mock テスト・上級用途のためそのまま残る。
///
/// 戻った [`Stream`] は **まだキャプチャしていない**。消費側が
/// [`Stream::start`] を呼んでから [`Stream::poll_chunk`] / [`Stream::poll_event`]
/// を周期的に呼ぶ（open と start の二段）。
///
/// # ソース → バックエンドの分岐
/// - [`SourceKind::Mic`] → [`flexaudio_mic::CpalMicBackend`]（cpal, **全 OS**）。
/// - [`SourceKind::SystemLoopback`] → **Linux のみ**
///   [`flexaudio_os_linux::PwSystemBackend`]（既定出力の monitor / PipeWire）。
///   非 Linux では [`Error::Unsupported`]。
/// - [`SourceKind::ProcessLoopback`] → **Linux のみ**
///   [`flexaudio_os_linux::PwProcessBackend`]。`config.target_pid` が必須で、
///   無ければ [`Error::InvalidArg`]。`config.exclude_self` をそのまま渡す。
///   非 Linux では [`Error::Unsupported`]。
///
/// # エラー
/// - 出力フォーマットが非対応 → [`Error::UnsupportedFormat`]（早期に弾く）。
/// - ProcessLoopback で `target_pid` 欠落 → [`Error::InvalidArg`]。
/// - 当該 OS で非対応のソース（非 Linux の system/process）→ [`Error::Unsupported`]。
/// - その他は [`Stream::open`] 由来（`ring_capacity_chunks == 0` 等）。
///
/// # 例
/// ```no_run
/// use flexaudio::{open, StreamConfig, SourceKind};
///
/// let config = StreamConfig {
///     kind: SourceKind::Mic,
///     ..Default::default()
/// };
/// let mut stream = open(config)?;
/// stream.start()?;
/// while let Some(chunk) = stream.poll_chunk() {
///     // chunk.data は出力フォーマットの interleaved f32
///     let _ = chunk;
/// }
/// stream.stop();
/// # Ok::<(), flexaudio::Error>(())
/// ```
pub fn open(config: StreamConfig) -> Result<Stream> {
    use flexaudio_core::types::Error;

    // 出力フォーマットを早期に検証して分かりやすく弾く（Stream::open でも
    // 再検証されるが、backend を構築する前に返したい）。
    config.output.validate()?;

    // config.kind でソース別にバックエンドを構築する（選択の一元化）。
    let backend: Box<dyn CaptureBackend> = match config.kind {
        // マイク入力は全 OS 共通（cpal）。
        SourceKind::Mic => Box::new(flexaudio_mic::CpalMicBackend::new()),

        // システム出力ループバックは Linux のみ。
        SourceKind::SystemLoopback => {
            #[cfg(target_os = "linux")]
            {
                Box::new(flexaudio_os_linux::PwSystemBackend::new())
            }
            #[cfg(not(target_os = "linux"))]
            {
                return Err(Error::Unsupported);
            }
        }

        // プロセス出力ループバックは Linux のみ・target_pid 必須。
        SourceKind::ProcessLoopback => {
            #[cfg(target_os = "linux")]
            {
                let pid = config.target_pid.ok_or_else(|| {
                    Error::InvalidArg("ProcessLoopback には target_pid が必要".into())
                })?;
                Box::new(flexaudio_os_linux::PwProcessBackend::new(
                    pid,
                    config.exclude_self,
                ))
            }
            #[cfg(not(target_os = "linux"))]
            {
                return Err(Error::Unsupported);
            }
        }
    };

    // 低レベル入口へ委譲（Normalizer 構成・スレッド配線はここが担う）。
    Stream::open(config, backend)
}
