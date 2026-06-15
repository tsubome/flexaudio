//! flexaudio-os-linux — Linux バックエンド: PipeWire (`pipewire` 0.10)
//!
//! 「システム音声出力（既定 sink の monitor）」をキャプチャする
//! [`PwSystemBackend`] を提供する。WASAPI ループバックの Linux 相当であり、
//! スピーカーへ流れている音そのものを `Stream/Input/Audio` ストリームの
//! `stream.capture.sink=true` 経由で録る（§0.6「Linux 実装方針」）。
//!
//! # アーキテクチャ（`!Send` 回避）
//!
//! PipeWire の `MainLoop` / `Context` / `Core` / `Stream` はいずれも `!Send`
//! （内部に生ポインタと thread-local な loop を抱える）。一方コア契約
//! [`CaptureBackend`] は `Send` を要求する。そこで cpal / `MockBackend` と
//! 同型に「**専用スレッド 1 本の上で PipeWire を生成・実行・破棄まで完結**」
//! させ、[`PwSystemBackend`] が保持するのは `Send` なものだけ（停止用
//! [`pipewire::channel::Sender`]・スレッドの [`JoinHandle`]・起動結果受信用の
//! [`std::sync::mpsc`]）にする。`MainLoop` 等は決してスレッド境界を跨がない。
//!
//! # フォーマット
//!
//! ネイティブフォーマットは **48000 Hz / 2ch / f32** を「要求」する。PipeWire は
//! グラフのレート/チャンネルが異なっても `audioconvert` を自動挿入して変換して
//! くれるため、コア側で別途リサンプル/リミックスせずに済む（要求がそのまま
//! ネイティブとして扱える）。
//!
//! # 非 Linux
//!
//! このクレートは Linux 専用。`#![cfg(target_os = "linux")]` により非 Linux では
//! 空コンパイルになり、`pipewire` 依存も `Cargo.toml` の `target.'cfg(...linux)'`
//! セクションでのみ引かれる。

#![cfg(target_os = "linux")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use flexaudio_core::backend::{CaptureBackend, RawSink};
use flexaudio_core::clock::monotonic_now_ns;
use flexaudio_core::types::{Error, Result};

use pipewire as pw;
use pw::spa;
use pw::{properties::properties, stream::StreamFlags};
use spa::param::format::{MediaSubtype, MediaType};
use spa::param::format_utils;
use spa::pod::Pod;

/// ネイティブサンプルレート（Hz）。48kHz を要求し PipeWire に変換させる。
const NATIVE_RATE: u32 = 48_000;
/// ネイティブチャンネル数。ステレオを要求し PipeWire に変換させる。
const NATIVE_CHANNELS: u16 = 2;

/// PipeWire 経由でシステム音声出力（既定 sink の monitor）をキャプチャする
/// [`CaptureBackend`]。
///
/// 専用スレッド上で PipeWire `MainLoop` + 入力 `Stream` を構築し、`process`
/// コールバックで dequeue した interleaved f32 サンプルを [`RawSink::push`] へ
/// 非ブロッキングに流す。`stream.capture.sink=true` を指定しているため、対象は
/// 録音デバイスではなく「sink（スピーカー）の monitor」＝システム音声出力となる。
///
/// PipeWire/sink が存在しない環境（ヘッドレスサーバ等）でも **panic せず**
/// [`start`](CaptureBackend::start) が [`Error::Backend`] を返すだけに留める。
///
/// ```no_run
/// use flexaudio_os_linux::PwSystemBackend;
/// use flexaudio_core::backend::CaptureBackend;
///
/// let backend = PwSystemBackend::new();
/// assert_eq!(backend.native_format(), (48_000, 2));
/// // let mut backend = backend;
/// // backend.start(sink)?;   // PipeWire 不在/動作中 sink 無しなら Err(Backend)
/// // ...
/// // backend.stop();
/// ```
pub struct PwSystemBackend {
    /// 起動中フラグ（二重 start ガード／drop 判定用）。`Send`。
    running: Arc<AtomicBool>,
    /// PipeWire ループスレッドへ停止を伝える送信端。`start` で `Some`。
    ///
    /// 送ると、ループスレッドに attach 済みの受信端コールバックが
    /// `main_loop.quit()` を**ループスレッド自身から**呼び、`run()` を抜ける。
    stop_tx: Option<pw::channel::Sender<Terminate>>,
    /// PipeWire ループスレッドのハンドル。`start` で `Some`。
    handle: Option<JoinHandle<()>>,
}

/// ループスレッドへ送る停止メッセージ（ゼロサイズ）。
struct Terminate;

impl PwSystemBackend {
    /// 新しいバックエンドを構築する（この時点では PipeWire へ接続しない）。
    ///
    /// 実際の接続・ストリーム作成は [`start`](CaptureBackend::start) 内で
    /// 専用スレッド上で行われる。
    pub fn new() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            stop_tx: None,
            handle: None,
        }
    }
}

impl Default for PwSystemBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CaptureBackend for PwSystemBackend {
    fn native_format(&self) -> (u32, u16) {
        (NATIVE_RATE, NATIVE_CHANNELS)
    }

    fn start(&mut self, sink: RawSink) -> Result<()> {
        // 二重 start に安全（既に動作中なら何もしない）。
        if self.running.load(Ordering::SeqCst) {
            return Ok(());
        }

        // ループスレッドへの停止チャネル（pipewire 製。受信端は loop に attach する）。
        let (stop_tx, stop_rx) = pw::channel::channel::<Terminate>();
        // ループスレッドのセットアップ成否を start() へ同期返却するためのチャネル。
        // セットアップ（init→mainloop→context→connect→stream→connect）まで成功
        // したら Ok(()) を、途中で失敗したら Err(エラー文字列) を返す。
        let (ready_tx, ready_rx) = mpsc::channel::<std::result::Result<(), String>>();

        let running = self.running.clone();
        running.store(true, Ordering::SeqCst);

        let handle = thread::Builder::new()
            .name("flexaudio-pw-system".into())
            .spawn(move || {
                run_pw_loop(sink, stop_rx, &ready_tx);
            })
            .map_err(|e| Error::Backend(format!("spawn pipewire thread: {e}")))?;

        // セットアップ結果を待つ。スレッドが ready を送らずに終了した場合
        // （recv エラー）も失敗として扱う。
        match ready_rx.recv() {
            Ok(Ok(())) => {
                // セットアップ成功。停止用の送信端とハンドルを保持。
                self.stop_tx = Some(stop_tx);
                self.handle = Some(handle);
                Ok(())
            }
            Ok(Err(msg)) => {
                // セットアップ失敗（pipewire 不在・sink 無し・connect 失敗等）。
                // スレッドは既に return しているので join して片付ける。
                running.store(false, Ordering::SeqCst);
                let _ = handle.join();
                Err(Error::Backend(msg))
            }
            Err(_) => {
                // ready を一度も送らずスレッドが消えた（想定外パニック等）。
                running.store(false, Ordering::SeqCst);
                let _ = handle.join();
                Err(Error::Backend(
                    "pipewire setup thread terminated before signaling readiness".into(),
                ))
            }
        }
    }

    fn stop(&mut self) {
        // 二重 stop / 未 start に安全。
        if !self.running.swap(false, Ordering::SeqCst) {
            // running が false → 未起動 or 既に停止済み。念のため残骸を join。
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
            self.stop_tx = None;
            return;
        }

        // ループスレッドへ停止を通知（受信端コールバックが loop.quit() を呼ぶ）。
        // 送信端を drop する前に send。失敗（受信端消失）は無視（既に終わっている）。
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(Terminate);
        }

        // run() を抜けてスレッドが終了するのを待つ。スレッド終了時に Stream→Core→
        // Context→MainLoop が drop 順に破棄される（すべてループスレッド上で）。
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for PwSystemBackend {
    fn drop(&mut self) {
        self.stop();
    }
}

/// `process` コールバックと `param_changed` の間で共有する状態。
///
/// 確定したフォーマット（channels）を `process` から参照するために保持する。
struct UserData {
    /// PipeWire が確定したキャプチャフォーマット。`param_changed` で更新。
    format: spa::param::audio::AudioInfoRaw,
    /// 生フレームを流す先。`process` から `&mut` で push する。
    sink: RawSink,
}

/// PipeWire ループスレッド本体。
///
/// この関数の中だけで `MainLoop`/`Context`/`Core`/`Stream` を生成・実行・破棄する
/// （いずれも `!Send` なのでスレッド境界を跨がせない）。セットアップ完了/失敗を
/// `ready_tx` で呼び出し元へ返し、成功時は `main_loop.run()` で停止指示まで回る。
fn run_pw_loop(
    sink: RawSink,
    stop_rx: pw::channel::Receiver<Terminate>,
    ready_tx: &mpsc::Sender<std::result::Result<(), String>>,
) {
    // セットアップを別関数にまとめ、`?`/エラー文字列化を一箇所に集約する。
    // 戻り値はループ実行に必要な所有物（drop されないよう run 中保持する）。
    let (main_loop, _stream, _listener) = match setup_pw(sink) {
        Ok(t) => t,
        Err(msg) => {
            // セットアップ失敗を通知して終了（panic しない）。
            let _ = ready_tx.send(Err(msg));
            return;
        }
    };

    // 停止チャネルの受信端を loop に attach。Terminate 受信で quit()。
    // attach はこのローカル `main_loop` を借用するだけなので、戻り値の
    // AttachedReceiver はこのスタックフレーム内に閉じる（自己参照構造体にならず、
    // unsafe な寿命延長も不要）。quit() は loop 駆動のコールバック内、すなわち
    // **このスレッド上から**呼ばれる。
    let main_loop_for_quit = main_loop.clone();
    let _attached = stop_rx.attach(main_loop.loop_(), move |_terminate| {
        main_loop_for_quit.quit();
    });

    // セットアップ成功を通知。以後は run() がブロックする。
    if ready_tx.send(Ok(())).is_err() {
        // 呼び出し元が消えている（start が drop 済み等）。起動しない。
        return;
    }

    // 停止指示（Terminate）受信 or プロセス終了まで回る。
    main_loop.run();
    // ここを抜けると _attached → _listener → _stream → main_loop の順
    // （宣言の逆順）で drop され、PipeWire リソースがこのスレッド上で安全に破棄される。
}

/// PipeWire のセットアップ一式。失敗は `Err(String)` で返す（panic しない）。
///
/// 返すのは run 中ずっと生かしておく必要のあるハンドル群:
/// - `MainLoopRc`: `run()`/`quit()` の主体
/// - `StreamRc`: キャプチャストリーム本体
/// - `StreamListener`: コールバック登録。drop するとコールバックが外れる
///
/// 停止チャネルの loop への attach は呼び出し元（[`run_pw_loop`]）が行う。
/// そうすることで `AttachedReceiver` が返り値タプル（`MainLoopRc` を含む）を
/// 借用する自己参照構造体にならずに済む。
#[allow(clippy::type_complexity)]
fn setup_pw(
    sink: RawSink,
) -> std::result::Result<
    (
        pw::main_loop::MainLoopRc,
        pw::stream::StreamRc,
        pw::stream::StreamListener<UserData>,
    ),
    String,
> {
    // pw::init は何度呼んでもよい（内部で参照カウント）。
    pw::init();

    let main_loop = pw::main_loop::MainLoopRc::new(None)
        .map_err(|e| format!("create pipewire main loop failed: {e}"))?;
    let context = pw::context::ContextRc::new(&main_loop, None)
        .map_err(|e| format!("create pipewire context failed: {e}"))?;
    // 既定の PipeWire デーモンへ接続。デーモン不在ならここで Err。
    let core = context
        .connect_rc(None)
        .map_err(|e| format!("connect to pipewire daemon failed (is PipeWire running?): {e}"))?;

    // 入力（キャプチャ）ストリームのプロパティ。
    // - media.type=Audio / media.category=Capture: 音声キャプチャストリーム
    // - media.class=Stream/Input/Audio: グラフ上の役割（入力＝録る側）
    // - stream.capture.sink=true: 録音デバイスではなく sink の monitor を録る
    //   ＝「システム音声出力」を取得する核心の指定（§0.6）
    // - media.role: 既定 sink への autoconnect 用ヒント
    let mut props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_CLASS => "Stream/Input/Audio",
        *pw::keys::MEDIA_ROLE => "Music",
    };
    // monitor（sink の出力＝システム音声）を録る指定。
    props.insert(*pw::keys::STREAM_CAPTURE_SINK, "true");

    let stream = pw::stream::StreamRc::new(core, "flexaudio-system-capture", props)
        .map_err(|e| format!("create pipewire capture stream failed: {e}"))?;

    let user_data = UserData {
        format: spa::param::audio::AudioInfoRaw::new(),
        sink,
    };

    // コールバック登録。`param_changed` で確定 format を控え、`process` で
    // dequeue したバッファを RawSink へ流す。
    let listener = stream
        .add_local_listener_with_user_data(user_data)
        .param_changed(|_stream, user_data, id, param| {
            // NULL は format クリア。
            let Some(param) = param else {
                return;
            };
            if id != pw::spa::param::ParamType::Format.as_raw() {
                return;
            }
            let (media_type, media_subtype) = match format_utils::parse_format(param) {
                Ok(v) => v,
                Err(_) => return,
            };
            // raw audio のみ受理。
            if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
                return;
            }
            // 確定フォーマットを控える（process でチャンネル数として使う）。
            if user_data.format.parse(param).is_err() {
                // パース失敗時は更新しない（直前の値を保持）。
            }
        })
        .process(|stream, user_data| {
            // RT スレッドで呼ばれる。ブロック禁止・確保禁止が望ましい。
            // バッファが無ければ何もしない（panic しない）。
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }
            let data = &mut datas[0];
            // 有効バイト数とオフセット（リング上の位置）を控えてから data() を借りる。
            let chunk = data.chunk();
            let size = chunk.size() as usize;
            let offset = chunk.offset() as usize;
            if size == 0 {
                return;
            }
            let Some(bytes) = data.data() else {
                return;
            };
            // [offset, offset+size) が有効領域。範囲外は弾く（防御的）。
            let end = offset.saturating_add(size);
            if end > bytes.len() {
                return;
            }
            let valid = &bytes[offset..end];
            // f32 の倍数だけ取り出す（端数バイトは無視）。
            let n_floats = valid.len() / std::mem::size_of::<f32>();
            if n_floats == 0 {
                return;
            }
            // バイト列を f32 interleaved として読む。`data` のアライメントは
            // 保証されないため、align_to ではなく from_le_bytes で安全に読む。
            // スタック上の小バッファに詰めてから 1 回で push する（RawSink::push は
            // 非ブロッキングで満杯時 DROP）。
            //
            // 典型ブロックは ~数百〜数千 frames。スレッドローカルな再利用バッファ
            // を使い、process ごとの確保を避ける。
            PROC_SCRATCH.with(|cell| {
                let mut scratch = cell.borrow_mut();
                scratch.clear();
                scratch.reserve(n_floats);
                for i in 0..n_floats {
                    let b = i * 4;
                    let v = f32::from_le_bytes([
                        valid[b],
                        valid[b + 1],
                        valid[b + 2],
                        valid[b + 3],
                    ]);
                    scratch.push(v);
                }
                // PTS は将来 pw_buffer.time の device クロックを使う（TODO）。
                // 現状は到着時刻の単調クロックで代用（§clock の ClockNormalizer が
                // 初回原点を取るため、単調近似でも下流は破綻しない）。
                user_data.sink.push(&scratch, monotonic_now_ns());
            });
        })
        .register()
        .map_err(|e| format!("register pipewire stream listener failed: {e}"))?;

    // 要求フォーマット param: f32 / 48000 / 2ch。
    // rate/channels を明示するので、グラフが異なれば PipeWire が audioconvert を
    // 自動挿入して 48k/stereo/f32 に変換してくれる（§0.6）。
    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
    audio_info.set_rate(NATIVE_RATE);
    audio_info.set_channels(NATIVE_CHANNELS as u32);

    let obj = pw::spa::pod::Object {
        type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: pw::spa::param::ParamType::EnumFormat.as_raw(),
        properties: audio_info.into(),
    };
    let values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(obj),
    )
    .map_err(|e| format!("serialize audio format pod failed: {e}"))?
    .0
    .into_inner();

    let pod = Pod::from_bytes(&values)
        .ok_or_else(|| "build audio format pod from bytes failed".to_string())?;
    let mut params = [pod];

    // 入力方向で connect。AUTOCONNECT で既定ターゲット（既定 sink の monitor）へ。
    // MAP_BUFFERS でバッファを直接読めるようにし、RT_PROCESS で process を RT 実行。
    stream
        .connect(
            spa::utils::Direction::Input,
            None,
            StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS | StreamFlags::RT_PROCESS,
            &mut params,
        )
        .map_err(|e| format!("connect pipewire capture stream failed: {e}"))?;

    Ok((main_loop, stream, listener))
}

thread_local! {
    /// `process` コールバックの f32 詰め替え用スクラッチ（確保回避）。
    static PROC_SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use flexaudio_core::raw_ring::raw_ring;

    /// [`CaptureBackend`] 契約どおり `PwSystemBackend: Send` であること
    /// （PipeWire の `!Send` を専用スレッドへ閉じ込められている証左）。
    /// コンパイルが通れば成立。
    #[test]
    fn backend_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<PwSystemBackend>();
    }

    /// 構築直後にネイティブフォーマットが固定契約どおり (48000, 2) であること。
    #[test]
    fn native_format_is_48k_stereo() {
        let be = PwSystemBackend::new();
        assert_eq!(be.native_format(), (NATIVE_RATE, NATIVE_CHANNELS));
        assert_eq!(be.native_format(), (48_000, 2));
    }

    /// 未 start での stop / 二重 stop が安全（panic しない）。
    #[test]
    fn stop_without_start_is_safe() {
        let mut be = PwSystemBackend::new();
        be.stop();
        be.stop();
    }

    /// スモークテスト: `start` は PipeWire/sink が無い homelab では
    /// `Err(Error::Backend)` になり得るが、**panic はしない**。
    /// Ok（PipeWire と動作中 sink がある環境）と Err(Backend) の両方を許容する。
    ///
    /// PipeWire の動くデスクトップ/ラップトップでは Ok になり、`stop()` まで
    /// 一巡できる。実際の音声 end-to-end 検証は下の `#[ignore]` テスト参照。
    #[test]
    fn start_is_graceful_without_pipewire() {
        let (prod, _cons) = raw_ring(1 << 16);
        let sink = RawSink::new(prod, NATIVE_RATE, NATIVE_CHANNELS);
        let mut be = PwSystemBackend::new();
        match be.start(sink) {
            Ok(()) => {
                // 動作中 PipeWire/sink がある環境。停止まで一巡できること。
                be.stop();
            }
            Err(Error::Backend(_)) => {
                // PipeWire 不在/sink 無し: 想定内。panic していないことが要点。
            }
            Err(other) => panic!("unexpected error variant: {other:?}"),
        }
    }

    /// 実キャプチャ end-to-end（PipeWire が動くデスクトップ/ラップトップでのみ）。
    ///
    /// 実行方法（ラップトップ等、PipeWire + 何か音を鳴らした状態で）:
    /// ```text
    /// cargo test -p flexaudio-os-linux -- --ignored capture_smoke
    /// ```
    /// 既定 sink の monitor を一定時間録り、サンプルが流れてくる
    /// （overflow か pop で観測）ことを期待する。homelab/CI では音源も
    /// PipeWire も無いため `#[ignore]`。
    #[test]
    #[ignore = "requires a running PipeWire session with audio playing (desktop/laptop)"]
    fn capture_smoke() {
        use std::time::Duration;
        let (prod, mut cons) = raw_ring(1 << 18);
        let sink = RawSink::new(prod, NATIVE_RATE, NATIVE_CHANNELS);
        let mut be = PwSystemBackend::new();
        be.start(sink).expect("start should succeed on a PipeWire desktop");
        // 録音が回るのを少し待つ。
        thread::sleep(Duration::from_millis(500));
        be.stop();
        // 何らかのサンプルが届いている（無音 sink でも 0.0 サンプルは流れる）。
        let mut out = vec![0.0f32; 1920];
        let got = cons.pop_slice(&mut out);
        assert!(got > 0, "expected captured samples from the default sink monitor");
    }
}
