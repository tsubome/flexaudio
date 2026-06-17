//! [`WasapiSystemBackend`] — 既定 render endpoint の古典 WASAPI loopback。
//!
//! 既定スピーカー（render endpoint）へ流れているミックスを
//! `AUDCLNT_STREAMFLAGS_LOOPBACK` で録る。Linux の
//! [`PwSystemBackend`](../flexaudio_os_linux) 相当。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use flexaudio_core::backend::{CaptureBackend, RawSink};
use flexaudio_core::types::{Error, Result};

use windows::core::Interface;
use windows::Win32::Media::Audio::{
    eConsole, eRender, IAudioClient, IMMDevice, IMMDeviceEnumerator, MMDeviceEnumerator,
    WAVEFORMATEX,
};
use windows::Win32::System::Com::{CoCreateInstance, CoTaskMemFree, CLSCTX_ALL};

use crate::common::{
    capture_loop, init_loopback_capture, map_hr, parse_mix_format, ComThread,
};

/// 既定 render endpoint を取得できなかったときに [`native_format`] が返す
/// 無難なフォールバック `(48000, 2)`（panic しない）。実際の `start` で
/// 取得失敗すれば [`Error`] を返す。
const FALLBACK_FORMAT: (u32, u16) = (48_000, 2);

/// 既定 render endpoint の古典 loopback でシステム音声出力をキャプチャする
/// [`CaptureBackend`]。
///
/// 専用スレッド上で COM を初期化し、`MMDeviceEnumerator` →
/// `GetDefaultAudioEndpoint(eRender, eConsole)` → `IAudioClient` を取得して
/// `AUDCLNT_STREAMFLAGS_LOOPBACK` で Initialize し、イベント駆動でパケットを
/// [`RawSink::push`] へ流す。
///
/// この型は `Send`（保持するのは停止フラグ・[`JoinHandle`]・キャッシュ済み
/// フォーマットのみ。`!Send` な COM インターフェイスは専用スレッド内に閉じ込める）。
pub struct WasapiSystemBackend {
    /// 起動中フラグ（二重 start ガード／停止指示／drop 判定）。`Send`。
    stop_flag: Arc<AtomicBool>,
    /// COM/キャプチャを所有するスレッドのハンドル（start 後に `Some`）。
    handle: Option<JoinHandle<()>>,
    /// `new` 時に問い合わせてキャッシュしたネイティブフォーマット。
    native: (u32, u16),
}

impl WasapiSystemBackend {
    /// 新しいシステム loopback バックエンドを構築する。
    ///
    /// 構築時に既定 render endpoint の MixFormat を一度問い合わせてキャッシュする。
    /// 取得失敗時は [`FALLBACK_FORMAT`]（`(48000, 2)`）をキャッシュする（panic しない）。
    pub fn new() -> Self {
        let native = query_native_format().unwrap_or(FALLBACK_FORMAT);
        Self {
            stop_flag: Arc::new(AtomicBool::new(false)),
            handle: None,
            native,
        }
    }
}

impl Default for WasapiSystemBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// 既定 render endpoint の MixFormat から `(rate, channels)` を取得する。
/// 取得できなければ `None`（panic しない）。一時的に COM を初期化して問い合わせる。
fn query_native_format() -> Option<(u32, u16)> {
    let _com = ComThread::new();
    unsafe {
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).ok()?;
        let device: IMMDevice = enumerator.GetDefaultAudioEndpoint(eRender, eConsole).ok()?;
        let client: IAudioClient = device.Activate(CLSCTX_ALL, None).ok()?;
        let pwfx = client.GetMixFormat().ok()?;
        if pwfx.is_null() {
            return None;
        }
        let rate = (*pwfx).nSamplesPerSec;
        let channels = (*pwfx).nChannels;
        CoTaskMemFree(Some(pwfx as *const _ as *const _));
        Some((rate, channels))
    }
}

impl CaptureBackend for WasapiSystemBackend {
    fn native_format(&self) -> (u32, u16) {
        self.native
    }

    fn start(&mut self, sink: RawSink) -> Result<()> {
        // 二重 start に安全: 既にスレッドが生きていれば何もしない。
        if self.handle.is_some() {
            return Ok(());
        }
        // 前回 stop 後でも再 start できるようフラグをリセット。
        self.stop_flag.store(false, Ordering::SeqCst);

        let stop_flag = self.stop_flag.clone();
        // setup（COM init〜Initialize〜Start 直前）の成否を同期返却するチャネル。
        let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();

        let handle = thread::Builder::new()
            .name("flexaudio-wasapi-system".into())
            .spawn(move || {
                run_system_thread(sink, stop_flag, ready_tx);
            })
            .map_err(|e| Error::Backend(format!("spawn wasapi system thread: {e}")))?;

        match ready_rx.recv() {
            Ok(Ok(())) => {
                self.handle = Some(handle);
                Ok(())
            }
            Ok(Err(e)) => {
                // setup 失敗。スレッドは ready 送信後すぐ終了するので join。
                self.stop_flag.store(false, Ordering::SeqCst);
                let _ = handle.join();
                Err(e)
            }
            Err(_) => {
                self.stop_flag.store(false, Ordering::SeqCst);
                let _ = handle.join();
                Err(Error::Backend(
                    "wasapi system thread exited before reporting readiness".into(),
                ))
            }
        }
    }

    fn stop(&mut self) {
        // 再入・二重 stop に安全。
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for WasapiSystemBackend {
    fn drop(&mut self) {
        self.stop();
    }
}

/// 所有スレッド本体。COM を初期化し、既定 render endpoint の loopback を構成して
/// キャプチャループを回す。setup の成否を `ready_tx` で [`WasapiSystemBackend::start`]
/// へ報告する。COM ガード（`_com`）はこの関数のスコープに閉じ、スレッド終了時に
/// `CoUninitialize` される（COM オブジェクトより後に drop される宣言順）。
fn run_system_thread(sink: RawSink, stop_flag: Arc<AtomicBool>, ready_tx: mpsc::Sender<Result<()>>) {
    // COM をこのスレッドで初期化（drop で uninit）。最初に宣言＝最後に drop。
    let _com = ComThread::new();

    // setup を行い、Initialize 済み client / capture / event / channels を得る。
    let setup = unsafe { setup_system_loopback(&sink) };
    let (client, capture, event, channels) = match setup {
        Ok(t) => t,
        Err(e) => {
            let _ = ready_tx.send(Err(e));
            return;
        }
    };

    // setup 成功を報告。以後はキャプチャループ（内部で client.Start()）。
    if ready_tx.send(Ok(())).is_err() {
        // 呼び出し元が消えている。Start せず戻る（COM は drop で片付く）。
        return;
    }

    unsafe { capture_loop(&client, &capture, event, channels, sink, &stop_flag) };
    // capture_loop が client.Stop() と CloseHandle(event) を行う。
    // ここを抜けると capture → client → _com の順で drop（宣言の逆順）。
}

/// 既定 render endpoint の古典 loopback をセットアップし、Initialize 済みの
/// `IAudioClient` / `IAudioCaptureClient` / イベントハンドル / チャンネル数を返す。
///
/// `sink` はチャンネル数のチェック（native と一致確認）には使わず、参照のみ受け取る
/// （所有はループへ渡す呼び出し側が持つ）。
///
/// # Safety
/// 呼び出しスレッドで COM が初期化済みであること。
#[allow(clippy::type_complexity)]
unsafe fn setup_system_loopback(
    _sink: &RawSink,
) -> Result<(
    IAudioClient,
    windows::Win32::Media::Audio::IAudioCaptureClient,
    windows::Win32::Foundation::HANDLE,
    u16,
)> {
    let enumerator: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
        .map_err(|e| map_hr("CoCreateInstance(MMDeviceEnumerator)", e))?;

    // 既定の render endpoint（loopback は render を録る）。無ければ DeviceNotFound。
    let device: IMMDevice = enumerator
        .GetDefaultAudioEndpoint(eRender, eConsole)
        .map_err(|_e| Error::DeviceNotFound)?;

    let client: IAudioClient = device
        .Activate(CLSCTX_ALL, None)
        .map_err(|e| map_hr("IMMDevice::Activate(IAudioClient)", e))?;

    // 共有 MixFormat（使用後 CoTaskMemFree 必須）。
    let pwfx: *mut WAVEFORMATEX = client
        .GetMixFormat()
        .map_err(|e| map_hr("IAudioClient::GetMixFormat", e))?;

    // フォーマット判定（IEEE float のみ MVP 直結）。rate/channels はここで控える。
    let parsed = parse_mix_format(pwfx as *const WAVEFORMATEX);
    let (_rate, channels) = match parsed {
        Ok(v) => v,
        Err(e) => {
            CoTaskMemFree(Some(pwfx as *const _ as *const _));
            return Err(e);
        }
    };

    // Initialize（LOOPBACK|EVENTCALLBACK）→ event → capture サービス。
    let init = init_loopback_capture(&client, pwfx as *const WAVEFORMATEX);
    // pwfx は Initialize がフォーマットをコピーするので、ここで解放してよい。
    CoTaskMemFree(Some(pwfx as *const _ as *const _));
    let (capture, event) = init?;

    // 念のため Interface が生きていることを保証（使わないが drop 順の明示）。
    let _ = client.as_raw();

    Ok((client, capture, event, channels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flexaudio_core::raw_ring;

    /// `new` + `native_format` が panic しないこと（render endpoint 有無を問わず）。
    #[test]
    fn new_and_native_format_do_not_panic() {
        let backend = WasapiSystemBackend::new();
        let (rate, channels) = backend.native_format();
        assert!(rate > 0);
        assert!(channels > 0);
    }

    /// `start` → `stop` がデバイス有無を問わず panic しないこと。
    /// render endpoint が無い/開けない環境では `Err` を許容（panic だけ不可）。
    #[test]
    fn start_then_stop_tolerates_missing_endpoint() {
        let mut backend = WasapiSystemBackend::new();
        let (rate, channels) = backend.native_format();
        let cap = (rate as usize * channels as usize).max(1);
        let (prod, _cons) = raw_ring(cap);
        let sink = RawSink::new(prod, rate, channels);

        match backend.start(sink) {
            Ok(()) => {
                backend.stop();
                backend.stop(); // 二重 stop も安全。
            }
            Err(_e) => { /* render endpoint 無し/非 float 等は許容 */ }
        }
    }
}
