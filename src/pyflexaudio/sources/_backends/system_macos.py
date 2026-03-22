"""macOS system audio capture via ScreenCaptureKit."""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

from pyflexaudio.types import AudioChunk

logger = logging.getLogger("pyflexaudio.sources.system_macos")

__all__ = ["SCKSystemAudioBackend"]


class _SCKBaseBackend:
    """ScreenCaptureKit バックエンドの共通基底クラス"""

    def __init__(self):
        self._data_queue: queue.Queue | None = None
        self._stop_event: threading.Event | None = None
        self._stream = None
        self._delegate = None
        self._is_open = False
        self._format_detected = False
        self._sample_rate = 48000  # SCK デフォルト
        self._channels = 2

    @property
    def is_open(self) -> bool:
        return self._is_open

    def _build_content_filter(self, content):
        """サブクラスで SCContentFilter を構築する。"""
        raise NotImplementedError

    def open(self, data_queue: queue.Queue, stop_event: threading.Event) -> None:
        if self._is_open:
            return
        self._data_queue = data_queue
        self._stop_event = stop_event

        # pyobjc 遅延 import
        from ScreenCaptureKit import (  # type: ignore[import]
            SCShareableContent,
            SCStreamConfiguration,
            SCStream,
        )
        from dispatch import dispatch_queue_create, DISPATCH_QUEUE_SERIAL  # type: ignore[import]
        import objc  # type: ignore[import]

        # 共有可能コンテンツの取得（同期的に待つ）
        content_event = threading.Event()
        content_result: dict = {}

        def content_handler(content, error):
            if error:
                content_result["error"] = str(error)
            else:
                content_result["content"] = content
            content_event.set()

        SCShareableContent.getShareableContentWithCompletionHandler_(content_handler)
        if not content_event.wait(timeout=10.0):
            raise TimeoutError("Timed out getting shareable content")
        if "error" in content_result:
            raise RuntimeError(
                f"Failed to get shareable content: {content_result['error']}"
            )

        content = content_result["content"]

        # フィルタ構築（サブクラスに委譲）
        content_filter = self._build_content_filter(content)

        # 設定
        config = SCStreamConfiguration.alloc().init()
        config.setWidth_(2)   # 映像最小化
        config.setHeight_(2)
        config.setCapturesAudio_(True)
        config.setExcludesCurrentProcessAudio_(True)  # 自プロセス除外
        config.setSampleRate_(48000)
        config.setChannelCount_(2)

        # デリゲートクラスが利用不可の場合はエラー
        if _StreamOutputDelegate is None:
            raise RuntimeError(
                "pyobjc is required for ScreenCaptureKit backend. "
                "Install it with: pip install pyobjc-framework-ScreenCaptureKit"
            )

        # デリゲート
        self._delegate = _StreamOutputDelegate.alloc().init()
        self._delegate._capture = self  # 循環参照（close() で切断）

        # ストリーム作成
        self._stream = SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter, config, None
        )

        # カスタム dispatch queue
        audio_queue = dispatch_queue_create(
            b"pyflexaudio.sck.system", DISPATCH_QUEUE_SERIAL
        )

        # 出力追加（type 1 = SCStreamOutputTypeAudio）
        error_ptr = objc.nil
        self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self._delegate, 1, audio_queue, error_ptr
        )

        # キャプチャ開始（同期的に待つ）
        start_event = threading.Event()
        start_result: dict = {}

        def start_handler(error):
            if error:
                start_result["error"] = str(error)
            start_event.set()

        self._stream.startCaptureWithCompletionHandler_(start_handler)
        if not start_event.wait(timeout=10.0):
            raise TimeoutError("Timed out starting capture")
        if "error" in start_result:
            raise RuntimeError(f"Failed to start capture: {start_result['error']}")

        self._is_open = True
        logger.info("Opened SCK audio capture (%s)", self.source_id)

    def close(self) -> None:
        if not self._is_open:
            return

        if self._stream is not None:
            stop_event = threading.Event()

            def stop_handler(error):
                if error:
                    logger.warning("SCK stop error: %s", error)
                stop_event.set()

            self._stream.stopCaptureWithCompletionHandler_(stop_handler)
            stop_event.wait(timeout=10.0)
            self._stream = None

        # 循環参照を切断
        if self._delegate is not None:
            self._delegate._capture = None
            self._delegate = None

        self._is_open = False
        logger.info("Closed SCK audio capture (%s)", self.source_id)

    def _on_audio_buffer(self, sample_buffer) -> None:
        """デリゲートから呼ばれるオーディオバッファ処理"""
        if self._stop_event is not None and self._stop_event.is_set():
            return

        try:
            from CoreMedia import (  # type: ignore[import]
                CMSampleBufferGetFormatDescription,
                CMAudioFormatDescriptionGetStreamBasicDescription,
                CMSampleBufferGetDataBuffer,
                CMBlockBufferGetDataLength,
                CMBlockBufferCopyDataBytes,
            )

            # フォーマット検出（初回のみ）
            if not self._format_detected:
                fmt_desc = CMSampleBufferGetFormatDescription(sample_buffer)
                if fmt_desc is not None:
                    asbd = CMAudioFormatDescriptionGetStreamBasicDescription(fmt_desc)
                    if asbd is not None:
                        self._sample_rate = int(asbd.mSampleRate)
                        self._channels = int(asbd.mChannelsPerFrame)
                        self._format_detected = True
                        logger.info(
                            "SCK format: %dHz, %dch",
                            self._sample_rate,
                            self._channels,
                        )

            # オーディオデータ取得
            data_buffer = CMSampleBufferGetDataBuffer(sample_buffer)
            if data_buffer is None:
                return
            data_length = CMBlockBufferGetDataLength(data_buffer)
            if data_length == 0:
                return

            # バッファからデータをコピー
            raw_data = bytearray(data_length)
            CMBlockBufferCopyDataBytes(data_buffer, 0, data_length, raw_data)

            # float32 に変換（SCK は non-interleaved/planar の場合あり）
            samples = np.frombuffer(raw_data, dtype=np.float32)
            frames = len(samples) // self._channels

            if frames == 0:
                return

            # プレーナー → インターリーブ変換
            if self._channels >= 2:
                # 各チャンネルが連続して格納されている想定
                channels_data = [
                    samples[i * frames : (i + 1) * frames]
                    for i in range(self._channels)
                ]
                interleaved = np.column_stack(channels_data)  # shape=(frames, channels)
            else:
                interleaved = samples.reshape(-1, 1)

            chunk = AudioChunk(
                data=interleaved.astype(np.float32),
                timestamp=time.monotonic(),
                sample_rate=self._sample_rate,
                channels=self._channels,
                source_id=self.source_id,
            )

            if self._data_queue is None:
                return

            try:
                self._data_queue.put_nowait(chunk)
            except queue.Full:
                # DROP_OLDEST: 古いチャンクを捨てて新しいものを入れる
                try:
                    self._data_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._data_queue.put_nowait(chunk)
                except queue.Full:
                    pass

        except Exception:
            logger.exception("Error processing SCK audio buffer")

    @property
    def source_id(self) -> str:
        raise NotImplementedError


class SCKSystemAudioBackend(_SCKBaseBackend):
    """ScreenCaptureKit によるシステム音声キャプチャ（macOS）"""

    @property
    def source_id(self) -> str:
        return "system_audio:default"

    def _build_content_filter(self, content):
        """全画面の音声を取得するフィルタを構築する。"""
        from ScreenCaptureKit import SCContentFilter  # type: ignore[import]

        displays = content.displays()
        if not displays:
            raise RuntimeError("No displays found")

        display = displays[0]
        content_filter = SCContentFilter.alloc().initWithDisplay_excludingWindows_(
            display, []
        )
        return content_filter


# ---------------------------------------------------------------------------
# NSObject サブクラスのデリゲート
# ---------------------------------------------------------------------------

def _create_delegate_class():
    """SCStreamOutput プロトコルに準拠したデリゲートクラスを動的に作成する。"""
    try:
        import objc  # type: ignore[import]
        from Foundation import NSObject  # type: ignore[import]

        class _StreamOutputDelegate(NSObject):  # type: ignore[misc]
            _capture = None

            def stream_didOutputSampleBuffer_ofType_(
                self, stream, sample_buffer, output_type
            ):
                # output_type 1 = SCStreamOutputTypeAudio
                if self._capture is not None and output_type == 1:
                    self._capture._on_audio_buffer(sample_buffer)

        # SCStreamOutput protocol のメソッドシグネチャを登録
        objc.classAddProtocol(
            _StreamOutputDelegate, objc.protocolNamed("SCStreamOutput")
        )

        return _StreamOutputDelegate
    except ImportError:
        return None


# pyobjc が利用可能な場合のみクラスを作成
try:
    _StreamOutputDelegate = _create_delegate_class()
except Exception:
    _StreamOutputDelegate = None
