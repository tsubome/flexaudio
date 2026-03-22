from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

from pyflexaudio.types import AudioChunk

logger = logging.getLogger("pyflexaudio.sources.system_windows")

__all__ = ["WasapiLoopbackBackend"]


class WasapiLoopbackBackend:
    """pyaudiowpatch による WASAPI Loopback（Windows システム音声キャプチャ）"""

    def __init__(self, device_index: int | None = None):
        self._device_index = device_index
        self._pwa = None  # pyaudiowpatch.PyAudio instance
        self._stream = None
        self._data_queue: queue.Queue | None = None
        self._stop_event: threading.Event | None = None
        self._is_open = False
        self._sample_rate = 44100
        self._channels = 2
        self._sample_width = 2  # bytes (int16)

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def source_id(self) -> str:
        idx = self._device_index if self._device_index is not None else "default"
        return f"system_audio:{idx}"

    def open(self, data_queue: queue.Queue, stop_event: threading.Event) -> None:
        if self._is_open:
            return
        self._data_queue = data_queue
        self._stop_event = stop_event

        import pyaudiowpatch as pwa

        self._pwa = pwa.PyAudio()

        # ループバックデバイスの検索
        if self._device_index is not None:
            device_info = self._pwa.get_device_info_by_index(self._device_index)
        else:
            # デフォルトの WASAPI ループバックデバイスを取得
            wasapi_info = self._pwa.get_host_api_info_by_type(pwa.paWASAPI)
            default_speakers = self._pwa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

            # ループバックデバイスを検索
            device_info = None
            for loopback in self._pwa.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    device_info = loopback
                    break

            if device_info is None:
                raise RuntimeError("No WASAPI loopback device found")

        self._sample_rate = int(device_info["defaultSampleRate"])
        self._channels = int(device_info["maxInputChannels"])

        # ストリームをコールバックモードで開く
        self._stream = self._pwa.open(
            format=pwa.paInt16,
            channels=self._channels,
            rate=self._sample_rate,
            input=True,
            input_device_index=device_info["index"],
            frames_per_buffer=1024,
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        self._is_open = True
        logger.info("Opened WASAPI loopback: rate=%d, ch=%d", self._sample_rate, self._channels)

    def close(self) -> None:
        if not self._is_open:
            return
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                logger.exception("Error closing WASAPI loopback stream")
            self._stream = None
        if self._pwa is not None:
            try:
                self._pwa.terminate()
            except Exception:
                pass
            self._pwa = None
        self._is_open = False

    def _callback(self, in_data, frame_count, time_info, status):
        """pyaudiowpatch コールバック"""
        import pyaudiowpatch as pwa

        if self._stop_event is not None and self._stop_event.is_set():
            return (None, pwa.paComplete)

        # int16 bytes → float32 numpy
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        samples = samples.reshape(-1, self._channels)

        chunk = AudioChunk(
            data=samples,
            timestamp=time.monotonic(),
            sample_rate=self._sample_rate,
            channels=self._channels,
            source_id=self.source_id,
        )

        try:
            self._data_queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._data_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._data_queue.put_nowait(chunk)
            except queue.Full:
                pass

        return (None, pwa.paContinue)
