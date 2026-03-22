from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np
import miniaudio

from pyflexaudio.types import AudioChunk

__all__ = ["MiniaudioMicBackend"]

logger = logging.getLogger("pyflexaudio.sources.mic_miniaudio")


class MiniaudioMicBackend:
    """miniaudio によるマイク入力（sounddevice のフォールバック）"""

    def __init__(self, device_index: int | None = None, sample_rate: int | None = None,
                 channels: int | None = None):
        self._device_index = device_index
        self._sample_rate = sample_rate or 44100
        self._channels = channels or 1
        self._capture: miniaudio.CaptureDevice | None = None
        self._data_queue: queue.Queue | None = None
        self._stop_event: threading.Event | None = None
        self._is_open = False
        self._source_id = f"microphone:{device_index if device_index is not None else 'default'}"

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def source_id(self) -> str:
        return self._source_id

    def open(self, data_queue: queue.Queue, stop_event: threading.Event) -> None:
        if self._is_open:
            return
        self._data_queue = data_queue
        self._stop_event = stop_event

        # デバイス一覧からデバイスIDを解決
        device_id = None
        if self._device_index is not None:
            devices = miniaudio.Devices()
            captures = devices.get_captures()
            if self._device_index < len(captures):
                device_id = captures[self._device_index]["id"]

        self._capture = miniaudio.CaptureDevice(
            input_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=self._channels,
            sample_rate=self._sample_rate,
            device_id=device_id,
            callback=self._callback,
        )
        self._capture.start()
        self._is_open = True
        logger.info("Opened miniaudio mic: rate=%d, ch=%d", self._sample_rate, self._channels)

    def close(self) -> None:
        if not self._is_open:
            return
        if self._capture is not None:
            try:
                self._capture.stop()
                self._capture.close()
            except Exception:
                logger.exception("Error closing miniaudio capture")
            self._capture = None
        self._is_open = False
        logger.info("Closed miniaudio mic")

    def _callback(self, data: bytes, num_frames: int) -> None:
        """miniaudio キャプチャコールバック"""
        if self._stop_event is not None and self._stop_event.is_set():
            return

        # bytes → numpy float32 配列
        samples = np.frombuffer(data, dtype=np.float32).copy()
        # reshape to 2D (frames, channels)
        if self._channels > 1:
            samples = samples.reshape(-1, self._channels)
        else:
            samples = samples.reshape(-1, 1)

        chunk = AudioChunk(
            data=samples,
            timestamp=time.monotonic(),
            sample_rate=self._sample_rate,
            channels=self._channels,
            source_id=self._source_id,
        )

        try:
            self._data_queue.put_nowait(chunk)
        except queue.Full:
            # DROP_OLDEST: 古いチャンクを捨てて新しいチャンクを入れる
            try:
                self._data_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._data_queue.put_nowait(chunk)
            except queue.Full:
                pass  # 最悪の場合はドロップ
