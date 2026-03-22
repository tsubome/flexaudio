from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd

from pyflexaudio.types import AudioChunk

__all__ = ["SounddeviceMicBackend"]

logger = logging.getLogger("pyflexaudio.sources.mic_sounddevice")


class SounddeviceMicBackend:
    """sounddevice コールバックモードによるマイク入力"""

    def __init__(self, device_index: int | None = None, sample_rate: int | None = None,
                 channels: int | None = None, blocksize: int = 1024):
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._channels = channels
        self._blocksize = blocksize
        self._stream: sd.InputStream | None = None
        self._data_queue: queue.Queue | None = None
        self._stop_event: threading.Event | None = None
        self._is_open = False
        self._source_id = f"microphone:{device_index if device_index is not None else 'default'}"

        # open() 後に確定する値
        self._actual_sample_rate: int = 0
        self._actual_channels: int = 0

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

        # デバイス情報からデフォルト値を取得
        if self._device_index is not None:
            dev_info = sd.query_devices(self._device_index)
        else:
            dev_info = sd.query_devices(kind='input')

        actual_sr = self._sample_rate or int(dev_info['default_samplerate'])
        actual_ch = self._channels or min(int(dev_info['max_input_channels']), 2)

        self._actual_sample_rate = actual_sr
        self._actual_channels = actual_ch

        # InputStream をコールバックモードで開く
        self._stream = sd.InputStream(
            device=self._device_index,
            samplerate=actual_sr,
            channels=actual_ch,
            dtype=np.float32,
            blocksize=self._blocksize,
            callback=self._callback,
        )
        self._stream.start()
        self._is_open = True
        logger.info("Opened sounddevice mic: device=%s, rate=%d, ch=%d",
                    self._device_index, actual_sr, actual_ch)

    def close(self) -> None:
        if not self._is_open:
            return
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("Error closing sounddevice stream")
            self._stream = None
        self._is_open = False
        logger.info("Closed sounddevice mic")

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice コールバック。PortAudio のリアルタイムスレッド上で実行される"""
        if status:
            logger.warning("sounddevice status: %s", status)

        if self._stop_event is not None and self._stop_event.is_set():
            return

        # ★ indata.copy() は必須。コールバックを抜けるとバッファが PortAudio に返却される
        chunk = AudioChunk(
            data=indata.copy(),  # float32, shape=(frames, channels)
            timestamp=time.monotonic(),
            sample_rate=self._actual_sample_rate,
            channels=self._actual_channels,
            source_id=self._source_id,
        )

        # ★ put_nowait を使用。コールバック内でブロッキング put すると PortAudio がクラッシュ
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
