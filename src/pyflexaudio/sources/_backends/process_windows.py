from __future__ import annotations

import ctypes
import logging
import os
import queue
import threading
import time

import numpy as np

from pyflexaudio.types import AudioChunk

logger = logging.getLogger("pyflexaudio.sources.process_windows")

__all__ = ["PacProcessBackend"]


# DLL エラーコード
PAC_OK = 0
PAC_ERROR_NOT_INITIALIZED = 1
PAC_ERROR_ALREADY_CAPTURING = 2
PAC_ERROR_INVALID_PARAMETER = 3
PAC_ERROR_CAPTURE_FAILED = 4

# キャプチャモード
PAC_MODE_INCLUDE = 0
PAC_MODE_EXCLUDE = 1


class PacProcessBackend:
    """ProcessAudioCapture DLL によるプロセス別音声キャプチャ（Windows）"""

    def __init__(self, pid: int, mode: str = "include"):
        self._pid = pid
        self._mode = PAC_MODE_INCLUDE if mode == "include" else PAC_MODE_EXCLUDE
        self._dll = None
        self._handle = None
        self._data_queue: queue.Queue | None = None
        self._stop_event: threading.Event | None = None
        self._io_thread: threading.Thread | None = None
        self._is_open = False
        self._sample_rate = 48000
        self._channels = 2
        self._callback_ref = None  # GC 防止

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def source_id(self) -> str:
        return f"process_audio:{self._pid}"

    def open(self, data_queue: queue.Queue, stop_event: threading.Event) -> None:
        if self._is_open:
            return
        self._data_queue = data_queue
        self._stop_event = stop_event

        # DLL ロード
        self._dll = self._load_dll()

        # レベルコールバック（必須だが pyflexaudio では使わない）
        LEVEL_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_float, ctypes.c_void_p)
        self._callback_ref = LEVEL_CALLBACK(self._level_callback)

        # キャプチャ開始
        handle = ctypes.c_void_p()
        result = self._dll.PacStartCaptureEx(
            ctypes.c_uint32(self._pid),
            ctypes.c_int(self._mode),
            ctypes.c_uint32(5000),  # リングバッファ 5秒
            self._callback_ref,
            None,
            ctypes.byref(handle),
        )
        if result != PAC_OK:
            raise RuntimeError(f"PacStartCaptureEx failed with code {result}")

        self._handle = handle

        # フォーマット取得
        channels = ctypes.c_int()
        sample_rate = ctypes.c_int()
        bits_per_sample = ctypes.c_int()
        result = self._dll.PacGetFormat(
            self._handle,
            ctypes.byref(channels),
            ctypes.byref(sample_rate),
            ctypes.byref(bits_per_sample),
        )
        if result == PAC_OK:
            self._channels = channels.value
            self._sample_rate = sample_rate.value

        # IO ポーリングスレッド開始
        self._io_thread = threading.Thread(
            target=self._polling_loop,
            name="pyflexaudio-pac-poll",
            daemon=True,
        )
        self._io_thread.start()
        self._is_open = True
        logger.info("Opened PAC: pid=%d, mode=%s, rate=%d, ch=%d",
                    self._pid, self._mode, self._sample_rate, self._channels)

    def close(self) -> None:
        if not self._is_open:
            return

        # IO スレッド停止を待機
        if self._io_thread is not None:
            self._io_thread.join(timeout=3.0)
            self._io_thread = None

        # DLL のキャプチャ停止
        if self._handle is not None and self._dll is not None:
            try:
                self._dll.PacStopCapture(self._handle)
            except Exception:
                logger.exception("Error stopping PAC")
            self._handle = None  # 二重 stop 防止

        self._callback_ref = None
        self._is_open = False
        logger.info("Closed PAC: pid=%d", self._pid)

    def _load_dll(self):
        """ProcessAudioCapture.dll をロード"""
        try:
            import pyflexaudio_win_dll
            dll_path = os.path.join(os.path.dirname(pyflexaudio_win_dll.__file__), "ProcessAudioCapture.dll")
        except ImportError:
            raise RuntimeError(
                "pyflexaudio-win-dll is not installed. "
                "Install it with: pip install pyflexaudio[win-process]"
            )

        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"DLL not found: {dll_path}")

        dll = ctypes.CDLL(dll_path)

        # 関数シグネチャ設定
        # PacStartCaptureEx
        dll.PacStartCaptureEx.argtypes = [
            ctypes.c_uint32,   # processId
            ctypes.c_int,      # mode
            ctypes.c_uint32,   # ringBufferSizeMs
            ctypes.CFUNCTYPE(None, ctypes.c_float, ctypes.c_void_p),  # levelCallback
            ctypes.c_void_p,   # userData
            ctypes.POINTER(ctypes.c_void_p),  # handle
        ]
        dll.PacStartCaptureEx.restype = ctypes.c_int

        # PacStopCapture
        dll.PacStopCapture.argtypes = [ctypes.c_void_p]
        dll.PacStopCapture.restype = ctypes.c_int

        # PacReadBuffer
        dll.PacReadBuffer.argtypes = [
            ctypes.c_void_p,                    # handle
            ctypes.POINTER(ctypes.c_float),     # buffer
            ctypes.c_int,                       # maxFrames
            ctypes.POINTER(ctypes.c_int),       # actualFrames
            ctypes.POINTER(ctypes.c_int),       # overrunCount
        ]
        dll.PacReadBuffer.restype = ctypes.c_int

        # PacGetFormat
        dll.PacGetFormat.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),  # channels
            ctypes.POINTER(ctypes.c_int),  # sampleRate
            ctypes.POINTER(ctypes.c_int),  # bitsPerSample
        ]
        dll.PacGetFormat.restype = ctypes.c_int

        return dll

    def _polling_loop(self) -> None:
        """IO スレッド: DLL リングバッファからポーリング"""
        max_frames = 4096
        buf = (ctypes.c_float * (max_frames * self._channels))()
        actual_frames = ctypes.c_int()
        overrun_count = ctypes.c_int()

        while not self._stop_event.is_set():
            result = self._dll.PacReadBuffer(
                self._handle,
                buf,
                max_frames,
                ctypes.byref(actual_frames),
                ctypes.byref(overrun_count),
            )

            if result != PAC_OK:
                logger.error("PacReadBuffer error: %d", result)
                break

            if overrun_count.value > 0:
                logger.warning("PAC ring buffer overrun: %d", overrun_count.value)

            if actual_frames.value > 0:
                # ctypes 配列 → numpy
                data = np.ctypeslib.as_array(buf, shape=(max_frames * self._channels,))
                data = data[:actual_frames.value * self._channels].copy()
                data = data.reshape(-1, self._channels).astype(np.float32)

                chunk = AudioChunk(
                    data=data,
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
            else:
                time.sleep(0.01)  # 10ms 待機

    def _level_callback(self, level: float, user_data) -> None:
        """DLL レベルコールバック（pyflexaudio では使用しないがシグネチャ必須）"""
        pass
