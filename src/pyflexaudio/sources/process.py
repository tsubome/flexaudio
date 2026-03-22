from __future__ import annotations

import queue
import threading

from pyflexaudio.types import AudioChunk


__all__ = ["ProcessAudioSource"]


class ProcessAudioSource:
    """プロセス別音声キャプチャ。OS に応じたバックエンドを使用"""

    def __init__(self, pid: int, mode: str = "include"):
        """
        Args:
            pid: 対象プロセスの PID
            mode: "include" = 対象プロセスのみ、"exclude" = 対象プロセス以外
        """
        self._pid = pid
        self._mode = mode
        self._backend = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def source_id(self) -> str:
        return f"process_audio:{self._pid}"

    def open(self, data_queue: queue.Queue[AudioChunk | None], stop_event: threading.Event) -> None:
        if self._is_open:
            return
        self._backend = self._create_backend()
        self._backend.open(data_queue, stop_event)
        self._is_open = True

    def close(self) -> None:
        if not self._is_open:
            return
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        self._is_open = False

    def _create_backend(self):
        from pyflexaudio._platform import IS_WINDOWS, IS_MACOS
        if IS_WINDOWS:
            from pyflexaudio.sources._backends.process_windows import PacProcessBackend
            return PacProcessBackend(pid=self._pid, mode=self._mode)
        elif IS_MACOS:
            from pyflexaudio.sources._backends.process_macos import SCKProcessAudioBackend
            return SCKProcessAudioBackend(pid=self._pid, mode=self._mode)
        else:
            raise NotImplementedError("Process audio capture is not yet supported on this platform")
