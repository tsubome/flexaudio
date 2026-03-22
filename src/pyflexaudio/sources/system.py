from __future__ import annotations

import queue
import threading

from pyflexaudio.types import AudioChunk


__all__ = ["SystemAudioSource"]


class SystemAudioSource:
    """システム音声キャプチャ。OS に応じたバックエンドを使用"""

    def __init__(self, device_index: int | None = None):
        self._device_index = device_index
        self._backend = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def source_id(self) -> str:
        idx = self._device_index if self._device_index is not None else "default"
        return f"system_audio:{idx}"

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
            from pyflexaudio.sources._backends.system_windows import WasapiLoopbackBackend
            return WasapiLoopbackBackend(device_index=self._device_index)
        elif IS_MACOS:
            from pyflexaudio.sources._backends.system_macos import SCKSystemAudioBackend
            return SCKSystemAudioBackend()
        else:
            raise NotImplementedError("System audio capture is not yet supported on this platform")
