from __future__ import annotations

from collections.abc import Callable

from pyflexaudio.types import AudioChunk

__all__ = ["CallbackSink"]


class CallbackSink:
    """コールバック関数に AudioChunk を配信する Sink"""

    def __init__(self, callback: Callable[[AudioChunk], None], *, enabled: bool = True):
        self.enabled = enabled
        self.pause_exempt = False
        self._callback = callback

    def write(self, chunk: AudioChunk) -> None:
        if self.enabled:
            self._callback(chunk)

    def flush(self) -> None:
        pass  # バッファなし

    def close(self) -> None:
        pass  # リソースなし
