from __future__ import annotations

from typing import Protocol, runtime_checkable

from pyflexaudio.types import AudioChunk

__all__ = ["AudioSink"]


@runtime_checkable
class AudioSink(Protocol):
    """音声チャンクを受け取る Sink の Protocol"""

    enabled: bool
    pause_exempt: bool  # True の場合、pause 中も配信される

    def write(self, chunk: AudioChunk) -> None:
        """チャンクを書き込む"""
        ...

    def flush(self) -> None:
        """バッファをフラッシュ"""
        ...

    def close(self) -> None:
        """リソースを解放"""
        ...
