from __future__ import annotations

from typing import Protocol, runtime_checkable

from pyflexaudio.types import AudioChunk


__all__ = ["AudioProcessor"]


@runtime_checkable
class AudioProcessor(Protocol):
    """音声チャンクを処理するプロセッサの Protocol"""

    def process(self, chunk: AudioChunk) -> AudioChunk:
        """チャンクを処理して返す。データ変換、メタデータ付加等"""
        ...

    def reset(self) -> None:
        """内部状態をリセット"""
        ...
