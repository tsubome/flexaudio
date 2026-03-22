from __future__ import annotations

from pyflexaudio.processors.base import AudioProcessor
from pyflexaudio.types import AudioChunk


__all__ = ["ProcessorChain"]


class ProcessorChain:
    """複数の AudioProcessor を順次実行するチェーン"""

    def __init__(self, processors: list[AudioProcessor] | None = None):
        self._processors: list[AudioProcessor] = list(processors) if processors else []

    def add(self, processor: AudioProcessor) -> None:
        """プロセッサを末尾に追加"""
        self._processors.append(processor)

    def process(self, chunk: AudioChunk) -> AudioChunk:
        """全プロセッサを順次実行"""
        for processor in self._processors:
            chunk = processor.process(chunk)
        return chunk

    def reset(self) -> None:
        """全プロセッサの内部状態をリセット"""
        for processor in self._processors:
            processor.reset()
