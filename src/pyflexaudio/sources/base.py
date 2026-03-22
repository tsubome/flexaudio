from __future__ import annotations

import queue
import threading
from typing import Protocol, runtime_checkable

from pyflexaudio.types import AudioChunk


__all__ = ["AudioSource"]


@runtime_checkable
class AudioSource(Protocol):
    """音声ソースの Protocol"""

    @property
    def is_open(self) -> bool:
        """ソースが開かれているか"""
        ...

    @property
    def source_id(self) -> str:
        """ソース識別子。形式: '{source_type}:{device_index_or_pid}'"""
        ...

    def open(self, data_queue: queue.Queue[AudioChunk | None], stop_event: threading.Event) -> None:
        """ソースを開き、data_queue へのチャンク配信を開始"""
        ...

    def close(self) -> None:
        """ソースを閉じ、リソースを解放"""
        ...
