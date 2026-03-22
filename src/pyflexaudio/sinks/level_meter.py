from __future__ import annotations

from pyflexaudio.types import AudioChunk, LevelEvent
from pyflexaudio.events import EventBus

__all__ = ["LevelMeterSink"]


class LevelMeterSink:
    """AudioChunk.level_db を読んで LevelEvent を emit する Sink。

    常時 ON (pause_exempt=True)。pause 中もレベルメーターは動作する。
    """

    def __init__(self, event_bus: EventBus):
        self.enabled = True
        self.pause_exempt = True
        self._event_bus = event_bus

    def write(self, chunk: AudioChunk) -> None:
        if chunk.level_db is not None:
            self._event_bus.emit(LevelEvent(db=chunk.level_db, source_id=chunk.source_id))

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
