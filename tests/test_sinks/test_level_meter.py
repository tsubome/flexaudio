"""Tests for LevelMeterSink."""

import time

import numpy as np
import pytest

from pyflexaudio.events import EventBus
from pyflexaudio.sinks.level_meter import LevelMeterSink
from pyflexaudio.types import AudioChunk, LevelEvent


def make_chunk(frames=1024, sample_rate=44100, channels=2, amplitude=0.5, level_db=None):
    data = (np.random.randn(frames, channels) * amplitude).astype(np.float32)
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=sample_rate,
        channels=channels,
        source_id="test:0",
        level_db=level_db,
    )


class TestLevelMeterEmit:
    def test_emits_level_event_when_level_db_set(self):
        bus = EventBus()
        events = []
        bus.on(LevelEvent, events.append)

        sink = LevelMeterSink(event_bus=bus)
        chunk = make_chunk(level_db=-12.0)
        sink.write(chunk)

        assert len(events) == 1
        assert isinstance(events[0], LevelEvent)

    def test_level_event_contains_correct_db(self):
        bus = EventBus()
        events = []
        bus.on(LevelEvent, events.append)

        sink = LevelMeterSink(event_bus=bus)
        sink.write(make_chunk(level_db=-20.5))

        assert events[0].db == pytest.approx(-20.5)

    def test_level_event_contains_correct_source_id(self):
        bus = EventBus()
        events = []
        bus.on(LevelEvent, events.append)

        sink = LevelMeterSink(event_bus=bus)
        sink.write(make_chunk(level_db=-6.0))

        assert events[0].source_id == "test:0"

    def test_no_emit_when_level_db_is_none(self):
        bus = EventBus()
        events = []
        bus.on(LevelEvent, events.append)

        sink = LevelMeterSink(event_bus=bus)
        chunk = make_chunk(level_db=None)
        sink.write(chunk)

        assert len(events) == 0

    def test_multiple_chunks_emit_multiple_events(self):
        bus = EventBus()
        events = []
        bus.on(LevelEvent, events.append)

        sink = LevelMeterSink(event_bus=bus)
        for db in [-10.0, -20.0, -30.0]:
            sink.write(make_chunk(level_db=db))

        assert len(events) == 3
        db_values = [e.db for e in events]
        assert db_values == pytest.approx([-10.0, -20.0, -30.0])

    def test_mixed_chunks_only_emit_for_set_level_db(self):
        bus = EventBus()
        events = []
        bus.on(LevelEvent, events.append)

        sink = LevelMeterSink(event_bus=bus)
        sink.write(make_chunk(level_db=-5.0))
        sink.write(make_chunk(level_db=None))
        sink.write(make_chunk(level_db=-15.0))

        assert len(events) == 2


class TestLevelMeterPauseExempt:
    def test_pause_exempt_is_true(self):
        bus = EventBus()
        sink = LevelMeterSink(event_bus=bus)
        assert sink.pause_exempt is True

    def test_enabled_is_true_by_default(self):
        bus = EventBus()
        sink = LevelMeterSink(event_bus=bus)
        assert sink.enabled is True


class TestLevelMeterDbRange:
    def test_db_value_zero(self):
        bus = EventBus()
        events = []
        bus.on(LevelEvent, events.append)

        sink = LevelMeterSink(event_bus=bus)
        sink.write(make_chunk(level_db=0.0))

        assert len(events) == 1
        assert events[0].db == pytest.approx(0.0)

    def test_db_value_very_negative(self):
        bus = EventBus()
        events = []
        bus.on(LevelEvent, events.append)

        sink = LevelMeterSink(event_bus=bus)
        sink.write(make_chunk(level_db=-96.0))

        assert len(events) == 1
        assert events[0].db == pytest.approx(-96.0)

    def test_db_value_positive(self):
        """Positive dB (clipping) should still be emitted."""
        bus = EventBus()
        events = []
        bus.on(LevelEvent, events.append)

        sink = LevelMeterSink(event_bus=bus)
        sink.write(make_chunk(level_db=3.0))

        assert len(events) == 1
        assert events[0].db == pytest.approx(3.0)

    def test_db_value_is_float(self):
        bus = EventBus()
        events = []
        bus.on(LevelEvent, events.append)

        sink = LevelMeterSink(event_bus=bus)
        sink.write(make_chunk(level_db=-12.3456))

        assert isinstance(events[0].db, float)


class TestLevelMeterLifecycle:
    def test_flush_does_not_raise(self):
        bus = EventBus()
        sink = LevelMeterSink(event_bus=bus)
        sink.flush()

    def test_close_does_not_raise(self):
        bus = EventBus()
        sink = LevelMeterSink(event_bus=bus)
        sink.close()
