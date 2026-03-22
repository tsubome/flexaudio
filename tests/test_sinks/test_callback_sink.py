"""Tests for CallbackSink."""

import time

import numpy as np
import pytest

from pyflexaudio.sinks.callback import CallbackSink
from pyflexaudio.types import AudioChunk


def make_chunk(frames=1024, sample_rate=44100, channels=2, amplitude=0.5):
    data = (np.random.randn(frames, channels) * amplitude).astype(np.float32)
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=sample_rate,
        channels=channels,
        source_id="test:0",
    )


class TestCallbackSinkEnabled:
    def test_callback_called_when_enabled(self):
        received = []
        sink = CallbackSink(callback=received.append, enabled=True)
        chunk = make_chunk()
        sink.write(chunk)
        assert len(received) == 1
        assert received[0] is chunk

    def test_callback_called_multiple_times(self):
        received = []
        sink = CallbackSink(callback=received.append, enabled=True)
        for _ in range(5):
            sink.write(make_chunk())
        assert len(received) == 5


class TestCallbackSinkDisabled:
    def test_callback_not_called_when_disabled(self):
        received = []
        sink = CallbackSink(callback=received.append, enabled=False)
        sink.write(make_chunk())
        assert len(received) == 0

    def test_callback_not_called_for_multiple_writes_when_disabled(self):
        received = []
        sink = CallbackSink(callback=received.append, enabled=False)
        for _ in range(3):
            sink.write(make_chunk())
        assert len(received) == 0


class TestCallbackSinkToggle:
    def test_toggle_enabled_to_disabled(self):
        received = []
        sink = CallbackSink(callback=received.append, enabled=True)
        sink.write(make_chunk())
        assert len(received) == 1

        sink.enabled = False
        sink.write(make_chunk())
        assert len(received) == 1  # no new calls

    def test_toggle_disabled_to_enabled(self):
        received = []
        sink = CallbackSink(callback=received.append, enabled=False)
        sink.write(make_chunk())
        assert len(received) == 0

        sink.enabled = True
        sink.write(make_chunk())
        assert len(received) == 1

    def test_multiple_toggles(self):
        received = []
        sink = CallbackSink(callback=received.append, enabled=True)

        sink.write(make_chunk())   # enabled  -> 1
        sink.enabled = False
        sink.write(make_chunk())   # disabled -> still 1
        sink.write(make_chunk())   # disabled -> still 1
        sink.enabled = True
        sink.write(make_chunk())   # enabled  -> 2
        sink.write(make_chunk())   # enabled  -> 3

        assert len(received) == 3


class TestCallbackSinkPauseExempt:
    def test_pause_exempt_defaults_to_false(self):
        sink = CallbackSink(callback=lambda c: None)
        assert sink.pause_exempt is False


class TestCallbackSinkLifecycle:
    def test_flush_does_not_raise(self):
        sink = CallbackSink(callback=lambda c: None)
        sink.flush()  # should be a no-op

    def test_close_does_not_raise(self):
        sink = CallbackSink(callback=lambda c: None)
        sink.close()  # should be a no-op
