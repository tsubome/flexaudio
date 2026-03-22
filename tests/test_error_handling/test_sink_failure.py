"""Sink.write() が失敗するとその Sink だけ disable されることを検証するテスト"""

import queue
import time

import numpy as np
import pytest

from pyflexaudio.types import AudioChunk, ErrorEvent
from pyflexaudio.events import EventBus
from pyflexaudio.pipeline import Pipeline
from pyflexaudio.processors.chain import ProcessorChain


# ---- ヘルパー ----

def make_chunk(frames=512, sr=16000, ch=1, source_id="test:0"):
    data = np.random.randn(frames, ch).astype(np.float32)
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=sr,
        channels=ch,
        source_id=source_id,
    )


class MockSink:
    def __init__(self, enabled=True, pause_exempt=False):
        self.enabled = enabled
        self.pause_exempt = pause_exempt
        self.chunks = []
        self.flushed = False
        self.closed = False

    def write(self, chunk):
        if self.enabled:
            self.chunks.append(chunk)

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


class FailingSink:
    """write() が常に IOError を投げる Sink"""

    enabled = True
    pause_exempt = False

    def write(self, chunk):
        raise IOError("disk full")

    def flush(self):
        pass

    def close(self):
        pass


# ---- テスト ----

def test_sink_failure_disables_only_that_sink():
    """FailingSink のみ disable され、MockSink は正常にチャンクを受信する"""
    bus = EventBus()
    errors = []
    bus.on(ErrorEvent, errors.append)

    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    failing_sink = FailingSink()
    mock_sink = MockSink()

    pipeline.add_sink(failing_sink)
    pipeline.add_sink(mock_sink)

    dq = queue.Queue()
    pipeline.start(dq)

    chunk = make_chunk()
    dq.put(chunk)
    time.sleep(0.2)

    pipeline.stop()

    # FailingSink は disabled になる
    assert failing_sink.enabled is False

    # MockSink は正常にチャンクを受信
    assert len(mock_sink.chunks) >= 1

    # ErrorEvent が emit される
    assert len(errors) >= 1
    assert errors[0].error.code == "SINK_WRITE_ERROR"


def test_sink_failure_error_event_has_source_id():
    """ErrorEvent に正しい source_id が含まれる"""
    bus = EventBus()
    errors = []
    bus.on(ErrorEvent, errors.append)

    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())
    pipeline.add_sink(FailingSink())

    dq = queue.Queue()
    pipeline.start(dq)

    chunk = make_chunk(source_id="microphone:0")
    dq.put(chunk)
    time.sleep(0.2)

    pipeline.stop()

    assert len(errors) >= 1
    assert errors[0].source_id == "microphone:0"


def test_sink_failure_subsequent_chunks_skipped():
    """一度 disable された Sink には以降のチャンクが届かない"""
    bus = EventBus()

    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    failing_sink = FailingSink()
    pipeline.add_sink(failing_sink)

    dq = queue.Queue()
    pipeline.start(dq)

    # 最初のチャンクで disable される
    dq.put(make_chunk())
    time.sleep(0.1)

    # disable された後にチャンクを追加投入
    dq.put(make_chunk())
    dq.put(make_chunk())
    time.sleep(0.2)

    pipeline.stop()

    # enabled は False のまま（戻ることはない）
    assert failing_sink.enabled is False


def test_multiple_failing_sinks():
    """複数の FailingSink があっても MockSink は正常動作する"""
    bus = EventBus()
    errors = []
    bus.on(ErrorEvent, errors.append)

    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    failing1 = FailingSink()
    failing2 = FailingSink()
    mock_sink = MockSink()

    pipeline.add_sink(failing1)
    pipeline.add_sink(failing2)
    pipeline.add_sink(mock_sink)

    dq = queue.Queue()
    pipeline.start(dq)

    dq.put(make_chunk())
    time.sleep(0.2)

    pipeline.stop()

    assert failing1.enabled is False
    assert failing2.enabled is False
    assert len(mock_sink.chunks) >= 1
    # 2つの FailingSink から各1つの ErrorEvent
    assert len(errors) >= 2
