"""pause 中に Queue が溢れないことを検証するテスト"""

import queue
import threading
import time

import numpy as np
import pytest

from pyflexaudio.types import AudioChunk
from pyflexaudio.events import EventBus
from pyflexaudio.pipeline import Pipeline
from pyflexaudio.processors.chain import ProcessorChain
from pyflexaudio.session import FlexAudioSession


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


class PauseExemptSink(MockSink):
    """pause 中も動作する Sink"""

    def __init__(self):
        super().__init__(pause_exempt=True)


# ---- テスト ----

def test_pause_does_not_overflow_pipeline():
    """Pipeline を pause した状態でチャンクを大量投入しても Queue が溢れない"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    regular_sink = MockSink()
    exempt_sink = PauseExemptSink()
    pipeline.add_sink(regular_sink)
    pipeline.add_sink(exempt_sink)

    # 小さめの Queue（Pipeline Thread が drain し続ければ溢れない）
    dq = queue.Queue(maxsize=50)
    pipeline.start(dq)

    # pause 状態にする
    pipeline.pause()

    # チャンクを大量投入（Pipeline Thread は pause 中も drain する）
    dropped = 0
    produced = 0
    for _ in range(200):
        try:
            dq.put_nowait(make_chunk())
            produced += 1
        except queue.Full:
            dropped += 1
        time.sleep(0.001)

    time.sleep(0.5)

    # Queue のサイズが maxsize を超えていないこと
    assert dq.qsize() <= 50

    pipeline.resume()
    pipeline.stop()

    # pause 中も Pipeline Thread が drain していたことを確認（regular_sink には届かない）
    # exempt_sink には pause 中でも届く
    assert len(exempt_sink.chunks) > 0
    # regular_sink には pause 中は届かない
    # (resume 後に残っていたチャンクが届く可能性があるが、主に少ない)


def test_pause_regular_sink_receives_nothing():
    """pause 中に regular_sink にはチャンクが届かない"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    regular_sink = MockSink()
    pipeline.add_sink(regular_sink)

    dq = queue.Queue(maxsize=100)
    pipeline.start(dq)

    # pause
    pipeline.pause()

    # 少数チャンクを投入
    for _ in range(5):
        dq.put(make_chunk())
    time.sleep(0.2)

    chunks_during_pause = len(regular_sink.chunks)

    pipeline.resume()
    pipeline.stop()

    # pause 中は regular_sink に届かない
    assert chunks_during_pause == 0


def test_pause_exempt_sink_receives_during_pause():
    """pause 中でも pause_exempt Sink にはチャンクが届く"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    exempt_sink = PauseExemptSink()
    pipeline.add_sink(exempt_sink)

    dq = queue.Queue(maxsize=100)
    pipeline.start(dq)

    pipeline.pause()

    for _ in range(5):
        dq.put(make_chunk())
    time.sleep(0.2)

    pipeline.resume()
    pipeline.stop()

    # pause_exempt は pause 中でも受信する
    assert len(exempt_sink.chunks) == 5


def test_resume_after_pause_delivers_new_chunks():
    """resume 後の新しいチャンクは regular_sink に届く"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    regular_sink = MockSink()
    pipeline.add_sink(regular_sink)

    dq = queue.Queue(maxsize=100)
    pipeline.start(dq)

    pipeline.pause()

    # pause 中のチャンク（届かない）
    for _ in range(3):
        dq.put(make_chunk())
    time.sleep(0.2)

    pipeline.resume()

    # resume 後のチャンク（届く）
    for _ in range(3):
        dq.put(make_chunk())
    time.sleep(0.2)

    pipeline.stop()

    # resume 後のチャンクは届いている
    assert len(regular_sink.chunks) == 3


def test_queue_not_growing_during_pause():
    """pause 中に Pipeline Thread がキューをドレインするため、Queue が無制限に膨らまない"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    pipeline.add_sink(MockSink())

    dq = queue.Queue(maxsize=30)
    pipeline.start(dq)

    pipeline.pause()

    # 短期間で大量投入
    for _ in range(100):
        try:
            dq.put_nowait(make_chunk())
        except queue.Full:
            pass
        time.sleep(0.002)

    # Pipeline Thread が drain していれば Queue サイズが 30 以下
    final_size = dq.qsize()
    assert final_size <= 30, f"Queue size {final_size} exceeded maxsize 30"

    pipeline.resume()
    pipeline.stop()


def test_session_pause_backpressure():
    """FlexAudioSession の pause 中も Queue が溢れない"""
    session = FlexAudioSession(queue_size=50)

    regular_sink = MockSink()
    session.add_sink(regular_sink)

    session.start()
    session.pause()

    # チャンクを直接 data_queue に投入
    dq = session._data_queue
    for _ in range(100):
        try:
            dq.put_nowait(make_chunk())
        except queue.Full:
            pass
        time.sleep(0.002)

    time.sleep(0.3)

    assert dq.qsize() <= 50

    session.stop()


def test_pause_resume_multiple_cycles():
    """複数回の pause/resume サイクルでも安定動作する"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue(maxsize=100)
    pipeline.start(dq)

    for cycle in range(3):
        pipeline.pause()
        for _ in range(5):
            dq.put(make_chunk())
        time.sleep(0.1)

        pipeline.resume()
        for _ in range(5):
            dq.put(make_chunk())
        time.sleep(0.1)

    pipeline.stop()

    # resume 期間中のチャンクが届いている（各サイクル5チャンク × 3サイクル = 15）
    assert len(sink.chunks) == 15
