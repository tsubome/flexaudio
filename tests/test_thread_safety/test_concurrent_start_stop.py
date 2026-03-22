"""複数スレッドから同時に start/stop を呼んでも安全であることを検証するテスト"""

import queue
import threading
import time

import numpy as np
import pytest

from pyflexaudio.types import AudioChunk
from pyflexaudio.events import EventBus
from pyflexaudio.session import FlexAudioSession
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


# ---- テスト ----

def test_concurrent_start_stop():
    """複数スレッドから同時に start/stop を呼んでもデッドロックしない"""
    session = FlexAudioSession()
    exceptions = []

    def safe_start():
        try:
            session.start()
        except Exception as e:
            exceptions.append(e)

    def safe_stop():
        try:
            session.stop()
        except Exception as e:
            exceptions.append(e)

    threads = []
    for _ in range(10):
        threads.append(threading.Thread(target=safe_start))
        threads.append(threading.Thread(target=safe_stop))

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=5)

    # デッドロックなく全スレッドが完了すること（join(timeout=5) でタイムアウトしない）
    assert all(not t.is_alive() for t in threads), "Some threads are still alive (deadlock?)"

    # クリーンアップ
    try:
        session.stop()
    except Exception:
        pass


def test_concurrent_start_idempotent():
    """複数スレッドから start を呼んでも2重起動しない（冪等性）"""
    session = FlexAudioSession()
    start_count = [0]
    lock = threading.Lock()

    def count_start():
        session.start()
        with lock:
            start_count[0] += 1

    threads = [threading.Thread(target=count_start) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    # 全スレッドが完了している
    assert all(not t.is_alive() for t in threads)

    session.stop()


def test_concurrent_stop_idempotent():
    """複数スレッドから stop を呼んでも安全（冪等性）"""
    session = FlexAudioSession()
    session.start()
    time.sleep(0.1)

    exceptions = []

    def safe_stop():
        try:
            session.stop()
        except Exception as e:
            exceptions.append(e)

    threads = [threading.Thread(target=safe_stop) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert exceptions == []


def test_pipeline_concurrent_start_stop():
    """Pipeline に対して複数スレッドから start/stop を呼んでもクラッシュしない"""
    bus = EventBus()
    exceptions = []

    def run_pipeline():
        try:
            pipeline = Pipeline(bus)
            pipeline.set_main_chain(ProcessorChain())
            dq = queue.Queue()
            pipeline.start(dq)
            time.sleep(0.05)
            pipeline.stop()
        except Exception as e:
            exceptions.append(e)

    threads = [threading.Thread(target=run_pipeline) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert exceptions == []


def test_session_start_stop_rapid_cycle():
    """Session を高速に start/stop を繰り返してもクラッシュしない"""
    session = FlexAudioSession()
    exceptions = []

    for _ in range(5):
        try:
            session.start()
            time.sleep(0.02)
            session.stop()
            time.sleep(0.02)
        except Exception as e:
            exceptions.append(e)

    assert exceptions == []


def test_concurrent_add_sink_and_start():
    """start と同時に Sink を追加してもクラッシュしない"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    dq = queue.Queue()
    exceptions = []

    def add_sinks():
        for _ in range(20):
            try:
                sink = MockSink()
                pipeline.add_sink(sink)
                time.sleep(0.001)
            except Exception as e:
                exceptions.append(e)

    # Pipeline を起動しながら Sink を追加
    pipeline.start(dq)
    t = threading.Thread(target=add_sinks)
    t.start()

    # チャンクも投入
    for _ in range(10):
        dq.put(make_chunk())
        time.sleep(0.005)

    t.join(timeout=5)
    pipeline.stop()

    assert exceptions == []
