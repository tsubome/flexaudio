"""デバイス切断のシミュレーションテスト"""

import queue
import threading
import time

import numpy as np
import pytest

from pyflexaudio.types import AudioChunk
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


class MockSource:
    """テスト用のソース。is_open を false にすると生成を停止する"""

    def __init__(self, source_id="test:0", interval_sec=0.02):
        self.source_id = source_id
        self.interval_sec = interval_sec
        self.is_open = False
        self.chunks_sent = 0
        self._thread = None
        self._data_queue = None
        self._stop_event = None

    def open(self, data_queue, stop_event):
        self._data_queue = data_queue
        self._stop_event = stop_event
        self.is_open = True
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def close(self):
        """デバイス切断をシミュレート"""
        self.is_open = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _produce(self):
        while self.is_open and not self._stop_event.is_set():
            chunk = make_chunk(source_id=self.source_id)
            try:
                self._data_queue.put_nowait(chunk)
                self.chunks_sent += 1
            except queue.Full:
                pass
            time.sleep(self.interval_sec)


# ---- テスト ----

def test_source_close_during_operation():
    """MockSource を途中で close しても Pipeline は停止しない"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue(maxsize=100)
    stop_event = threading.Event()
    pipeline.start(dq)

    source = MockSource()
    source.open(dq, stop_event)

    # しばらく動かす
    time.sleep(0.3)
    chunks_before_close = len(sink.chunks)

    # デバイス切断をシミュレート
    source.close()
    assert not source.is_open

    # Pipeline スレッドはまだ生きている（チャンクが来なくなっただけ）
    time.sleep(0.1)
    assert pipeline._thread is not None
    assert pipeline._thread.is_alive()

    pipeline.stop()

    # 切断前にチャンクが届いていた
    assert chunks_before_close > 0


def test_source_close_then_reopen():
    """Source を close した後に別の Source を開いても Pipeline は正常動作"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue(maxsize=100)
    stop_event = threading.Event()
    pipeline.start(dq)

    # 最初の Source
    source_a = MockSource(source_id="test:a")
    source_a.open(dq, stop_event)
    time.sleep(0.2)
    source_a.close()

    chunks_from_a = len(sink.chunks)
    assert chunks_from_a > 0

    # 2つ目の Source（再接続）
    source_b = MockSource(source_id="test:b")
    source_b.open(dq, stop_event)
    time.sleep(0.2)
    source_b.close()

    pipeline.stop()

    chunks_from_b = len(sink.chunks) - chunks_from_a
    assert chunks_from_b > 0


def test_pipeline_not_stopped_after_source_close():
    """Source.close() 後も Pipeline._stop_event はセットされていない"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue(maxsize=50)
    stop_event = threading.Event()
    pipeline.start(dq)

    source = MockSource()
    source.open(dq, stop_event)
    time.sleep(0.1)

    source.close()
    time.sleep(0.1)

    # Pipeline の stop_event はセットされていないこと
    assert not pipeline._stop_event.is_set()

    pipeline.stop()


def test_source_close_sends_no_sentinel():
    """MockSource の close は Queue にセンチネルを送らない"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue(maxsize=50)
    stop_event = threading.Event()
    pipeline.start(dq)

    source = MockSource(interval_sec=0.1)
    source.open(dq, stop_event)
    time.sleep(0.15)
    source.close()

    # source.close() 後も Pipeline は動き続ける
    time.sleep(0.1)
    assert pipeline._thread.is_alive()

    # 手動でチャンクを投入しても処理される
    dq.put(make_chunk(source_id="manual:0"))
    time.sleep(0.1)

    pipeline.stop()

    # 手動チャンクが届いている
    assert any(c.source_id == "manual:0" for c in sink.chunks)
