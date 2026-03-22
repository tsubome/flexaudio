"""Source.read() タイムアウト時にパイプラインが継続することを検証するテスト"""

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


class DelayedSource:
    """一定時間後にチャンクを1つだけ送るソース"""

    def __init__(self, delay_sec=1.0, source_id="test:delayed"):
        self.delay_sec = delay_sec
        self.source_id = source_id
        self.is_open = False
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
        self.is_open = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _produce(self):
        # delay_sec 待機してからチャンクを投入
        start = time.monotonic()
        while time.monotonic() - start < self.delay_sec:
            if self._stop_event.is_set() or not self.is_open:
                return
            time.sleep(0.05)

        if not self._stop_event.is_set() and self.is_open:
            chunk = make_chunk(source_id=self.source_id)
            try:
                self._data_queue.put_nowait(chunk)
            except queue.Full:
                pass


# ---- テスト ----

def test_pipeline_continues_on_empty_queue():
    """空の Queue で Pipeline を開始し、1秒後にチャンクを投入しても正常に処理される"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue()
    pipeline.start(dq)

    # Pipeline スレッドは空の Queue に対して timeout=1.0 で待機し続けるはず
    time.sleep(0.3)
    assert pipeline._thread is not None
    assert pipeline._thread.is_alive()

    # 1秒後にチャンクを投入
    time.sleep(0.7)
    chunk = make_chunk()
    dq.put(chunk)
    time.sleep(0.2)

    pipeline.stop()

    # チャンクは正常に処理される
    assert len(sink.chunks) == 1


def test_pipeline_survives_long_empty_period():
    """長期間チャンクが来なくても Pipeline は停止しない"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue()
    pipeline.start(dq)

    # 2秒間チャンクを送らない（Pipeline は timeout=1.0 を2回経験）
    time.sleep(2.0)

    assert pipeline._thread is not None
    assert pipeline._thread.is_alive()

    # その後チャンクを投入
    dq.put(make_chunk())
    time.sleep(0.2)

    pipeline.stop()

    assert len(sink.chunks) == 1


def test_pipeline_processes_delayed_source_chunk():
    """DelayedSource からのチャンクが遅延後に正しく処理される"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue()
    stop_event = threading.Event()
    pipeline.start(dq)

    # 1秒後にチャンクを生成するソースを開始
    source = DelayedSource(delay_sec=1.0)
    source.open(dq, stop_event)

    # チャンク到着を待つ（2秒程度）
    time.sleep(2.0)
    source.close()

    pipeline.stop()

    # チャンクが処理されていること
    assert len(sink.chunks) >= 1


def test_pipeline_empty_then_burst():
    """空の状態からバーストチャンクが来ても全て処理される"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue(maxsize=100)
    pipeline.start(dq)

    # しばらく空
    time.sleep(0.3)

    # バーストで5チャンク投入
    for _ in range(5):
        dq.put(make_chunk())

    time.sleep(0.3)
    pipeline.stop()

    assert len(sink.chunks) == 5
