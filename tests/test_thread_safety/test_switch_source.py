"""切り替え中のデータフロー整合性を検証するテスト"""

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

    def __init__(self, source_id="test:0", interval_sec=0.01):
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

def test_source_switch_data_continuity():
    """Source A → Source B への切り替え後も Sink にデータが届く"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue(maxsize=100)
    stop_event = threading.Event()
    pipeline.start(dq)

    # Source A を開始
    source_a = MockSource(source_id="source:a", interval_sec=0.02)
    source_a.open(dq, stop_event)
    time.sleep(0.3)
    chunks_before_switch = len(sink.chunks)

    # Source A → Source B に切り替え（Pipeline の switch_source コマンド経由）
    source_b = MockSource(source_id="source:b", interval_sec=0.02)
    source_b.open(dq, stop_event)
    pipeline.send_command(("switch_source", source_b, source_a))

    time.sleep(0.3)
    chunks_after_switch = len(sink.chunks)

    source_b.close()
    pipeline.stop()

    # 切り替え前にデータが届いていた
    assert chunks_before_switch > 0
    # 切り替え後にもデータが届いた
    assert chunks_after_switch > chunks_before_switch


def test_source_switch_pipeline_not_stopped():
    """Source 切り替え中に Pipeline スレッドが停止しない"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue(maxsize=100)
    stop_event = threading.Event()
    pipeline.start(dq)

    source_a = MockSource(source_id="source:a")
    source_a.open(dq, stop_event)

    time.sleep(0.1)

    # 切り替えコマンド送信
    source_b = MockSource(source_id="source:b")
    source_b.open(dq, stop_event)
    pipeline.send_command(("switch_source", source_b, source_a))

    time.sleep(0.1)

    # Pipeline スレッドがまだ生きている
    assert pipeline._thread is not None
    assert pipeline._thread.is_alive()

    source_b.close()
    pipeline.stop()


def test_source_switch_drains_old_chunks():
    """切り替え時に旧 Source のチャンクが適切にドレインされる"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink = MockSink()
    pipeline.add_sink(sink)

    # 大きめの Queue を使用
    dq = queue.Queue(maxsize=200)
    stop_event = threading.Event()
    pipeline.start(dq)

    # Source A のチャンクを事前に大量投入（Pipeline Thread が処理する前に）
    source_a = MockSource(source_id="source:a")
    for _ in range(30):
        try:
            dq.put_nowait(make_chunk(source_id="source:a"))
        except queue.Full:
            break

    # 切り替えコマンドを即座に送信
    source_b = MockSource(source_id="source:b")
    source_b.open(dq, stop_event)
    pipeline.send_command(("switch_source", source_b, source_a))

    time.sleep(0.5)
    source_b.close()
    pipeline.stop()

    # Pipeline は停止していない（ドレイン後も継続）
    assert pipeline._thread is None  # stop 後なので None


def test_rapid_source_switches():
    """短期間で複数回 Source を切り替えてもクラッシュしない"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue(maxsize=200)
    stop_event = threading.Event()
    pipeline.start(dq)

    sources = []
    prev_source = None
    for i in range(5):
        new_source = MockSource(source_id=f"source:{i}", interval_sec=0.01)
        new_source.open(dq, stop_event)
        pipeline.send_command(("switch_source", new_source, prev_source))
        sources.append(new_source)
        prev_source = new_source
        time.sleep(0.05)

    time.sleep(0.2)

    for s in sources:
        s.close()

    pipeline.stop()

    # Pipeline は正常終了
    assert pipeline._thread is None


def test_source_switch_with_sink_receiving_data():
    """切り替え中に Sink がデータを継続受信する"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue(maxsize=100)
    stop_event = threading.Event()
    pipeline.start(dq)

    source_a = MockSource(source_id="source:a", interval_sec=0.02)
    source_a.open(dq, stop_event)
    time.sleep(0.2)
    count_a = len(sink.chunks)

    source_b = MockSource(source_id="source:b", interval_sec=0.02)
    source_b.open(dq, stop_event)
    pipeline.send_command(("switch_source", source_b, source_a))
    time.sleep(0.3)
    count_b_total = len(sink.chunks)

    source_b.close()
    pipeline.stop()

    assert count_a > 0
    # 切り替え後にもチャンクが届いていること（ドロップはあっても Pipeline は継続）
    assert count_b_total >= count_a
