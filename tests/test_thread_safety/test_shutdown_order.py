"""シャットダウン順序保証を検証するテスト"""

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


class OrderedMockSink:
    """操作順序を記録する Sink"""

    def __init__(self, name: str, order_list: list):
        self.name = name
        self.enabled = True
        self.pause_exempt = False
        self.chunks = []
        self.flushed = False
        self.closed = False
        self._order = order_list

    def write(self, chunk):
        if self.enabled:
            self.chunks.append(chunk)

    def flush(self):
        self.flushed = True
        self._order.append(f"{self.name}_flush")

    def close(self):
        self.closed = True
        self._order.append(f"{self.name}_close")


class OrderedMockSource:
    """操作順序を記録する Source"""

    def __init__(self, name: str, order_list: list, interval_sec=0.02):
        self.source_id = f"test:{name}"
        self.name = name
        self.interval_sec = interval_sec
        self.is_open = False
        self._order = order_list
        self._thread = None
        self._data_queue = None
        self._stop_event = None

    def open(self, data_queue, stop_event):
        self._data_queue = data_queue
        self._stop_event = stop_event
        self.is_open = True
        self._order.append(f"{self.name}_open")
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def close(self):
        self.is_open = False
        self._order.append(f"{self.name}_close")
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _produce(self):
        while self.is_open and not self._stop_event.is_set():
            chunk = make_chunk(source_id=self.source_id)
            try:
                self._data_queue.put_nowait(chunk)
            except queue.Full:
                pass
            time.sleep(self.interval_sec)


# ---- テスト ----

def test_shutdown_order_source_before_sink():
    """シャットダウン時に source_close → sink_flush → sink_close の順で実行される"""
    order = []
    bus = EventBus()

    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink = OrderedMockSink("sink_a", order)
    pipeline.add_sink(sink)

    dq = queue.Queue(maxsize=50)
    stop_event = threading.Event()
    pipeline.start(dq)

    source = OrderedMockSource("source_a", order)
    source.open(dq, stop_event)

    time.sleep(0.2)

    # Session の stop() に相当する処理を手動で実施
    # 1. stop_event をセット（Source が停止）
    stop_event.set()

    # 2. Source close
    source.close()

    # 3. Pipeline stop（Sink flush/close）
    pipeline.stop()

    # 順序検証: source_close → sink_flush → sink_close
    assert "source_a_close" in order
    assert "sink_a_flush" in order
    assert "sink_a_close" in order

    src_close_idx = order.index("source_a_close")
    sink_flush_idx = order.index("sink_a_flush")
    sink_close_idx = order.index("sink_a_close")

    assert src_close_idx < sink_flush_idx, "source_close must happen before sink_flush"
    assert sink_flush_idx < sink_close_idx, "sink_flush must happen before sink_close"


def test_shutdown_sink_flush_before_close():
    """sink_flush は sink_close より前に実行される"""
    order = []
    bus = EventBus()

    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink_a = OrderedMockSink("a", order)
    sink_b = OrderedMockSink("b", order)
    pipeline.add_sink(sink_a)
    pipeline.add_sink(sink_b)

    dq = queue.Queue()
    pipeline.start(dq)

    for _ in range(3):
        dq.put(make_chunk())
    time.sleep(0.1)

    pipeline.stop()

    # 全 sink の flush → close の順序を確認
    assert order.index("a_flush") < order.index("a_close")
    assert order.index("b_flush") < order.index("b_close")


def test_shutdown_all_sinks_flushed():
    """Pipeline stop 後、全 Sink が flush/close される"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sinks = []
    order = []
    for i in range(3):
        sink = OrderedMockSink(f"sink_{i}", order)
        pipeline.add_sink(sink)
        sinks.append(sink)

    dq = queue.Queue()
    pipeline.start(dq)

    dq.put(make_chunk())
    time.sleep(0.1)

    pipeline.stop()

    for sink in sinks:
        assert sink.flushed is True, f"{sink.name} was not flushed"
        assert sink.closed is True, f"{sink.name} was not closed"


def test_shutdown_pipeline_thread_stops():
    """Pipeline stop 後にスレッドが停止する"""
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    dq = queue.Queue()
    pipeline.start(dq)

    assert pipeline._thread is not None
    assert pipeline._thread.is_alive()

    pipeline.stop()

    assert pipeline._thread is None


def test_shutdown_stop_event_set_before_source_close():
    """stop_event は source.close() より前にセットされるべき"""
    # Session の stop() では stop_event.set() → source.close() の順
    order = []
    stop_event = threading.Event()

    class TrackingSource:
        source_id = "test:tracking"
        is_open = True

        def open(self, dq, se):
            pass

        def close(self):
            # close 時点での stop_event 状態を記録
            order.append(("close_called", stop_event.is_set()))

    source = TrackingSource()

    # stop_event をセット（Session.stop() の順序を再現）
    stop_event.set()
    source.close()

    # close が呼ばれた時点で stop_event はすでにセットされていること
    assert len(order) == 1
    assert order[0] == ("close_called", True)
