"""Pipeline のユニットテスト"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest

from pyflexaudio.events import EventBus
from pyflexaudio.pipeline import Pipeline
from pyflexaudio.processors.chain import ProcessorChain
from pyflexaudio.processors.level import LevelMeterProcessor
from pyflexaudio.types import AudioChunk

# conftest の MockSource / MockSink をインポート
from conftest import MockSink, MockSource


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_chunk(source_id: str = "microphone:test") -> AudioChunk:
    """テスト用シングルチャンクを生成"""
    data = np.random.randn(512, 1).astype(np.float32) * 0.1
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=16000,
        channels=1,
        source_id=source_id,
    )


def _start_pipeline_with_queue(pipeline: Pipeline) -> queue.Queue:
    """Pipeline を起動し data_queue を返す"""
    data_queue: queue.Queue = queue.Queue(maxsize=100)
    pipeline.start(data_queue)
    return data_queue


def _put_and_wait(pipeline: Pipeline, data_queue: queue.Queue, chunk: AudioChunk,
                  wait_sec: float = 0.5) -> None:
    """チャンクを投入してパイプラインスレッドが処理するのを待つ"""
    data_queue.put(chunk)
    time.sleep(wait_sec)
    pipeline.stop()


# ---------------------------------------------------------------------------
# テスト
# ---------------------------------------------------------------------------

class TestDataFlow:
    """MockSource → Pipeline → MockSink にチャンクが届く"""

    def test_chunk_delivered_to_sink(self):
        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = MockSink()
        pipeline.add_sink(sink)

        data_queue = _start_pipeline_with_queue(pipeline)

        chunk = _make_chunk()
        data_queue.put(chunk)
        time.sleep(0.3)

        pipeline.stop()

        assert len(sink.chunks) > 0, "Sink がチャンクを受信していない"

    def test_sink_flushed_and_closed_after_stop(self):
        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = MockSink()
        pipeline.add_sink(sink)

        data_queue = _start_pipeline_with_queue(pipeline)
        data_queue.put(_make_chunk())
        time.sleep(0.3)
        pipeline.stop()

        assert sink.flushed, "stop() 後に sink.flush() が呼ばれていない"
        assert sink.closed, "stop() 後に sink.close() が呼ばれていない"

    def test_multiple_chunks_all_delivered(self):
        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = MockSink()
        pipeline.add_sink(sink)

        data_queue = _start_pipeline_with_queue(pipeline)

        n = 5
        for _ in range(n):
            data_queue.put(_make_chunk())

        time.sleep(0.5)
        pipeline.stop()

        assert len(sink.chunks) == n, f"期待 {n} チャンク、受信 {len(sink.chunks)}"


class TestFanOut:
    """複数 Sink に同一チャンクが配信される"""

    def test_fanout_to_multiple_sinks(self):
        bus = EventBus()
        pipeline = Pipeline(bus)
        sinks = [MockSink() for _ in range(3)]
        for s in sinks:
            pipeline.add_sink(s)

        data_queue = _start_pipeline_with_queue(pipeline)
        data_queue.put(_make_chunk())
        time.sleep(0.3)
        pipeline.stop()

        for i, s in enumerate(sinks):
            assert len(s.chunks) > 0, f"Sink[{i}] がチャンクを受信していない"

    def test_fanout_same_chunk_object(self):
        """fan-out では全 Sink が同一 data オブジェクトを受け取る（read-only 契約）"""
        bus = EventBus()
        pipeline = Pipeline(bus)
        sinks = [MockSink() for _ in range(3)]
        for s in sinks:
            pipeline.add_sink(s)

        data_queue = _start_pipeline_with_queue(pipeline)
        data_queue.put(_make_chunk())
        time.sleep(0.3)
        pipeline.stop()

        # 全 Sink が同じ chunk オブジェクト（同一 data 配列）を受け取る
        chunks = [s.chunks[0] for s in sinks if s.chunks]
        assert len(chunks) == 3, "全 Sink がチャンクを受信していない"
        ref_data = chunks[0].data
        for c in chunks[1:]:
            assert c.data is ref_data, "fan-out で異なる data オブジェクトが渡された"


class TestPause:
    """pause 中は通常 Sink に配信されない"""

    def test_pause_stops_delivery_to_normal_sink(self):
        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = MockSink()
        pipeline.add_sink(sink)

        data_queue = _start_pipeline_with_queue(pipeline)
        pipeline.pause()

        for _ in range(3):
            data_queue.put(_make_chunk())
        time.sleep(0.3)
        pipeline.stop()

        assert len(sink.chunks) == 0, "pause 中に通常 Sink にチャンクが配信された"

    def test_pause_resume_restores_delivery(self):
        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = MockSink()
        pipeline.add_sink(sink)

        data_queue = _start_pipeline_with_queue(pipeline)
        pipeline.pause()
        time.sleep(0.1)
        pipeline.resume()

        data_queue.put(_make_chunk())
        time.sleep(0.3)
        pipeline.stop()

        assert len(sink.chunks) > 0, "resume 後にチャンクが配信されない"


class TestPauseExempt:
    """pause 中でも pause_exempt=True の Sink には配信される"""

    def test_pause_exempt_sink_receives_during_pause(self):
        bus = EventBus()
        pipeline = Pipeline(bus)
        normal_sink = MockSink(pause_exempt=False)
        exempt_sink = MockSink(pause_exempt=True)
        pipeline.add_sink(normal_sink)
        pipeline.add_sink(exempt_sink)

        data_queue = _start_pipeline_with_queue(pipeline)
        pipeline.pause()

        data_queue.put(_make_chunk())
        time.sleep(0.3)
        pipeline.stop()

        assert len(normal_sink.chunks) == 0, "pause 中に通常 Sink にチャンクが配信された"
        assert len(exempt_sink.chunks) > 0, "pause 中に pause_exempt Sink にチャンクが配信されない"


class TestSinkDisabled:
    """enabled=False の Sink にはスキップ"""

    def test_disabled_sink_not_written(self):
        bus = EventBus()
        pipeline = Pipeline(bus)
        disabled_sink = MockSink(enabled=False)
        enabled_sink = MockSink(enabled=True)
        pipeline.add_sink(disabled_sink)
        pipeline.add_sink(enabled_sink)

        data_queue = _start_pipeline_with_queue(pipeline)
        data_queue.put(_make_chunk())
        time.sleep(0.3)
        pipeline.stop()

        assert len(disabled_sink.chunks) == 0, "disabled Sink にチャンクが書き込まれた"
        assert len(enabled_sink.chunks) > 0, "enabled Sink にチャンクが配信されない"

    def test_disabled_sink_still_flushed_and_closed(self):
        """disabled でも stop() 時に flush/close が呼ばれる"""
        bus = EventBus()
        pipeline = Pipeline(bus)
        disabled_sink = MockSink(enabled=False)
        pipeline.add_sink(disabled_sink)

        data_queue = _start_pipeline_with_queue(pipeline)
        pipeline.stop()

        assert disabled_sink.flushed
        assert disabled_sink.closed


class TestShutdown:
    """stop() で全 Sink が flush + close される"""

    def test_stop_flushes_and_closes_all_sinks(self):
        bus = EventBus()
        pipeline = Pipeline(bus)
        sinks = [MockSink() for _ in range(3)]
        for s in sinks:
            pipeline.add_sink(s)

        data_queue = _start_pipeline_with_queue(pipeline)
        pipeline.stop()

        for i, s in enumerate(sinks):
            assert s.flushed, f"Sink[{i}] が flush されていない"
            assert s.closed, f"Sink[{i}] が close されていない"

    def test_stop_idempotent_no_exception(self):
        """stop() を 2 回呼んでもエラーにならない"""
        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = MockSink()
        pipeline.add_sink(sink)

        data_queue = _start_pipeline_with_queue(pipeline)
        pipeline.stop()
        # 2 回目の stop（data_queue が None のケースを確認）
        pipeline.stop()


class TestProcessorChain:
    """main_chain のプロセッサが実行される"""

    def test_level_meter_processor_sets_level_db(self):
        """LevelMeterProcessor が level_db を設定する"""
        bus = EventBus()
        pipeline = Pipeline(bus)
        chain = ProcessorChain([LevelMeterProcessor()])
        pipeline.set_main_chain(chain)

        received: list[AudioChunk] = []

        class CaptureSink:
            enabled = True
            pause_exempt = False

            def write(self, chunk):
                received.append(chunk)

            def flush(self):
                pass

            def close(self):
                pass

        pipeline.add_sink(CaptureSink())

        data_queue = _start_pipeline_with_queue(pipeline)
        data_queue.put(_make_chunk())
        time.sleep(0.3)
        pipeline.stop()

        assert len(received) > 0, "Sink がチャンクを受信していない"
        assert received[0].level_db is not None, "level_db が設定されていない"
        assert isinstance(received[0].level_db, float), "level_db が float でない"

    def test_processor_chain_executed_in_order(self):
        """プロセッサが追加順に実行される"""
        call_order: list[str] = []

        class OrderProc:
            def __init__(self, name: str):
                self._name = name

            def process(self, chunk: AudioChunk) -> AudioChunk:
                call_order.append(self._name)
                return chunk

            def reset(self):
                pass

        bus = EventBus()
        pipeline = Pipeline(bus)
        chain = ProcessorChain([OrderProc("A"), OrderProc("B"), OrderProc("C")])
        pipeline.set_main_chain(chain)

        sink = MockSink()
        pipeline.add_sink(sink)

        data_queue = _start_pipeline_with_queue(pipeline)
        data_queue.put(_make_chunk())
        time.sleep(0.3)
        pipeline.stop()

        assert call_order == ["A", "B", "C"], f"実行順序が正しくない: {call_order}"


class TestAddRemoveSink:
    """Sink の動的追加/除去"""

    def test_remove_sink_stops_delivery(self):
        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = MockSink()
        sink_id = pipeline.add_sink(sink)

        data_queue = _start_pipeline_with_queue(pipeline)

        # Sink を除去してからチャンクを投入
        pipeline.remove_sink(sink_id)
        data_queue.put(_make_chunk())
        time.sleep(0.3)
        pipeline.stop()

        assert len(sink.chunks) == 0, "除去した Sink にチャンクが配信された"

    def test_add_sink_after_start(self):
        """Pipeline 起動後に Sink を追加しても配信される"""
        bus = EventBus()
        pipeline = Pipeline(bus)

        data_queue = _start_pipeline_with_queue(pipeline)

        # 起動後に Sink を追加
        sink = MockSink()
        pipeline.add_sink(sink)

        data_queue.put(_make_chunk())
        time.sleep(0.3)
        pipeline.stop()

        assert len(sink.chunks) > 0, "起動後に追加した Sink にチャンクが配信されない"


class TestMockSourceIntegration:
    """MockSource → Pipeline → MockSink の end-to-end"""

    def test_mock_source_feeds_pipeline(self):
        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = MockSink()
        pipeline.add_sink(sink)

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=100)
        pipeline.start(data_queue)

        source = MockSource(sample_rate=16000, channels=1, chunk_frames=512)
        source.open(data_queue, stop_event)

        time.sleep(0.5)
        stop_event.set()
        source.close()
        pipeline.stop()

        assert len(sink.chunks) > 0, "MockSource から Pipeline にチャンクが届いていない"
        assert sink.flushed
        assert sink.closed
