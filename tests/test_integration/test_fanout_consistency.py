"""統合テスト: fan-out で各 Sink が同一データを受け取ることの検証"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest

from pyflexaudio.events import EventBus
from pyflexaudio.pipeline import Pipeline
from pyflexaudio.types import AudioChunk

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import MockSink, MockSource


class TestFanoutConsistency:
    """fan-out で各 Sink が同一データを受け取ることの検証"""

    def test_three_sinks_receive_same_data_object(self):
        """3 つの Sink が同じ data オブジェクトを受け取る（read-only 契約）"""
        bus = EventBus()
        pipeline = Pipeline(bus)
        sinks = [MockSink() for _ in range(3)]
        for s in sinks:
            pipeline.add_sink(s)

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=100)
        pipeline.start(data_queue)

        source = MockSource(sample_rate=16000, channels=1, chunk_frames=512)
        source.open(data_queue, stop_event)

        time.sleep(0.5)

        stop_event.set()
        source.close()
        pipeline.stop()

        # 全 Sink が同じチャンク数を受け取る
        counts = [len(s.chunks) for s in sinks]
        assert all(c > 0 for c in counts), f"チャンクが届いていない Sink がある: {counts}"
        assert counts[0] == counts[1] == counts[2], \
            f"Sink 間でチャンク数が一致しない: {counts}"

        # 各 Sink の i 番目のチャンクが同一オブジェクト
        n = counts[0]
        for i in range(n):
            chunks = [sinks[j].chunks[i] for j in range(3)]
            # 同一 chunk オブジェクト（is で確認）
            assert chunks[0] is chunks[1] is chunks[2], \
                f"chunk[{i}] が同一オブジェクトでない"
            # data は同一配列
            assert chunks[0].data is chunks[1].data is chunks[2].data, \
                f"chunk[{i}].data が同一オブジェクトでない"

    def test_chunk_data_content_identical(self):
        """各 Sink で受け取った data の内容が同一であることを検証（値チェック）"""
        bus = EventBus()
        pipeline = Pipeline(bus)
        sinks = [MockSink() for _ in range(3)]
        for s in sinks:
            pipeline.add_sink(s)

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=100)
        pipeline.start(data_queue)

        source = MockSource(sample_rate=16000, channels=1, chunk_frames=512)
        source.open(data_queue, stop_event)

        time.sleep(0.5)

        stop_event.set()
        source.close()
        pipeline.stop()

        assert all(len(s.chunks) > 0 for s in sinks)
        n = min(len(s.chunks) for s in sinks)

        for i in range(n):
            data_0 = sinks[0].chunks[i].data
            data_1 = sinks[1].chunks[i].data
            data_2 = sinks[2].chunks[i].data
            assert np.array_equal(data_0, data_1), f"chunk[{i}] の data が Sink0/1 で異なる"
            assert np.array_equal(data_0, data_2), f"chunk[{i}] の data が Sink0/2 で異なる"

    def test_single_chunk_fanout(self):
        """単一チャンクを直接投入した場合の fan-out 検証"""
        bus = EventBus()
        pipeline = Pipeline(bus)
        sinks = [MockSink() for _ in range(3)]
        for s in sinks:
            pipeline.add_sink(s)

        data_queue: queue.Queue = queue.Queue(maxsize=10)
        pipeline.start(data_queue)

        # 既知の値を持つチャンク
        fixed_data = np.ones((512, 1), dtype=np.float32) * 0.42
        chunk = AudioChunk(
            data=fixed_data,
            timestamp=time.monotonic(),
            sample_rate=16000,
            channels=1,
            source_id="test:fanout",
        )
        data_queue.put(chunk)
        time.sleep(0.3)
        pipeline.stop()

        for i, s in enumerate(sinks):
            assert len(s.chunks) == 1, f"Sink[{i}] のチャンク数が 1 でない: {len(s.chunks)}"
            assert s.chunks[0].data is fixed_data, \
                f"Sink[{i}] が元の data オブジェクトを受け取っていない"

    def test_chunk_metadata_identical_across_sinks(self):
        """各 Sink のチャンクのメタデータが同一であることを検証"""
        bus = EventBus()
        pipeline = Pipeline(bus)
        sinks = [MockSink() for _ in range(3)]
        for s in sinks:
            pipeline.add_sink(s)

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=100)
        pipeline.start(data_queue)

        source = MockSource(
            sample_rate=16000,
            channels=1,
            chunk_frames=512,
            source_id="microphone:fanout_test",
        )
        source.open(data_queue, stop_event)

        time.sleep(0.5)

        stop_event.set()
        source.close()
        pipeline.stop()

        assert all(len(s.chunks) > 0 for s in sinks)
        n = min(len(s.chunks) for s in sinks)

        for i in range(n):
            chunks = [sinks[j].chunks[i] for j in range(3)]
            # メタデータが全て同一オブジェクト（chunk 自体が同一）
            assert chunks[0] is chunks[1], f"chunk[{i}] が Sink0/1 で同一オブジェクトでない"
            assert chunks[0] is chunks[2], f"chunk[{i}] が Sink0/2 で同一オブジェクトでない"

    def test_fanout_with_enabled_disabled_mix(self):
        """enabled/disabled Sink が混在しても enabled Sink が同一データを受け取る"""
        bus = EventBus()
        pipeline = Pipeline(bus)

        enabled_sinks = [MockSink(enabled=True) for _ in range(2)]
        disabled_sink = MockSink(enabled=False)

        for s in enabled_sinks:
            pipeline.add_sink(s)
        pipeline.add_sink(disabled_sink)

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=100)
        pipeline.start(data_queue)

        source = MockSource(sample_rate=16000, channels=1, chunk_frames=512)
        source.open(data_queue, stop_event)

        time.sleep(0.5)

        stop_event.set()
        source.close()
        pipeline.stop()

        # disabled Sink にはチャンクが届いていない
        assert len(disabled_sink.chunks) == 0, "disabled Sink にチャンクが配信された"

        # enabled Sink は同じチャンクを受け取っている
        assert len(enabled_sinks[0].chunks) > 0
        assert len(enabled_sinks[1].chunks) > 0
        n = min(len(s.chunks) for s in enabled_sinks)
        for i in range(n):
            assert enabled_sinks[0].chunks[i] is enabled_sinks[1].chunks[i], \
                f"chunk[{i}] が enabled Sink 間で同一オブジェクトでない"

    def test_fanout_ordering_preserved(self):
        """各 Sink でチャンクが到着順に格納される（timestamp で確認）"""
        bus = EventBus()
        pipeline = Pipeline(bus)
        sinks = [MockSink() for _ in range(2)]
        for s in sinks:
            pipeline.add_sink(s)

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=100)
        pipeline.start(data_queue)

        source = MockSource(sample_rate=16000, channels=1, chunk_frames=512)
        source.open(data_queue, stop_event)

        time.sleep(0.5)

        stop_event.set()
        source.close()
        pipeline.stop()

        assert all(len(s.chunks) > 1 for s in sinks)

        for s in sinks:
            timestamps = [c.timestamp for c in s.chunks]
            for j in range(1, len(timestamps)):
                assert timestamps[j] >= timestamps[j - 1], \
                    f"チャンクの順序が乱れている: {timestamps[j-1]} > {timestamps[j]}"
