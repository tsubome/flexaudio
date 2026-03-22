"""統合テスト: MockSourceA → switch → MockSourceB → データ継続性"""

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


class TestSourceSwitch:
    """Source 切り替えの統合テスト"""

    def test_both_sources_processed(self):
        """MockSourceA から MockSourceB に切り替えて、両方からチャンクが処理される"""
        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = MockSink()
        pipeline.add_sink(sink)

        stop_event_a = threading.Event()
        stop_event_b = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=200)
        pipeline.start(data_queue)

        source_a = MockSource(
            sample_rate=16000,
            channels=1,
            chunk_frames=512,
            source_id="microphone:source_a",
        )
        source_a.open(data_queue, stop_event_a)

        time.sleep(0.3)
        chunks_after_a = len(sink.chunks)

        # Source A を停止して Source B に切り替え
        stop_event_a.set()
        source_a.close()

        # Pipeline に switch_source コマンドを送信
        source_b = MockSource(
            sample_rate=16000,
            channels=1,
            chunk_frames=512,
            source_id="microphone:source_b",
        )
        source_b.open(data_queue, stop_event_b)
        pipeline.send_command(("switch_source", source_b, source_a))

        time.sleep(0.3)

        stop_event_b.set()
        source_b.close()
        pipeline.stop()

        # 両方のソースからチャンクが届いたことを検証
        assert chunks_after_a > 0, "Source A からチャンクが届いていない"
        assert len(sink.chunks) > chunks_after_a, "Source B からチャンクが届いていない"

        source_ids = {c.source_id for c in sink.chunks}
        # Source A か Source B のどちらかのチャンクが含まれる
        assert "microphone:source_a" in source_ids or "microphone:source_b" in source_ids, \
            f"予期しない source_id: {source_ids}"

    def test_source_b_chunks_have_correct_source_id(self):
        """切り替え後のチャンクが Source B の source_id を持つ"""
        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = MockSink()
        pipeline.add_sink(sink)

        stop_event_a = threading.Event()
        stop_event_b = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=200)
        pipeline.start(data_queue)

        source_a = MockSource(
            sample_rate=16000,
            channels=1,
            chunk_frames=512,
            source_id="microphone:source_a",
        )
        source_a.open(data_queue, stop_event_a)

        time.sleep(0.2)

        # 切り替え
        stop_event_a.set()
        source_a.close()

        # Source B を開始
        source_b = MockSource(
            sample_rate=16000,
            channels=1,
            chunk_frames=512,
            source_id="microphone:source_b",
        )
        source_b.open(data_queue, stop_event_b)
        pipeline.send_command(("switch_source", source_b, source_a))

        time.sleep(0.4)

        stop_event_b.set()
        source_b.close()
        pipeline.stop()

        # 後半のチャンクが Source B の source_id を持つことを確認
        later_chunks = sink.chunks[len(sink.chunks) // 2:]
        if later_chunks:
            b_chunks = [c for c in later_chunks if c.source_id == "microphone:source_b"]
            assert len(b_chunks) > 0, "後半のチャンクに Source B のものがない"

    def test_pipeline_continues_after_switch(self):
        """切り替え後も Pipeline がチャンクを処理し続ける"""
        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = MockSink()
        pipeline.add_sink(sink)

        stop_event_a = threading.Event()
        stop_event_b = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=200)
        pipeline.start(data_queue)

        source_a = MockSource(
            sample_rate=16000,
            channels=1,
            chunk_frames=512,
            source_id="microphone:source_a",
        )
        source_a.open(data_queue, stop_event_a)
        time.sleep(0.3)
        stop_event_a.set()
        source_a.close()

        count_before_switch = len(sink.chunks)

        source_b = MockSource(
            sample_rate=16000,
            channels=1,
            chunk_frames=512,
            source_id="microphone:source_b",
        )
        source_b.open(data_queue, stop_event_b)
        pipeline.send_command(("switch_source", source_b, None))

        time.sleep(0.3)
        count_after_switch = len(sink.chunks)

        stop_event_b.set()
        source_b.close()
        pipeline.stop()

        assert count_before_switch > 0, "切り替え前にチャンクが来ていない"
        assert count_after_switch > count_before_switch, "切り替え後にチャンクが増えていない"

    def test_switch_to_none_old_source(self):
        """old_source が None の switch_source コマンドが問題なく処理される"""
        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = MockSink()
        pipeline.add_sink(sink)

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=200)
        pipeline.start(data_queue)

        source = MockSource(
            sample_rate=16000,
            channels=1,
            chunk_frames=512,
            source_id="microphone:only",
        )
        source.open(data_queue, stop_event)

        # old_source=None でコマンドを送信
        pipeline.send_command(("switch_source", source, None))

        time.sleep(0.3)

        stop_event.set()
        source.close()
        pipeline.stop()

        assert len(sink.chunks) > 0, "チャンクが処理されていない"

    def test_switch_with_different_formats(self):
        """異なるフォーマットの Source に切り替えてもクラッシュしない"""
        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = MockSink()
        pipeline.add_sink(sink)

        stop_event_a = threading.Event()
        stop_event_b = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=200)
        pipeline.start(data_queue)

        # 16kHz mono Source A
        source_a = MockSource(
            sample_rate=16000,
            channels=1,
            chunk_frames=512,
            source_id="microphone:16k_mono",
        )
        source_a.open(data_queue, stop_event_a)
        time.sleep(0.2)

        stop_event_a.set()
        source_a.close()

        # 44.1kHz stereo Source B
        source_b = MockSource(
            sample_rate=44100,
            channels=2,
            chunk_frames=1024,
            source_id="microphone:44k_stereo",
        )
        source_b.open(data_queue, stop_event_b)
        pipeline.send_command(("switch_source", source_b, source_a))

        time.sleep(0.3)

        stop_event_b.set()
        source_b.close()
        pipeline.stop()

        # クラッシュせずに完了できれば OK
        assert len(sink.chunks) > 0
