"""統合テスト: MockSource → CallbackSink → 受信チャンク検証"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest

from pyflexaudio.events import EventBus
from pyflexaudio.pipeline import Pipeline
from pyflexaudio.sinks.callback import CallbackSink
from pyflexaudio.types import AudioChunk

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import MockSource


class TestMicToCallback:
    """MockSource → Pipeline → CallbackSink の統合テスト"""

    def test_callback_called_with_chunks(self):
        """コールバックが呼ばれ、チャンクが届くことを検証"""
        received: list[AudioChunk] = []

        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = CallbackSink(callback=received.append)
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

        assert len(received) > 0, "コールバックが呼ばれていない"

    def test_chunk_shape_correct(self):
        """チャンクの shape が (frames, channels) であることを検証"""
        received: list[AudioChunk] = []

        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = CallbackSink(callback=received.append)
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

        assert len(received) > 0
        chunk = received[0]
        assert chunk.data.ndim == 2, f"data.ndim が 2 でない: {chunk.data.ndim}"
        assert chunk.data.shape[1] == 1, f"channels が 1 でない: {chunk.data.shape[1]}"
        assert chunk.data.shape[0] == 512, f"frames が 512 でない: {chunk.data.shape[0]}"

    def test_chunk_dtype_float32(self):
        """チャンクの dtype が float32 であることを検証"""
        received: list[AudioChunk] = []

        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = CallbackSink(callback=received.append)
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

        assert len(received) > 0
        assert received[0].data.dtype == np.float32, \
            f"dtype が float32 でない: {received[0].data.dtype}"

    def test_chunk_metadata_correct(self):
        """チャンクのメタデータ（sample_rate, channels, source_id）が正しいことを検証"""
        received: list[AudioChunk] = []

        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = CallbackSink(callback=received.append)
        pipeline.add_sink(sink)

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=100)
        pipeline.start(data_queue)

        source_id = "microphone:integration_test"
        source = MockSource(
            sample_rate=16000,
            channels=1,
            chunk_frames=512,
            source_id=source_id,
        )
        source.open(data_queue, stop_event)

        time.sleep(0.5)

        stop_event.set()
        source.close()
        pipeline.stop()

        assert len(received) > 0
        chunk = received[0]
        assert chunk.sample_rate == 16000, f"sample_rate が正しくない: {chunk.sample_rate}"
        assert chunk.channels == 1, f"channels が正しくない: {chunk.channels}"
        assert chunk.source_id == source_id, f"source_id が正しくない: {chunk.source_id}"

    def test_stereo_chunk_shape(self):
        """ステレオチャンクの shape が (frames, 2) であることを検証"""
        received: list[AudioChunk] = []

        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = CallbackSink(callback=received.append)
        pipeline.add_sink(sink)

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=100)
        pipeline.start(data_queue)

        source = MockSource(sample_rate=44100, channels=2, chunk_frames=1024)
        source.open(data_queue, stop_event)

        time.sleep(0.5)

        stop_event.set()
        source.close()
        pipeline.stop()

        assert len(received) > 0
        chunk = received[0]
        assert chunk.data.ndim == 2
        assert chunk.data.shape[1] == 2, f"channels が 2 でない: {chunk.data.shape[1]}"

    def test_callback_sink_disabled_not_called(self):
        """enabled=False の CallbackSink はコールバックが呼ばれない"""
        received: list[AudioChunk] = []

        bus = EventBus()
        pipeline = Pipeline(bus)
        sink = CallbackSink(callback=received.append, enabled=False)
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

        assert len(received) == 0, "disabled CallbackSink のコールバックが呼ばれた"

    def test_multiple_callbacks(self):
        """複数の CallbackSink にそれぞれコールバックが届く"""
        received_a: list[AudioChunk] = []
        received_b: list[AudioChunk] = []

        bus = EventBus()
        pipeline = Pipeline(bus)
        pipeline.add_sink(CallbackSink(callback=received_a.append))
        pipeline.add_sink(CallbackSink(callback=received_b.append))

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=100)
        pipeline.start(data_queue)

        source = MockSource(sample_rate=16000, channels=1, chunk_frames=512)
        source.open(data_queue, stop_event)

        time.sleep(0.5)

        stop_event.set()
        source.close()
        pipeline.stop()

        assert len(received_a) > 0, "CallbackSink A にチャンクが届いていない"
        assert len(received_b) > 0, "CallbackSink B にチャンクが届いていない"
        assert len(received_a) == len(received_b), \
            f"受信数が異なる: A={len(received_a)}, B={len(received_b)}"
