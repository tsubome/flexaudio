"""Processor が例外を投げてもパイプラインが継続することを検証するテスト"""

import queue
import threading
import time

import numpy as np
import pytest

from pyflexaudio.types import AudioChunk, ErrorEvent, FlexAudioError
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


class FailingProcessor:
    """常に RuntimeError を投げるプロセッサ"""

    def process(self, chunk):
        raise RuntimeError("test error")

    def reset(self):
        pass


class PassThroughProcessor:
    """チャンクをそのまま返すプロセッサ"""

    def __init__(self):
        self.processed_count = 0

    def process(self, chunk):
        self.processed_count += 1
        return chunk

    def reset(self):
        self.processed_count = 0


# ---- テスト ----

def test_processor_error_emits_error_event():
    """Processor がエラーを投げると ErrorEvent が emit される"""
    bus = EventBus()
    errors = []
    bus.on(ErrorEvent, errors.append)

    pipeline = Pipeline(bus)
    chain = ProcessorChain([FailingProcessor()])
    pipeline.set_main_chain(chain)

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue()
    pipeline.start(dq)

    chunk = make_chunk()
    dq.put(chunk)
    time.sleep(0.1)

    pipeline.stop()

    assert len(errors) >= 1
    assert errors[0].error.code == "PROCESSOR_ERROR"


def test_processor_error_skips_chunk():
    """Processor がエラーを投げるとそのチャンクは Sink に届かない"""
    bus = EventBus()

    pipeline = Pipeline(bus)
    chain = ProcessorChain([FailingProcessor()])
    pipeline.set_main_chain(chain)

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue()
    pipeline.start(dq)

    chunk = make_chunk()
    dq.put(chunk)
    time.sleep(0.1)

    pipeline.stop()

    # FailingProcessor がエラーを投げるため sink には届かない
    assert len(sink.chunks) == 0


def test_pipeline_continues_after_processor_error():
    """Processor がエラーを投げた後もパイプラインは次のチャンクを処理できる"""
    bus = EventBus()
    errors = []
    bus.on(ErrorEvent, errors.append)

    pipeline = Pipeline(bus)
    # FailingProcessor を設定（全チャンクでエラー）
    chain = ProcessorChain([FailingProcessor()])
    pipeline.set_main_chain(chain)

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue()
    pipeline.start(dq)

    # 複数チャンクを投入
    for _ in range(3):
        dq.put(make_chunk())

    time.sleep(0.3)

    # パイプラインスレッドがまだ生きていること
    assert pipeline._thread is not None
    assert pipeline._thread.is_alive()

    pipeline.stop()

    # 全チャンクでエラーが発生する
    assert len(errors) == 3


def test_pipeline_continues_after_processor_error_with_passthrough():
    """エラープロセッサの後に正常プロセッサを置いた場合、エラー後の連続チャンクは正常処理される"""
    bus = EventBus()
    errors = []
    bus.on(ErrorEvent, errors.append)

    # まず FailingProcessor を main_chain に設定して1チャンク投入
    pipeline = Pipeline(bus)
    failing_chain = ProcessorChain([FailingProcessor()])
    pipeline.set_main_chain(failing_chain)

    sink = MockSink()
    pipeline.add_sink(sink)

    dq = queue.Queue()
    pipeline.start(dq)

    # エラーを起こすチャンク
    dq.put(make_chunk())
    time.sleep(0.1)

    # PassThroughProcessor に切り替え
    passthrough = PassThroughProcessor()
    pipeline.set_main_chain(ProcessorChain([passthrough]))

    # 正常チャンクを投入
    dq.put(make_chunk())
    time.sleep(0.1)

    pipeline.stop()

    # エラー後のチャンクは正常処理される
    assert len(errors) == 1
    assert passthrough.processed_count == 1
    assert len(sink.chunks) == 1
