"""ProcessorChain のテスト"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import pytest

from pyflexaudio.processors.base import AudioProcessor
from pyflexaudio.processors.chain import ProcessorChain
from pyflexaudio.processors.channels import ChannelConvertProcessor
from pyflexaudio.processors.level import LevelMeterProcessor
from pyflexaudio.processors.resample import ResampleProcessor
from pyflexaudio.types import AudioChunk


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def make_chunk(frames: int = 1024, sample_rate: int = 48000, channels: int = 1) -> AudioChunk:
    t = np.arange(frames) / sample_rate
    data = np.sin(2 * np.pi * 440 * t).astype(np.float32).reshape(-1, channels)
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=sample_rate,
        channels=channels,
        source_id="test:0",
    )


class OrderTrackingProcessor:
    """処理順序を記録するモックプロセッサ"""

    def __init__(self, order_log: list, tag: str):
        self._order_log = order_log
        self._tag = tag
        self.reset_called = False

    def process(self, chunk: AudioChunk) -> AudioChunk:
        self._order_log.append(self._tag)
        return chunk

    def reset(self) -> None:
        self.reset_called = True


class TransformProcessor:
    """チャンクの sample_rate を変えるモックプロセッサ（素通し確認用）"""

    def __init__(self, tag: str):
        self.tag = tag
        self.processed_chunks: list[AudioChunk] = []

    def process(self, chunk: AudioChunk) -> AudioChunk:
        self.processed_chunks.append(chunk)
        return chunk

    def reset(self) -> None:
        self.processed_chunks.clear()


# ---------------------------------------------------------------------------
# テスト
# ---------------------------------------------------------------------------

class TestProcessorChainEmpty:
    """空チェーンの動作"""

    def test_empty_chain_returns_same_chunk(self):
        """空チェーン: chunk がそのまま返る"""
        chain = ProcessorChain()
        chunk = make_chunk()
        result = chain.process(chunk)
        assert result is chunk

    def test_empty_chain_with_none_processors(self):
        chain = ProcessorChain(processors=None)
        chunk = make_chunk()
        result = chain.process(chunk)
        assert result is chunk

    def test_empty_chain_with_empty_list(self):
        chain = ProcessorChain(processors=[])
        chunk = make_chunk()
        result = chain.process(chunk)
        assert result is chunk


class TestProcessorChainSingle:
    """単一プロセッサの動作"""

    def test_single_processor_executed(self):
        """単一プロセッサが正しく実行される"""
        proc = LevelMeterProcessor()
        chain = ProcessorChain(processors=[proc])
        chunk = make_chunk(frames=1024)
        result = chain.process(chunk)
        # LevelMeterProcessor は同一オブジェクトを返して level_db をセットする
        assert result is chunk
        assert result.level_db is not None

    def test_single_channel_convert(self):
        proc = ChannelConvertProcessor(target_channels=1)
        chain = ProcessorChain(processors=[proc])
        chunk = make_chunk(frames=1024, channels=2)
        result = chain.process(chunk)
        assert result.channels == 1

    def test_single_resample(self):
        proc = ResampleProcessor(target_sample_rate=16000)
        chain = ProcessorChain(processors=[proc])
        chunk = make_chunk(frames=4800, sample_rate=48000)
        result = chain.process(chunk)
        assert result.sample_rate == 16000


class TestProcessorChainOrder:
    """複数プロセッサの実行順序"""

    def test_order_is_preserved(self):
        """複数プロセッサが順序通りに実行される"""
        log: list[str] = []
        p1 = OrderTrackingProcessor(log, "first")
        p2 = OrderTrackingProcessor(log, "second")
        p3 = OrderTrackingProcessor(log, "third")
        chain = ProcessorChain(processors=[p1, p2, p3])
        chunk = make_chunk()
        chain.process(chunk)
        assert log == ["first", "second", "third"]

    def test_channel_then_level(self):
        """チャンネル変換 → レベル計算の順で動作"""
        channel_proc = ChannelConvertProcessor(target_channels=1)
        level_proc = LevelMeterProcessor()
        chain = ProcessorChain(processors=[channel_proc, level_proc])

        # stereo チャンクを入力
        t = np.arange(1024) / 48000
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
        data = np.stack([left, right], axis=1)
        chunk = AudioChunk(
            data=data,
            timestamp=time.monotonic(),
            sample_rate=48000,
            channels=2,
            source_id="test:0",
        )
        result = chain.process(chunk)
        # チャンネル変換が先に実行 → mono
        assert result.channels == 1
        assert result.data.shape == (1024, 1)
        # レベルが計算されている
        assert result.level_db is not None

    def test_output_reflects_all_transforms(self):
        """各プロセッサの変換が順次反映される"""
        log: list[str] = []
        p1 = OrderTrackingProcessor(log, "A")
        p2 = OrderTrackingProcessor(log, "B")
        chain = ProcessorChain(processors=[p1, p2])
        chain.process(make_chunk())
        assert log[0] == "A"
        assert log[1] == "B"


class TestProcessorChainAdd:
    """add() メソッドの動作"""

    def test_add_appends_processor(self):
        """add() でプロセッサが追加される"""
        chain = ProcessorChain()
        assert len(chain._processors) == 0

        level_proc = LevelMeterProcessor()
        chain.add(level_proc)
        assert len(chain._processors) == 1
        assert chain._processors[0] is level_proc

    def test_add_multiple(self):
        chain = ProcessorChain()
        p1 = LevelMeterProcessor()
        p2 = ChannelConvertProcessor(target_channels=1)
        chain.add(p1)
        chain.add(p2)
        assert len(chain._processors) == 2
        assert chain._processors[0] is p1
        assert chain._processors[1] is p2

    def test_add_executes_in_order(self):
        log: list[str] = []
        chain = ProcessorChain()
        chain.add(OrderTrackingProcessor(log, "X"))
        chain.add(OrderTrackingProcessor(log, "Y"))
        chain.process(make_chunk())
        assert log == ["X", "Y"]

    def test_add_to_existing_chain(self):
        p1 = LevelMeterProcessor()
        chain = ProcessorChain(processors=[p1])
        p2 = ChannelConvertProcessor(target_channels=1)
        chain.add(p2)
        assert len(chain._processors) == 2


class TestProcessorChainReset:
    """reset() で全プロセッサがリセットされる"""

    def test_reset_calls_all_processors(self):
        """reset() が全プロセッサの reset() を呼び出す"""
        log: list[str] = []
        p1 = OrderTrackingProcessor(log, "p1")
        p2 = OrderTrackingProcessor(log, "p2")
        p3 = OrderTrackingProcessor(log, "p3")
        chain = ProcessorChain(processors=[p1, p2, p3])
        chain.reset()
        assert p1.reset_called
        assert p2.reset_called
        assert p3.reset_called

    def test_reset_empty_chain_does_not_raise(self):
        chain = ProcessorChain()
        chain.reset()  # 例外が出なければ OK

    def test_process_after_reset(self):
        """reset() 後も process() が正常に動作する"""
        level_proc = LevelMeterProcessor()
        chain = ProcessorChain(processors=[level_proc])
        chunk1 = make_chunk(frames=1024)
        chain.process(chunk1)
        chain.reset()
        chunk2 = make_chunk(frames=1024)
        result = chain.process(chunk2)
        assert result.level_db is not None

    def test_resample_reset_via_chain(self):
        """ProcessorChain.reset() が ResampleProcessor の内部状態をクリアする"""
        resample_proc = ResampleProcessor(target_sample_rate=16000)
        chain = ProcessorChain(processors=[resample_proc])

        chunk = make_chunk(frames=4800, sample_rate=48000)
        chain.process(chunk)
        assert resample_proc._resampler is not None

        chain.reset()
        assert resample_proc._resampler is None


class TestProcessorChainProtocol:
    """ProcessorChain が AudioProcessor Protocol を満たす"""

    def test_isinstance_audio_processor(self):
        """ProcessorChain は AudioProcessor Protocol を満たす"""
        chain = ProcessorChain()
        assert isinstance(chain, AudioProcessor)

    def test_chain_usable_as_processor_in_another_chain(self):
        """ProcessorChain を別の ProcessorChain に追加できる（ネスト）"""
        inner_chain = ProcessorChain(processors=[LevelMeterProcessor()])
        outer_chain = ProcessorChain(processors=[inner_chain])

        chunk = make_chunk(frames=1024)
        result = outer_chain.process(chunk)
        assert result.level_db is not None

    def test_has_process_method(self):
        chain = ProcessorChain()
        assert callable(getattr(chain, "process", None))

    def test_has_reset_method(self):
        chain = ProcessorChain()
        assert callable(getattr(chain, "reset", None))
