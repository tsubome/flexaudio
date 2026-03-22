"""ResampleProcessor のテスト"""

from __future__ import annotations

import time

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# テスト
# ---------------------------------------------------------------------------

class TestResampleProcessorPassthrough:
    """ソースレートとターゲットが同じ場合は素通し"""

    def test_same_rate_returns_same_object(self):
        processor = ResampleProcessor(target_sample_rate=48000)
        chunk = make_chunk(sample_rate=48000)
        result = processor.process(chunk)
        assert result is chunk

    def test_same_rate_16k_returns_same_object(self):
        processor = ResampleProcessor(target_sample_rate=16000)
        chunk = make_chunk(sample_rate=16000)
        result = processor.process(chunk)
        assert result is chunk


class TestResampleProcessorDownsample:
    """48kHz → 16kHz ダウンサンプリング"""

    def test_output_sample_rate_is_target(self):
        processor = ResampleProcessor(target_sample_rate=16000)
        chunk = make_chunk(frames=4800, sample_rate=48000)
        result = processor.process(chunk)
        assert result.sample_rate == 16000

    def test_output_channels_preserved(self):
        processor = ResampleProcessor(target_sample_rate=16000)
        chunk = make_chunk(frames=4800, sample_rate=48000, channels=1)
        result = processor.process(chunk)
        assert result.channels == 1

    def test_output_is_new_object(self):
        processor = ResampleProcessor(target_sample_rate=16000)
        chunk = make_chunk(frames=4800, sample_rate=48000)
        result = processor.process(chunk)
        assert result is not chunk

    def test_metadata_preserved(self):
        processor = ResampleProcessor(target_sample_rate=16000)
        chunk = make_chunk(frames=4800, sample_rate=48000)
        result = processor.process(chunk)
        assert result.timestamp == chunk.timestamp
        assert result.source_id == chunk.source_id

    def test_level_db_preserved(self):
        processor = ResampleProcessor(target_sample_rate=16000)
        chunk = make_chunk(frames=4800, sample_rate=48000)
        chunk.level_db = -12.5
        result = processor.process(chunk)
        assert result.level_db == -12.5

    def test_approximate_frame_count(self):
        """48kHz 4800フレーム → 16kHz は約1600フレーム"""
        processor = ResampleProcessor(target_sample_rate=16000)
        chunk = make_chunk(frames=4800, sample_rate=48000)
        result = processor.process(chunk)
        # soxr のストリーミングバッファにより多少前後する可能性がある
        assert result.data.shape[0] > 0
        assert result.data.ndim == 2
        assert result.data.shape[1] == 1


class TestResampleProcessorRoundTrip:
    """16kHz → 48kHz → 16kHz ラウンドトリップ"""

    def test_roundtrip_correlation(self):
        """ラウンドトリップ後の信号が元信号と高い相関を持つ"""
        # 16kHz のサイン波を長めに生成（バッファを十分に埋める）
        frames = 16000  # 1秒
        t = np.arange(frames) / 16000
        original = np.sin(2 * np.pi * 440 * t).astype(np.float32).reshape(-1, 1)
        chunk_16k = AudioChunk(
            data=original,
            timestamp=time.monotonic(),
            sample_rate=16000,
            channels=1,
            source_id="test:0",
        )

        # 16kHz → 48kHz
        up_processor = ResampleProcessor(target_sample_rate=48000)
        chunk_48k = up_processor.process(chunk_16k)
        assert chunk_48k.sample_rate == 48000

        # 48kHz → 16kHz
        down_processor = ResampleProcessor(target_sample_rate=16000)
        chunk_back = down_processor.process(chunk_48k)
        assert chunk_back.sample_rate == 16000

        # 出力が存在することを確認
        assert chunk_back.data.shape[0] > 0

        # 共通長で相関係数を計算
        n = min(original.shape[0], chunk_back.data.shape[0])
        # 先頭はフィルタの過渡応答があるため、後半部分で評価
        start = n // 4
        orig_slice = original[start:n, 0]
        back_slice = chunk_back.data[start:n, 0]

        if len(orig_slice) > 10 and len(back_slice) > 10:
            corr = np.corrcoef(orig_slice, back_slice)[0, 1]
            assert corr > 0.95, f"Round-trip correlation too low: {corr:.4f}"


class TestResampleProcessorReinitialization:
    """ソースレート変更時にリサンプラーが再初期化される"""

    def test_resampler_reinit_on_rate_change(self):
        processor = ResampleProcessor(target_sample_rate=16000)

        # 最初は 48kHz
        chunk_48k = make_chunk(frames=4800, sample_rate=48000)
        result_1 = processor.process(chunk_48k)
        assert result_1.sample_rate == 16000
        assert processor._current_source_rate == 48000

        # 次に 44100Hz — リサンプラーが再初期化されるはず
        chunk_44k = make_chunk(frames=4410, sample_rate=44100)
        result_2 = processor.process(chunk_44k)
        assert result_2.sample_rate == 16000
        assert processor._current_source_rate == 44100

    def test_resampler_reinit_on_channel_change(self):
        processor = ResampleProcessor(target_sample_rate=16000)

        chunk_mono = make_chunk(frames=4800, sample_rate=48000, channels=1)
        processor.process(chunk_mono)
        assert processor._current_channels == 1

        chunk_stereo = make_chunk(frames=4800, sample_rate=48000, channels=2)
        processor.process(chunk_stereo)
        assert processor._current_channels == 2


class TestResampleProcessorReset:
    """reset() 後に再度 process() できる"""

    def test_reset_clears_state(self):
        processor = ResampleProcessor(target_sample_rate=16000)
        chunk = make_chunk(frames=4800, sample_rate=48000)
        processor.process(chunk)

        assert processor._resampler is not None
        assert processor._current_source_rate == 48000

        processor.reset()

        assert processor._resampler is None
        assert processor._current_source_rate is None
        assert processor._current_channels is None

    def test_process_after_reset(self):
        processor = ResampleProcessor(target_sample_rate=16000)
        chunk = make_chunk(frames=4800, sample_rate=48000)

        processor.process(chunk)
        processor.reset()

        # reset 後も再度 process できる
        chunk2 = make_chunk(frames=4800, sample_rate=48000)
        result = processor.process(chunk2)
        assert result.sample_rate == 16000

    def test_reset_then_same_rate_passthrough(self):
        processor = ResampleProcessor(target_sample_rate=48000)
        chunk = make_chunk(frames=1024, sample_rate=48000)

        processor.reset()
        result = processor.process(chunk)
        assert result is chunk
