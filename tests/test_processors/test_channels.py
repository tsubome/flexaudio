"""ChannelConvertProcessor のテスト"""

from __future__ import annotations

import time

import numpy as np
import pytest

from pyflexaudio.processors.channels import ChannelConvertProcessor
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


def make_stereo_chunk(frames: int = 1024) -> AudioChunk:
    """左右で異なる値を持つステレオチャンク"""
    left = np.sin(2 * np.pi * 440 * np.arange(frames) / 48000).astype(np.float32)
    right = np.sin(2 * np.pi * 880 * np.arange(frames) / 48000).astype(np.float32)
    data = np.stack([left, right], axis=1)
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=48000,
        channels=2,
        source_id="test:0",
    )


# ---------------------------------------------------------------------------
# テスト
# ---------------------------------------------------------------------------

class TestChannelConvertProcessorInit:
    """コンストラクタのバリデーション"""

    def test_target_channels_3_raises_value_error(self):
        with pytest.raises(ValueError, match="target_channels must be 1 or 2"):
            ChannelConvertProcessor(target_channels=3)

    def test_target_channels_0_raises_value_error(self):
        with pytest.raises(ValueError, match="target_channels must be 1 or 2"):
            ChannelConvertProcessor(target_channels=0)

    def test_target_channels_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            ChannelConvertProcessor(target_channels=-1)

    def test_valid_mono(self):
        proc = ChannelConvertProcessor(target_channels=1)
        assert proc._target_channels == 1

    def test_valid_stereo(self):
        proc = ChannelConvertProcessor(target_channels=2)
        assert proc._target_channels == 2


class TestChannelConvertProcessorPassthrough:
    """同じチャンネル数なら素通し（同一オブジェクト返却）"""

    def test_mono_to_mono_same_object(self):
        proc = ChannelConvertProcessor(target_channels=1)
        chunk = make_chunk(channels=1)
        result = proc.process(chunk)
        assert result is chunk

    def test_stereo_to_stereo_same_object(self):
        proc = ChannelConvertProcessor(target_channels=2)
        chunk = make_chunk(channels=2)
        result = proc.process(chunk)
        assert result is chunk


class TestChannelConvertStereoToMono:
    """stereo → mono 変換"""

    def test_output_shape_is_mono(self):
        proc = ChannelConvertProcessor(target_channels=1)
        chunk = make_stereo_chunk(frames=1024)
        result = proc.process(chunk)
        assert result.data.shape == (1024, 1)

    def test_output_channels_attribute(self):
        proc = ChannelConvertProcessor(target_channels=1)
        chunk = make_stereo_chunk(frames=512)
        result = proc.process(chunk)
        assert result.channels == 1

    def test_value_is_mean_of_channels(self):
        """stereo → mono の値は2チャンネルの平均"""
        proc = ChannelConvertProcessor(target_channels=1)
        chunk = make_stereo_chunk(frames=256)
        result = proc.process(chunk)
        expected = np.mean(chunk.data, axis=1, keepdims=True)
        assert np.allclose(result.data, expected.astype(np.float32), atol=1e-6)

    def test_output_dtype_is_float32(self):
        proc = ChannelConvertProcessor(target_channels=1)
        chunk = make_stereo_chunk(frames=256)
        result = proc.process(chunk)
        assert result.data.dtype == np.float32

    def test_metadata_preserved(self):
        proc = ChannelConvertProcessor(target_channels=1)
        chunk = make_stereo_chunk(frames=256)
        result = proc.process(chunk)
        assert result.timestamp == chunk.timestamp
        assert result.sample_rate == chunk.sample_rate
        assert result.source_id == chunk.source_id

    def test_level_db_preserved(self):
        proc = ChannelConvertProcessor(target_channels=1)
        chunk = make_stereo_chunk(frames=256)
        chunk.level_db = -10.0
        result = proc.process(chunk)
        assert result.level_db == -10.0

    def test_output_is_new_object(self):
        proc = ChannelConvertProcessor(target_channels=1)
        chunk = make_stereo_chunk(frames=256)
        result = proc.process(chunk)
        assert result is not chunk


class TestChannelConvertMonoToStereo:
    """mono → stereo 変換"""

    def test_output_shape_is_stereo(self):
        proc = ChannelConvertProcessor(target_channels=2)
        chunk = make_chunk(frames=1024, channels=1)
        result = proc.process(chunk)
        assert result.data.shape == (1024, 2)

    def test_output_channels_attribute(self):
        proc = ChannelConvertProcessor(target_channels=2)
        chunk = make_chunk(frames=512, channels=1)
        result = proc.process(chunk)
        assert result.channels == 2

    def test_value_is_duplicated(self):
        """mono → stereo の左右チャンネルは同一データ"""
        proc = ChannelConvertProcessor(target_channels=2)
        chunk = make_chunk(frames=256, channels=1)
        result = proc.process(chunk)
        assert np.allclose(result.data[:, 0], result.data[:, 1], atol=1e-7)

    def test_value_matches_original(self):
        """stereo の各チャンネルが元の mono データと一致"""
        proc = ChannelConvertProcessor(target_channels=2)
        chunk = make_chunk(frames=256, channels=1)
        result = proc.process(chunk)
        assert np.allclose(result.data[:, 0], chunk.data[:, 0], atol=1e-7)
        assert np.allclose(result.data[:, 1], chunk.data[:, 0], atol=1e-7)

    def test_metadata_preserved(self):
        proc = ChannelConvertProcessor(target_channels=2)
        chunk = make_chunk(frames=256, channels=1)
        result = proc.process(chunk)
        assert result.timestamp == chunk.timestamp
        assert result.sample_rate == chunk.sample_rate
        assert result.source_id == chunk.source_id

    def test_output_is_new_object(self):
        proc = ChannelConvertProcessor(target_channels=2)
        chunk = make_chunk(frames=256, channels=1)
        result = proc.process(chunk)
        assert result is not chunk


class TestChannelConvertProcessorReset:
    """reset() は副作用なし（ステートレスなので常に成功）"""

    def test_reset_does_not_raise(self):
        proc = ChannelConvertProcessor(target_channels=1)
        proc.reset()  # 例外が出なければ OK

    def test_process_after_reset(self):
        proc = ChannelConvertProcessor(target_channels=1)
        proc.reset()
        chunk = make_stereo_chunk(frames=256)
        result = proc.process(chunk)
        assert result.data.shape == (256, 1)
