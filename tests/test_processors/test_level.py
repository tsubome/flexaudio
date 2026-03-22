"""LevelMeterProcessor のテスト"""

from __future__ import annotations

import time

import numpy as np
import pytest

from pyflexaudio.processors.level import LevelMeterProcessor
from pyflexaudio.types import AudioChunk


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def make_chunk(frames: int = 1024, sample_rate: int = 48000, channels: int = 1,
               amplitude: float = 1.0) -> AudioChunk:
    t = np.arange(frames) / sample_rate
    data = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32).reshape(-1, channels)
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=sample_rate,
        channels=channels,
        source_id="test:0",
    )


def make_constant_chunk(frames: int = 1024, value: float = 1.0, channels: int = 1) -> AudioChunk:
    """全サンプルが同一値のチャンク（RMS = |value|）"""
    data = np.full((frames, channels), value, dtype=np.float32)
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=48000,
        channels=channels,
        source_id="test:0",
    )


def make_silent_chunk(frames: int = 1024, channels: int = 1) -> AudioChunk:
    """無音チャンク（全ゼロ）"""
    data = np.zeros((frames, channels), dtype=np.float32)
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=48000,
        channels=channels,
        source_id="test:0",
    )


# ---------------------------------------------------------------------------
# テスト
# ---------------------------------------------------------------------------

class TestLevelMeterBasic:
    """基本的な dB 計算"""

    def test_silent_level_is_very_low(self):
        """無音データ: level_db が非常に小さい（≈ -200dB）"""
        proc = LevelMeterProcessor()
        chunk = make_silent_chunk(frames=1024)
        result = proc.process(chunk)
        # 実装では max(rms, 1e-10) を使用しているため -200dB 付近になる
        assert result.level_db < -100.0

    def test_full_scale_level_is_near_zero(self):
        """フルスケール (all 1.0): level_db ≈ 0dB"""
        proc = LevelMeterProcessor()
        chunk = make_constant_chunk(frames=1024, value=1.0)
        result = proc.process(chunk)
        # RMS(1.0) = 1.0 → 20*log10(1.0) = 0.0
        assert result.level_db is not None
        assert abs(result.level_db - 0.0) < 1.0, f"Expected ≈0 dB, got {result.level_db:.2f} dB"

    def test_half_amplitude_is_near_minus6db(self):
        """0.5 振幅: level_db ≈ -6dB (理論値: 20*log10(0.5) ≈ -6.02)"""
        proc = LevelMeterProcessor()
        chunk = make_constant_chunk(frames=1024, value=0.5)
        result = proc.process(chunk)
        expected_db = 20.0 * np.log10(0.5)  # ≈ -6.02
        assert result.level_db is not None
        assert abs(result.level_db - expected_db) < 1.0, (
            f"Expected ≈{expected_db:.2f} dB, got {result.level_db:.2f} dB"
        )

    def test_01_amplitude_is_near_minus20db(self):
        """0.1 振幅: level_db ≈ -20dB (理論値: 20*log10(0.1) = -20.0)"""
        proc = LevelMeterProcessor()
        chunk = make_constant_chunk(frames=1024, value=0.1)
        result = proc.process(chunk)
        expected_db = 20.0 * np.log10(0.1)  # = -20.0
        assert result.level_db is not None
        assert abs(result.level_db - expected_db) < 1.0, (
            f"Expected ≈{expected_db:.2f} dB, got {result.level_db:.2f} dB"
        )

    def test_sine_wave_amplitude(self):
        """サイン波の RMS は振幅 / sqrt(2)"""
        amplitude = 0.8
        proc = LevelMeterProcessor()
        chunk = make_chunk(frames=48000, amplitude=amplitude)  # 1秒 → 正確な RMS
        result = proc.process(chunk)
        rms = amplitude / np.sqrt(2)
        expected_db = 20.0 * np.log10(rms)
        assert result.level_db is not None
        assert abs(result.level_db - expected_db) < 1.0, (
            f"Expected ≈{expected_db:.2f} dB, got {result.level_db:.2f} dB"
        )


class TestLevelMeterSideEffects:
    """副作用の確認"""

    def test_same_object_returned(self):
        """データが変更されていないこと: 同一オブジェクトを返す"""
        proc = LevelMeterProcessor()
        chunk = make_chunk(frames=1024)
        result = proc.process(chunk)
        assert result is chunk

    def test_level_db_is_set(self):
        """level_db がセットされること"""
        proc = LevelMeterProcessor()
        chunk = make_chunk(frames=1024)
        assert chunk.level_db is None
        result = proc.process(chunk)
        assert result.level_db is not None

    def test_data_not_modified(self):
        """元の data 配列が変更されていないこと"""
        proc = LevelMeterProcessor()
        chunk = make_chunk(frames=1024, amplitude=0.5)
        original_data = chunk.data.copy()
        proc.process(chunk)
        assert np.array_equal(chunk.data, original_data)

    def test_level_db_is_float(self):
        """level_db が float 型"""
        proc = LevelMeterProcessor()
        chunk = make_chunk(frames=1024)
        result = proc.process(chunk)
        assert isinstance(result.level_db, float)

    def test_level_db_overwritten_on_second_call(self):
        """2回呼ぶと level_db が上書きされる"""
        proc = LevelMeterProcessor()
        chunk1 = make_constant_chunk(frames=1024, value=1.0)
        chunk2 = make_constant_chunk(frames=1024, value=0.1)
        proc.process(chunk1)
        proc.process(chunk2)
        # chunk2 の level_db は -20dB 付近
        assert chunk2.level_db < -10.0


class TestLevelMeterMultichannel:
    """マルチチャンネル処理"""

    def test_stereo_full_scale(self):
        """ステレオ全チャンネルが 1.0 → ≈ 0dB"""
        proc = LevelMeterProcessor()
        chunk = make_constant_chunk(frames=1024, value=1.0, channels=2)
        result = proc.process(chunk)
        assert abs(result.level_db - 0.0) < 1.0

    def test_stereo_half_amplitude(self):
        """ステレオ 0.5 → ≈ -6dB"""
        proc = LevelMeterProcessor()
        chunk = make_constant_chunk(frames=1024, value=0.5, channels=2)
        result = proc.process(chunk)
        expected_db = 20.0 * np.log10(0.5)
        assert abs(result.level_db - expected_db) < 1.0


class TestLevelMeterReset:
    """reset() は副作用なし（ステートレス）"""

    def test_reset_does_not_raise(self):
        proc = LevelMeterProcessor()
        proc.reset()  # 例外が出なければ OK

    def test_process_after_reset(self):
        proc = LevelMeterProcessor()
        proc.reset()
        chunk = make_constant_chunk(frames=1024, value=1.0)
        result = proc.process(chunk)
        assert abs(result.level_db - 0.0) < 1.0
