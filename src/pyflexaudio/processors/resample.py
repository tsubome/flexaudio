from __future__ import annotations

import logging

import numpy as np
import soxr

from pyflexaudio.types import AudioChunk


__all__ = ["ResampleProcessor"]

logger = logging.getLogger("pyflexaudio.processors.resample")


class ResampleProcessor:
    """soxr-python を使ったストリーミングリサンプリング"""

    def __init__(self, target_sample_rate: int, quality: str = "HQ"):
        """
        Args:
            target_sample_rate: ターゲットサンプルレート (e.g., 16000)
            quality: soxr 品質 ("QQ", "LQ", "MQ", "HQ", "VHQ")
        """
        self._target_rate = target_sample_rate
        self._quality = quality
        self._resampler: soxr.ResampleStream | None = None
        self._current_source_rate: int | None = None
        self._current_channels: int | None = None

    def process(self, chunk: AudioChunk) -> AudioChunk:
        # ソースレートがターゲットと同じなら素通し
        if chunk.sample_rate == self._target_rate:
            return chunk

        # リサンプラーの初期化/再初期化（ソースレートかチャンネル数が変わった場合）
        if (self._current_source_rate != chunk.sample_rate
                or self._current_channels != chunk.channels):
            self._init_resampler(chunk.sample_rate, chunk.channels)

        # ストリーミングリサンプリング
        resampled = self._resampler.resample_chunk(chunk.data)

        return AudioChunk(
            data=resampled,
            timestamp=chunk.timestamp,
            sample_rate=self._target_rate,
            channels=chunk.channels,
            source_id=chunk.source_id,
            level_db=chunk.level_db,
        )

    def reset(self) -> None:
        """内部状態をリセット（リサンプラーを破棄）"""
        self._resampler = None
        self._current_source_rate = None
        self._current_channels = None

    def _init_resampler(self, source_rate: int, channels: int) -> None:
        """リサンプラーを (再)初期化"""
        self._resampler = soxr.ResampleStream(
            source_rate,
            self._target_rate,
            num_channels=channels,
            dtype=np.float32,
            quality=self._quality,
        )
        self._current_source_rate = source_rate
        self._current_channels = channels
        logger.debug(
            "Resampler initialized: %dHz -> %dHz, %dch, quality=%s",
            source_rate, self._target_rate, channels, self._quality
        )
