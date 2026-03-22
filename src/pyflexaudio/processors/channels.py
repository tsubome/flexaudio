import numpy as np

from pyflexaudio.types import AudioChunk

__all__ = ["ChannelConvertProcessor"]


class ChannelConvertProcessor:
    """チャンネル数を変換するプロセッサ"""

    def __init__(self, target_channels: int):
        """
        Args:
            target_channels: ターゲットチャンネル数 (1=mono, 2=stereo)
        """
        if target_channels not in (1, 2):
            raise ValueError(f"target_channels must be 1 or 2, got {target_channels}")
        self._target_channels = target_channels

    def process(self, chunk: AudioChunk) -> AudioChunk:
        if chunk.channels == self._target_channels:
            return chunk

        if self._target_channels == 1:
            mono_data = np.mean(chunk.data, axis=1, keepdims=True)
            return AudioChunk(
                data=mono_data.astype(np.float32),
                timestamp=chunk.timestamp,
                sample_rate=chunk.sample_rate,
                channels=1,
                source_id=chunk.source_id,
                level_db=chunk.level_db,
            )
        else:
            stereo_data = np.repeat(chunk.data, 2, axis=1)
            return AudioChunk(
                data=stereo_data,
                timestamp=chunk.timestamp,
                sample_rate=chunk.sample_rate,
                channels=2,
                source_id=chunk.source_id,
                level_db=chunk.level_db,
            )

    def reset(self) -> None:
        pass
