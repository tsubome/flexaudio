import numpy as np

from pyflexaudio.types import AudioChunk

__all__ = ["LevelMeterProcessor"]


class LevelMeterProcessor:
    """RMS → dB レベルを計算して AudioChunk.level_db に設定"""

    def process(self, chunk: AudioChunk) -> AudioChunk:
        rms = np.sqrt(np.mean(chunk.data ** 2))
        db = 20.0 * np.log10(max(float(rms), 1e-10))
        chunk.level_db = float(db)
        return chunk

    def reset(self) -> None:
        pass
