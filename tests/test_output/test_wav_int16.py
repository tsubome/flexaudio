"""Tests for float32 → int16 conversion in FileSink."""

import time
import wave

import numpy as np
import pytest

from pyflexaudio.sinks.file import FileSink
from pyflexaudio.types import AudioChunk


def make_chunk_from_data(data: np.ndarray, sample_rate=44100):
    """Create a chunk from a specific numpy array."""
    channels = data.shape[1] if data.ndim == 2 else 1
    return AudioChunk(
        data=data.astype(np.float32),
        timestamp=time.monotonic(),
        sample_rate=sample_rate,
        channels=channels,
        source_id="test:0",
    )


def read_wav_samples_int16(path: str) -> np.ndarray:
    """Read int16 samples from a WAV file and return as numpy array."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)
    return samples


class TestFloat32ToInt16Range:
    def test_output_samples_are_in_int16_range(self, tmp_path):
        path = str(tmp_path / "output.wav")
        data = np.random.uniform(-1.0, 1.0, (1024, 2)).astype(np.float32)
        sink = FileSink(path=path)
        sink.write(make_chunk_from_data(data))
        sink.close()

        samples = read_wav_samples_int16(path)
        assert samples.min() >= -32768
        assert samples.max() <= 32767

    def test_full_scale_positive_maps_near_32767(self, tmp_path):
        path = str(tmp_path / "output.wav")
        # 1.0 * 32767 = 32767 (clipped to int16 max)
        data = np.ones((64, 1), dtype=np.float32)
        sink = FileSink(path=path)
        sink.write(make_chunk_from_data(data, sample_rate=44100))
        sink.close()

        samples = read_wav_samples_int16(path)
        assert np.all(samples == 32767)

    def test_full_scale_negative_maps_near_minus_32767(self, tmp_path):
        path = str(tmp_path / "output.wav")
        data = np.full((64, 1), -1.0, dtype=np.float32)
        sink = FileSink(path=path)
        sink.write(make_chunk_from_data(data, sample_rate=44100))
        sink.close()

        samples = read_wav_samples_int16(path)
        # -1.0 * 32767 = -32767 (np.clip range is -32768..32767)
        assert np.all(samples == -32767)

    def test_zero_input_maps_to_zero(self, tmp_path):
        path = str(tmp_path / "output.wav")
        data = np.zeros((64, 2), dtype=np.float32)
        sink = FileSink(path=path)
        sink.write(make_chunk_from_data(data))
        sink.close()

        samples = read_wav_samples_int16(path)
        assert np.all(samples == 0)


class TestFloat32ToInt16Clipping:
    def test_positive_overflow_clips_to_32767(self, tmp_path):
        path = str(tmp_path / "output.wav")
        data = np.full((64, 1), 1.5, dtype=np.float32)
        sink = FileSink(path=path)
        sink.write(make_chunk_from_data(data))
        sink.close()

        samples = read_wav_samples_int16(path)
        assert np.all(samples == 32767)

    def test_negative_overflow_clips_to_minus_32768(self, tmp_path):
        path = str(tmp_path / "output.wav")
        data = np.full((64, 1), -1.5, dtype=np.float32)
        sink = FileSink(path=path)
        sink.write(make_chunk_from_data(data))
        sink.close()

        samples = read_wav_samples_int16(path)
        assert np.all(samples == -32768)

    def test_large_positive_value_clips(self, tmp_path):
        path = str(tmp_path / "output.wav")
        data = np.full((64, 1), 100.0, dtype=np.float32)
        sink = FileSink(path=path)
        sink.write(make_chunk_from_data(data))
        sink.close()

        samples = read_wav_samples_int16(path)
        assert np.all(samples == 32767)

    def test_large_negative_value_clips(self, tmp_path):
        path = str(tmp_path / "output.wav")
        data = np.full((64, 1), -100.0, dtype=np.float32)
        sink = FileSink(path=path)
        sink.write(make_chunk_from_data(data))
        sink.close()

        samples = read_wav_samples_int16(path)
        assert np.all(samples == -32768)

    def test_mixed_clipped_and_normal_values(self, tmp_path):
        path = str(tmp_path / "output.wav")
        # First 32 frames clip, last 32 are zero
        data = np.vstack([
            np.full((32, 1), 2.0, dtype=np.float32),
            np.zeros((32, 1), dtype=np.float32),
        ])
        sink = FileSink(path=path)
        sink.write(make_chunk_from_data(data))
        sink.close()

        samples = read_wav_samples_int16(path)
        assert np.all(samples[:32] == 32767)
        assert np.all(samples[32:] == 0)


class TestFloat32ToInt16Precision:
    def test_half_amplitude_conversion(self, tmp_path):
        path = str(tmp_path / "output.wav")
        data = np.full((64, 1), 0.5, dtype=np.float32)
        sink = FileSink(path=path)
        sink.write(make_chunk_from_data(data))
        sink.close()

        samples = read_wav_samples_int16(path)
        # 0.5 * 32767 = 16383 (integer truncation)
        expected = int(0.5 * 32767)
        assert np.all(samples == expected)
