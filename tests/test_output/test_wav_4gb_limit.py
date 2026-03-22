"""Tests for WAV 4 GB limit splitting in FileSink.

Instead of writing 4 GB of real audio data, we temporarily patch
FileSink._MAX_DATA_BYTES to a small value so that the split is triggered
after a tiny amount of data. After each test the original value is restored.
"""

import os
import time
import wave

import numpy as np
import pytest

from pyflexaudio.sinks.file import FileSink
from pyflexaudio.types import AudioChunk


# Bytes produced by one chunk of 256 frames, 2 channels, int16
_CHUNK_FRAMES = 256
_CHUNK_CHANNELS = 2
_BYTES_PER_CHUNK = _CHUNK_FRAMES * _CHUNK_CHANNELS * 2  # = 1024 bytes


def make_chunk(frames=_CHUNK_FRAMES, sample_rate=44100, channels=_CHUNK_CHANNELS, amplitude=0.5):
    data = (np.random.randn(frames, channels) * amplitude).astype(np.float32)
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=sample_rate,
        channels=channels,
        source_id="test:0",
    )


@pytest.fixture
def small_max_data_bytes():
    """Temporarily set _MAX_DATA_BYTES so exactly 2 chunks fit, and the 3rd triggers a split.

    FileSink checks: total_data_bytes + len(raw_bytes) > _MAX_DATA_BYTES
    After chunk 1:  1024 + 1024 > limit?  must be False (allow chunk 2)
    After chunk 2:  2048 + 1024 > limit?  must be True  (split before chunk 3)
    So limit must satisfy: 2048 <= limit < 3072 — use exactly 2 * chunk_size.
    """
    original = FileSink._MAX_DATA_BYTES
    FileSink._MAX_DATA_BYTES = _BYTES_PER_CHUNK * 2  # = 2048; 2048+1024 > 2048 is True
    yield FileSink._MAX_DATA_BYTES
    FileSink._MAX_DATA_BYTES = original


@pytest.fixture
def one_chunk_max():
    """Trigger split after every single chunk."""
    original = FileSink._MAX_DATA_BYTES
    FileSink._MAX_DATA_BYTES = _BYTES_PER_CHUNK - 1
    yield FileSink._MAX_DATA_BYTES
    FileSink._MAX_DATA_BYTES = original


class TestFourGbSplitTriggered:
    def test_split_creates_second_file(self, tmp_path, small_max_data_bytes):
        base = str(tmp_path / "output.wav")
        sink = FileSink(path=base)

        # Write 3 chunks: first two fill the limit, third triggers split
        sink.write(make_chunk())
        sink.write(make_chunk())
        sink.write(make_chunk())
        sink.close()

        assert os.path.exists(base)
        assert os.path.exists(str(tmp_path / "output_002.wav"))

    def test_no_split_below_limit(self, tmp_path, small_max_data_bytes):
        base = str(tmp_path / "output.wav")
        sink = FileSink(path=base)

        # Only 2 chunks — stays within limit
        sink.write(make_chunk())
        sink.write(make_chunk())
        sink.close()

        assert os.path.exists(base)
        assert not os.path.exists(str(tmp_path / "output_002.wav"))

    def test_multiple_splits_create_sequential_files(self, tmp_path, one_chunk_max):
        base = str(tmp_path / "output.wav")
        sink = FileSink(path=base)

        # Each chunk overflows the limit -> new file per chunk after the first
        for _ in range(4):
            sink.write(make_chunk())
        sink.close()

        assert os.path.exists(str(tmp_path / "output.wav"))
        assert os.path.exists(str(tmp_path / "output_002.wav"))
        assert os.path.exists(str(tmp_path / "output_003.wav"))
        assert os.path.exists(str(tmp_path / "output_004.wav"))


class TestSplitFileHeaders:
    def test_first_file_has_valid_header(self, tmp_path, small_max_data_bytes):
        base = str(tmp_path / "output.wav")
        sink = FileSink(path=base)
        sink.write(make_chunk())
        sink.write(make_chunk())
        sink.write(make_chunk())
        sink.close()

        with wave.open(base, "rb") as wf:
            assert wf.getnchannels() == _CHUNK_CHANNELS
            assert wf.getframerate() == 44100
            assert wf.getsampwidth() == 2

    def test_second_file_has_valid_header(self, tmp_path, small_max_data_bytes):
        base = str(tmp_path / "output.wav")
        sink = FileSink(path=base)
        sink.write(make_chunk())
        sink.write(make_chunk())
        sink.write(make_chunk())
        sink.close()

        second = str(tmp_path / "output_002.wav")
        with wave.open(second, "rb") as wf:
            assert wf.getnchannels() == _CHUNK_CHANNELS
            assert wf.getframerate() == 44100
            assert wf.getsampwidth() == 2

    def test_all_split_files_have_valid_headers(self, tmp_path, one_chunk_max):
        base = str(tmp_path / "output.wav")
        sink = FileSink(path=base)
        for _ in range(3):
            sink.write(make_chunk())
        sink.close()

        names = ["output.wav", "output_002.wav", "output_003.wav"]
        for name in names:
            path = str(tmp_path / name)
            assert os.path.exists(path), f"Missing: {path}"
            with wave.open(path, "rb") as wf:
                assert wf.getsampwidth() == 2
                assert wf.getnchannels() == _CHUNK_CHANNELS


class TestSplitFileFrameCounts:
    def test_second_file_contains_overflow_chunk(self, tmp_path, small_max_data_bytes):
        """The chunk that caused the overflow goes into the new file."""
        base = str(tmp_path / "output.wav")
        sink = FileSink(path=base)
        sink.write(make_chunk())
        sink.write(make_chunk())
        # Third chunk triggers split and lands in output_002.wav
        sink.write(make_chunk())
        sink.close()

        second = str(tmp_path / "output_002.wav")
        with wave.open(second, "rb") as wf:
            assert wf.getnframes() == _CHUNK_FRAMES

    def test_original_value_restored_after_test(self):
        """Sanity check: _MAX_DATA_BYTES is back to its original large value."""
        assert FileSink._MAX_DATA_BYTES > 1_000_000
