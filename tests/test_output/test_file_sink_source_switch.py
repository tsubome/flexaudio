"""Tests for FileSink source-format-change splitting behavior.

When the incoming AudioChunk's sample_rate or channels differ from the
currently open file, FileSink closes the current file and opens a new one
with a numbered suffix (_002, _003, …).
"""

import os
import time
import wave

import numpy as np
import pytest

from pyflexaudio.sinks.file import FileSink
from pyflexaudio.types import AudioChunk


def make_chunk(frames=1024, sample_rate=44100, channels=2, amplitude=0.5):
    data = (np.random.randn(frames, channels) * amplitude).astype(np.float32)
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=sample_rate,
        channels=channels,
        source_id="test:0",
    )


class TestSampleRateChange:
    def test_new_file_created_on_sample_rate_change(self, tmp_path):
        base = str(tmp_path / "output.wav")
        sink = FileSink(path=base)

        sink.write(make_chunk(sample_rate=44100))
        sink.write(make_chunk(sample_rate=16000))  # different rate -> split
        sink.close()

        assert os.path.exists(base)
        second = str(tmp_path / "output_002.wav")
        assert os.path.exists(second)

    def test_first_file_has_original_sample_rate(self, tmp_path):
        base = str(tmp_path / "output.wav")
        sink = FileSink(path=base)

        sink.write(make_chunk(sample_rate=44100))
        sink.write(make_chunk(sample_rate=16000))
        sink.close()

        with wave.open(base, "rb") as wf:
            assert wf.getframerate() == 44100

    def test_second_file_has_new_sample_rate(self, tmp_path):
        base = str(tmp_path / "output.wav")
        sink = FileSink(path=base)

        sink.write(make_chunk(sample_rate=44100))
        sink.write(make_chunk(sample_rate=16000))
        sink.close()

        second = str(tmp_path / "output_002.wav")
        with wave.open(second, "rb") as wf:
            assert wf.getframerate() == 16000

    def test_multiple_sample_rate_changes(self, tmp_path):
        base = str(tmp_path / "output.wav")
        sink = FileSink(path=base)

        sink.write(make_chunk(sample_rate=44100))
        sink.write(make_chunk(sample_rate=16000))
        sink.write(make_chunk(sample_rate=48000))
        sink.close()

        assert os.path.exists(str(tmp_path / "output.wav"))
        assert os.path.exists(str(tmp_path / "output_002.wav"))
        assert os.path.exists(str(tmp_path / "output_003.wav"))


class TestChannelChange:
    def test_new_file_created_on_channel_change(self, tmp_path):
        base = str(tmp_path / "output.wav")
        sink = FileSink(path=base)

        sink.write(make_chunk(channels=2))
        sink.write(make_chunk(channels=1))  # different channels -> split
        sink.close()

        assert os.path.exists(base)
        second = str(tmp_path / "output_002.wav")
        assert os.path.exists(second)

    def test_first_file_has_original_channels(self, tmp_path):
        base = str(tmp_path / "output.wav")
        sink = FileSink(path=base)

        sink.write(make_chunk(channels=2))
        sink.write(make_chunk(channels=1))
        sink.close()

        with wave.open(base, "rb") as wf:
            assert wf.getnchannels() == 2

    def test_second_file_has_new_channels(self, tmp_path):
        base = str(tmp_path / "output.wav")
        sink = FileSink(path=base)

        sink.write(make_chunk(channels=2))
        sink.write(make_chunk(channels=1))
        sink.close()

        second = str(tmp_path / "output_002.wav")
        with wave.open(second, "rb") as wf:
            assert wf.getnchannels() == 1


class TestSplitFileNaming:
    def test_suffix_pattern_is_underscore_003(self, tmp_path):
        """Third file gets _003 suffix."""
        base = str(tmp_path / "rec.wav")
        sink = FileSink(path=base)

        sink.write(make_chunk(sample_rate=44100))
        sink.write(make_chunk(sample_rate=16000))
        sink.write(make_chunk(sample_rate=48000))
        sink.close()

        third = str(tmp_path / "rec_003.wav")
        assert os.path.exists(third)

    def test_base_path_is_first_file(self, tmp_path):
        """The very first file uses the exact path provided, no suffix."""
        base = str(tmp_path / "audio.wav")
        sink = FileSink(path=base)

        sink.write(make_chunk(sample_rate=44100))
        sink.write(make_chunk(sample_rate=16000))
        sink.close()

        # First file: exact path
        assert os.path.exists(base)
        # Second file: _002 suffix
        assert os.path.exists(str(tmp_path / "audio_002.wav"))

    def test_extension_preserved_in_split_files(self, tmp_path):
        base = str(tmp_path / "myfile.wav")
        sink = FileSink(path=base)

        sink.write(make_chunk(sample_rate=44100))
        sink.write(make_chunk(sample_rate=16000))
        sink.close()

        second = str(tmp_path / "myfile_002.wav")
        assert second.endswith(".wav")
        assert os.path.exists(second)

    def test_no_split_when_format_unchanged(self, tmp_path):
        """Same sample_rate and channels across multiple chunks: only one file."""
        base = str(tmp_path / "output.wav")
        sink = FileSink(path=base)

        for _ in range(5):
            sink.write(make_chunk(sample_rate=44100, channels=2))
        sink.close()

        assert os.path.exists(base)
        assert not os.path.exists(str(tmp_path / "output_002.wav"))
