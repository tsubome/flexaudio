"""Tests for FileSink."""

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


class TestFileSinkCreation:
    def test_wav_file_is_created(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        chunk = make_chunk()
        sink.write(chunk)
        sink.close()

        assert os.path.exists(path)

    def test_wav_file_has_nonzero_size(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk())
        sink.close()

        assert os.path.getsize(path) > 0


class TestFileSinkGrows:
    def test_file_size_increases_with_each_chunk(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)

        sink.write(make_chunk(frames=512))
        sink.flush()
        size_after_one = os.path.getsize(path)

        sink.write(make_chunk(frames=512))
        sink.flush()
        size_after_two = os.path.getsize(path)

        sink.close()

        assert size_after_two > size_after_one

    def test_file_grows_proportionally_to_frames(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)

        frames = 1024
        channels = 2
        bytes_per_sample = 2  # int16

        sink.write(make_chunk(frames=frames, channels=channels))
        sink.close()

        expected_data_bytes = frames * channels * bytes_per_sample
        expected_total = 44 + expected_data_bytes  # 44-byte header + data
        assert os.path.getsize(path) == expected_total


class TestFileSinkWavValidity:
    def test_wav_header_is_valid_after_close(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk(sample_rate=44100, channels=2))
        sink.close()

        with wave.open(path, "rb") as wf:
            assert wf.getnchannels() == 2
            assert wf.getframerate() == 44100
            assert wf.getsampwidth() == 2  # int16 = 2 bytes

    def test_wav_nframes_correct_after_close(self, tmp_path):
        path = str(tmp_path / "output.wav")
        frames = 2048
        channels = 1
        sink = FileSink(path=path)
        sink.write(make_chunk(frames=frames, channels=channels))
        sink.close()

        with wave.open(path, "rb") as wf:
            assert wf.getnframes() == frames

    def test_wav_readable_with_mono(self, tmp_path):
        path = str(tmp_path / "mono.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk(channels=1, sample_rate=16000))
        sink.close()

        with wave.open(path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000


class TestFileSinkDisable:
    def test_disabled_sink_writes_no_data(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path, enabled=False)
        sink.write(make_chunk())
        sink.write(make_chunk())

        # File should not exist because enabled=False prevents open
        assert not os.path.exists(path)

    def test_disable_mid_write_stops_data(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path, enabled=True)

        sink.write(make_chunk(frames=1024))
        sink.flush()
        size_after_enabled = os.path.getsize(path)

        sink.enabled = False
        sink.write(make_chunk(frames=1024))
        sink.flush()
        size_after_disabled = os.path.getsize(path)

        sink.close()

        # Size should not have changed after disabling
        assert size_after_disabled == size_after_enabled


class TestFileSinkFlush:
    def test_flush_updates_wav_header(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk(frames=1024, channels=2))
        sink.flush()

        # After flush, file should be readable as valid WAV
        with wave.open(path, "rb") as wf:
            assert wf.getnframes() > 0

    def test_flush_before_any_write_does_not_raise(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        sink.flush()  # no file open yet, should be a no-op
        sink.close()

    def test_multiple_flushes_keep_file_valid(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)

        for _ in range(3):
            sink.write(make_chunk(frames=512))
            sink.flush()

        sink.close()

        with wave.open(path, "rb") as wf:
            assert wf.getnframes() == 512 * 3
