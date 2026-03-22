"""Tests for FileSink enabled/disabled on-off behavior.

Verifies that disabling the sink skips writes (no silent padding),
and re-enabling resumes writing from where it left off.
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


class TestFileSinkOnOff:
    def test_enabled_writes_data(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path, enabled=True)
        sink.write(make_chunk(frames=1024))
        sink.close()

        with wave.open(path, "rb") as wf:
            assert wf.getnframes() == 1024

    def test_disabled_skips_write(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path, enabled=True)

        sink.write(make_chunk(frames=1024))
        sink.enabled = False
        sink.write(make_chunk(frames=1024))  # should be skipped
        sink.close()

        with wave.open(path, "rb") as wf:
            # Only the first chunk should be in the file
            assert wf.getnframes() == 1024

    def test_reenable_resumes_writing(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path, enabled=True)

        sink.write(make_chunk(frames=512))   # written
        sink.enabled = False
        sink.write(make_chunk(frames=512))   # skipped
        sink.write(make_chunk(frames=512))   # skipped
        sink.enabled = True
        sink.write(make_chunk(frames=512))   # written
        sink.close()

        with wave.open(path, "rb") as wf:
            # Only the two enabled chunks (512 + 512 = 1024)
            assert wf.getnframes() == 1024

    def test_no_silence_gap_when_disabled(self, tmp_path):
        """Disabled period is not padded with silence; frames count reflects only enabled writes."""
        path = str(tmp_path / "output.wav")
        frames_on = 256
        frames_off = 1024  # this duration is "paused"
        frames_on2 = 256

        sink = FileSink(path=path, enabled=True)
        sink.write(make_chunk(frames=frames_on))
        sink.enabled = False
        sink.write(make_chunk(frames=frames_off))
        sink.enabled = True
        sink.write(make_chunk(frames=frames_on2))
        sink.close()

        with wave.open(path, "rb") as wf:
            total_frames = wf.getnframes()

        # Silence gap would add frames_off; verify it's absent
        assert total_frames == frames_on + frames_on2
        assert total_frames != frames_on + frames_off + frames_on2

    def test_multiple_on_off_cycles(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path, enabled=True)

        expected_frames = 0
        for i in range(5):
            sink.write(make_chunk(frames=100))
            expected_frames += 100
            sink.enabled = False
            sink.write(make_chunk(frames=200))  # skipped
            sink.enabled = True

        sink.close()

        with wave.open(path, "rb") as wf:
            assert wf.getnframes() == expected_frames

    def test_initial_disabled_creates_no_file(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path, enabled=False)

        sink.write(make_chunk(frames=1024))
        sink.write(make_chunk(frames=1024))
        # close without ever enabling
        sink.close()

        assert not os.path.exists(path)

    def test_file_size_reflects_only_enabled_writes(self, tmp_path):
        path = str(tmp_path / "output.wav")
        channels = 2
        frames_written = 1024

        sink = FileSink(path=path, enabled=True)
        sink.write(make_chunk(frames=frames_written, channels=channels))
        sink.enabled = False
        sink.write(make_chunk(frames=4096, channels=channels))  # large skipped chunk
        sink.close()

        expected_size = 44 + frames_written * channels * 2
        assert os.path.getsize(path) == expected_size
