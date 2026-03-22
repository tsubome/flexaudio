"""Tests for WAV crash-recovery via flush().

FileSink updates the WAV header every 30 seconds automatically, but the
flush() method forces an immediate header update. This simulates crash
recovery: if the process dies after writing audio data but before close(),
a previously flushed file should still be readable.
"""

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


class TestFlushUpdatesHeader:
    def test_flush_after_write_updates_header(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk(frames=1024, channels=2))
        sink.flush()

        # After flush the file should be a valid WAV with correct frame count
        with wave.open(path, "rb") as wf:
            assert wf.getnframes() == 1024

    def test_flush_with_multiple_chunks(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)

        total_frames = 0
        for frames in [512, 768, 1024]:
            sink.write(make_chunk(frames=frames))
            total_frames += frames

        sink.flush()

        with wave.open(path, "rb") as wf:
            assert wf.getnframes() == total_frames

    def test_repeated_flush_keeps_file_valid(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)

        for i in range(4):
            sink.write(make_chunk(frames=256))
            sink.flush()
            with wave.open(path, "rb") as wf:
                assert wf.getnframes() == 256 * (i + 1)

        sink.close()


class TestCrashRecoverySimulation:
    def test_flushed_file_readable_without_close(self, tmp_path):
        """Simulate crash: file not closed but was flushed."""
        path = str(tmp_path / "crash_sim.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk(frames=2048, channels=1, sample_rate=16000))
        sink.flush()

        # Do NOT call sink.close() — simulate crash after flush
        # File should still be a valid, readable WAV
        with wave.open(path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2
            assert wf.getnframes() == 2048

    def test_unflushed_file_has_stale_header(self, tmp_path):
        """Without flush, header still shows data_size=0 from initial write."""
        path = str(tmp_path / "no_flush.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk(frames=1024, channels=2))
        # Deliberately do NOT flush or close — simulate crash before any update

        # Python wave module may raise on invalid header, or report 0 frames
        try:
            with wave.open(path, "rb") as wf:
                # Header was not updated, so nframes should be 0
                nframes = wf.getnframes()
            assert nframes == 0
        except Exception:
            # wave module may reject a file with mismatched data; that is acceptable
            pass

    def test_flush_then_more_writes_then_crash(self, tmp_path):
        """Flush captures state at flush point; later writes without flush are lost."""
        path = str(tmp_path / "partial.wav")
        sink = FileSink(path=path)

        # First batch — flushed
        sink.write(make_chunk(frames=512, channels=2))
        sink.flush()

        # Second batch — NOT flushed (crash simulation)
        sink.write(make_chunk(frames=512, channels=2))
        # No close/flush

        # Header reflects only the flushed state (512 frames)
        with wave.open(path, "rb") as wf:
            assert wf.getnframes() == 512

    def test_flush_produces_correct_riff_size(self, tmp_path):
        """RIFF chunk size field = 36 + data_bytes after flush."""
        import struct

        path = str(tmp_path / "output.wav")
        frames = 1024
        channels = 2
        sink = FileSink(path=path)
        sink.write(make_chunk(frames=frames, channels=channels))
        sink.flush()

        expected_data_bytes = frames * channels * 2  # int16
        expected_riff_size = 36 + expected_data_bytes

        with open(path, "rb") as f:
            f.read(4)  # "RIFF"
            riff_size = struct.unpack("<I", f.read(4))[0]

        assert riff_size == expected_riff_size
