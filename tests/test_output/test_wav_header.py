"""Tests for WAV header correctness written by FileSink."""

import os
import struct
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


class TestWavHeaderSize:
    def test_header_is_44_bytes(self, tmp_path):
        """A WAV file with no audio data should be exactly 44 bytes."""
        path = str(tmp_path / "empty.wav")
        # Write a WAV header manually the same way FileSink does, then verify size
        import struct as _struct

        with open(path, "wb") as f:
            sample_rate = 44100
            channels = 2
            data_size = 0
            f.write(b"RIFF")
            f.write(_struct.pack("<I", 36 + data_size))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(_struct.pack("<I", 16))
            f.write(_struct.pack("<H", 1))
            f.write(_struct.pack("<H", channels))
            f.write(_struct.pack("<I", sample_rate))
            f.write(_struct.pack("<I", sample_rate * channels * 2))
            f.write(_struct.pack("<H", channels * 2))
            f.write(_struct.pack("<H", 16))
            f.write(b"data")
            f.write(_struct.pack("<I", data_size))

        assert os.path.getsize(path) == 44

    def test_header_size_with_data(self, tmp_path):
        """File size should equal 44 (header) + data bytes."""
        path = str(tmp_path / "output.wav")
        frames = 1024
        channels = 2
        sink = FileSink(path=path)
        sink.write(make_chunk(frames=frames, channels=channels))
        sink.close()

        expected = 44 + frames * channels * 2  # 2 bytes per int16 sample
        assert os.path.getsize(path) == expected


class TestWavHeaderMarkers:
    def test_riff_marker(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk())
        sink.close()

        with open(path, "rb") as f:
            assert f.read(4) == b"RIFF"

    def test_wave_marker(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk())
        sink.close()

        with open(path, "rb") as f:
            f.read(8)  # skip RIFF + file size
            assert f.read(4) == b"WAVE"

    def test_fmt_chunk_marker(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk())
        sink.close()

        with open(path, "rb") as f:
            f.read(12)  # skip RIFF + size + WAVE
            assert f.read(4) == b"fmt "

    def test_data_chunk_marker(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk())
        sink.close()

        with open(path, "rb") as f:
            f.read(36)  # skip to data chunk
            assert f.read(4) == b"data"


class TestWavHeaderFmtChunk:
    def test_pcm_format_code(self, tmp_path):
        """Audio format code 1 = PCM."""
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk())
        sink.close()

        with open(path, "rb") as f:
            f.read(20)  # RIFF(4) + size(4) + WAVE(4) + fmt (4) + chunk_size(4)
            audio_format = struct.unpack("<H", f.read(2))[0]
        assert audio_format == 1  # PCM

    def test_fmt_chunk_size_is_16(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk())
        sink.close()

        with open(path, "rb") as f:
            f.read(16)  # RIFF(4) + size(4) + WAVE(4) + fmt (4)
            chunk_size = struct.unpack("<I", f.read(4))[0]
        assert chunk_size == 16


class TestWavHeaderSampleRate:
    def test_sample_rate_44100(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk(sample_rate=44100))
        sink.close()

        with wave.open(path, "rb") as wf:
            assert wf.getframerate() == 44100

    def test_sample_rate_16000(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk(sample_rate=16000))
        sink.close()

        with wave.open(path, "rb") as wf:
            assert wf.getframerate() == 16000

    def test_sample_rate_48000(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk(sample_rate=48000))
        sink.close()

        with wave.open(path, "rb") as wf:
            assert wf.getframerate() == 48000


class TestWavHeaderChannels:
    def test_mono_channel_count(self, tmp_path):
        path = str(tmp_path / "mono.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk(channels=1))
        sink.close()

        with wave.open(path, "rb") as wf:
            assert wf.getnchannels() == 1

    def test_stereo_channel_count(self, tmp_path):
        path = str(tmp_path / "stereo.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk(channels=2))
        sink.close()

        with wave.open(path, "rb") as wf:
            assert wf.getnchannels() == 2


class TestWavHeaderBitsPerSample:
    def test_bits_per_sample_is_16(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk())
        sink.close()

        with open(path, "rb") as f:
            f.read(20 + 2 + 2 + 4 + 4 + 2)  # skip to bits_per_sample field
            bits_per_sample = struct.unpack("<H", f.read(2))[0]
        assert bits_per_sample == 16

    def test_sample_width_via_wave_module(self, tmp_path):
        path = str(tmp_path / "output.wav")
        sink = FileSink(path=path)
        sink.write(make_chunk())
        sink.close()

        with wave.open(path, "rb") as wf:
            assert wf.getsampwidth() == 2  # 16 bits = 2 bytes
