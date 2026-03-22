"""統合テスト: MockSource → ProcessorChain(LevelMeter) → FileSink → WAV 検証"""

from __future__ import annotations

import queue
import threading
import time
import wave

import numpy as np
import pytest

from pyflexaudio.events import EventBus
from pyflexaudio.pipeline import Pipeline
from pyflexaudio.processors.chain import ProcessorChain
from pyflexaudio.processors.level import LevelMeterProcessor
from pyflexaudio.sinks.file import FileSink

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import MockSource


class TestMicToFile:
    """MockSource → Pipeline(LevelMeterProcessor) → FileSink → WAV ファイル検証"""

    def test_wav_file_created_and_readable(self, tmp_path):
        """WAV ファイルが作成されて wave.open で読めることを検証"""
        wav_path = str(tmp_path / "output.wav")

        bus = EventBus()
        pipeline = Pipeline(bus)

        # LevelMeterProcessor を含む main chain
        chain = ProcessorChain([LevelMeterProcessor()])
        pipeline.set_main_chain(chain)

        sink = FileSink(path=wav_path)
        pipeline.add_sink(sink)

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=200)
        pipeline.start(data_queue)

        source = MockSource(sample_rate=16000, channels=1, chunk_frames=512)
        source.open(data_queue, stop_event)

        # 約 1 秒分のデータを生成
        time.sleep(1.0)

        stop_event.set()
        source.close()
        pipeline.stop()

        # WAV ファイルが存在する
        assert os.path.exists(wav_path), "WAV ファイルが作成されていない"

        # wave.open で読めること
        with wave.open(wav_path, "rb") as wf:
            assert wf.getnchannels() == 1, "チャンネル数が正しくない"
            assert wf.getframerate() == 16000, "サンプルレートが正しくない"
            assert wf.getnframes() > 0, "フレーム数が 0"
            assert wf.getsampwidth() == 2, "サンプル幅が 2 バイト (int16) でない"

    def test_wav_file_has_audio_data(self, tmp_path):
        """WAV ファイルに実際の音声データが含まれることを検証"""
        wav_path = str(tmp_path / "audio_data.wav")

        bus = EventBus()
        pipeline = Pipeline(bus)
        chain = ProcessorChain([LevelMeterProcessor()])
        pipeline.set_main_chain(chain)

        sink = FileSink(path=wav_path)
        pipeline.add_sink(sink)

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=200)
        pipeline.start(data_queue)

        source = MockSource(sample_rate=16000, channels=1, chunk_frames=512, frequency=440.0)
        source.open(data_queue, stop_event)

        time.sleep(1.0)

        stop_event.set()
        source.close()
        pipeline.stop()

        # WAV ファイルの内容を確認
        with wave.open(wav_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)

        # サイン波なので RMS > 0
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        assert rms > 100, f"RMS が低すぎる（無音の可能性）: {rms:.1f}"

    def test_level_db_set_on_chunks(self, tmp_path):
        """LevelMeterProcessor が level_db を設定することを検証"""
        from pyflexaudio.types import LevelEvent

        wav_path = str(tmp_path / "level.wav")

        bus = EventBus()
        pipeline = Pipeline(bus)

        chain = ProcessorChain([LevelMeterProcessor()])
        pipeline.set_main_chain(chain)

        level_events: list[LevelEvent] = []
        bus.on(LevelEvent, level_events.append)

        # LevelMeterSink を模した Sink
        received_chunks = []

        class ChunkCaptureSink:
            enabled = True
            pause_exempt = False

            def write(self, chunk):
                received_chunks.append(chunk)

            def flush(self):
                pass

            def close(self):
                pass

        pipeline.add_sink(FileSink(path=wav_path))
        pipeline.add_sink(ChunkCaptureSink())

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=200)
        pipeline.start(data_queue)

        source = MockSource(sample_rate=16000, channels=1, chunk_frames=512)
        source.open(data_queue, stop_event)

        time.sleep(0.8)

        stop_event.set()
        source.close()
        pipeline.stop()

        assert len(received_chunks) > 0, "チャンクが受信されていない"
        for chunk in received_chunks:
            assert chunk.level_db is not None, "level_db が設定されていない"
            assert isinstance(chunk.level_db, float), "level_db が float でない"

    def test_stereo_wav_file(self, tmp_path):
        """ステレオ WAV ファイルが正しく作成される"""
        wav_path = str(tmp_path / "stereo.wav")

        bus = EventBus()
        pipeline = Pipeline(bus)

        sink = FileSink(path=wav_path)
        pipeline.add_sink(sink)

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=200)
        pipeline.start(data_queue)

        # ステレオ Source
        source = MockSource(sample_rate=44100, channels=2, chunk_frames=1024)
        source.open(data_queue, stop_event)

        time.sleep(0.8)

        stop_event.set()
        source.close()
        pipeline.stop()

        with wave.open(wav_path, "rb") as wf:
            assert wf.getnchannels() == 2, "ステレオでない"
            assert wf.getframerate() == 44100
            assert wf.getnframes() > 0
