"""統合テスト: MockSource(合成音声) → 解析チェーン → SpeechStart/End 検証"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest

# onnxruntime がなければスキップ
onnxruntime = pytest.importorskip("onnxruntime")

from pyflexaudio.events import EventBus
from pyflexaudio.pipeline import Pipeline
from pyflexaudio.processors.chain import ProcessorChain
from pyflexaudio.types import AudioChunk, SpeechEndEvent, SpeechStartEvent

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import MockSource


def _vad_model_available() -> bool:
    """Silero VAD モデルファイルが存在するか確認"""
    from pathlib import Path
    model_path = Path(__file__).parent.parent.parent / "src" / "pyflexaudio" / "processors" / "models" / "silero_vad.onnx"
    return model_path.exists()


pytestmark = pytest.mark.skipif(
    not _vad_model_available(),
    reason="Silero VAD モデルファイルが見つからない"
)


class SilenceSource:
    """無音を生成するテスト用ソース"""

    def __init__(self, sample_rate: int = 16000, chunk_frames: int = 512,
                 duration_sec: float = 1.0, source_id: str = "microphone:silence"):
        self.sample_rate = sample_rate
        self.chunk_frames = chunk_frames
        self.duration_sec = duration_sec
        self._source_id = source_id
        self._is_open = False
        self._thread: threading.Thread | None = None
        self._data_queue: queue.Queue | None = None
        self._stop_event: threading.Event | None = None

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def source_id(self) -> str:
        return self._source_id

    def open(self, data_queue: queue.Queue, stop_event: threading.Event) -> None:
        self._data_queue = data_queue
        self._stop_event = stop_event
        self._is_open = True
        self._thread = threading.Thread(target=self._generate, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._is_open = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _generate(self) -> None:
        total_chunks = int(self.duration_sec * self.sample_rate / self.chunk_frames)
        for _ in range(total_chunks):
            if not self._is_open or self._stop_event.is_set():
                break
            data = np.zeros((self.chunk_frames, 1), dtype=np.float32)
            chunk = AudioChunk(
                data=data,
                timestamp=time.monotonic(),
                sample_rate=self.sample_rate,
                channels=1,
                source_id=self._source_id,
            )
            try:
                self._data_queue.put(chunk, timeout=1.0)
            except queue.Full:
                pass
            time.sleep(self.chunk_frames / self.sample_rate * 0.5)


class PatternSource:
    """無音→サイン波→無音のパターンを生成するテスト用ソース"""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_frames: int = 512,
        silence_chunks_before: int = 20,
        speech_chunks: int = 60,
        silence_chunks_after: int = 40,
        amplitude: float = 0.8,
        frequency: float = 440.0,
        source_id: str = "microphone:pattern",
    ):
        self.sample_rate = sample_rate
        self.chunk_frames = chunk_frames
        self.silence_chunks_before = silence_chunks_before
        self.speech_chunks = speech_chunks
        self.silence_chunks_after = silence_chunks_after
        self.amplitude = amplitude
        self.frequency = frequency
        self._source_id = source_id
        self._is_open = False
        self._thread: threading.Thread | None = None
        self._data_queue: queue.Queue | None = None
        self._stop_event: threading.Event | None = None
        self._phase = 0.0

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def source_id(self) -> str:
        return self._source_id

    def open(self, data_queue: queue.Queue, stop_event: threading.Event) -> None:
        self._data_queue = data_queue
        self._stop_event = stop_event
        self._is_open = True
        self._thread = threading.Thread(target=self._generate, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._is_open = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _make_silence_chunk(self) -> AudioChunk:
        data = np.zeros((self.chunk_frames, 1), dtype=np.float32)
        return AudioChunk(
            data=data,
            timestamp=time.monotonic(),
            sample_rate=self.sample_rate,
            channels=1,
            source_id=self._source_id,
        )

    def _make_speech_chunk(self) -> AudioChunk:
        t = np.arange(self.chunk_frames) / self.sample_rate + self._phase
        data = (self.amplitude * np.sin(2 * np.pi * self.frequency * t)).astype(np.float32)
        self._phase += self.chunk_frames / self.sample_rate
        return AudioChunk(
            data=data.reshape(-1, 1),
            timestamp=time.monotonic(),
            sample_rate=self.sample_rate,
            channels=1,
            source_id=self._source_id,
        )

    def _put_chunk(self, chunk: AudioChunk) -> None:
        try:
            self._data_queue.put(chunk, timeout=1.0)
        except queue.Full:
            pass
        time.sleep(self.chunk_frames / self.sample_rate * 0.5)

    def _generate(self) -> None:
        # 無音
        for _ in range(self.silence_chunks_before):
            if not self._is_open or self._stop_event.is_set():
                return
            self._put_chunk(self._make_silence_chunk())

        # 発話（サイン波）
        for _ in range(self.speech_chunks):
            if not self._is_open or self._stop_event.is_set():
                return
            self._put_chunk(self._make_speech_chunk())

        # 無音（発話終了を確定させる）
        for _ in range(self.silence_chunks_after):
            if not self._is_open or self._stop_event.is_set():
                return
            self._put_chunk(self._make_silence_chunk())


class TestVADPipeline:
    """MockSource(合成音声) → VAD 解析チェーン → SpeechStart/End 検証"""

    def _build_vad_pipeline(self, bus: EventBus) -> Pipeline:
        """VAD 解析チェーン付きの Pipeline を構築"""
        from pyflexaudio.processors.vad import SileroVADProcessor

        pipeline = Pipeline(bus)
        analysis_chain = ProcessorChain([SileroVADProcessor(bus)])
        pipeline.set_analysis_chain(analysis_chain)
        return pipeline

    def test_speech_start_event_emitted(self):
        """発話区間で SpeechStartEvent が emit される"""
        bus = EventBus()
        pipeline = self._build_vad_pipeline(bus)

        speech_start_events: list[SpeechStartEvent] = []
        bus.on(SpeechStartEvent, speech_start_events.append)

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=500)
        pipeline.start(data_queue)

        source = PatternSource(
            sample_rate=16000,
            chunk_frames=512,
            silence_chunks_before=10,
            speech_chunks=60,
            silence_chunks_after=40,
            amplitude=0.9,
        )
        source.open(data_queue, stop_event)

        # パターン生成完了を待つ
        time.sleep(
            (source.silence_chunks_before + source.speech_chunks + source.silence_chunks_after)
            * source.chunk_frames / source.sample_rate * 0.6 + 3.0
        )

        stop_event.set()
        source.close()
        pipeline.stop()

        assert len(speech_start_events) > 0, "SpeechStartEvent が emit されていない"

    def test_speech_end_event_emitted(self):
        """発話区間終了後に SpeechEndEvent が emit される"""
        bus = EventBus()
        pipeline = self._build_vad_pipeline(bus)

        speech_end_events: list[SpeechEndEvent] = []
        bus.on(SpeechEndEvent, speech_end_events.append)

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=500)
        pipeline.start(data_queue)

        source = PatternSource(
            sample_rate=16000,
            chunk_frames=512,
            silence_chunks_before=10,
            speech_chunks=60,
            silence_chunks_after=50,  # 十分な無音でSpeechEndを確定させる
            amplitude=0.9,
        )
        source.open(data_queue, stop_event)

        time.sleep(
            (source.silence_chunks_before + source.speech_chunks + source.silence_chunks_after)
            * source.chunk_frames / source.sample_rate * 0.6 + 3.0
        )

        stop_event.set()
        source.close()
        pipeline.stop()

        assert len(speech_end_events) > 0, "SpeechEndEvent が emit されていない"

    def test_speech_end_has_audio_data(self):
        """SpeechEndEvent に音声データが含まれることを検証"""
        bus = EventBus()
        pipeline = self._build_vad_pipeline(bus)

        speech_end_events: list[SpeechEndEvent] = []
        bus.on(SpeechEndEvent, speech_end_events.append)

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=500)
        pipeline.start(data_queue)

        source = PatternSource(
            sample_rate=16000,
            chunk_frames=512,
            silence_chunks_before=10,
            speech_chunks=60,
            silence_chunks_after=50,
            amplitude=0.9,
        )
        source.open(data_queue, stop_event)

        time.sleep(
            (source.silence_chunks_before + source.speech_chunks + source.silence_chunks_after)
            * source.chunk_frames / source.sample_rate * 0.6 + 3.0
        )

        stop_event.set()
        source.close()
        pipeline.stop()

        assert len(speech_end_events) > 0
        event = speech_end_events[0]
        assert isinstance(event.audio_data, np.ndarray), "audio_data が ndarray でない"
        assert len(event.audio_data) > 0, "audio_data が空"
        assert event.audio_data.dtype == np.float32, "audio_data の dtype が float32 でない"
        assert event.duration_sec > 0, "duration_sec が 0 以下"

    def test_silence_no_speech_events(self):
        """無音のみのストリームでは SpeechStart/End が emit されない"""
        bus = EventBus()
        pipeline = self._build_vad_pipeline(bus)

        speech_start_events: list[SpeechStartEvent] = []
        speech_end_events: list[SpeechEndEvent] = []
        bus.on(SpeechStartEvent, speech_start_events.append)
        bus.on(SpeechEndEvent, speech_end_events.append)

        stop_event = threading.Event()
        data_queue: queue.Queue = queue.Queue(maxsize=500)
        pipeline.start(data_queue)

        # 無音のみ
        silence_source = SilenceSource(
            sample_rate=16000,
            chunk_frames=512,
            duration_sec=2.0,
        )
        silence_source.open(data_queue, stop_event)

        time.sleep(3.0)

        stop_event.set()
        silence_source.close()
        pipeline.stop()

        assert len(speech_start_events) == 0, \
            f"無音なのに SpeechStartEvent が {len(speech_start_events)} 件 emit された"
        assert len(speech_end_events) == 0, \
            f"無音なのに SpeechEndEvent が {len(speech_end_events)} 件 emit された"
