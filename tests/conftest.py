import queue
import threading
import time
import numpy as np
import pytest
from pyflexaudio.types import AudioChunk
from pyflexaudio.events import EventBus


class MockSource:
    """テスト用の音声ソース。サイン波を生成"""

    def __init__(self, sample_rate=16000, channels=1, chunk_frames=512,
                 frequency=440.0, source_id="microphone:test"):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_frames = chunk_frames
        self.frequency = frequency
        self._source_id = source_id
        self._is_open = False
        self._thread = None
        self._data_queue = None
        self._stop_event = None
        self._phase = 0.0

    @property
    def is_open(self):
        return self._is_open

    @property
    def source_id(self):
        return self._source_id

    def open(self, data_queue, stop_event):
        self._data_queue = data_queue
        self._stop_event = stop_event
        self._is_open = True
        self._thread = threading.Thread(target=self._generate, daemon=True)
        self._thread.start()

    def close(self):
        self._is_open = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _generate(self):
        while self._is_open and not self._stop_event.is_set():
            t = np.arange(self.chunk_frames) / self.sample_rate + self._phase
            data = np.sin(2 * np.pi * self.frequency * t).astype(np.float32)
            self._phase += self.chunk_frames / self.sample_rate
            if self.channels > 1:
                data = np.column_stack([data] * self.channels)
            else:
                data = data.reshape(-1, 1)
            chunk = AudioChunk(
                data=data,
                timestamp=time.monotonic(),
                sample_rate=self.sample_rate,
                channels=self.channels,
                source_id=self._source_id,
            )
            try:
                self._data_queue.put_nowait(chunk)
            except queue.Full:
                pass
            time.sleep(self.chunk_frames / self.sample_rate * 0.5)  # 半分の速度で生成


class MockSink:
    """テスト用の Sink。受信チャンクを蓄積"""

    def __init__(self, enabled=True, pause_exempt=False):
        self.enabled = enabled
        self.pause_exempt = pause_exempt
        self.chunks: list[AudioChunk] = []
        self.flushed = False
        self.closed = False

    def write(self, chunk):
        if self.enabled:
            self.chunks.append(chunk)

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def mock_source():
    return MockSource()


@pytest.fixture
def mock_sink():
    return MockSink()


@pytest.fixture
def sample_chunk():
    """テスト用の AudioChunk (16kHz, mono, 512 frames)"""
    data = np.random.randn(512, 1).astype(np.float32) * 0.5
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=16000,
        channels=1,
        source_id="microphone:test",
    )


@pytest.fixture
def stereo_chunk():
    """テスト用の AudioChunk (48kHz, stereo, 1024 frames)"""
    data = np.random.randn(1024, 2).astype(np.float32) * 0.5
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=48000,
        channels=2,
        source_id="microphone:test",
    )
