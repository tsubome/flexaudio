"""pyflexaudio の型定義モジュール"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy


__all__ = [
    "SourceType",
    "QueuePolicy",
    "AudioFormat",
    "AudioChunk",
    "DeviceInfo",
    "AudioProcess",
    "FlexAudioError",
    "LevelEvent",
    "SpeechStartEvent",
    "SpeechEndEvent",
    "SourceSwitchedEvent",
    "DeviceDisconnectedEvent",
    "ErrorEvent",
    "ChunkDroppedEvent",
    "StateChangedEvent",
    "PermissionDeniedEvent",
]


class SourceType(enum.Enum):
    MICROPHONE = "microphone"
    SYSTEM_AUDIO = "system_audio"
    PROCESS_AUDIO = "process_audio"


class QueuePolicy(enum.Enum):
    DROP_OLDEST = "drop_oldest"
    BACKPRESSURE = "backpressure"


@dataclass(frozen=True)
class AudioFormat:
    sample_rate: int
    channels: int
    dtype: str  # 'int16' | 'float32'

    @property
    def sample_width(self) -> int:
        return {"int16": 2, "float32": 4}[self.dtype]


@dataclass
class AudioChunk:
    data: numpy.ndarray  # float32, shape=(frames, channels)
    timestamp: float
    sample_rate: int
    channels: int
    source_id: str  # "{source_type.value}:{device_index_or_pid}"
    level_db: float | None = None


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    default_sample_rate: int
    is_loopback: bool


@dataclass(frozen=True)
class AudioProcess:
    pid: int
    name: str
    window_title: str


class FlexAudioError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        platform: str = "",
        traceback_str: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.platform = platform
        self.traceback_str = traceback_str


@dataclass(frozen=True)
class LevelEvent:
    db: float
    source_id: str


@dataclass(frozen=True)
class SpeechStartEvent:
    timestamp: float
    source_id: str


@dataclass(frozen=True)
class SpeechEndEvent:
    timestamp: float
    duration_sec: float
    audio_data: numpy.ndarray  # float32, 16kHz, mono, shape=(frames,)
    source_id: str


@dataclass(frozen=True)
class SourceSwitchedEvent:
    old_source_id: str
    new_source_id: str


@dataclass(frozen=True)
class DeviceDisconnectedEvent:
    device_info: DeviceInfo


@dataclass(frozen=True)
class ErrorEvent:
    error: FlexAudioError
    source_id: str


@dataclass(frozen=True)
class ChunkDroppedEvent:
    drop_count: int
    queue_size: int
    source_id: str


@dataclass(frozen=True)
class StateChangedEvent:
    old_state: str
    new_state: str


@dataclass(frozen=True)
class PermissionDeniedEvent:
    permission_type: str
    platform: str
    message: str
