"""pyflexaudio — Flexible cross-platform audio capture library."""

import logging

from pyflexaudio._version import __version__
from pyflexaudio.types import (
    AudioChunk,
    AudioFormat,
    AudioProcess,
    ChunkDroppedEvent,
    DeviceDisconnectedEvent,
    DeviceInfo,
    ErrorEvent,
    FlexAudioError,
    LevelEvent,
    PermissionDeniedEvent,
    QueuePolicy,
    SourceSwitchedEvent,
    SourceType,
    SpeechEndEvent,
    SpeechStartEvent,
    StateChangedEvent,
)
from pyflexaudio.devices import (
    list_input_devices,
    list_output_devices,
    list_loopback_devices,
    list_audio_processes,
)
from pyflexaudio.events import EventBus
from pyflexaudio.permissions import PermissionStatus
from pyflexaudio.session import FlexAudioSession

logging.getLogger("pyflexaudio").addHandler(logging.NullHandler())

__all__ = [
    "__version__",
    # session.py
    "FlexAudioSession",
    # types.py
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
    # devices.py
    "list_input_devices",
    "list_output_devices",
    "list_loopback_devices",
    "list_audio_processes",
    # permissions.py
    "PermissionStatus",
    # events.py
    "EventBus",
]
