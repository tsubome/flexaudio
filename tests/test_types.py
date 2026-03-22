import time
import numpy as np
import pytest
from pyflexaudio.types import (
    SourceType,
    QueuePolicy,
    AudioFormat,
    AudioChunk,
    DeviceInfo,
    AudioProcess,
    FlexAudioError,
    LevelEvent,
    SpeechStartEvent,
    SpeechEndEvent,
    SourceSwitchedEvent,
    DeviceDisconnectedEvent,
    ErrorEvent,
    ChunkDroppedEvent,
    StateChangedEvent,
    PermissionDeniedEvent,
)


# --- SourceType ---

def test_source_type_microphone():
    assert SourceType.MICROPHONE.value == "microphone"


def test_source_type_system_audio():
    assert SourceType.SYSTEM_AUDIO.value == "system_audio"


def test_source_type_process_audio():
    assert SourceType.PROCESS_AUDIO.value == "process_audio"


def test_source_type_members():
    members = {e.value for e in SourceType}
    assert members == {"microphone", "system_audio", "process_audio"}


# --- QueuePolicy ---

def test_queue_policy_drop_oldest():
    assert QueuePolicy.DROP_OLDEST.value == "drop_oldest"


def test_queue_policy_backpressure():
    assert QueuePolicy.BACKPRESSURE.value == "backpressure"


def test_queue_policy_members():
    members = {e.value for e in QueuePolicy}
    assert members == {"drop_oldest", "backpressure"}


# --- AudioFormat ---

def test_audio_format_creation():
    fmt = AudioFormat(sample_rate=16000, channels=1, dtype="float32")
    assert fmt.sample_rate == 16000
    assert fmt.channels == 1
    assert fmt.dtype == "float32"


def test_audio_format_frozen():
    fmt = AudioFormat(sample_rate=16000, channels=1, dtype="float32")
    with pytest.raises((AttributeError, TypeError)):
        fmt.sample_rate = 8000  # type: ignore[misc]


def test_audio_format_sample_width_int16():
    fmt = AudioFormat(sample_rate=16000, channels=1, dtype="int16")
    assert fmt.sample_width == 2


def test_audio_format_sample_width_float32():
    fmt = AudioFormat(sample_rate=48000, channels=2, dtype="float32")
    assert fmt.sample_width == 4


# --- AudioChunk ---

def test_audio_chunk_creation(sample_chunk):
    assert sample_chunk.data.shape == (512, 1)
    assert sample_chunk.sample_rate == 16000
    assert sample_chunk.channels == 1
    assert sample_chunk.source_id == "microphone:test"


def test_audio_chunk_level_db_default_none(sample_chunk):
    assert sample_chunk.level_db is None


def test_audio_chunk_level_db_settable(sample_chunk):
    sample_chunk.level_db = -20.5
    assert sample_chunk.level_db == -20.5


def test_audio_chunk_stereo(stereo_chunk):
    assert stereo_chunk.data.shape == (1024, 2)
    assert stereo_chunk.sample_rate == 48000
    assert stereo_chunk.channels == 2


def test_audio_chunk_not_frozen():
    data = np.zeros((256, 1), dtype=np.float32)
    chunk = AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=16000,
        channels=1,
        source_id="microphone:0",
    )
    # AudioChunk is a regular (non-frozen) dataclass; mutation should succeed
    chunk.level_db = -30.0
    assert chunk.level_db == -30.0


# --- DeviceInfo ---

def test_device_info_creation():
    dev = DeviceInfo(
        index=0,
        name="Built-in Microphone",
        host_api="Core Audio",
        max_input_channels=2,
        default_sample_rate=44100,
        is_loopback=False,
    )
    assert dev.index == 0
    assert dev.name == "Built-in Microphone"
    assert dev.host_api == "Core Audio"
    assert dev.max_input_channels == 2
    assert dev.default_sample_rate == 44100
    assert dev.is_loopback is False


def test_device_info_frozen():
    dev = DeviceInfo(
        index=0,
        name="Test Device",
        host_api="WASAPI",
        max_input_channels=1,
        default_sample_rate=16000,
        is_loopback=True,
    )
    with pytest.raises((AttributeError, TypeError)):
        dev.index = 99  # type: ignore[misc]


# --- AudioProcess ---

def test_audio_process_creation():
    proc = AudioProcess(pid=1234, name="Spotify", window_title="Spotify — Playing")
    assert proc.pid == 1234
    assert proc.name == "Spotify"
    assert proc.window_title == "Spotify — Playing"


def test_audio_process_frozen():
    proc = AudioProcess(pid=1234, name="Spotify", window_title="Spotify")
    with pytest.raises((AttributeError, TypeError)):
        proc.pid = 9999  # type: ignore[misc]


# --- FlexAudioError ---

def test_flex_audio_error_basic():
    err = FlexAudioError(code="ERR_DEVICE_NOT_FOUND", message="Device not found")
    assert err.code == "ERR_DEVICE_NOT_FOUND"
    assert err.message == "Device not found"
    assert str(err) == "Device not found"


def test_flex_audio_error_platform():
    err = FlexAudioError(
        code="ERR_PERMISSION",
        message="Permission denied",
        platform="darwin",
    )
    assert err.platform == "darwin"


def test_flex_audio_error_traceback_str():
    err = FlexAudioError(
        code="ERR_IO",
        message="I/O error",
        traceback_str="Traceback (most recent call last): ...",
    )
    assert "Traceback" in err.traceback_str


def test_flex_audio_error_defaults():
    err = FlexAudioError(code="ERR_UNKNOWN", message="Unknown error")
    assert err.platform == ""
    assert err.traceback_str == ""


def test_flex_audio_error_is_exception():
    err = FlexAudioError(code="ERR_TIMEOUT", message="Timeout")
    assert isinstance(err, Exception)


# --- Event types ---

def test_level_event():
    ev = LevelEvent(db=-20.0, source_id="microphone:0")
    assert ev.db == -20.0
    assert ev.source_id == "microphone:0"


def test_level_event_frozen():
    ev = LevelEvent(db=-20.0, source_id="microphone:0")
    with pytest.raises((AttributeError, TypeError)):
        ev.db = 0.0  # type: ignore[misc]


def test_speech_start_event():
    ev = SpeechStartEvent(timestamp=1.0, source_id="microphone:0")
    assert ev.timestamp == 1.0
    assert ev.source_id == "microphone:0"


def test_speech_end_event_with_numpy_array():
    audio = np.zeros(16000, dtype=np.float32)
    ev = SpeechEndEvent(
        timestamp=2.0,
        duration_sec=1.0,
        audio_data=audio,
        source_id="microphone:0",
    )
    assert ev.duration_sec == 1.0
    assert ev.audio_data.shape == (16000,)
    assert ev.source_id == "microphone:0"


def test_speech_end_event_frozen():
    audio = np.zeros(8000, dtype=np.float32)
    ev = SpeechEndEvent(
        timestamp=1.0,
        duration_sec=0.5,
        audio_data=audio,
        source_id="microphone:0",
    )
    with pytest.raises((AttributeError, TypeError)):
        ev.duration_sec = 99.0  # type: ignore[misc]


def test_source_switched_event():
    ev = SourceSwitchedEvent(old_source_id="microphone:0", new_source_id="microphone:1")
    assert ev.old_source_id == "microphone:0"
    assert ev.new_source_id == "microphone:1"


def test_device_disconnected_event():
    dev = DeviceInfo(
        index=2,
        name="USB Mic",
        host_api="ALSA",
        max_input_channels=1,
        default_sample_rate=48000,
        is_loopback=False,
    )
    ev = DeviceDisconnectedEvent(device_info=dev)
    assert ev.device_info is dev


def test_error_event():
    err = FlexAudioError(code="ERR_IO", message="I/O error")
    ev = ErrorEvent(error=err, source_id="microphone:0")
    assert ev.error is err
    assert ev.source_id == "microphone:0"


def test_chunk_dropped_event():
    ev = ChunkDroppedEvent(drop_count=5, queue_size=100, source_id="microphone:0")
    assert ev.drop_count == 5
    assert ev.queue_size == 100
    assert ev.source_id == "microphone:0"


def test_state_changed_event():
    ev = StateChangedEvent(old_state="idle", new_state="recording")
    assert ev.old_state == "idle"
    assert ev.new_state == "recording"


def test_permission_denied_event():
    ev = PermissionDeniedEvent(
        permission_type="microphone",
        platform="darwin",
        message="Microphone access denied",
    )
    assert ev.permission_type == "microphone"
    assert ev.platform == "darwin"
    assert ev.message == "Microphone access denied"
