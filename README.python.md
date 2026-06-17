[🇯🇵 日本語](README.ja.md) | **🇺🇸 English**

# pyflexaudio

**Flexible cross-platform audio capture library for Python**

pyflexaudio provides a single, unified API for capturing audio from microphones, system audio (loopback), and individual processes — across Windows, macOS, and Linux.

```python
from pyflexaudio import FlexAudioSession, SourceType
from pyflexaudio.sinks.file import FileSink
from pyflexaudio.types import LevelEvent

with FlexAudioSession() as session:
    session.on(LevelEvent, lambda e: print(f"Level: {e.db:.1f} dB"))
    session.add_sink(FileSink("output.wav"))
    session.set_source(SourceType.MICROPHONE)
    session.start()
    input("Recording… press Enter to stop.\n")
```

---

## Features

### Audio Sources

| Source | Description |
|--------|-------------|
| **Microphone** | Via `sounddevice` (PortAudio) with `miniaudio` fallback |
| **System audio** | WASAPI Loopback (Windows), ScreenCaptureKit (macOS) |
| **Per-process audio** | ProcessAudioCapture DLL (Windows 11), ScreenCaptureKit filter (macOS) |
| **Hot-swappable** | Call `set_source()` while running to switch sources on-the-fly |

### Processing Pipeline

- **Source → Processor → Sink** fan-out architecture
- **`LevelMeterProcessor`** — Real-time RMS → dB level calculation attached to every chunk
- **`ResampleProcessor`** — High-quality streaming resampling via soxr
- **`ChannelConvertProcessor`** — Stereo ↔ Mono conversion
- **Silero VAD** — Voice Activity Detection with `SpeechStartEvent` / `SpeechEndEvent`
- **`ProcessorChain`** — Compose multiple processors sequentially

### Output Sinks

| Sink | Description |
|------|-------------|
| **`FileSink`** | WAV int16 output; crash-safe header updates every 30 s; automatic file splitting |
| **`CallbackSink`** | Real-time `AudioChunk` delivery to any Python callable |
| **`LevelMeterSink`** | Continuous `LevelEvent` emission — active even during pause |

### Event System

Type-safe `EventBus` with the following built-in events:

`LevelEvent` · `SpeechStartEvent` · `SpeechEndEvent` · `DeviceDisconnectedEvent` · `StateChangedEvent` · `ErrorEvent` · `ChunkDroppedEvent` · `SourceSwitchedEvent` · `PermissionDeniedEvent`

### Resilience

- **Partial failure tolerance** — one sink failure does not affect the others
- **Crash-safe WAV** — headers flushed to disk every 30 seconds via `fsync`
- **Automatic file splitting** — triggered on source format change or WAV 4 GB limit
- **Graceful device disconnection** — emits `DeviceDisconnectedEvent` and keeps running
- **Idempotent controls** — calling `start()` / `stop()` / `pause()` / `resume()` multiple times is always safe

---

## Platform Support

| Feature | Windows | macOS | Linux |
|---------|:-------:|:-----:|:-----:|
| Microphone capture | ✓ | ✓ | ✓ |
| System audio (loopback) | ✓ (WASAPI) | ✓ (ScreenCaptureKit) | — |
| Per-process audio | ✓ (Win 11, DLL) | ✓ (ScreenCaptureKit) | — |
| Microphone fallback (miniaudio) | ✓ | ✓ | ✓ |

---

## Requirements

- Python ≥ 3.10
- [numpy](https://numpy.org/) ≥ 1.24
- [sounddevice](https://python-sounddevice.readthedocs.io/) ≥ 0.5
- [soxr](https://github.com/dofuuz/python-soxr) ≥ 0.5

---

## Installation

**Basic (microphone only)**

```bash
pip install pyflexaudio
```

**With VAD (Silero Voice Activity Detection)**

```bash
pip install "pyflexaudio[vad]"
```

**macOS (system / process audio)**

```bash
pip install "pyflexaudio[mac]"
```

**Windows system audio**

```bash
pip install "pyflexaudio[win-system]"
```

**Windows per-process audio**

```bash
pip install "pyflexaudio[win-process]"
```

**Everything**

```bash
pip install "pyflexaudio[full]"
```

---

## Quick Start

### 1. Microphone Recording

```python
import time
from pyflexaudio import FlexAudioSession, SourceType
from pyflexaudio.sinks.file import FileSink
from pyflexaudio.types import LevelEvent

with FlexAudioSession() as session:
    session.on(LevelEvent, lambda e: print(f"  {e.db:+.1f} dB", end="\r"))

    sink_id = session.add_sink(FileSink("recording.wav"))
    session.set_source(SourceType.MICROPHONE)
    session.start()

    time.sleep(10)  # record for 10 seconds
# WAV file is safely closed on __exit__
```

### 2. Voice Activity Detection

```python
from pyflexaudio import FlexAudioSession, SourceType
from pyflexaudio.types import SpeechStartEvent, SpeechEndEvent

with FlexAudioSession(vad_enabled=True) as session:
    session.on(SpeechStartEvent, lambda e: print("Speech started"))

    def on_speech_end(e: SpeechEndEvent):
        print(f"Speech ended — {e.duration_sec:.2f} s, {len(e.audio_data)} frames")

    session.on(SpeechEndEvent, on_speech_end)
    session.set_source(SourceType.MICROPHONE)
    session.start()

    input("Listening for speech… press Enter to stop.\n")
```

### 3. System Audio Capture

```python
from pyflexaudio import FlexAudioSession, SourceType
from pyflexaudio.sinks.file import FileSink

with FlexAudioSession() as session:
    session.add_sink(FileSink("system_audio.wav"))
    session.set_source(SourceType.SYSTEM_AUDIO)
    session.start()

    input("Capturing system audio… press Enter to stop.\n")
```

### 4. Per-Process Audio Capture

```python
from pyflexaudio import FlexAudioSession, SourceType
from pyflexaudio.sinks.file import FileSink
from pyflexaudio.sinks.callback import CallbackSink
from pyflexaudio.types import AudioChunk

TARGET_PID = 12345  # replace with the target process PID

def on_chunk(chunk: AudioChunk) -> None:
    print(f"Received {len(chunk.data)} frames from {chunk.source_id}")

with FlexAudioSession() as session:
    session.add_sink(FileSink("process_audio.wav"))
    session.add_sink(CallbackSink(on_chunk))
    session.set_source(SourceType.PROCESS_AUDIO, pid=TARGET_PID)
    session.start()

    input("Capturing process audio… press Enter to stop.\n")
```

### 5. Live Source Switching

```python
import time
from pyflexaudio import FlexAudioSession, SourceType
from pyflexaudio.sinks.file import FileSink
from pyflexaudio.types import SourceSwitchedEvent

with FlexAudioSession() as session:
    session.on(
        SourceSwitchedEvent,
        lambda e: print(f"Switched: {e.old_source_id} → {e.new_source_id}"),
    )
    session.add_sink(FileSink("mixed.wav"))
    session.set_source(SourceType.MICROPHONE)
    session.start()

    time.sleep(5)
    print("Switching to system audio…")
    session.set_source(SourceType.SYSTEM_AUDIO)  # hot-swap while running

    time.sleep(5)
# Both segments are written to the same WAV file (auto-split on format change)
```

### 6. Pause / Resume

```python
import time
from pyflexaudio import FlexAudioSession, SourceType
from pyflexaudio.sinks.file import FileSink
from pyflexaudio.types import LevelEvent

with FlexAudioSession() as session:
    # LevelEvent keeps firing during pause — useful for VU meters in UI
    session.on(LevelEvent, lambda e: print(f"Level: {e.db:+.1f} dB", end="\r"))

    session.add_sink(FileSink("output.wav"))
    session.set_source(SourceType.MICROPHONE)
    session.start()

    time.sleep(3)
    print("\nPausing…")
    session.pause()         # FileSink stops writing; LevelMeterSink stays active

    time.sleep(2)
    print("Resuming…")
    session.resume()        # FileSink resumes writing

    time.sleep(3)
```

---

## API Reference

### `FlexAudioSession`

```python
FlexAudioSession(
    vad_enabled: bool = False,
    vad_sample_rate: int = 16000,
    vad_channels: int = 1,
    source_timeout_sec: float = 10.0,
    queue_policy: QueuePolicy = QueuePolicy.DROP_OLDEST,
    queue_size: int = 200,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vad_enabled` | `bool` | `False` | Enable Silero VAD analysis chain |
| `vad_sample_rate` | `int` | `16000` | Target sample rate for VAD processing |
| `vad_channels` | `int` | `1` | Target channel count for VAD processing |
| `source_timeout_sec` | `float` | `10.0` | Timeout waiting for first audio frame |
| `queue_policy` | `QueuePolicy` | `DROP_OLDEST` | Overflow behaviour (`DROP_OLDEST` or `BACKPRESSURE`) |
| `queue_size` | `int` | `200` | Internal audio chunk queue depth |

**Methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `set_source` | `(source_type, *, device_index=None, pid=None, mode="include") → None` | Set or hot-swap the audio source |
| `add_sink` | `(sink) → str` | Add a sink; returns its `sink_id` |
| `remove_sink` | `(sink_id: str) → None` | Remove a sink by ID |
| `enable_sink` | `(sink_id: str) → None` | Enable a previously disabled sink |
| `disable_sink` | `(sink_id: str) → None` | Disable a sink without removing it |
| `start` | `() → None` | Start the session (idempotent) |
| `stop` | `() → None` | Stop the session and flush all sinks (idempotent) |
| `pause` | `() → None` | Pause sink delivery (idempotent) |
| `resume` | `() → None` | Resume sink delivery (idempotent) |
| `on` | `(event_type: type, handler) → None` | Register an event handler |
| `off` | `(event_type: type, handler) → None` | Unregister an event handler |

**Properties**

| Property | Type | Description |
|----------|------|-------------|
| `is_running` | `bool` | `True` if state is `RUNNING` or `PAUSED` |
| `is_paused` | `bool` | `True` if state is `PAUSED` |
| `current_source_type` | `SourceType \| None` | The currently configured source type |
| `level_db` | `float \| None` | Latest level (use `LevelEvent` handler for real-time updates) |

Implements the context manager protocol — `stop()` is called automatically on `__exit__`.

---

### Sinks

#### `FileSink`

```python
FileSink(
    path: str,
    sample_rate: int | None = None,
    channels: int | None = None,
    *,
    enabled: bool = True,
)
```

| Parameter | Description |
|-----------|-------------|
| `path` | Output file path (`.wav`) |
| `sample_rate` | Target sample rate; `None` uses the source rate |
| `channels` | Target channel count; `None` uses the source count |
| `enabled` | Set `False` to skip writing without removing the sink |

- Output format: WAV PCM int16
- WAV header is updated every 30 seconds and on `close()` — crash-safe
- File is split automatically on source format change or when the 4 GB WAV limit is reached (suffix `_002.wav`, `_003.wav`, …)

#### `CallbackSink`

```python
CallbackSink(
    callback: Callable[[AudioChunk], None],
    *,
    enabled: bool = True,
)
```

Calls `callback(chunk)` synchronously on the pipeline thread for every `AudioChunk`. Keep the callback fast; offload heavy work to another thread if needed.

#### `LevelMeterSink`

```python
LevelMeterSink(event_bus: EventBus)
```

Managed internally by `FlexAudioSession`. Emits a `LevelEvent` for every chunk, even while the session is paused (`pause_exempt = True`). You do not need to add this sink manually.

---

### Events

| Event | Fields | Description |
|-------|--------|-------------|
| `LevelEvent` | `db: float`, `source_id: str` | RMS level in dBFS for each chunk |
| `SpeechStartEvent` | `timestamp: float`, `source_id: str` | VAD detected speech onset |
| `SpeechEndEvent` | `timestamp: float`, `duration_sec: float`, `audio_data: ndarray`, `source_id: str` | VAD detected speech end; includes raw float32 16 kHz mono audio |
| `SourceSwitchedEvent` | `old_source_id: str`, `new_source_id: str` | Source hot-swap completed |
| `DeviceDisconnectedEvent` | `device_info: DeviceInfo` | Capture device was unplugged |
| `StateChangedEvent` | `old_state: str`, `new_state: str` | Session state machine transition |
| `ErrorEvent` | `error: FlexAudioError`, `source_id: str` | Non-fatal error in source or sink |
| `ChunkDroppedEvent` | `drop_count: int`, `queue_size: int`, `source_id: str` | Queue overflow; chunks were dropped |
| `PermissionDeniedEvent` | `permission_type: str`, `platform: str`, `message: str` | OS permission denied (e.g. microphone access) |

---

### Data Types

#### `AudioChunk`

```python
@dataclass
class AudioChunk:
    data: numpy.ndarray   # float32, shape=(frames, channels)
    timestamp: float      # Unix timestamp of the first frame
    sample_rate: int
    channels: int
    source_id: str        # "{source_type}:{device_index_or_pid}"
    level_db: float | None
```

#### `DeviceInfo`

```python
@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    default_sample_rate: int
    is_loopback: bool
```

#### `AudioProcess`

```python
@dataclass(frozen=True)
class AudioProcess:
    pid: int
    name: str
    window_title: str
```

---

## Architecture

```
[Source]                  [Processors]              [Fan-out Sinks]
MicrophoneSource    ─┐
SystemAudioSource   ─┼─►  ProcessorChain  ─────────►  FileSink
ProcessAudioSource  ─┘     (LevelMeter)              ├─► CallbackSink
                                │                    ├─► LevelMeterSink  (pause_exempt)
                                └─► Analysis Chain   └─► [custom sinks…]
                                     (vad_enabled)
                                      Resample → 16 kHz
                                      ChannelConvert → Mono
                                      SileroVAD
                                       ├─► SpeechStartEvent
                                       └─► SpeechEndEvent
```

**Key design points**

- **Internal format** — All audio is carried as `float32` 2-D arrays `(frames, channels)` throughout the pipeline.
- **One device = one stream** — Each source opens exactly one OS-level audio stream.
- **Pipeline thread** — A dedicated non-daemon thread drains the chunk queue and dispatches to sinks, keeping the capture callback free.
- **Partial failure isolation** — Each sink is wrapped in a try/except; one failing sink does not interrupt the others.

---

## CLI

List available audio devices:

```bash
pyflexaudio devices
```

Check platform capabilities and optional dependency status:

```bash
pyflexaudio check
```

---

## License

[MIT](LICENSE)
