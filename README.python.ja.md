**🇯🇵 日本語** | [🇺🇸 English](README.md)

# pyflexaudio

Python 向けの柔軟なクロスプラットフォーム音声キャプチャライブラリ。

マイク入力、システム音声キャプチャ、プロセス別音声キャプチャを、Windows・macOS・Linux 上で統一された API で扱えます。

```python
from pyflexaudio import FlexAudioSession, SourceType
from pyflexaudio.sinks.file import FileSink
from pyflexaudio.types import LevelEvent

with FlexAudioSession() as session:
    session.on(LevelEvent, lambda e: print(f"レベル: {e.db:.1f} dB"))
    session.add_sink(FileSink("output.wav"))
    session.set_source(SourceType.MICROPHONE)
    session.start()
    input("録音中… Enter で停止\n")
```

---

## 特長

### 音声ソース

| ソース | 説明 |
|--------|------|
| **マイク** | `sounddevice` (PortAudio) 経由、`miniaudio` フォールバック付き |
| **システム音声** | WASAPI Loopback (Windows)、ScreenCaptureKit (macOS) |
| **プロセス別音声** | ProcessAudioCapture DLL (Windows 11+)、ScreenCaptureKit フィルタ (macOS) |
| **ホットスワップ** | 実行中に `set_source()` を呼ぶことでソースをリアルタイム切り替え |

### 処理パイプライン

- **Source → Processor → Sink** fan-out アーキテクチャ
- **`LevelMeterProcessor`** — リアルタイム RMS → dB レベル計算（全チャンクに付与）
- **`ResampleProcessor`** — soxr による高品質ストリーミングリサンプリング
- **`ChannelConvertProcessor`** — ステレオ ↔ モノラル変換
- **Silero VAD** — 音声区間検出（`SpeechStartEvent` / `SpeechEndEvent` 付き）
- **`ProcessorChain`** — 複数プロセッサの連鎖実行（ネスト可能）

### 出力 Sink

| Sink | 説明 |
|------|------|
| **`FileSink`** | WAV int16 出力、30秒ごとのクラッシュセーフなヘッダ更新、自動ファイル分割 |
| **`CallbackSink`** | リアルタイムな `AudioChunk` をコールバック関数へ配信 |
| **`LevelMeterSink`** | 連続的な `LevelEvent` を発行（一時停止中も動作） |

### イベントシステム

型安全な `EventBus` と以下の組み込みイベント:

`LevelEvent` · `SpeechStartEvent` · `SpeechEndEvent` · `DeviceDisconnectedEvent` · `StateChangedEvent` · `ErrorEvent` · `ChunkDroppedEvent` · `SourceSwitchedEvent` · `PermissionDeniedEvent`

### 耐障害性

- **部分障害耐性** — 1つの Sink が故障しても他の Sink は動作継続
- **クラッシュセーフな WAV** — 30秒ごとに `fsync` でヘッダをディスクへ書き出し
- **自動ファイル分割** — ソースフォーマット変更時・WAV 4GB 制限到達時（サフィックス `_002.wav`、`_003.wav`、…）
- **デバイス切断の検知** — `DeviceDisconnectedEvent` を発行し、安全にシャットダウン
- **冪等な制御** — `start()` / `stop()` / `pause()` / `resume()` を何回呼んでも安全

---

## プラットフォーム対応

| 機能 | Windows | macOS | Linux |
|------|:-------:|:-----:|:-----:|
| マイクキャプチャ | ✓ | ✓ | ✓ |
| システム音声（ループバック） | ✓ (WASAPI) | ✓ (ScreenCaptureKit) | — |
| プロセス別音声 | ✓ (Win 11, DLL) | ✓ (ScreenCaptureKit) | — |
| マイクフォールバック (miniaudio) | ✓ | ✓ | ✓ |

---

## 動作要件

- Python ≥ 3.10
- [numpy](https://numpy.org/) ≥ 1.24
- [sounddevice](https://python-sounddevice.readthedocs.io/) ≥ 0.5
- [soxr](https://github.com/dofuuz/python-soxr) ≥ 0.5

---

## インストール

**基本（マイクのみ）**

```bash
pip install pyflexaudio
```

**VAD 付き（Silero 音声区間検出）**

```bash
pip install "pyflexaudio[vad]"
```

**macOS（システム / プロセス音声）**

```bash
pip install "pyflexaudio[mac]"
```

**Windows システム音声**

```bash
pip install "pyflexaudio[win-system]"
```

**Windows プロセス別音声**

```bash
pip install "pyflexaudio[win-process]"
```

**全部入り**

```bash
pip install "pyflexaudio[full]"
```

---

## クイックスタート

### 1. マイク録音

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

    time.sleep(10)  # 10秒間録音
# __exit__ 時に WAV ファイルが安全にクローズされる
```

### 2. 音声区間検出（VAD）

```python
from pyflexaudio import FlexAudioSession, SourceType
from pyflexaudio.types import SpeechStartEvent, SpeechEndEvent

with FlexAudioSession(vad_enabled=True) as session:
    session.on(SpeechStartEvent, lambda e: print("発話開始"))

    def on_speech_end(e: SpeechEndEvent):
        print(f"発話終了 — {e.duration_sec:.2f} 秒、{len(e.audio_data)} フレーム")

    session.on(SpeechEndEvent, on_speech_end)
    session.set_source(SourceType.MICROPHONE)
    session.start()

    input("発話を検出中… Enter で停止\n")
```

### 3. システム音声キャプチャ

```python
from pyflexaudio import FlexAudioSession, SourceType
from pyflexaudio.sinks.file import FileSink

with FlexAudioSession() as session:
    session.add_sink(FileSink("system_audio.wav"))
    session.set_source(SourceType.SYSTEM_AUDIO)
    session.start()

    input("システム音声をキャプチャ中… Enter で停止\n")
```

### 4. プロセス別音声キャプチャ

```python
from pyflexaudio import FlexAudioSession, SourceType
from pyflexaudio.sinks.file import FileSink
from pyflexaudio.sinks.callback import CallbackSink
from pyflexaudio.types import AudioChunk

TARGET_PID = 12345  # キャプチャ対象プロセスの PID に変更

def on_chunk(chunk: AudioChunk) -> None:
    print(f"{chunk.source_id} から {len(chunk.data)} フレーム受信")

with FlexAudioSession() as session:
    session.add_sink(FileSink("process_audio.wav"))
    session.add_sink(CallbackSink(on_chunk))
    session.set_source(SourceType.PROCESS_AUDIO, pid=TARGET_PID)
    session.start()

    input("プロセス音声をキャプチャ中… Enter で停止\n")
```

### 5. ソースのライブ切り替え

```python
import time
from pyflexaudio import FlexAudioSession, SourceType
from pyflexaudio.sinks.file import FileSink
from pyflexaudio.types import SourceSwitchedEvent

with FlexAudioSession() as session:
    session.on(
        SourceSwitchedEvent,
        lambda e: print(f"切替: {e.old_source_id} → {e.new_source_id}"),
    )
    session.add_sink(FileSink("mixed.wav"))
    session.set_source(SourceType.MICROPHONE)
    session.start()

    time.sleep(5)
    print("システム音声に切り替え中…")
    session.set_source(SourceType.SYSTEM_AUDIO)  # 実行中にホットスワップ

    time.sleep(5)
# 両セグメントは同一 WAV ファイルへ書き込まれる（フォーマット変更時は自動分割）
```

### 6. 一時停止 / 再開

```python
import time
from pyflexaudio import FlexAudioSession, SourceType
from pyflexaudio.sinks.file import FileSink
from pyflexaudio.types import LevelEvent

with FlexAudioSession() as session:
    # 一時停止中も LevelEvent は発行され続ける — UI の VU メーターに有用
    session.on(LevelEvent, lambda e: print(f"レベル: {e.db:+.1f} dB", end="\r"))

    session.add_sink(FileSink("output.wav"))
    session.set_source(SourceType.MICROPHONE)
    session.start()

    time.sleep(3)
    print("\n一時停止中…")
    session.pause()         # FileSink は書き込み停止、LevelMeterSink は動作継続

    time.sleep(2)
    print("再開中…")
    session.resume()        # FileSink が書き込みを再開

    time.sleep(3)
```

---

## API リファレンス

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

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `vad_enabled` | `bool` | `False` | Silero VAD 解析チェーンを有効化 |
| `vad_sample_rate` | `int` | `16000` | VAD 処理のターゲットサンプルレート |
| `vad_channels` | `int` | `1` | VAD 処理のターゲットチャンネル数 |
| `source_timeout_sec` | `float` | `10.0` | 最初の音声フレームを待つタイムアウト秒数 |
| `queue_policy` | `QueuePolicy` | `DROP_OLDEST` | オーバーフロー時の挙動（`DROP_OLDEST` または `BACKPRESSURE`） |
| `queue_size` | `int` | `200` | 内部音声チャンクキューの深さ |

**メソッド一覧**

| メソッド | シグネチャ | 説明 |
|---------|-----------|------|
| `set_source` | `(source_type, *, device_index=None, pid=None, mode="include") → None` | 音声ソースを設定またはホットスワップ |
| `add_sink` | `(sink) → str` | Sink を追加し、`sink_id` を返す |
| `remove_sink` | `(sink_id: str) → None` | ID を指定して Sink を除去 |
| `enable_sink` | `(sink_id: str) → None` | 無効化された Sink を有効化 |
| `disable_sink` | `(sink_id: str) → None` | Sink を除去せず無効化 |
| `start` | `() → None` | セッションを開始（冪等） |
| `stop` | `() → None` | セッションを停止し、全 Sink をフラッシュ（冪等） |
| `pause` | `() → None` | Sink への配信を一時停止（冪等） |
| `resume` | `() → None` | Sink への配信を再開（冪等） |
| `on` | `(event_type: type, handler) → None` | イベントハンドラを登録 |
| `off` | `(event_type: type, handler) → None` | イベントハンドラを解除 |

**プロパティ一覧**

| プロパティ | 型 | 説明 |
|-----------|------|------|
| `is_running` | `bool` | 状態が `RUNNING` または `PAUSED` なら `True` |
| `is_paused` | `bool` | 状態が `PAUSED` なら `True` |
| `current_source_type` | `SourceType \| None` | 現在設定されているソースタイプ |
| `level_db` | `float \| None` | 最新のレベル（リアルタイム更新は `LevelEvent` ハンドラを使用） |

コンテキストマネージャプロトコルを実装 — `__exit__` 時に `stop()` が自動呼び出しされます。

---

### Sink

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

| パラメータ | 説明 |
|-----------|------|
| `path` | 出力ファイルパス（`.wav`） |
| `sample_rate` | ターゲットサンプルレート。`None` の場合はソースのレートを使用 |
| `channels` | ターゲットチャンネル数。`None` の場合はソースのチャンネル数を使用 |
| `enabled` | `False` にすると Sink を除去せず書き込みをスキップ |

- 出力フォーマット: WAV PCM int16
- WAV ヘッダは30秒ごとおよび `close()` 時に更新 — クラッシュセーフ
- ソースフォーマット変更時・WAV 4GB 制限到達時に自動ファイル分割（サフィックス `_002.wav`、`_003.wav`、…）

#### `CallbackSink`

```python
CallbackSink(
    callback: Callable[[AudioChunk], None],
    *,
    enabled: bool = True,
)
```

パイプラインスレッド上で `AudioChunk` ごとに `callback(chunk)` を同期呼び出しします。コールバックは高速に保ち、重い処理は別スレッドにオフロードしてください。

#### `LevelMeterSink`

```python
LevelMeterSink(event_bus: EventBus)
```

`FlexAudioSession` が内部で管理します。セッションの一時停止中も（`pause_exempt = True`）、チャンクごとに `LevelEvent` を発行します。このシンクを手動で追加する必要はありません。

---

### イベント一覧

| イベント | フィールド | 説明 |
|---------|-----------|------|
| `LevelEvent` | `db: float`、`source_id: str` | 各チャンクの dBFS RMS レベル |
| `SpeechStartEvent` | `timestamp: float`、`source_id: str` | VAD が発話開始を検出 |
| `SpeechEndEvent` | `timestamp: float`、`duration_sec: float`、`audio_data: ndarray`、`source_id: str` | VAD が発話終了を検出。float32 16kHz モノラルの生音声データを含む |
| `SourceSwitchedEvent` | `old_source_id: str`、`new_source_id: str` | ソースのホットスワップ完了 |
| `DeviceDisconnectedEvent` | `device_info: DeviceInfo` | キャプチャデバイスが取り外された |
| `StateChangedEvent` | `old_state: str`、`new_state: str` | セッション状態マシンの遷移 |
| `ErrorEvent` | `error: FlexAudioError`、`source_id: str` | ソースまたは Sink の非致命的エラー |
| `ChunkDroppedEvent` | `drop_count: int`、`queue_size: int`、`source_id: str` | キューオーバーフローによりチャンクが破棄された |
| `PermissionDeniedEvent` | `permission_type: str`、`platform: str`、`message: str` | OS 権限の拒否（例: マイクアクセス）— 主に macOS |

---

### データ型

#### `AudioChunk`

```python
@dataclass
class AudioChunk:
    data: numpy.ndarray   # float32、shape=(frames, channels)
    timestamp: float      # 最初のフレームの Unix タイムスタンプ
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

## アーキテクチャ

```
[ソース]                   [プロセッサ]               [fan-out Sink 群]
MicrophoneSource    ─┐
SystemAudioSource   ─┼─►  ProcessorChain  ─────────►  FileSink
ProcessAudioSource  ─┘     (LevelMeter)              ├─► CallbackSink
                                │                    ├─► LevelMeterSink  (pause_exempt)
                                └─► 解析チェーン      └─► [カスタム Sink…]
                                     (vad_enabled 時)
                                      リサンプリング → 16kHz
                                      チャンネル変換 → モノラル
                                      SileroVAD
                                       ├─► SpeechStartEvent
                                       └─► SpeechEndEvent
```

**設計上のポイント**

- **内部フォーマット** — パイプライン全体を通じて、音声は `float32` の 2D 配列 `(frames, channels)` として扱われます。
- **1デバイス = 1ストリーム** — 各ソースは OS レベルの音声ストリームを1つだけオープンします。
- **Pipeline Thread** — 専用の非デーモンスレッドがチャンクキューを処理し Sink へ配信するため、キャプチャコールバックが常に軽量に保たれます。
- **部分障害の分離** — 各 Sink は try/except でラップされており、1つの Sink が失敗しても他の Sink に影響しません。

---

## CLI

利用可能な音声デバイスの一覧:

```bash
pyflexaudio devices
```

プラットフォームの対応状況とオプション依存関係の確認:

```bash
pyflexaudio check
```

---

## ライセンス

[MIT](LICENSE)
