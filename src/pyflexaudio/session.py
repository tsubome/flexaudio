"""pyflexaudio のメインエントリポイント — FlexAudioSession"""

from __future__ import annotations

import enum
import logging
import queue
import threading

from pyflexaudio.types import (
    AudioChunk,
    ErrorEvent,
    FlexAudioError,
    QueuePolicy,
    SourceSwitchedEvent,
    SourceType,
    StateChangedEvent,
)
from pyflexaudio.events import EventBus
from pyflexaudio.pipeline import Pipeline
from pyflexaudio.processors.chain import ProcessorChain
from pyflexaudio.processors.level import LevelMeterProcessor
from pyflexaudio.sinks.level_meter import LevelMeterSink

__all__ = ["FlexAudioSession"]

logger = logging.getLogger("pyflexaudio.session")


class _SessionState(enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"


class FlexAudioSession:
    """pyflexaudio のメインエントリポイント"""

    def __init__(
        self,
        vad_enabled: bool = False,
        vad_sample_rate: int = 16000,
        vad_channels: int = 1,
        source_timeout_sec: float = 10.0,
        queue_policy: QueuePolicy = QueuePolicy.DROP_OLDEST,
        queue_size: int = 200,
    ):
        self._vad_enabled = vad_enabled
        self._vad_sample_rate = vad_sample_rate
        self._vad_channels = vad_channels
        self._source_timeout_sec = source_timeout_sec
        self._queue_policy = queue_policy
        self._queue_size = queue_size

        # 状態管理
        self._state = _SessionState.STOPPED
        self._state_lock = threading.Lock()

        # イベント
        self._event_bus = EventBus()

        # Pipeline
        self._pipeline = Pipeline(self._event_bus)
        self._data_queue: queue.Queue[AudioChunk | None] = queue.Queue(maxsize=queue_size)

        # 共通プロセッサチェーン（LevelMeterProcessor は常時 ON）
        main_chain = ProcessorChain([LevelMeterProcessor()])
        self._pipeline.set_main_chain(main_chain)

        # LevelMeterSink（常時 ON, pause 中も動作）
        self._level_meter_sink = LevelMeterSink(self._event_bus)
        self._pipeline.add_sink(self._level_meter_sink)

        # Source 管理
        self._source = None
        self._source_type: SourceType | None = None
        self._source_config: dict | None = None  # set_source の引数を保存
        self._stop_event = threading.Event()

        # 解析チェーン（VAD用、start時に構築）
        self._analysis_chain: ProcessorChain | None = None

    # ---- Source 管理 ----

    def set_source(
        self,
        source_type: SourceType,
        *,
        device_index: int | None = None,
        pid: int | None = None,
        mode: str = "include",
    ) -> None:
        """Source を設定/切り替え"""
        self._source_config = {
            "source_type": source_type,
            "device_index": device_index,
            "pid": pid,
            "mode": mode,
        }
        self._source_type = source_type

        # 実行中の場合はスムーズ切り替え
        if self._state in (_SessionState.RUNNING, _SessionState.PAUSED):
            self._switch_source_live(source_type, device_index=device_index, pid=pid, mode=mode)

    def _switch_source_live(self, source_type: SourceType, **kwargs) -> None:
        """実行中の Source 切り替え"""
        old_source = self._source
        try:
            new_source = self._create_source(source_type, **kwargs)
            new_source.open(self._data_queue, self._stop_event)
        except Exception as e:
            logger.error("Source switch failed: %s", e)
            self._event_bus.emit(ErrorEvent(
                error=FlexAudioError(code="SOURCE_OPEN_FAILED", message=str(e)),
                source_id=f"{source_type.value}:unknown",
            ))
            return  # 旧 Source を維持

        self._source = new_source
        # Pipeline Thread に切替指令を送信
        self._pipeline.send_command(("switch_source", new_source, old_source))

        old_id = self._get_source_id(old_source) if old_source else "none"
        new_id = self._get_source_id(new_source)
        self._event_bus.emit(SourceSwitchedEvent(old_source_id=old_id, new_source_id=new_id))

    def _create_source(self, source_type: SourceType, **kwargs):
        """Source インスタンスを作成"""
        from pyflexaudio.sources.mic import MicrophoneSource
        from pyflexaudio.sources.system import SystemAudioSource
        from pyflexaudio.sources.process import ProcessAudioSource

        if source_type == SourceType.MICROPHONE:
            return MicrophoneSource(
                device_index=kwargs.get("device_index"),
            )
        elif source_type == SourceType.SYSTEM_AUDIO:
            return SystemAudioSource(
                device_index=kwargs.get("device_index"),
            )
        elif source_type == SourceType.PROCESS_AUDIO:
            pid = kwargs.get("pid")
            if pid is None:
                raise ValueError("pid is required for PROCESS_AUDIO source")
            return ProcessAudioSource(
                pid=pid,
                mode=kwargs.get("mode", "include"),
            )
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    def _get_source_id(self, source) -> str:
        """Source から source_id 文字列を生成"""
        if hasattr(source, "source_id"):
            return source.source_id
        return "unknown"

    # ---- Sink 管理 ----

    def add_sink(self, sink) -> str:
        """Sink を追加。sink_id を返す"""
        return self._pipeline.add_sink(sink)

    def remove_sink(self, sink_id: str) -> None:
        """Sink を除去"""
        self._pipeline.remove_sink(sink_id)

    def enable_sink(self, sink_id: str) -> None:
        """Sink を有効化"""
        sink = self._pipeline.get_sink(sink_id)
        if sink is not None:
            sink.enabled = True

    def disable_sink(self, sink_id: str) -> None:
        """Sink を無効化"""
        sink = self._pipeline.get_sink(sink_id)
        if sink is not None:
            sink.enabled = False

    # ---- 制御（冪等性保証） ----

    def start(self) -> None:
        """セッションを開始"""
        with self._state_lock:
            if self._state != _SessionState.STOPPED:
                return  # 冪等: 既に開始済み
            old_state = self._state
            self._state = _SessionState.STARTING

        self._event_bus.emit(StateChangedEvent(old_state=old_state.value, new_state=self._state.value))

        try:
            # stop_event をリセット（再起動可能にする）
            self._stop_event.clear()

            # VAD 解析チェーン構築（vad_enabled 時）
            if self._vad_enabled:
                self._build_analysis_chain()

            # Pipeline Thread 開始
            self._pipeline.start(self._data_queue)

            # Source 開始（設定済みの場合）
            if self._source_config is not None:
                source_type = self._source_config["source_type"]
                try:
                    self._source = self._create_source(
                        source_type,
                        device_index=self._source_config.get("device_index"),
                        pid=self._source_config.get("pid"),
                        mode=self._source_config.get("mode", "include"),
                    )
                    self._source.open(self._data_queue, self._stop_event)
                except Exception as e:
                    logger.error("Source open failed: %s", e)
                    self._event_bus.emit(ErrorEvent(
                        error=FlexAudioError(code="SOURCE_OPEN_FAILED", message=str(e)),
                        source_id=f"{source_type.value}:unknown",
                    ))

            with self._state_lock:
                self._state = _SessionState.RUNNING
            self._event_bus.emit(StateChangedEvent(
                old_state=_SessionState.STARTING.value,
                new_state=self._state.value,
            ))

        except Exception:
            logger.exception("Session start failed")
            with self._state_lock:
                self._state = _SessionState.STOPPED
            raise

    def stop(self) -> None:
        """セッションを停止"""
        with self._state_lock:
            if self._state == _SessionState.STOPPED:
                return  # 冪等
            if self._state == _SessionState.STOPPING:
                return
            old_state = self._state
            self._state = _SessionState.STOPPING

        self._event_bus.emit(StateChangedEvent(old_state=old_state.value, new_state=self._state.value))

        # シャットダウン順序:
        # 1. stop_event をセット（Source のコールバックが data_queue への push を停止）
        self._stop_event.set()

        # 2. Source close
        if self._source is not None:
            try:
                self._source.close()
            except Exception:
                logger.exception("Source close error")
            self._source = None

        # 3. Pipeline stop（センチネル送信 → Thread join → Sink flush/close）
        self._pipeline.stop()

        # 4. data_queue を drain
        while True:
            try:
                self._data_queue.get_nowait()
            except queue.Empty:
                break

        # 5. EventBus クリア
        self._event_bus.clear()

        with self._state_lock:
            self._state = _SessionState.STOPPED
        # EventBus は clear 済みなので StateChangedEvent は emit しない

    def pause(self) -> None:
        """Sink 配信を一時停止"""
        with self._state_lock:
            if self._state != _SessionState.RUNNING:
                return
            old_state = self._state
            self._state = _SessionState.PAUSED

        self._pipeline.pause()
        self._event_bus.emit(StateChangedEvent(old_state=old_state.value, new_state=self._state.value))

    def resume(self) -> None:
        """Sink 配信を再開"""
        with self._state_lock:
            if self._state != _SessionState.PAUSED:
                return
            old_state = self._state
            self._state = _SessionState.RUNNING

        self._pipeline.resume()
        self._event_bus.emit(StateChangedEvent(old_state=old_state.value, new_state=self._state.value))

    # ---- VAD 解析チェーン ----

    def _build_analysis_chain(self) -> None:
        """VAD 用の解析サブチェーンを構築"""
        from pyflexaudio.processors.resample import ResampleProcessor
        from pyflexaudio.processors.channels import ChannelConvertProcessor
        from pyflexaudio.processors.vad import SileroVADProcessor

        processors = [
            ResampleProcessor(self._vad_sample_rate),
            ChannelConvertProcessor(self._vad_channels),
            SileroVADProcessor(self._event_bus),
        ]
        self._analysis_chain = ProcessorChain(processors)
        self._pipeline.set_analysis_chain(self._analysis_chain)

    # ---- イベント ----

    def on(self, event_type: type, handler) -> None:
        """イベントハンドラを登録"""
        self._event_bus.on(event_type, handler)

    def off(self, event_type: type, handler) -> None:
        """イベントハンドラを解除"""
        self._event_bus.off(event_type, handler)

    # ---- プロパティ ----

    @property
    def is_running(self) -> bool:
        return self._state in (_SessionState.RUNNING, _SessionState.PAUSED)

    @property
    def is_paused(self) -> bool:
        return self._state == _SessionState.PAUSED

    @property
    def level_db(self) -> float | None:
        """最新のレベル（LevelMeterSink が emit した値）"""
        # LevelEvent のハンドラで更新される最新値を返す設計もあるが、
        # シンプルにイベント経由で取得する方式を推奨
        return None

    @property
    def current_source_type(self) -> SourceType | None:
        return self._source_type

    # ---- コンテキストマネージャ ----

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
