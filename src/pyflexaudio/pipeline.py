"""Pipeline Thread を管理し、Source→Processor→Sink のデータフローを実行するモジュール"""

from __future__ import annotations

import logging
import queue
import threading
import time

from pyflexaudio.types import AudioChunk, ErrorEvent, FlexAudioError
from pyflexaudio.events import EventBus
from pyflexaudio.processors.chain import ProcessorChain

__all__ = ["Pipeline"]

logger = logging.getLogger("pyflexaudio.pipeline")


class Pipeline:
    """Pipeline Thread を管理し、Source→Processor→Sink のデータフローを実行"""

    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._main_chain = ProcessorChain()      # 共通プロセッサチェーン
        self._analysis_chain: ProcessorChain | None = None  # 解析サブチェーン（VAD用）

        # Sink管理
        self._sinks: dict[str, object] = {}       # sink_id -> AudioSink
        self._sink_lock = threading.Lock()
        self._next_sink_id = 0

        # スレッド管理
        self._thread: threading.Thread | None = None
        self._data_queue: queue.Queue | None = None
        self._command_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._paused = threading.Event()  # set = not paused (通常動作), clear = paused
        self._paused.set()  # 初期状態: 非一時停止

    # ---- 構成 ----

    def set_main_chain(self, chain: ProcessorChain) -> None:
        """共通プロセッサチェーンを設定"""
        self._main_chain = chain

    def set_analysis_chain(self, chain: ProcessorChain | None) -> None:
        """解析サブチェーン（VAD用）を設定。None で無効化"""
        self._analysis_chain = chain

    def add_sink(self, sink) -> str:
        """Sink を追加。sink_id を返す"""
        with self._sink_lock:
            sink_id = f"sink_{self._next_sink_id}"
            self._next_sink_id += 1
            self._sinks[sink_id] = sink
        return sink_id

    def remove_sink(self, sink_id: str) -> None:
        """Sink を除去"""
        with self._sink_lock:
            self._sinks.pop(sink_id, None)

    def get_sink(self, sink_id: str):
        """Sink を取得"""
        with self._sink_lock:
            return self._sinks.get(sink_id)

    # ---- 制御 ----

    def start(self, data_queue: queue.Queue) -> None:
        """Pipeline Thread を開始"""
        self._data_queue = data_queue
        self._stop_event.clear()
        self._paused.set()
        self._thread = threading.Thread(
            target=self._run,
            name="pyflexaudio-pipeline",
            daemon=False,  # FileSink の close/flush を保証
        )
        self._thread.start()

    def stop(self) -> None:
        """Pipeline Thread を停止"""
        self._stop_event.set()
        # センチネル値を送信して Pipeline Thread を起こす
        if self._data_queue is not None:
            self._data_queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Pipeline thread did not stop within timeout")
            self._thread = None
        # 全 Sink を flush + close
        self._close_all_sinks()

    def pause(self) -> None:
        """Sink への配信を一時停止（pause_exempt は除外）"""
        self._paused.clear()

    def resume(self) -> None:
        """Sink への配信を再開"""
        self._paused.set()

    def send_command(self, command: tuple) -> None:
        """Pipeline Thread にコマンドを送信"""
        self._command_queue.put(command)

    # ---- Pipeline Thread メインループ ----

    def _run(self) -> None:
        """Pipeline Thread のメインループ"""
        logger.debug("Pipeline thread started")
        try:
            while not self._stop_event.is_set():
                # コマンド処理（ノンブロッキング）
                self._process_commands()

                # チャンク取得
                try:
                    chunk = self._data_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # センチネル値チェック
                if chunk is None:
                    break

                # チャンク処理
                self._process_chunk(chunk)
        except Exception:
            logger.exception("Pipeline thread unexpected error")
        finally:
            logger.debug("Pipeline thread stopped")

    def _process_commands(self) -> None:
        """command_queue のコマンドを処理（ノンブロッキング）"""
        while True:
            try:
                command = self._command_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_command(command)

    def _handle_command(self, command: tuple) -> None:
        """個別コマンドの処理"""
        cmd_type = command[0]
        if cmd_type == "switch_source":
            new_source, old_source = command[1], command[2]
            # 旧チャンクを drain（最大50チャンク）
            drained = 0
            while drained < 50:
                try:
                    item = self._data_queue.get_nowait()
                    if item is None:
                        self._data_queue.put(None)  # センチネルは戻す
                        break
                    drained += 1
                except queue.Empty:
                    break
            if drained > 0:
                logger.debug("Drained %d old source chunks", drained)

            # 旧 Source を別スレッドで非同期 close
            if old_source is not None:
                threading.Thread(
                    target=self._close_source_async,
                    args=(old_source,),
                    name="pyflexaudio-source-close",
                    daemon=True,
                ).start()

    def _close_source_async(self, source) -> None:
        """旧 Source を非同期で close"""
        try:
            source.close()
        except Exception:
            logger.exception("Error closing old source")

    def _process_chunk(self, chunk: AudioChunk) -> None:
        """チャンクを共通チェーン → fan-out で処理"""
        # 共通プロセッサチェーン実行
        try:
            chunk = self._main_chain.process(chunk)
        except Exception:
            logger.exception("Main processor chain error, skipping chunk")
            self._event_bus.emit(ErrorEvent(
                error=FlexAudioError(code="PROCESSOR_ERROR", message="Main chain processing failed"),
                source_id=chunk.source_id,
            ))
            return

        # fan-out: 全 Sink にチャンクを配信
        is_paused = not self._paused.is_set()
        with self._sink_lock:
            sinks_snapshot = list(self._sinks.items())

        for sink_id, sink in sinks_snapshot:
            # pause 中は pause_exempt の Sink のみ配信
            if is_paused and not sink.pause_exempt:
                continue
            if not sink.enabled:
                continue
            try:
                sink.write(chunk)
            except Exception:
                logger.exception("Sink %s write error, disabling", sink_id)
                sink.enabled = False
                self._event_bus.emit(ErrorEvent(
                    error=FlexAudioError(code="SINK_WRITE_ERROR", message=f"Sink {sink_id} write failed"),
                    source_id=chunk.source_id,
                ))

        # 解析サブチェーン実行（設定されている場合のみ）
        if self._analysis_chain is not None:
            try:
                self._analysis_chain.process(chunk)
            except Exception:
                logger.exception("Analysis chain error, skipping")
                self._event_bus.emit(ErrorEvent(
                    error=FlexAudioError(code="PROCESSOR_ERROR", message="Analysis chain processing failed"),
                    source_id=chunk.source_id,
                ))

    def _close_all_sinks(self) -> None:
        """全 Sink を flush → close"""
        with self._sink_lock:
            sinks = list(self._sinks.values())
        for sink in sinks:
            try:
                sink.flush()
            except Exception:
                logger.exception("Sink flush error")
            try:
                sink.close()
            except Exception:
                logger.exception("Sink close error")
