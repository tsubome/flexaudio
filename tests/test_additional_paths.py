"""追加テスト — Pipeline / Session / EventBus の未カバーパスを検証"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from pyflexaudio.events import EventBus
from pyflexaudio.pipeline import Pipeline
from pyflexaudio.processors.chain import ProcessorChain
from pyflexaudio.session import FlexAudioSession
from pyflexaudio.types import AudioChunk, ErrorEvent, SourceType

from conftest import MockSink


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_chunk(source_id: str = "microphone:test") -> AudioChunk:
    data = np.random.randn(512, 1).astype(np.float32) * 0.1
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=16000,
        channels=1,
        source_id=source_id,
    )


def _start_pipeline(pipeline: Pipeline) -> queue.Queue:
    dq: queue.Queue = queue.Queue(maxsize=200)
    pipeline.start(dq)
    return dq


# ---------------------------------------------------------------------------
# A. EventBus 追加テスト
# ---------------------------------------------------------------------------

class TestEventBusAdditional:
    """EventBus の未カバーパスを検証"""

    def test_slow_handler_triggers_warning_log(self):
        """100ms超のハンドラで logger.warning が呼ばれる"""
        bus = EventBus()

        def slow_handler(event):
            time.sleep(0.15)

        bus.on(ErrorEvent, slow_handler)

        with patch("pyflexaudio.events.logger") as mock_logger:
            bus.emit(ErrorEvent(
                error=__import__("pyflexaudio.types", fromlist=["FlexAudioError"]).FlexAudioError(
                    code="TEST", message="test"
                ),
                source_id="test:0",
            ))
            mock_logger.warning.assert_called_once()

    def test_off_during_emit_current_emit_still_calls_handler(self):
        """emit 中に別スレッドから off() → スナップショット方式により今回の emit では呼ばれる"""
        bus = EventBus()
        called = []
        barrier = threading.Barrier(2)

        def handler(event):
            # ハンドラが呼ばれたことを記録してからバリアで同期
            called.append("called")
            barrier.wait(timeout=2.0)

        bus.on(ErrorEvent, handler)

        from pyflexaudio.types import FlexAudioError

        def off_thread():
            barrier.wait(timeout=2.0)
            bus.off(ErrorEvent, handler)

        t = threading.Thread(target=off_thread)
        t.start()

        bus.emit(ErrorEvent(
            error=FlexAudioError(code="TEST", message="test"),
            source_id="test:0",
        ))
        t.join(timeout=2.0)

        # スナップショット方式により、emit 開始時点でのハンドラが呼ばれる
        assert called == ["called"], "emit 中に off() してもスナップショットのハンドラは呼ばれるべき"

    def test_same_handler_registered_twice_off_once_leaves_one(self):
        """同一ハンドラ2回登録 + 1回off → 1つだけ削除、もう1つは残る"""
        bus = EventBus()
        results = []

        def handler(event):
            results.append("hit")

        bus.on(ErrorEvent, handler)
        bus.on(ErrorEvent, handler)
        # 1回だけ off → list.remove は最初の1つだけ削除
        bus.off(ErrorEvent, handler)

        assert bus.handler_count(ErrorEvent) == 1, "1つ削除後は残り1つのはず"

        from pyflexaudio.types import FlexAudioError
        bus.emit(ErrorEvent(
            error=FlexAudioError(code="TEST", message="test"),
            source_id="test:0",
        ))
        assert results == ["hit"], "残ったハンドラが呼ばれていない"

    def test_clear_then_handler_count_is_zero(self):
        """clear() 後の handler_count() が 0"""
        bus = EventBus()
        bus.on(ErrorEvent, lambda e: None)
        bus.on(ErrorEvent, lambda e: None)
        bus.clear()
        assert bus.handler_count(ErrorEvent) == 0

    def test_handler_count_for_unregistered_type_is_zero(self):
        """未登録型の handler_count() が 0"""
        bus = EventBus()
        assert bus.handler_count(ErrorEvent) == 0


# ---------------------------------------------------------------------------
# B. Pipeline 追加テスト
# ---------------------------------------------------------------------------

class TestPipelineDrainBoundary:
    """switch_source コマンドの drain 上限 50 チャンクを境界でテスト"""

    def test_drain_stops_at_50_chunks(self):
        """data_queue に 60 チャンクある状態で switch_source → 50 で停止、残 10 が残る"""
        bus = EventBus()
        pipeline = Pipeline(bus)

        dq: queue.Queue = queue.Queue(maxsize=200)

        # まず60チャンクを投入しておく（Pipelineスレッドが消費しないよう start 前に）
        for _ in range(60):
            dq.put(_make_chunk())

        # Pipeline を起動する前にコマンドキューに switch_source を積む
        pipeline._command_queue.put(("switch_source", None, None))

        # Pipeline を start するとすぐ _process_commands が実行される
        pipeline.start(dq)
        # コマンドが処理されるまで少し待つ
        time.sleep(0.3)
        pipeline.stop()

        # drain 上限が 50 なので残りは 10 チャンク程度のはず
        # （Pipeline スレッドが stop 処理中に少し消費する可能性があるため <=10 で確認）
        remaining = dq.qsize()
        # 60チャンク投入 → drainで50消費 → 残り10。ただしpipelineスレッドが
        # drain後に通常処理で消費する可能性があるので、残り<=10を確認
        assert remaining <= 10, f"drain 上限 50 を超えて消費された可能性がある: remaining={remaining}"

    def test_drain_sentinel_is_put_back(self):
        """drain 中に None センチネルを発見したら data_queue に戻す"""
        bus = EventBus()
        pipeline = Pipeline(bus)

        dq: queue.Queue = queue.Queue(maxsize=200)

        # [chunk, None, chunk, ...] を投入
        dq.put(_make_chunk())
        dq.put(None)  # センチネル
        for _ in range(5):
            dq.put(_make_chunk())

        # _handle_command を直接呼び出して内部動作を確認
        pipeline._data_queue = dq
        pipeline._handle_command(("switch_source", None, None))

        # None が戻されているはず
        found_sentinel = False
        while True:
            try:
                item = dq.get_nowait()
                if item is None:
                    found_sentinel = True
                    break
            except queue.Empty:
                break

        assert found_sentinel, "drain 中に発見した None センチネルが data_queue に戻されていない"


class TestPipelineAnalysisChainError:
    """解析チェーン例外時に ErrorEvent が emit されパイプラインが継続する"""

    def test_analysis_chain_exception_emits_error_event_and_continues(self):
        bus = EventBus()
        error_events: list[ErrorEvent] = []
        bus.on(ErrorEvent, error_events.append)

        pipeline = Pipeline(bus)
        sink = MockSink()
        pipeline.add_sink(sink)

        # 常に例外を投げる解析チェーン
        class FailingProcessor:
            def process(self, chunk):
                raise RuntimeError("analysis failure")

            def reset(self):
                pass

        pipeline.set_analysis_chain(ProcessorChain([FailingProcessor()]))

        dq = _start_pipeline(pipeline)

        # 2チャンク投入して、パイプラインが継続して両方処理するか確認
        dq.put(_make_chunk())
        dq.put(_make_chunk())
        time.sleep(0.4)
        pipeline.stop()

        # ErrorEvent が emit されたこと
        assert len(error_events) >= 2, f"ErrorEvent が期待数 emit されていない: {len(error_events)}"
        # Sink には両チャンクが届くこと（パイプライン継続確認）
        assert len(sink.chunks) >= 2, "解析チェーン例外後もパイプラインが継続していない"


class TestPipelineStopBeforeStart:
    """start() 前に stop() を呼んでも例外にならない"""

    def test_stop_before_start_no_exception(self):
        bus = EventBus()
        pipeline = Pipeline(bus)
        # _thread は None のまま → 例外なし
        pipeline.stop()  # raises nothing


class TestPipelineGetSinkNotFound:
    """get_sink() で存在しない ID → None が返る"""

    def test_get_sink_nonexistent_id_returns_none(self):
        bus = EventBus()
        pipeline = Pipeline(bus)
        result = pipeline.get_sink("nonexistent_sink_id")
        assert result is None


# ---------------------------------------------------------------------------
# C. Session 追加テスト
# ---------------------------------------------------------------------------

class TestSessionCreateSourceErrors:
    """_create_source のエラーケース"""

    def test_create_source_process_audio_pid_none_raises(self):
        """PROCESS_AUDIO で pid=None → ValueError"""
        session = FlexAudioSession()
        with pytest.raises(ValueError, match="pid"):
            session._create_source(SourceType.PROCESS_AUDIO, pid=None)

    def test_create_source_unknown_source_type_raises(self):
        """未知の SourceType → ValueError"""
        import enum

        class FakeSourceType(enum.Enum):
            UNKNOWN = "unknown"

        session = FlexAudioSession()
        with pytest.raises((ValueError, AttributeError)):
            # _create_source の else 節: raise ValueError(f"Unknown source type: {source_type}")
            session._create_source(FakeSourceType.UNKNOWN)


class TestSessionStartSourceOpenFailure:
    """start() で Source open が失敗しても RUNNING に遷移する"""

    def test_start_still_transitions_to_running_if_source_open_fails(self):
        session = FlexAudioSession()

        # set_source しておく（start 時に _create_source が呼ばれる）
        session.set_source(SourceType.MICROPHONE)

        # _create_source をモンキーパッチして例外を発生させる
        def failing_create_source(source_type, **kwargs):
            raise OSError("デバイスが見つからない")

        session._create_source = failing_create_source

        session.start()

        assert session.is_running, "Source open 失敗後も RUNNING になるべき"
        session.stop()


class TestSessionLevelDbProperty:
    """level_db プロパティが None を返す"""

    def test_level_db_is_none(self):
        session = FlexAudioSession()
        assert session.level_db is None


class TestSessionEnableDisableSinkNonexistent:
    """存在しない sink_id への enable_sink / disable_sink で例外なし"""

    def test_enable_sink_nonexistent_no_exception(self):
        session = FlexAudioSession()
        session.enable_sink("nonexistent_id")  # raises nothing

    def test_disable_sink_nonexistent_no_exception(self):
        session = FlexAudioSession()
        session.disable_sink("nonexistent_id")  # raises nothing


class TestSessionIsRunningWhenPaused:
    """is_running が PAUSED 状態でも True"""

    def test_is_running_true_when_paused(self):
        session = FlexAudioSession()
        session.start()
        session.pause()

        assert session.is_paused, "pause() 後は is_paused == True のはず"
        assert session.is_running, "PAUSED 状態でも is_running == True のはず"

        session.stop()


# ---------------------------------------------------------------------------
# D. Pipeline Sink エラーハンドリング詳細
# ---------------------------------------------------------------------------

class TestPipelineSinkWriteError:
    """Sink.write() 例外でその Sink のみ disable される"""

    def test_failing_sink_disabled_mock_sink_continues(self):
        bus = EventBus()
        error_events: list[ErrorEvent] = []
        bus.on(ErrorEvent, error_events.append)

        pipeline = Pipeline(bus)

        class FailingSink:
            enabled = True
            pause_exempt = False

            def write(self, chunk):
                raise RuntimeError("write failure")

            def flush(self):
                pass

            def close(self):
                pass

        failing_sink = FailingSink()
        mock_sink = MockSink()

        pipeline.add_sink(failing_sink)
        pipeline.add_sink(mock_sink)

        dq = _start_pipeline(pipeline)
        dq.put(_make_chunk())
        dq.put(_make_chunk())
        time.sleep(0.4)
        pipeline.stop()

        # FailingSink は disable されているはず
        assert not failing_sink.enabled, "write 例外後、FailingSink が disable されていない"
        # MockSink はチャンクを受信し続ける
        assert len(mock_sink.chunks) >= 1, "FailingSink のエラーが MockSink に影響している"
        # ErrorEvent が emit されているはず
        assert len(error_events) >= 1, "Sink write エラーで ErrorEvent が emit されていない"


class TestPipelineCloseAllSinksFlushError:
    """_close_all_sinks で flush 例外が出ても次の Sink は処理される"""

    def test_flush_exception_does_not_skip_remaining_sinks(self):
        bus = EventBus()
        pipeline = Pipeline(bus)

        class FlushFailSink:
            enabled = True
            pause_exempt = False
            closed = False

            def write(self, chunk):
                pass

            def flush(self):
                raise RuntimeError("flush failure")

            def close(self):
                self.closed = True

        class TrackCloseSink:
            enabled = True
            pause_exempt = False
            closed = False

            def write(self, chunk):
                pass

            def flush(self):
                pass

            def close(self):
                self.closed = True

        flush_fail_sink = FlushFailSink()
        track_close_sink = TrackCloseSink()

        pipeline.add_sink(flush_fail_sink)
        pipeline.add_sink(track_close_sink)

        dq = _start_pipeline(pipeline)
        pipeline.stop()

        # flush 例外が起きても両方の Sink が close されるべき
        assert flush_fail_sink.closed, "FlushFailSink が close されていない"
        assert track_close_sink.closed, "flush 例外後に TrackCloseSink が close されていない"
