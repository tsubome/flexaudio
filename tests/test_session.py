"""FlexAudioSession のユニットテスト"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyflexaudio.events import EventBus
from pyflexaudio.session import FlexAudioSession
from pyflexaudio.types import (
    AudioChunk,
    SourceType,
    StateChangedEvent,
)

# conftest の MockSource / MockSink をインポート
from conftest import MockSink, MockSource


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_session(**kwargs) -> FlexAudioSession:
    """デフォルト設定の FlexAudioSession を生成"""
    return FlexAudioSession(**kwargs)


def _mock_source_factory(source_id: str = "microphone:test") -> MockSource:
    """テスト用 MockSource を返すファクトリ"""
    return MockSource(source_id=source_id)


# ---------------------------------------------------------------------------
# 冪等性テスト
# ---------------------------------------------------------------------------

class TestIdempotency:
    """start() / stop() の冪等性"""

    def test_start_twice_no_error(self):
        """start() を 2 回呼んでもエラーにならない"""
        session = _make_session()
        # _create_source がないので start() しても Source open は失敗するが
        # Pipeline 自体は動く
        session.start()
        session.start()  # 2 回目は冪等
        session.stop()

    def test_stop_twice_no_error(self):
        """stop() を 2 回呼んでもエラーにならない"""
        session = _make_session()
        session.start()
        session.stop()
        session.stop()  # 2 回目は冪等

    def test_stop_before_start_no_error(self):
        """start() する前に stop() を呼んでもエラーにならない"""
        session = _make_session()
        session.stop()


# ---------------------------------------------------------------------------
# コンテキストマネージャテスト
# ---------------------------------------------------------------------------

class TestContextManager:
    """with 文での使用"""

    def test_context_manager_calls_stop_on_exit(self):
        """__exit__ で stop() が呼ばれる"""
        session = _make_session()
        session.start()
        assert session.is_running

        with session:
            pass  # __exit__ が呼ばれる

        assert not session.is_running

    def test_context_manager_stop_on_exception(self):
        """例外が発生しても __exit__ で stop() が呼ばれる"""
        session = _make_session()
        session.start()

        try:
            with session:
                raise ValueError("テスト例外")
        except ValueError:
            pass

        assert not session.is_running

    def test_context_manager_enter_returns_session(self):
        """__enter__ がセッション自身を返す"""
        session = _make_session()
        with session as s:
            assert s is session
        # stop() 呼び出し後なので is_running は False


# ---------------------------------------------------------------------------
# 状態プロパティテスト
# ---------------------------------------------------------------------------

class TestStateProperties:
    """is_running, is_paused, current_source_type"""

    def test_is_running_false_before_start(self):
        session = _make_session()
        assert not session.is_running

    def test_is_running_true_after_start(self):
        session = _make_session()
        session.start()
        assert session.is_running
        session.stop()

    def test_is_running_false_after_stop(self):
        session = _make_session()
        session.start()
        session.stop()
        assert not session.is_running

    def test_is_paused_false_before_pause(self):
        session = _make_session()
        session.start()
        assert not session.is_paused
        session.stop()

    def test_is_paused_true_after_pause(self):
        session = _make_session()
        session.start()
        session.pause()
        assert session.is_paused
        session.stop()

    def test_is_paused_false_after_resume(self):
        session = _make_session()
        session.start()
        session.pause()
        session.resume()
        assert not session.is_paused
        session.stop()

    def test_current_source_type_none_initially(self):
        session = _make_session()
        assert session.current_source_type is None

    def test_current_source_type_set_after_set_source(self):
        session = _make_session()
        session.set_source(SourceType.MICROPHONE)
        assert session.current_source_type == SourceType.MICROPHONE


# ---------------------------------------------------------------------------
# イベント委譲テスト
# ---------------------------------------------------------------------------

class TestEventDelegation:
    """on / off が EventBus に委譲される"""

    def test_on_registers_handler(self):
        session = _make_session()
        received: list[StateChangedEvent] = []
        session.on(StateChangedEvent, received.append)

        session.start()
        assert len(received) > 0, "StateChangedEvent が emit されていない"

        session.stop()

    def test_off_unregisters_handler(self):
        session = _make_session()
        received: list[StateChangedEvent] = []

        def handler(event):
            received.append(event)

        session.on(StateChangedEvent, handler)
        session.off(StateChangedEvent, handler)

        session.start()
        # off 後なのでハンドラは呼ばれない
        assert len(received) == 0
        session.stop()


# ---------------------------------------------------------------------------
# StateChangedEvent テスト
# ---------------------------------------------------------------------------

class TestStateChangedEvent:
    """start/stop/pause/resume で StateChangedEvent が emit される"""

    def test_start_emits_state_changed(self):
        session = _make_session()
        events: list[StateChangedEvent] = []
        session.on(StateChangedEvent, events.append)

        session.start()
        session.stop()

        new_states = [e.new_state for e in events]
        assert "starting" in new_states, f"starting イベントがない: {new_states}"
        assert "running" in new_states, f"running イベントがない: {new_states}"

    def test_pause_emits_state_changed(self):
        session = _make_session()
        events: list[StateChangedEvent] = []
        session.on(StateChangedEvent, events.append)

        session.start()
        session.pause()
        session.stop()

        new_states = [e.new_state for e in events]
        assert "paused" in new_states, f"paused イベントがない: {new_states}"

    def test_resume_emits_state_changed(self):
        session = _make_session()
        events: list[StateChangedEvent] = []
        session.on(StateChangedEvent, events.append)

        session.start()
        session.pause()
        session.resume()
        session.stop()

        new_states = [e.new_state for e in events]
        assert "running" in new_states, f"running イベントがない: {new_states}"

    def test_state_transition_sequence(self):
        """状態遷移の順序が正しい"""
        session = _make_session()
        events: list[StateChangedEvent] = []
        session.on(StateChangedEvent, events.append)

        session.start()
        session.pause()
        session.resume()
        session.stop()

        new_states = [e.new_state for e in events]
        # starting → running → paused → running → stopping の順
        assert new_states.index("starting") < new_states.index("running"), \
            "starting が running より前でない"
        assert new_states.index("running") < new_states.index("paused"), \
            "running が paused より前でない"


# ---------------------------------------------------------------------------
# set_source テスト
# ---------------------------------------------------------------------------

class TestSetSource:
    """set_source が _source_config を保存する"""

    def test_set_source_saves_config(self):
        session = _make_session()
        session.set_source(SourceType.MICROPHONE, device_index=2)

        assert session._source_config is not None
        assert session._source_config["source_type"] == SourceType.MICROPHONE
        assert session._source_config["device_index"] == 2

    def test_set_source_updates_current_source_type(self):
        session = _make_session()
        session.set_source(SourceType.SYSTEM_AUDIO)
        assert session.current_source_type == SourceType.SYSTEM_AUDIO

    def test_set_source_changes_type(self):
        session = _make_session()
        session.set_source(SourceType.MICROPHONE)
        session.set_source(SourceType.SYSTEM_AUDIO)
        assert session.current_source_type == SourceType.SYSTEM_AUDIO


# ---------------------------------------------------------------------------
# pause/resume 状態遷移テスト
# ---------------------------------------------------------------------------

class TestPauseResume:
    """pause/resume が正しく状態遷移する"""

    def test_pause_only_from_running(self):
        """RUNNING 以外では pause() は無効"""
        session = _make_session()
        # STOPPED 状態で pause() しても状態は変わらない
        session.pause()
        assert not session.is_paused

    def test_resume_only_from_paused(self):
        """PAUSED 以外では resume() は無効"""
        session = _make_session()
        session.start()
        # RUNNING 状態で resume() しても is_paused は変わらない
        session.resume()
        assert not session.is_paused
        session.stop()

    def test_pause_resume_cycle(self):
        """pause/resume を複数回繰り返しても正しく動作"""
        session = _make_session()
        session.start()

        for _ in range(3):
            session.pause()
            assert session.is_paused
            session.resume()
            assert not session.is_paused

        session.stop()


# ---------------------------------------------------------------------------
# Sink 管理テスト
# ---------------------------------------------------------------------------

class TestSinkManagement:
    """Session 経由の Sink 追加/除去/有効化/無効化"""

    def test_add_sink_returns_id(self):
        session = _make_session()
        sink = MockSink()
        sink_id = session.add_sink(sink)
        assert isinstance(sink_id, str)
        assert sink_id.startswith("sink_")

    def test_remove_sink(self):
        session = _make_session()
        sink = MockSink()
        sink_id = session.add_sink(sink)
        session.remove_sink(sink_id)
        # 除去後は get_sink で None が返る
        assert session._pipeline.get_sink(sink_id) is None

    def test_enable_disable_sink(self):
        session = _make_session()
        sink = MockSink(enabled=True)
        sink_id = session.add_sink(sink)

        session.disable_sink(sink_id)
        assert not sink.enabled

        session.enable_sink(sink_id)
        assert sink.enabled


# ---------------------------------------------------------------------------
# Pipeline との統合テスト（Source なし）
# ---------------------------------------------------------------------------

class TestPipelineIntegrationNoSource:
    """Source を使わずに Pipeline に直接チャンクを投入"""

    def test_pipeline_processes_chunk_without_source(self):
        """Source なしで Pipeline に直接チャンクを投入して処理できる"""
        session = _make_session()
        sink = MockSink()
        session.add_sink(sink)
        session.start()

        # 直接 data_queue にチャンクを投入
        data = np.random.randn(512, 1).astype(np.float32) * 0.1
        chunk = AudioChunk(
            data=data,
            timestamp=time.monotonic(),
            sample_rate=16000,
            channels=1,
            source_id="test:direct",
        )
        session._data_queue.put(chunk)
        time.sleep(0.3)
        session.stop()

        assert len(sink.chunks) > 0, "直接投入したチャンクが処理されていない"

    def test_level_meter_sink_always_receives(self):
        """LevelMeterSink は pause 中も chunk を受け取る"""
        from pyflexaudio.types import LevelEvent

        session = _make_session()
        level_events: list[LevelEvent] = []
        session.on(LevelEvent, level_events.append)
        session.start()
        session.pause()

        # pause 中でも LevelMeterSink は pause_exempt=True なのでレベルが来る
        data = np.ones((512, 1), dtype=np.float32) * 0.5
        chunk = AudioChunk(
            data=data,
            timestamp=time.monotonic(),
            sample_rate=16000,
            channels=1,
            source_id="test:direct",
        )
        session._data_queue.put(chunk)
        time.sleep(0.3)
        session.stop()

        assert len(level_events) > 0, "LevelEvent が emit されていない"
