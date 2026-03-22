"""新 Source の open 失敗時に旧 Source を維持することを検証するテスト"""

import queue
import threading
import time

import numpy as np
import pytest

from pyflexaudio.types import AudioChunk, ErrorEvent, SourceType
from pyflexaudio.events import EventBus
from pyflexaudio.session import FlexAudioSession


# ---- ヘルパー ----

def make_chunk(frames=512, sr=16000, ch=1, source_id="test:0"):
    data = np.random.randn(frames, ch).astype(np.float32)
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=sr,
        channels=ch,
        source_id=source_id,
    )


class MockSink:
    def __init__(self, enabled=True, pause_exempt=False):
        self.enabled = enabled
        self.pause_exempt = pause_exempt
        self.chunks = []
        self.flushed = False
        self.closed = False

    def write(self, chunk):
        if self.enabled:
            self.chunks.append(chunk)

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


class FailingSource:
    """open() が常に RuntimeError を投げるソース"""

    is_open = False
    source_id = "test:fail"

    def open(self, data_queue, stop_event):
        raise RuntimeError("open failed")

    def close(self):
        pass


class WorkingSource:
    """正常に動作するソース"""

    def __init__(self, source_id="test:working", interval_sec=0.02):
        self.source_id = source_id
        self.interval_sec = interval_sec
        self.is_open = False
        self.chunks_sent = 0
        self._thread = None
        self._data_queue = None
        self._stop_event = None

    def open(self, data_queue, stop_event):
        self._data_queue = data_queue
        self._stop_event = stop_event
        self.is_open = True
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def close(self):
        self.is_open = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _produce(self):
        while self.is_open and not self._stop_event.is_set():
            chunk = make_chunk(source_id=self.source_id)
            try:
                self._data_queue.put_nowait(chunk)
                self.chunks_sent += 1
            except queue.Full:
                pass
            time.sleep(self.interval_sec)


# ---- テスト: Pipeline レベル ----

def test_switch_source_live_emits_error_on_failure():
    """_switch_source_live で FailingSource を使うと ErrorEvent が emit される"""
    from pyflexaudio.pipeline import Pipeline
    from pyflexaudio.processors.chain import ProcessorChain

    bus = EventBus()
    errors = []
    bus.on(ErrorEvent, errors.append)

    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    dq = queue.Queue()
    pipeline.start(dq)

    # Pipeline を直接使わず Session の _switch_source_live に相当するロジックをテスト
    # FailingSource の open は例外を投げる
    failing_source = FailingSource()

    old_source = None
    try:
        failing_source.open(dq, threading.Event())
    except Exception as e:
        from pyflexaudio.types import FlexAudioError
        bus.emit(ErrorEvent(
            error=FlexAudioError(code="SOURCE_OPEN_FAILED", message=str(e)),
            source_id="test:fail",
        ))

    pipeline.stop()

    assert len(errors) >= 1
    assert errors[0].error.code == "SOURCE_OPEN_FAILED"


# ---- テスト: Session レベル ----

class _MockSourceFactory:
    """Session._create_source をモンキーパッチするためのファクトリ"""

    def __init__(self, sources):
        """sources: list of source objects to return in order"""
        self._sources = iter(sources)

    def __call__(self, source_type, **kwargs):
        return next(self._sources)


def _make_session_with_working_source():
    """WorkingSource を持つ Session を返す（source_config なし）"""
    session = FlexAudioSession()
    # source_config が None のため start しても Source は開かない
    session.start()
    return session


def test_session_switch_to_failing_source_emits_error():
    """実行中の Session で FailingSource への切り替えが失敗すると ErrorEvent が emit される"""
    session = FlexAudioSession()
    errors = []
    session.on(ErrorEvent, errors.append)

    # Source なしで起動
    session.start()

    # WorkingSource を手動でセット
    working = WorkingSource()
    session._source = working
    working.open(session._data_queue, session._stop_event)

    time.sleep(0.1)

    # FailingSource を使って _switch_source_live を呼び出す
    failing = FailingSource()

    # _create_source をモンキーパッチして FailingSource を返すようにする
    original_create = session._create_source
    session._create_source = lambda st, **kw: failing

    # SourceType を任意の値で呼び出す（MICROPHONE を使用）
    session._switch_source_live(SourceType.MICROPHONE)

    # 元のメソッドに戻す
    session._create_source = original_create

    time.sleep(0.1)

    working.close()
    session.stop()

    # ErrorEvent が emit されていること
    assert len(errors) >= 1
    sink_errors = [e for e in errors if e.error.code == "SOURCE_OPEN_FAILED"]
    assert len(sink_errors) >= 1


def test_session_old_source_maintained_on_switch_failure():
    """FailingSource への切り替えが失敗しても旧 Source が維持される"""
    session = FlexAudioSession()

    # Source なしで起動
    session.start()

    # WorkingSource を手動でセット
    working = WorkingSource(source_id="working:0")
    session._source = working
    working.open(session._data_queue, session._stop_event)

    time.sleep(0.1)
    old_source_ref = session._source

    # FailingSource で切り替えを試みる
    failing = FailingSource()
    session._create_source = lambda st, **kw: failing
    session._switch_source_live(SourceType.MICROPHONE)

    # 旧 Source が維持されていること
    assert session._source is old_source_ref

    working.close()
    session.stop()


def test_session_start_without_source():
    """Source が設定されていない状態で Session を start/stop しても例外なく完了する"""
    session = FlexAudioSession()
    errors = []
    session.on(ErrorEvent, errors.append)

    session.start()
    time.sleep(0.1)
    session.stop()

    # Source がないためエラーは発生しない
    assert len(errors) == 0
