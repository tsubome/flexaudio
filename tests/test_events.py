import threading
import time
import pytest
from pyflexaudio.events import EventBus
from pyflexaudio.types import LevelEvent, SpeechStartEvent, StateChangedEvent


# --- ハンドラ登録・発火・解除 ---

def test_on_and_emit(event_bus):
    received = []
    event_bus.on(LevelEvent, received.append)
    ev = LevelEvent(db=-10.0, source_id="microphone:0")
    event_bus.emit(ev)
    assert received == [ev]


def test_emit_calls_handler_with_correct_event(event_bus):
    calls = []
    event_bus.on(LevelEvent, lambda e: calls.append(e.db))
    event_bus.emit(LevelEvent(db=-20.0, source_id="microphone:0"))
    assert calls == [-20.0]


def test_off_unregisters_handler(event_bus):
    received = []
    handler = received.append
    event_bus.on(LevelEvent, handler)
    event_bus.off(LevelEvent, handler)
    event_bus.emit(LevelEvent(db=-10.0, source_id="microphone:0"))
    assert received == []


def test_off_nonexistent_handler_no_error(event_bus):
    # off を呼んでも例外が出ないこと
    event_bus.off(LevelEvent, lambda e: None)


def test_off_unregistered_type_no_error(event_bus):
    event_bus.off(StateChangedEvent, lambda e: None)


# --- 複数ハンドラ ---

def test_multiple_handlers_same_event_type(event_bus):
    results = []
    event_bus.on(LevelEvent, lambda e: results.append("h1"))
    event_bus.on(LevelEvent, lambda e: results.append("h2"))
    event_bus.emit(LevelEvent(db=-5.0, source_id="microphone:0"))
    assert results == ["h1", "h2"]


def test_multiple_handlers_called_in_registration_order(event_bus):
    order = []
    for i in range(5):
        idx = i
        event_bus.on(LevelEvent, lambda e, n=idx: order.append(n))
    event_bus.emit(LevelEvent(db=0.0, source_id="microphone:0"))
    assert order == [0, 1, 2, 3, 4]


# --- 異なるイベント型の独立性 ---

def test_different_event_types_are_independent(event_bus):
    level_calls = []
    state_calls = []
    event_bus.on(LevelEvent, level_calls.append)
    event_bus.on(StateChangedEvent, state_calls.append)

    event_bus.emit(LevelEvent(db=-10.0, source_id="microphone:0"))
    assert len(level_calls) == 1
    assert len(state_calls) == 0

    event_bus.emit(StateChangedEvent(old_state="idle", new_state="recording"))
    assert len(level_calls) == 1
    assert len(state_calls) == 1


def test_emit_unregistered_type_no_error(event_bus):
    event_bus.emit(SpeechStartEvent(timestamp=1.0, source_id="microphone:0"))


# --- スナップショット方式 ---

def test_emit_snapshot_new_handler_during_emit_not_called(event_bus):
    """emit 中に on() で登録したハンドラは今回の emit では呼ばれない"""
    called_second = []

    def first_handler(e):
        event_bus.on(LevelEvent, called_second.append)

    event_bus.on(LevelEvent, first_handler)
    event_bus.emit(LevelEvent(db=0.0, source_id="microphone:0"))
    assert called_second == []

    # 次の emit では呼ばれる
    event_bus.emit(LevelEvent(db=0.0, source_id="microphone:0"))
    assert len(called_second) == 1


# --- ハンドラ例外の分離 ---

def test_handler_exception_does_not_stop_other_handlers(event_bus):
    results = []

    def bad_handler(e):
        raise RuntimeError("intentional error")

    event_bus.on(LevelEvent, bad_handler)
    event_bus.on(LevelEvent, lambda e: results.append("ok"))

    event_bus.emit(LevelEvent(db=0.0, source_id="microphone:0"))
    assert results == ["ok"]


def test_multiple_handlers_all_exceptions_handled(event_bus):
    results = []

    def make_raiser(n):
        def handler(e):
            raise ValueError(f"error {n}")
        return handler

    for i in range(3):
        event_bus.on(LevelEvent, make_raiser(i))
    event_bus.on(LevelEvent, lambda e: results.append("survived"))

    event_bus.emit(LevelEvent(db=0.0, source_id="microphone:0"))
    assert results == ["survived"]


# --- has_handlers / handler_count ---

def test_has_handlers_false_when_none_registered(event_bus):
    assert event_bus.has_handlers(LevelEvent) is False


def test_has_handlers_true_after_registration(event_bus):
    event_bus.on(LevelEvent, lambda e: None)
    assert event_bus.has_handlers(LevelEvent) is True


def test_has_handlers_false_after_off(event_bus):
    handler = lambda e: None
    event_bus.on(LevelEvent, handler)
    event_bus.off(LevelEvent, handler)
    assert event_bus.has_handlers(LevelEvent) is False


def test_handler_count_zero_initially(event_bus):
    assert event_bus.handler_count(LevelEvent) == 0


def test_handler_count_increments(event_bus):
    event_bus.on(LevelEvent, lambda e: None)
    event_bus.on(LevelEvent, lambda e: None)
    assert event_bus.handler_count(LevelEvent) == 2


def test_handler_count_decrements_after_off(event_bus):
    h1 = lambda e: None
    h2 = lambda e: None
    event_bus.on(LevelEvent, h1)
    event_bus.on(LevelEvent, h2)
    event_bus.off(LevelEvent, h1)
    assert event_bus.handler_count(LevelEvent) == 1


# --- clear ---

def test_clear_removes_all_handlers(event_bus):
    event_bus.on(LevelEvent, lambda e: None)
    event_bus.on(StateChangedEvent, lambda e: None)
    event_bus.clear()
    assert event_bus.has_handlers(LevelEvent) is False
    assert event_bus.has_handlers(StateChangedEvent) is False


def test_clear_then_emit_no_error(event_bus):
    event_bus.on(LevelEvent, lambda e: None)
    event_bus.clear()
    event_bus.emit(LevelEvent(db=0.0, source_id="microphone:0"))


def test_clear_then_reregister(event_bus):
    results = []
    event_bus.on(LevelEvent, lambda e: None)
    event_bus.clear()
    event_bus.on(LevelEvent, results.append)
    ev = LevelEvent(db=-5.0, source_id="microphone:0")
    event_bus.emit(ev)
    assert results == [ev]


# --- スレッド安全性 ---

def test_thread_safety_concurrent_on_and_emit():
    bus = EventBus()
    errors = []
    call_count = [0]
    lock = threading.Lock()

    def register_and_emit():
        try:
            ev = LevelEvent(db=0.0, source_id="microphone:0")
            bus.on(LevelEvent, lambda e: None)
            bus.emit(ev)
            with lock:
                call_count[0] += 1
        except Exception as e:
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=register_and_emit) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert errors == [], f"Errors in threads: {errors}"
    assert call_count[0] == 20


def test_thread_safety_concurrent_off():
    bus = EventBus()
    errors = []
    handlers = [lambda e: None for _ in range(50)]

    for h in handlers:
        bus.on(LevelEvent, h)

    def unregister(h):
        try:
            bus.off(LevelEvent, h)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=unregister, args=(h,)) for h in handlers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert errors == []
    assert bus.handler_count(LevelEvent) == 0


def test_thread_safety_emit_from_multiple_threads():
    bus = EventBus()
    results = []
    lock = threading.Lock()

    def handler(e):
        with lock:
            results.append(e.db)

    bus.on(LevelEvent, handler)

    def emit_events():
        for i in range(10):
            bus.emit(LevelEvent(db=float(-i), source_id="microphone:0"))

    threads = [threading.Thread(target=emit_events) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    # 5 threads x 10 events = 50 total
    assert len(results) == 50
