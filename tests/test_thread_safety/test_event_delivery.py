"""EventBus のスレッド安全性を検証するテスト"""

import threading
import time

import pytest

from pyflexaudio.types import ErrorEvent, FlexAudioError, StateChangedEvent
from pyflexaudio.events import EventBus


# ---- ヘルパー ----

def make_error_event(source_id="test:0"):
    return ErrorEvent(
        error=FlexAudioError(code="TEST_ERROR", message="test"),
        source_id=source_id,
    )


# ---- テスト ----

def test_concurrent_emit_and_register():
    """1つのスレッドで on/off、別スレッドで emit — 例外なく完了する"""
    bus = EventBus()
    results = []
    lock = threading.Lock()
    exceptions = []

    def handler(e):
        with lock:
            results.append(e)

    def emit_loop():
        try:
            for _ in range(100):
                bus.emit(make_error_event())
                time.sleep(0.001)
        except Exception as e:
            exceptions.append(e)

    def register_loop():
        try:
            for _ in range(50):
                bus.on(ErrorEvent, handler)
                time.sleep(0.001)
                bus.off(ErrorEvent, handler)
                time.sleep(0.001)
        except Exception as e:
            exceptions.append(e)

    t_emit = threading.Thread(target=emit_loop)
    t_register = threading.Thread(target=register_loop)

    t_emit.start()
    t_register.start()

    t_emit.join(timeout=5)
    t_register.join(timeout=5)

    assert exceptions == []
    assert not t_emit.is_alive()
    assert not t_register.is_alive()


def test_concurrent_multiple_handlers():
    """複数スレッドから同時にハンドラを登録して emit しても安全"""
    bus = EventBus()
    results = []
    lock = threading.Lock()
    exceptions = []
    handlers = []

    def make_handler(idx):
        def handler(e):
            with lock:
                results.append(idx)
        return handler

    def register_and_emit(idx):
        try:
            h = make_handler(idx)
            bus.on(ErrorEvent, h)
            with lock:
                handlers.append(h)
            bus.emit(make_error_event())
        except Exception as e:
            exceptions.append(e)

    threads = [threading.Thread(target=register_and_emit, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert exceptions == []


def test_concurrent_clear_and_emit():
    """clear と emit が同時に実行されてもクラッシュしない"""
    bus = EventBus()
    results = []
    exceptions = []

    def handler(e):
        results.append(e)

    bus.on(ErrorEvent, handler)

    def emit_loop():
        try:
            for _ in range(50):
                bus.emit(make_error_event())
                time.sleep(0.001)
        except Exception as e:
            exceptions.append(e)

    def clear_loop():
        try:
            for _ in range(10):
                bus.clear()
                bus.on(ErrorEvent, handler)
                time.sleep(0.005)
        except Exception as e:
            exceptions.append(e)

    t_emit = threading.Thread(target=emit_loop)
    t_clear = threading.Thread(target=clear_loop)

    t_emit.start()
    t_clear.start()

    t_emit.join(timeout=5)
    t_clear.join(timeout=5)

    assert exceptions == []


def test_handler_exception_does_not_affect_other_handlers():
    """1つのハンドラが例外を投げても他のハンドラは実行される"""
    bus = EventBus()
    second_results = []

    def bad_handler(e):
        raise RuntimeError("handler error")

    def good_handler(e):
        second_results.append(e)

    bus.on(ErrorEvent, bad_handler)
    bus.on(ErrorEvent, good_handler)

    bus.emit(make_error_event())

    # good_handler は呼ばれる
    assert len(second_results) == 1


def test_concurrent_emit_multiple_event_types():
    """複数の Event タイプを同時に emit してもスレッド安全"""
    bus = EventBus()
    error_results = []
    state_results = []
    exceptions = []

    bus.on(ErrorEvent, error_results.append)
    bus.on(StateChangedEvent, state_results.append)

    def emit_errors():
        try:
            for _ in range(50):
                bus.emit(make_error_event())
                time.sleep(0.001)
        except Exception as e:
            exceptions.append(e)

    def emit_states():
        try:
            for _ in range(50):
                bus.emit(StateChangedEvent(old_state="stopped", new_state="running"))
                time.sleep(0.001)
        except Exception as e:
            exceptions.append(e)

    t1 = threading.Thread(target=emit_errors)
    t2 = threading.Thread(target=emit_states)

    t1.start()
    t2.start()

    t1.join(timeout=5)
    t2.join(timeout=5)

    assert exceptions == []
    assert len(error_results) == 50
    assert len(state_results) == 50


def test_handler_count_thread_safe():
    """handler_count が並行アクセス下で正確に動作する"""
    bus = EventBus()
    exceptions = []

    def handler(e):
        pass

    def modify_loop():
        try:
            for _ in range(20):
                bus.on(ErrorEvent, handler)
                count = bus.handler_count(ErrorEvent)
                assert count >= 1
                bus.off(ErrorEvent, handler)
                time.sleep(0.001)
        except Exception as e:
            exceptions.append(e)

    threads = [threading.Thread(target=modify_loop) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert exceptions == []
