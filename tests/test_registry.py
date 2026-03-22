import threading
import pytest
from pyflexaudio.registry import StreamRegistry


@pytest.fixture
def registry():
    return StreamRegistry()


# --- acquire ---

def test_acquire_calls_factory(registry):
    sentinel = object()
    result = registry.acquire("microphone:0", lambda: sentinel)
    assert result is sentinel


def test_acquire_returns_factory_result(registry):
    class FakeStream:
        pass

    stream = FakeStream()
    acquired = registry.acquire("microphone:0", lambda: stream)
    assert acquired is stream


def test_acquire_duplicate_raises_runtime_error(registry):
    registry.acquire("microphone:0", object)
    with pytest.raises(RuntimeError, match="already in use"):
        registry.acquire("microphone:0", object)


def test_acquire_different_keys_independent(registry):
    s1 = registry.acquire("microphone:0", object)
    s2 = registry.acquire("microphone:1", object)
    assert s1 is not None
    assert s2 is not None


# --- release ---

def test_release_allows_reacquire(registry):
    registry.acquire("microphone:0", object)
    registry.release("microphone:0")
    # 再取得できること
    result = registry.acquire("microphone:0", object)
    assert result is not None


def test_release_idempotent(registry):
    registry.acquire("microphone:0", object)
    registry.release("microphone:0")
    # 2 回目の release が例外を起こさないこと
    registry.release("microphone:0")


def test_release_nonexistent_key_no_error(registry):
    registry.release("microphone:99")


# --- is_acquired ---

def test_is_acquired_false_initially(registry):
    assert registry.is_acquired("microphone:0") is False


def test_is_acquired_true_after_acquire(registry):
    registry.acquire("microphone:0", object)
    assert registry.is_acquired("microphone:0") is True


def test_is_acquired_false_after_release(registry):
    registry.acquire("microphone:0", object)
    registry.release("microphone:0")
    assert registry.is_acquired("microphone:0") is False


# --- get ---

def test_get_returns_none_initially(registry):
    assert registry.get("microphone:0") is None


def test_get_returns_stream_after_acquire(registry):
    sentinel = object()
    registry.acquire("microphone:0", lambda: sentinel)
    assert registry.get("microphone:0") is sentinel


def test_get_returns_none_after_release(registry):
    registry.acquire("microphone:0", object)
    registry.release("microphone:0")
    assert registry.get("microphone:0") is None


# --- clear ---

def test_clear_removes_all_entries(registry):
    registry.acquire("microphone:0", object)
    registry.acquire("microphone:1", object)
    registry.acquire("system_audio:default", object)
    registry.clear()
    assert registry.is_acquired("microphone:0") is False
    assert registry.is_acquired("microphone:1") is False
    assert registry.is_acquired("system_audio:default") is False


def test_clear_allows_reacquire(registry):
    registry.acquire("microphone:0", object)
    registry.clear()
    result = registry.acquire("microphone:0", object)
    assert result is not None


def test_clear_empty_registry_no_error(registry):
    registry.clear()


# --- スレッド安全性 ---

def test_thread_safety_acquire_same_key_only_one_succeeds():
    registry = StreamRegistry()
    success_count = [0]
    error_count = [0]
    lock = threading.Lock()

    def try_acquire():
        try:
            registry.acquire("microphone:0", object)
            with lock:
                success_count[0] += 1
        except RuntimeError:
            with lock:
                error_count[0] += 1

    threads = [threading.Thread(target=try_acquire) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert success_count[0] == 1
    assert error_count[0] == 9
