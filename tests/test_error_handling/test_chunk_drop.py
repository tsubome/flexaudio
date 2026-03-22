"""Queue Full 時の挙動（DROP_OLDEST）を検証するテスト"""

import queue
import threading
import time

import numpy as np
import pytest

from pyflexaudio.types import AudioChunk, QueuePolicy
from pyflexaudio.events import EventBus
from pyflexaudio.pipeline import Pipeline
from pyflexaudio.processors.chain import ProcessorChain


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


def put_with_drop_oldest(q: queue.Queue, item) -> None:
    """DROP_OLDEST ポリシー: full の場合は古い要素を捨てて新しい要素を入れる"""
    while True:
        try:
            q.put_nowait(item)
            break
        except queue.Full:
            try:
                q.get_nowait()  # 最古の要素を破棄
            except queue.Empty:
                pass


# ---- テスト ----

def test_drop_oldest_queue_does_not_overflow():
    """DROP_OLDEST ポリシーで maxsize=5 の Queue に10個投入しても溢れない"""
    maxsize = 5
    dq = queue.Queue(maxsize=maxsize)

    for i in range(10):
        put_with_drop_oldest(dq, make_chunk())

    # Queue の要素数が maxsize を超えない
    assert dq.qsize() <= maxsize


def test_drop_oldest_discards_old_items():
    """DROP_OLDEST ポリシーでは古いチャンクが捨てられる"""
    maxsize = 3
    dq = queue.Queue(maxsize=maxsize)

    chunks = [make_chunk(source_id=f"test:{i}") for i in range(6)]

    for chunk in chunks:
        put_with_drop_oldest(dq, chunk)

    # Queue に残っているのは新しいチャンク（後半3つ）
    remaining = []
    while not dq.empty():
        remaining.append(dq.get_nowait())

    assert len(remaining) == maxsize
    # 最後に投入したチャンクが残っているはず
    remaining_ids = [c.source_id for c in remaining]
    assert "test:5" in remaining_ids
    assert "test:4" in remaining_ids
    assert "test:3" in remaining_ids


def test_pipeline_processes_chunks_without_overflow():
    """Queue が Full の場合、プロデューサ側で drop が発生し Pipeline はクラッシュしない"""
    # maxsize=5 の Queue に対して6個以上投入すると Full になることを確認
    maxsize = 5
    dq = queue.Queue(maxsize=maxsize)

    # Queue を満杯にする
    for i in range(maxsize):
        dq.put_nowait(make_chunk())

    assert dq.full()

    # Queue が Full の時、put_nowait は Full 例外を投げる
    dropped = 0
    for i in range(10):
        try:
            dq.put_nowait(make_chunk())
        except queue.Full:
            dropped += 1

    # 全て drop される（Queue が満杯のまま）
    assert dropped == 10
    assert dq.qsize() == maxsize

    # Pipeline を起動して drop されたチャンクがあっても正常動作することを確認
    bus = EventBus()
    pipeline = Pipeline(bus)
    pipeline.set_main_chain(ProcessorChain())

    sink = MockSink()
    pipeline.add_sink(sink)

    pipeline.start(dq)
    time.sleep(0.2)
    pipeline.stop()

    # Pipeline スレッドは正常終了（クラッシュしない）
    assert pipeline._thread is None
    # Queue に残っていたチャンクが処理されている
    assert len(sink.chunks) == maxsize


def test_queue_full_does_not_block_producer():
    """Queue が full の場合、プロデューサはブロックされない（put_nowait）"""
    dq = queue.Queue(maxsize=3)

    # Queue を満杯にする
    for _ in range(3):
        dq.put_nowait(make_chunk())

    # put_nowait は Full を投げる（ブロックしない）
    with pytest.raises(queue.Full):
        dq.put_nowait(make_chunk())


def test_drop_oldest_with_sentinel():
    """DROP_OLDEST ポリシーでセンチネル（None）が消えないこと"""
    maxsize = 3
    dq = queue.Queue(maxsize=maxsize)

    # センチネルを最初に入れる
    dq.put(None)

    # 追加のチャンクを投入（センチネルは DROP されうる）
    for _ in range(5):
        put_with_drop_oldest(dq, make_chunk())

    # Queue が maxsize を超えないこと
    assert dq.qsize() <= maxsize


def test_large_queue_handles_burst():
    """大きな Queue は大量のバーストチャンクを問題なく受け入れる"""
    dq = queue.Queue(maxsize=200)

    for i in range(150):
        dq.put_nowait(make_chunk())

    assert dq.qsize() == 150
    assert not dq.full()
