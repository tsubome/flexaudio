"""エッジケーステスト for pyflexaudio

カバー範囲:
  A. 空フレーム（frames=0）チャンク
  B. NaN / Inf 入力
  C. ProcessorChain 例外伝播
  D. CallbackSink コールバック例外
  E. LevelMeterSink enabled テスト
  F. FileSink 追加テスト
"""

from __future__ import annotations

import struct
import time

import numpy as np
import pytest

from pyflexaudio.events import EventBus
from pyflexaudio.processors.chain import ProcessorChain
from pyflexaudio.processors.channels import ChannelConvertProcessor
from pyflexaudio.processors.level import LevelMeterProcessor
from pyflexaudio.processors.resample import ResampleProcessor
from pyflexaudio.sinks.callback import CallbackSink
from pyflexaudio.sinks.file import FileSink
from pyflexaudio.sinks.level_meter import LevelMeterSink
from pyflexaudio.types import AudioChunk, LevelEvent


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def make_empty_chunk(sr: int = 16000, ch: int = 1) -> AudioChunk:
    """frames=0 の空チャンクを生成"""
    return AudioChunk(
        data=np.zeros((0, ch), dtype=np.float32),
        timestamp=time.monotonic(),
        sample_rate=sr,
        channels=ch,
        source_id="test:0",
    )


def make_chunk(data: np.ndarray, sr: int = 16000) -> AudioChunk:
    """任意の data から AudioChunk を生成（shape=(frames, channels)）"""
    ch = data.shape[1] if data.ndim == 2 else 1
    return AudioChunk(
        data=data,
        timestamp=time.monotonic(),
        sample_rate=sr,
        channels=ch,
        source_id="test:0",
    )


def make_nan_chunk() -> AudioChunk:
    data = np.array([[float("nan")], [0.5], [float("nan")]], dtype=np.float32)
    return make_chunk(data)


def make_inf_chunk() -> AudioChunk:
    data = np.array([[float("inf")], [-float("inf")], [0.5]], dtype=np.float32)
    return make_chunk(data)


# ---------------------------------------------------------------------------
# A. 空フレーム（frames=0）テスト
# ---------------------------------------------------------------------------


class TestEmptyChunk:
    def test_resample_empty_same_rate(self):
        """A-1a: ResampleProcessor — ソースレート==ターゲット → 素通し、例外なし"""
        proc = ResampleProcessor(target_sample_rate=16000)
        chunk = make_empty_chunk(sr=16000, ch=1)
        out = proc.process(chunk)
        assert out.data.shape[0] == 0
        assert out.sample_rate == 16000

    def test_resample_empty_different_rate(self):
        """A-1b: ResampleProcessor — ソースレート!=ターゲット → 例外なし、出力も空"""
        proc = ResampleProcessor(target_sample_rate=16000)
        chunk = make_empty_chunk(sr=48000, ch=1)
        out = proc.process(chunk)
        # soxr は空配列を渡されても空配列を返す
        assert out.data.shape[0] == 0
        assert out.sample_rate == 16000

    def test_channel_convert_empty_stereo_to_mono(self):
        """A-2: ChannelConvertProcessor — 空チャンク stereo→mono → 例外なし"""
        proc = ChannelConvertProcessor(target_channels=1)
        chunk = make_empty_chunk(sr=16000, ch=2)
        out = proc.process(chunk)
        assert out.data.shape == (0, 1)
        assert out.channels == 1

    def test_channel_convert_empty_mono_to_stereo(self):
        """A-3: ChannelConvertProcessor — 空チャンク mono→stereo → 例外なし"""
        proc = ChannelConvertProcessor(target_channels=2)
        chunk = make_empty_chunk(sr=16000, ch=1)
        out = proc.process(chunk)
        assert out.data.shape == (0, 2)
        assert out.channels == 2

    def test_level_meter_empty_chunk(self):
        """A-4: LevelMeterProcessor — 空チャンク → level_db が NaN になる

        np.mean(empty_array) は NaN を返す。
        max(float('nan'), 1e-10) == nan なので log10(nan) == nan となり
        level_db には NaN がセットされる。
        """
        proc = LevelMeterProcessor()
        chunk = make_empty_chunk()
        out = proc.process(chunk)
        assert out.level_db is not None
        assert np.isnan(out.level_db), (
            f"空チャンクの level_db は NaN であるべきだが {out.level_db!r} だった"
        )

    def test_processor_chain_empty_chunk(self):
        """A-5: ProcessorChain — 空チャンク → 例外なし"""
        chain = ProcessorChain([
            ResampleProcessor(target_sample_rate=16000),
            ChannelConvertProcessor(target_channels=1),
        ])
        chunk = make_empty_chunk(sr=48000, ch=2)
        out = chain.process(chunk)
        assert out.data.shape[0] == 0

    def test_file_sink_empty_chunk_no_size_increase(self, tmp_path):
        """A-6: FileSink.write() — 空チャンク → データサイズが増加しない"""
        path = str(tmp_path / "empty_test.wav")
        sink = FileSink(path)
        try:
            # まず通常チャンクでファイルを開く
            normal_data = np.random.randn(512, 1).astype(np.float32) * 0.1
            normal_chunk = make_chunk(normal_data)
            sink.write(normal_chunk)
            size_before = sink._total_data_bytes

            # 空チャンクを書き込み
            empty_chunk = make_empty_chunk(sr=16000, ch=1)
            sink.write(empty_chunk)
            size_after = sink._total_data_bytes

            assert size_after == size_before, (
                f"空チャンク書き込み後にサイズが変化した: {size_before} -> {size_after}"
            )
        finally:
            sink.close()

    def test_callback_sink_empty_chunk_calls_callback(self):
        """A-7: CallbackSink.write() — 空チャンク → コールバックは呼ばれる"""
        received: list[AudioChunk] = []
        sink = CallbackSink(callback=received.append)
        chunk = make_empty_chunk()
        sink.write(chunk)
        assert len(received) == 1
        assert received[0].data.shape[0] == 0


# ---------------------------------------------------------------------------
# B. NaN / Inf 入力テスト
# ---------------------------------------------------------------------------


class TestNanInfInput:
    def test_level_meter_nan_input(self):
        """B-1: LevelMeterProcessor — NaN 入力 → level_db が NaN"""
        proc = LevelMeterProcessor()
        out = proc.process(make_nan_chunk())
        assert out.level_db is not None
        assert np.isnan(out.level_db), (
            f"NaN 入力の level_db は NaN であるべきだが {out.level_db!r} だった"
        )

    def test_level_meter_inf_input(self):
        """B-2: LevelMeterProcessor — Inf 入力 → level_db が Inf"""
        proc = LevelMeterProcessor()
        out = proc.process(make_inf_chunk())
        assert out.level_db is not None
        assert np.isinf(out.level_db), (
            f"Inf 入力の level_db は Inf であるべきだが {out.level_db!r} だった"
        )

    def test_channel_convert_nan_no_crash(self):
        """B-3: ChannelConvertProcessor — NaN 入力 → クラッシュしない、出力に NaN 含む"""
        proc = ChannelConvertProcessor(target_channels=1)
        # stereo NaN → mono
        nan_stereo_data = np.array(
            [[float("nan"), 0.5], [0.3, float("nan")]], dtype=np.float32
        )
        chunk = make_chunk(nan_stereo_data)
        out = proc.process(chunk)
        assert out.data.shape == (2, 1)
        assert np.any(np.isnan(out.data)), "出力に NaN が含まれるべき"

    def test_file_sink_nan_no_crash(self, tmp_path):
        """B-4: FileSink.write() — NaN 入力 → np.clip 後の int16 変換がクラッシュしない

        np.clip(nan * 32767, -32768, 32767) は nan のまま。
        .astype(np.int16) は実装依存（通常 0 か不定値）だが RuntimeWarning が出るだけで
        例外は上がらない。
        """
        path = str(tmp_path / "nan_test.wav")
        sink = FileSink(path)
        try:
            sink.write(make_nan_chunk())
            # ファイルが開かれ、何らかのデータが書かれている（クラッシュしていない）
            assert sink._file is not None
        finally:
            sink.close()

    def test_file_sink_inf_clips_to_int16_range(self, tmp_path):
        """B-5: FileSink.write() — Inf 入力 → np.clip が 32767/-32768 にクリップ"""
        path = str(tmp_path / "inf_test.wav")
        sink = FileSink(path)
        try:
            sink.write(make_inf_chunk())
            sink.flush()
        finally:
            sink.close()

        # WAV ファイルを開いてサンプル値を確認
        with open(path, "rb") as f:
            f.seek(44)  # WAV ヘッダをスキップ
            raw = f.read()
        samples = np.frombuffer(raw, dtype=np.int16)
        # +Inf → 32767, -Inf → -32768 にクリップされているはず
        assert 32767 in samples, "+Inf は 32767 にクリップされるべき"
        assert -32768 in samples, "-Inf は -32768 にクリップされるべき"


# ---------------------------------------------------------------------------
# C. ProcessorChain 例外伝播テスト
# ---------------------------------------------------------------------------


class BrokenProcessor:
    """常に例外を投げるプロセッサ"""

    def __init__(self, error_msg: str = "broken"):
        self._error_msg = error_msg
        self.call_count = 0

    def process(self, chunk: AudioChunk) -> AudioChunk:
        self.call_count += 1
        raise RuntimeError(self._error_msg)

    def reset(self) -> None:
        raise RuntimeError(f"reset failed: {self._error_msg}")


class TrackingProcessor:
    """呼び出しを追跡するプロセッサ"""

    def __init__(self):
        self.process_called = False
        self.reset_called = False

    def process(self, chunk: AudioChunk) -> AudioChunk:
        self.process_called = True
        return chunk

    def reset(self) -> None:
        self.reset_called = True


class TestProcessorChainExceptions:
    def test_exception_propagates_and_stops_chain(self):
        """C-1: チェーン内のプロセッサが例外 → 後続プロセッサは呼ばれない → 例外が伝播"""
        broken = BrokenProcessor("process error")
        tracker = TrackingProcessor()
        chain = ProcessorChain([broken, tracker])

        chunk = make_empty_chunk()
        with pytest.raises(RuntimeError, match="process error"):
            chain.process(chunk)

        assert broken.call_count == 1
        assert not tracker.process_called, "例外後の後続プロセッサは呼ばれてはいけない"

    def test_reset_exception_aborts_remaining_resets(self):
        """C-2: reset() 中に例外 → 残りのプロセッサの reset() は呼ばれない

        # BUG: ProcessorChain.reset() に try/except がないため、最初の reset() が
        # 例外を投げると後続プロセッサの reset() が呼ばれない。
        # これが意図された仕様かバグかは設計次第だが、現状の挙動をテストで文書化する。
        """
        broken = BrokenProcessor("reset error")
        tracker = TrackingProcessor()
        chain = ProcessorChain([broken, tracker])

        with pytest.raises(RuntimeError, match="reset error"):
            chain.reset()

        # BUG: 例外が伝播したため tracker.reset() は呼ばれていない
        assert not tracker.reset_called, (
            "BUG: ProcessorChain.reset() は例外をキャッチしないため、"
            "先行プロセッサが例外を投げると後続プロセッサの reset() が呼ばれない"
        )


# ---------------------------------------------------------------------------
# D. CallbackSink コールバック例外テスト
# ---------------------------------------------------------------------------


class TestCallbackSinkException:
    def test_callback_exception_propagates_from_write(self):
        """D-1: コールバックが例外を投げた場合 → write() から例外が伝播する

        Pipeline._process_chunk() では sink.write() の例外をキャッチして
        該当 Sink を disabled にするが、CallbackSink.write() 自体は例外をキャッチしない。
        Pipeline を経由しない直接呼び出しでは例外がそのまま上がる。
        """

        def bad_callback(chunk: AudioChunk) -> None:
            raise ValueError("callback failed")

        sink = CallbackSink(callback=bad_callback)
        chunk = make_empty_chunk()
        with pytest.raises(ValueError, match="callback failed"):
            sink.write(chunk)

    def test_callback_not_called_when_disabled(self):
        """enabled=False の CallbackSink はコールバックを呼ばない"""
        received: list[AudioChunk] = []
        sink = CallbackSink(callback=received.append, enabled=False)
        sink.write(make_empty_chunk())
        assert len(received) == 0


# ---------------------------------------------------------------------------
# E. LevelMeterSink enabled テスト
# ---------------------------------------------------------------------------


class TestLevelMeterSinkEnabled:
    def test_level_meter_sink_emits_when_enabled(self):
        """E 前提確認: enabled=True のとき LevelEvent が emit される"""
        bus = EventBus()
        events: list[LevelEvent] = []
        bus.on(LevelEvent, events.append)

        sink = LevelMeterSink(event_bus=bus)
        assert sink.enabled is True

        chunk = make_chunk(np.array([[0.5]], dtype=np.float32))
        chunk.level_db = -6.0
        sink.write(chunk)
        assert len(events) == 1

    def test_level_meter_sink_enabled_false_still_emits(self):
        """E-1: enabled=False に変更しても write() が LevelEvent を emit し続ける

        # BUG: LevelMeterSink.write() は self.enabled をチェックしていない。
        # enabled=False に設定しても、chunk.level_db が None でなければ
        # LevelEvent が emit され続ける。
        # Pipeline._process_chunk() では「not sink.enabled」のチェックで
        # write() 呼び出し自体をスキップするため Pipeline 経由では問題が出ないが、
        # 直接 write() を呼ぶ場合や、将来 Pipeline のチェックが変わった場合に
        # 意図しない emit が発生しうる。
        """
        bus = EventBus()
        events: list[LevelEvent] = []
        bus.on(LevelEvent, events.append)

        sink = LevelMeterSink(event_bus=bus)
        sink.enabled = False  # 無効化

        chunk = make_chunk(np.array([[0.5]], dtype=np.float32))
        chunk.level_db = -6.0
        sink.write(chunk)

        # BUG: enabled=False でも emit される
        assert len(events) == 1, (
            "BUG: LevelMeterSink.write() は self.enabled をチェックしていないため、"
            "enabled=False でも LevelEvent が emit される"
        )

    def test_level_meter_sink_no_emit_when_level_db_none(self):
        """level_db=None のチャンクは emit しない（これは正しい挙動）"""
        bus = EventBus()
        events: list[LevelEvent] = []
        bus.on(LevelEvent, events.append)

        sink = LevelMeterSink(event_bus=bus)
        chunk = make_empty_chunk()
        # level_db は None のまま
        sink.write(chunk)
        assert len(events) == 0


# ---------------------------------------------------------------------------
# F. FileSink 追加テスト
# ---------------------------------------------------------------------------


class TestFileSinkExtra:
    def test_close_twice_is_idempotent(self, tmp_path):
        """F-1: close() 2回呼び出し → 例外なし（冪等性）"""
        path = str(tmp_path / "double_close.wav")
        sink = FileSink(path)
        data = np.random.randn(512, 1).astype(np.float32) * 0.1
        sink.write(make_chunk(data))
        sink.close()
        sink.close()  # 2回目は例外なし

    def test_flush_after_close_no_exception(self, tmp_path):
        """F-2: close() 後に flush() → 例外なし"""
        path = str(tmp_path / "flush_after_close.wav")
        sink = FileSink(path)
        data = np.random.randn(512, 1).astype(np.float32) * 0.1
        sink.write(make_chunk(data))
        sink.close()
        sink.flush()  # close 後の flush は無視される

    def test_wav_header_sample_rate_after_resample(self, tmp_path):
        """F-3: sample_rate 指定ありの書き出し → WAV ヘッダが指定サンプルレートになる"""
        path = str(tmp_path / "resampled.wav")
        target_sr = 16000
        sink = FileSink(path, sample_rate=target_sr)
        data = np.random.randn(512, 1).astype(np.float32) * 0.1
        chunk = make_chunk(data, sr=48000)  # ソースは 48kHz
        sink.write(chunk)
        sink.close()

        with open(path, "rb") as f:
            header = f.read(44)

        # WAV fmt チャンクのサンプルレートフィールド (offset 24, 4 bytes, little endian)
        wav_sr = struct.unpack("<I", header[24:28])[0]
        assert wav_sr == target_sr, (
            f"WAV ヘッダのサンプルレートは {target_sr} であるべきだが {wav_sr} だった"
        )

    def test_write_to_nonexistent_directory_raises(self, tmp_path):
        """F-4: 書き込み先ディレクトリが存在しない → FileNotFoundError"""
        path = str(tmp_path / "nonexistent_dir" / "output.wav")
        sink = FileSink(path)
        data = np.random.randn(512, 1).astype(np.float32) * 0.1
        with pytest.raises(FileNotFoundError):
            sink.write(make_chunk(data))

    def test_write_empty_chunk_to_new_file_opens_file(self, tmp_path):
        """F 補足: 最初のチャンクが空でも write() はファイルを開く（ヘッダのみ書き出し）"""
        path = str(tmp_path / "empty_first.wav")
        sink = FileSink(path)
        try:
            empty_chunk = make_empty_chunk(sr=16000, ch=1)
            sink.write(empty_chunk)
            # ファイルが開かれている
            assert sink._file is not None
            # データバイトは 0
            assert sink._total_data_bytes == 0
        finally:
            sink.close()

        # ファイルは存在する（44バイトのヘッダのみ）
        import os
        assert os.path.getsize(path) == 44, (
            "空チャンクのみ書き込んだ WAV は 44 バイト（ヘッダのみ）であるべき"
        )
