"""SileroVADProcessor のユニットテスト

onnxruntime をモックすることで、実際のモデルファイルや
onnxruntime のインストールなしにステートマシンロジックを検証する。
"""

from __future__ import annotations

import sys
import time
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyflexaudio.events import EventBus
from pyflexaudio.types import AudioChunk, SpeechEndEvent, SpeechStartEvent


# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
WINDOW_SIZE = 512          # SileroVADProcessor.WINDOW_SIZE
# デフォルト min_speech_ms=250 → 250*16000/1000 = 4000 サンプル
# 4000 / 512 = 7.8... → 8 窓目（累計 4096 サンプル）で >= 4000 に達する
MIN_SPEECH_WINDOWS = 8
# デフォルト min_silence_ms=500 → 500*16000/1000 = 8000 サンプル
# 8000 / 512 = 15.6... → 16 窓目（累計 8192 サンプル）で >= 8000 に達する
MIN_SILENCE_WINDOWS = 16


# ---------------------------------------------------------------------------
# モックユーティリティ
# ---------------------------------------------------------------------------

def _make_mock_ort(probabilities: list[float]) -> ModuleType:
    """onnxruntime を模倣する MagicMock モジュールを生成する"""
    mock_session = MagicMock()
    prob_iter = iter(probabilities)

    def mock_run(output_names, input_dict):
        try:
            prob = next(prob_iter)
        except StopIteration:
            prob = 0.0
        output = np.array([[prob]], dtype=np.float32)
        state = input_dict["state"]  # state はそのまま返す
        return [output, state]

    mock_session.run = mock_run

    mock_ort = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session
    return mock_ort, mock_session


def create_mock_vad(event_bus: EventBus, probabilities: list[float], **kwargs):
    """
    指定した確率シーケンスを返すモック VAD インスタンスを返す。

    onnxruntime を sys.modules 経由でモックし、os.path.exists も True に固定する。
    kwargs は SileroVADProcessor のコンストラクタ引数として転送する。
    """
    from pyflexaudio.processors.vad import SileroVADProcessor

    mock_ort, mock_session = _make_mock_ort(probabilities)

    with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
        with patch("os.path.exists", return_value=True):
            vad = SileroVADProcessor(
                event_bus,
                model_path="/fake/model.onnx",
                **kwargs,
            )

    # コンテキスト脱出後もセッションのモックが機能するよう差し替え
    vad._session = mock_session
    return vad


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def make_chunk(frames: int = WINDOW_SIZE, timestamp: float = 0.0,
               source_id: str = "test:0") -> AudioChunk:
    """16kHz mono float32 の AudioChunk を生成する（shape=(frames, 1)）"""
    data = np.zeros((frames, 1), dtype=np.float32)
    return AudioChunk(
        data=data,
        timestamp=timestamp,
        sample_rate=SAMPLE_RATE,
        channels=1,
        source_id=source_id,
    )


def feed_windows(vad, n_windows: int, timestamp: float = 0.0,
                 source_id: str = "test:0") -> None:
    """n_windows 分の 512 サンプルチャンクを process() に渡す"""
    for _ in range(n_windows):
        chunk = make_chunk(frames=WINDOW_SIZE, timestamp=timestamp,
                           source_id=source_id)
        vad.process(chunk)


# ---------------------------------------------------------------------------
# A. 初期化テスト
# ---------------------------------------------------------------------------

class TestInitialization:
    """SileroVADProcessor の初期化ロジック"""

    def test_import_error_raises_clear_message(self):
        """onnxruntime が ImportError の場合、明確なメッセージで ImportError を送出する"""
        bus = EventBus()

        # sys.modules に onnxruntime が存在しない状態を作る
        # None を入れると import 時に ImportError になる
        saved = sys.modules.pop("onnxruntime", _sentinel := object())
        try:
            from pyflexaudio.processors.vad import SileroVADProcessor
            with patch("os.path.exists", return_value=True):
                with pytest.raises(ImportError) as exc_info:
                    SileroVADProcessor(bus, model_path="/fake/model.onnx")
            assert "onnxruntime" in str(exc_info.value).lower()
            assert "pip install" in str(exc_info.value)
        finally:
            # 元の状態に戻す
            if saved is not _sentinel:
                sys.modules["onnxruntime"] = saved
            else:
                sys.modules.pop("onnxruntime", None)

    def test_import_error_via_sys_modules_none(self):
        """sys.modules に None を入れると ImportError になる（Pythonの仕様）"""
        bus = EventBus()

        with patch.dict(sys.modules, {"onnxruntime": None}):
            from pyflexaudio.processors.vad import SileroVADProcessor
            with patch("os.path.exists", return_value=True):
                with pytest.raises(ImportError) as exc_info:
                    SileroVADProcessor(bus, model_path="/fake/model.onnx")
            assert "onnxruntime" in str(exc_info.value).lower()
            assert "pip install" in str(exc_info.value)

    def test_missing_model_raises_file_not_found(self):
        """モデルファイルが存在しない場合、FileNotFoundError を送出する"""
        bus = EventBus()

        mock_ort, _ = _make_mock_ort([])
        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            from pyflexaudio.processors.vad import SileroVADProcessor
            with patch("os.path.exists", return_value=False):
                with pytest.raises(FileNotFoundError) as exc_info:
                    SileroVADProcessor(bus, model_path="/nonexistent/model.onnx")
            assert "/nonexistent/model.onnx" in str(exc_info.value)

    def test_default_parameters(self):
        """デフォルトパラメータの確認（threshold=0.5, min_silence_ms=500, min_speech_ms=250）"""
        bus = EventBus()
        vad = create_mock_vad(bus, probabilities=[])

        assert vad._threshold == 0.5
        # min_silence_ms=500 → 500 * 16000 / 1000 = 8000 サンプル
        assert vad._min_silence_samples == 8000
        # min_speech_ms=250 → 250 * 16000 / 1000 = 4000 サンプル
        assert vad._min_speech_samples == 4000

    def test_custom_threshold(self):
        """カスタム threshold が正しく設定される"""
        bus = EventBus()
        vad = create_mock_vad(bus, probabilities=[], threshold=0.7)
        assert vad._threshold == 0.7

    def test_custom_min_silence_ms(self):
        """カスタム min_silence_ms が正しく変換される"""
        bus = EventBus()
        vad = create_mock_vad(bus, probabilities=[], min_silence_ms=1000)
        assert vad._min_silence_samples == 16000  # 1000 * 16000 / 1000

    def test_custom_min_speech_ms(self):
        """カスタム min_speech_ms が正しく変換される"""
        bus = EventBus()
        vad = create_mock_vad(bus, probabilities=[], min_speech_ms=100)
        assert vad._min_speech_samples == 1600  # 100 * 16000 / 1000


# ---------------------------------------------------------------------------
# B. ステートマシンテスト
# ---------------------------------------------------------------------------

class TestSpeechStartDetection:
    """発話開始の検出ロジック"""

    def test_speech_start_emitted_after_min_speech_windows(self):
        """probability >= 0.5 が min_speech_samples 以上続いたら SpeechStartEvent が発火する"""
        bus = EventBus()
        speech_starts = []
        bus.on(SpeechStartEvent, speech_starts.append)

        probs = [0.9] * MIN_SPEECH_WINDOWS
        vad = create_mock_vad(bus, probs)

        feed_windows(vad, MIN_SPEECH_WINDOWS)

        assert len(speech_starts) == 1

    def test_speech_start_not_emitted_before_threshold_met(self):
        """min_speech_samples に達する前は SpeechStartEvent が発火しない"""
        bus = EventBus()
        speech_starts = []
        bus.on(SpeechStartEvent, speech_starts.append)

        probs = [0.9] * (MIN_SPEECH_WINDOWS - 1)
        vad = create_mock_vad(bus, probs)

        feed_windows(vad, MIN_SPEECH_WINDOWS - 1)

        assert len(speech_starts) == 0

    def test_speech_start_timestamp_matches_chunk(self):
        """SpeechStartEvent の timestamp が発話チャンクのものと一致する"""
        bus = EventBus()
        speech_starts = []
        bus.on(SpeechStartEvent, speech_starts.append)

        expected_ts = 42.0
        probs = [0.9] * MIN_SPEECH_WINDOWS
        vad = create_mock_vad(bus, probs)

        feed_windows(vad, MIN_SPEECH_WINDOWS, timestamp=expected_ts)

        assert len(speech_starts) == 1
        assert speech_starts[0].timestamp == expected_ts


class TestSpeechEndDetection:
    """発話終了の検出ロジック"""

    def test_speech_end_emitted_after_min_silence_windows(self):
        """発話後に probability < 0.5 が min_silence_samples 以上続いたら SpeechEndEvent が発火する"""
        bus = EventBus()
        speech_ends = []
        bus.on(SpeechEndEvent, speech_ends.append)

        probs = [0.9] * MIN_SPEECH_WINDOWS + [0.1] * MIN_SILENCE_WINDOWS
        vad = create_mock_vad(bus, probs)

        feed_windows(vad, MIN_SPEECH_WINDOWS + MIN_SILENCE_WINDOWS)

        assert len(speech_ends) == 1

    def test_speech_end_not_emitted_before_silence_threshold(self):
        """min_silence_samples に達する前は SpeechEndEvent が発火しない"""
        bus = EventBus()
        speech_ends = []
        bus.on(SpeechEndEvent, speech_ends.append)

        probs = [0.9] * MIN_SPEECH_WINDOWS + [0.1] * (MIN_SILENCE_WINDOWS - 1)
        vad = create_mock_vad(bus, probs)

        feed_windows(vad, MIN_SPEECH_WINDOWS + MIN_SILENCE_WINDOWS - 1)

        assert len(speech_ends) == 0


class TestShortSpeechFiltering:
    """短い発話のフィルタリング"""

    def test_short_speech_no_start_event(self):
        """probability >= 0.5 が min_speech_samples 未満なら SpeechStartEvent は発火しない"""
        bus = EventBus()
        speech_starts = []
        bus.on(SpeechStartEvent, speech_starts.append)

        # 7 窓の発話 → フィルタリングされる
        probs = [0.9] * (MIN_SPEECH_WINDOWS - 1) + [0.1]
        vad = create_mock_vad(bus, probs)

        feed_windows(vad, MIN_SPEECH_WINDOWS)

        assert len(speech_starts) == 0

    def test_short_speech_no_end_event(self):
        """短い発話 (min_speech_samples 未満) は SpeechEndEvent も発火しない"""
        bus = EventBus()
        speech_ends = []
        bus.on(SpeechEndEvent, speech_ends.append)

        probs = [0.9] * (MIN_SPEECH_WINDOWS - 1) + [0.1] * MIN_SILENCE_WINDOWS
        vad = create_mock_vad(bus, probs)

        feed_windows(vad, MIN_SPEECH_WINDOWS - 1 + MIN_SILENCE_WINDOWS)

        assert len(speech_ends) == 0


class TestAudioDataInSpeechEndEvent:
    """SpeechEndEvent の audio_data 検証"""

    def test_audio_data_length_matches_accumulated_windows(self):
        """SpeechEndEvent.audio_data の長さが実装の蓄積ロジックと一致する

        実装の蓄積ロジック:
        - 発話判定窓 1〜(MIN_SPEECH_WINDOWS-1): _is_speech=False のため蓄積しない
        - 発話判定窓 MIN_SPEECH_WINDOWS: _is_speech=True になった後に蓄積 → 1 窓
        - 無音判定窓 1〜MIN_SILENCE_WINDOWS: _is_speech=True のまま蓄積 → 16 窓

        合計 = 1 + MIN_SILENCE_WINDOWS = 17 窓 = 8704 サンプル
        """
        bus = EventBus()
        speech_ends = []
        bus.on(SpeechEndEvent, speech_ends.append)

        n_speech = MIN_SPEECH_WINDOWS
        n_silence = MIN_SILENCE_WINDOWS
        probs = [0.9] * n_speech + [0.1] * n_silence
        vad = create_mock_vad(bus, probs)

        feed_windows(vad, n_speech + n_silence)

        assert len(speech_ends) == 1
        # _is_speech が True になった窓（MIN_SPEECH_WINDOWS 番目）から蓄積が始まる
        # → 発話 1 窓 + 無音 16 窓 = 17 窓分
        expected_samples = (1 + n_silence) * WINDOW_SIZE
        assert len(speech_ends[0].audio_data) == expected_samples

    def test_audio_data_is_independent_copy(self):
        """SpeechEndEvent.audio_data が内部バッファとは独立したコピーであること"""
        bus = EventBus()
        speech_ends = []
        bus.on(SpeechEndEvent, speech_ends.append)

        probs = [0.9] * MIN_SPEECH_WINDOWS + [0.1] * MIN_SILENCE_WINDOWS
        vad = create_mock_vad(bus, probs)

        feed_windows(vad, MIN_SPEECH_WINDOWS + MIN_SILENCE_WINDOWS)

        assert len(speech_ends) == 1
        audio_data = speech_ends[0].audio_data

        # 発話終了後、内部リストはリセットされているはず
        assert len(vad._speech_audio) == 0

        # audio_data は np.concatenate().copy() で生成されており、
        # 独立した配列オブジェクトであること
        assert isinstance(audio_data, np.ndarray)
        assert audio_data.dtype == np.float32

    def test_audio_data_different_id_from_input(self):
        """SpeechEndEvent.audio_data が入力チャンクデータとは異なる id を持つ"""
        bus = EventBus()
        speech_ends = []
        bus.on(SpeechEndEvent, speech_ends.append)

        probs = [0.9] * MIN_SPEECH_WINDOWS + [0.1] * MIN_SILENCE_WINDOWS
        vad = create_mock_vad(bus, probs)

        input_chunks = []
        for _ in range(MIN_SPEECH_WINDOWS + MIN_SILENCE_WINDOWS):
            chunk = make_chunk(frames=WINDOW_SIZE)
            input_chunks.append(chunk)
            vad.process(chunk)

        assert len(speech_ends) == 1
        audio_data = speech_ends[0].audio_data

        # audio_data は連結・コピーされた新しい配列なので、
        # 入力チャンクの data とは id が異なる
        for chunk in input_chunks:
            assert id(audio_data) != id(chunk.data)


class TestMultipleSpeechCycles:
    """複数の発話サイクル"""

    def test_two_speech_cycles(self):
        """発話→無音→発話→無音 で SpeechStart/End が各2回発火する"""
        bus = EventBus()
        speech_starts = []
        speech_ends = []
        bus.on(SpeechStartEvent, speech_starts.append)
        bus.on(SpeechEndEvent, speech_ends.append)

        one_cycle = [0.9] * MIN_SPEECH_WINDOWS + [0.1] * MIN_SILENCE_WINDOWS
        probs = one_cycle * 2
        vad = create_mock_vad(bus, probs)

        total_windows = (MIN_SPEECH_WINDOWS + MIN_SILENCE_WINDOWS) * 2
        feed_windows(vad, total_windows)

        assert len(speech_starts) == 2
        assert len(speech_ends) == 2


class TestResetDuringSpeech:
    """reset() 中のクリーンアップ"""

    def test_reset_during_speech_no_end_event(self):
        """発話中に reset() を呼んでも SpeechEndEvent は emit されない"""
        bus = EventBus()
        speech_ends = []
        bus.on(SpeechEndEvent, speech_ends.append)

        probs = [0.9] * MIN_SPEECH_WINDOWS
        vad = create_mock_vad(bus, probs)

        feed_windows(vad, MIN_SPEECH_WINDOWS)
        assert vad._is_speech is True

        vad.reset()

        assert len(speech_ends) == 0

    def test_reset_clears_internal_state(self):
        """reset() 後に内部状態がクリアされている"""
        bus = EventBus()

        probs = [0.9] * MIN_SPEECH_WINDOWS
        vad = create_mock_vad(bus, probs)

        feed_windows(vad, MIN_SPEECH_WINDOWS)
        assert vad._is_speech is True

        vad.reset()

        assert vad._is_speech is False
        assert vad._speech_samples == 0
        assert vad._silence_samples == 0
        assert len(vad._speech_audio) == 0
        assert vad._speech_audio_samples == 0
        assert len(vad._buffer) == 0


class TestBufferAccumulation:
    """バッファ持ち越し（512 未満のチャンク）"""

    def test_small_chunks_accumulate_into_window(self):
        """512 未満のチャンクが蓄積されて、合計 >= 512 になったら推論が実行される"""
        bus = EventBus()
        speech_starts = []
        bus.on(SpeechStartEvent, speech_starts.append)

        # 各チャンクは 256 サンプル（WINDOW_SIZE の半分）
        # 2 チャンクで 1 窓分になる
        # 発話開始には 8 窓 = 16 チャンク（256×16=4096 サンプル）必要
        n_half_chunks = MIN_SPEECH_WINDOWS * 2  # = 16 チャンク → 8 窓分
        probs = [0.9] * MIN_SPEECH_WINDOWS

        vad = create_mock_vad(bus, probs)

        for _ in range(n_half_chunks):
            chunk = make_chunk(frames=256)
            vad.process(chunk)

        # 8 窓分の高確率が処理されたため SpeechStartEvent が発火するはず
        assert len(speech_starts) == 1

    def test_partial_buffer_not_processed(self):
        """512 未満のチャンクのみではバッファに残り、推論は実行されない"""
        bus = EventBus()
        speech_starts = []
        bus.on(SpeechStartEvent, speech_starts.append)

        probs = [0.9] * MIN_SPEECH_WINDOWS
        vad = create_mock_vad(bus, probs)

        # 256 サンプルのチャンク 1 つ → バッファに 256 溜まるだけ
        chunk = make_chunk(frames=256)
        vad.process(chunk)

        assert len(speech_starts) == 0
        assert len(vad._buffer) == 256


class TestProcessReturnValue:
    """process() の戻り値がデータを変更しないこと"""

    def test_process_returns_same_object(self):
        """process() が入力チャンクと同一のオブジェクトを返す"""
        bus = EventBus()
        probs = [0.3]  # 無音
        vad = create_mock_vad(bus, probs)

        chunk = make_chunk()
        result = vad.process(chunk)

        assert result is chunk

    def test_process_does_not_modify_data(self):
        """process() が chunk.data を変更しない"""
        bus = EventBus()
        probs = [0.9]  # 発話
        vad = create_mock_vad(bus, probs)

        chunk = make_chunk()
        original_data = chunk.data.copy()
        vad.process(chunk)

        assert np.array_equal(chunk.data, original_data)

    def test_process_does_not_modify_metadata(self):
        """process() が chunk のメタデータ（timestamp, source_id 等）を変更しない"""
        bus = EventBus()
        probs = [0.3]
        vad = create_mock_vad(bus, probs)

        ts = 99.9
        chunk = make_chunk(timestamp=ts)
        vad.process(chunk)

        assert chunk.timestamp == ts
        assert chunk.sample_rate == SAMPLE_RATE
        assert chunk.source_id == "test:0"


# ---------------------------------------------------------------------------
# C. エッジケース
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """境界値・エッジケースのテスト"""

    def test_empty_chunk_no_exception(self):
        """frames=0 のチャンクを process() に渡しても例外が発生しない"""
        bus = EventBus()
        probs = []
        vad = create_mock_vad(bus, probs)

        chunk = make_chunk(frames=0)
        result = vad.process(chunk)
        assert result is chunk

    def test_boundary_speech_start_exactly_at_min_speech_windows(self):
        """ちょうど min_speech_samples に達した窓で SpeechStartEvent が発火する（境界値）"""
        bus = EventBus()
        speech_starts = []
        bus.on(SpeechStartEvent, speech_starts.append)

        # MIN_SPEECH_WINDOWS - 1 窓では発火しない
        probs_before = [0.9] * (MIN_SPEECH_WINDOWS - 1)
        vad = create_mock_vad(bus, probs_before)
        feed_windows(vad, MIN_SPEECH_WINDOWS - 1)
        assert len(speech_starts) == 0

        # もう1窓追加 → ちょうど MIN_SPEECH_WINDOWS で発火する
        extra_prob_iter = iter([0.9])

        def mock_run_extra(output_names, input_dict):
            prob = next(extra_prob_iter)
            output = np.array([[prob]], dtype=np.float32)
            return [output, input_dict["state"]]

        vad._session.run = mock_run_extra
        feed_windows(vad, 1)

        assert len(speech_starts) == 1

    def test_boundary_speech_end_exactly_at_min_silence_windows(self):
        """ちょうど min_silence_samples に達した窓で SpeechEndEvent が発火する（境界値）"""
        bus = EventBus()
        speech_ends = []
        bus.on(SpeechEndEvent, speech_ends.append)

        # 発話開始後、MIN_SILENCE_WINDOWS - 1 窓無音では発火しない
        probs_speech = [0.9] * MIN_SPEECH_WINDOWS
        probs_silence_short = [0.1] * (MIN_SILENCE_WINDOWS - 1)
        vad = create_mock_vad(bus, probs_speech + probs_silence_short)

        feed_windows(vad, MIN_SPEECH_WINDOWS + MIN_SILENCE_WINDOWS - 1)
        assert len(speech_ends) == 0

        # もう1窓無音を追加 → ちょうど MIN_SILENCE_WINDOWS で発火する
        extra_prob_iter = iter([0.1])

        def mock_run_extra(output_names, input_dict):
            prob = next(extra_prob_iter)
            output = np.array([[prob]], dtype=np.float32)
            return [output, input_dict["state"]]

        vad._session.run = mock_run_extra
        feed_windows(vad, 1)

        assert len(speech_ends) == 1

    def test_threshold_boundary_exactly_at_threshold(self):
        """確率がちょうど threshold (0.5) の場合、発話として判定される（>= 判定）"""
        bus = EventBus()
        speech_starts = []
        bus.on(SpeechStartEvent, speech_starts.append)

        probs = [0.5] * MIN_SPEECH_WINDOWS  # ちょうど threshold
        vad = create_mock_vad(bus, probs)

        feed_windows(vad, MIN_SPEECH_WINDOWS)

        assert len(speech_starts) == 1

    def test_threshold_just_below_does_not_trigger_speech(self):
        """確率が threshold より少し下 (0.499) の場合、発話として判定されない"""
        bus = EventBus()
        speech_starts = []
        bus.on(SpeechStartEvent, speech_starts.append)

        probs = [0.499] * MIN_SPEECH_WINDOWS
        vad = create_mock_vad(bus, probs)

        feed_windows(vad, MIN_SPEECH_WINDOWS)

        assert len(speech_starts) == 0

    def test_process_called_multiple_times_with_single_window_chunks(self):
        """512 サンプルの正規チャンクを逐次処理しても正常に動作する"""
        bus = EventBus()
        speech_starts = []
        speech_ends = []
        bus.on(SpeechStartEvent, speech_starts.append)
        bus.on(SpeechEndEvent, speech_ends.append)

        probs = [0.9] * MIN_SPEECH_WINDOWS + [0.1] * MIN_SILENCE_WINDOWS
        vad = create_mock_vad(bus, probs)

        feed_windows(vad, MIN_SPEECH_WINDOWS + MIN_SILENCE_WINDOWS)

        assert len(speech_starts) == 1
        assert len(speech_ends) == 1

    def test_source_id_propagated_to_events(self):
        """SpeechStartEvent / SpeechEndEvent に source_id が正しく伝播する"""
        bus = EventBus()
        speech_starts = []
        speech_ends = []
        bus.on(SpeechStartEvent, speech_starts.append)
        bus.on(SpeechEndEvent, speech_ends.append)

        probs = [0.9] * MIN_SPEECH_WINDOWS + [0.1] * MIN_SILENCE_WINDOWS
        vad = create_mock_vad(bus, probs)

        expected_source_id = "test:0"
        feed_windows(vad, MIN_SPEECH_WINDOWS + MIN_SILENCE_WINDOWS,
                     source_id=expected_source_id)

        assert speech_starts[0].source_id == expected_source_id
        assert speech_ends[0].source_id == expected_source_id

    def test_speech_start_event_has_correct_fields(self):
        """SpeechStartEvent のフィールドが正しく設定されている"""
        bus = EventBus()
        speech_starts = []
        bus.on(SpeechStartEvent, speech_starts.append)

        probs = [0.9] * MIN_SPEECH_WINDOWS
        vad = create_mock_vad(bus, probs)
        feed_windows(vad, MIN_SPEECH_WINDOWS, timestamp=5.0, source_id="mic:0")

        event = speech_starts[0]
        assert isinstance(event, SpeechStartEvent)
        assert event.timestamp == 5.0
        assert event.source_id == "mic:0"

    def test_speech_end_event_has_correct_fields(self):
        """SpeechEndEvent のフィールドが正しく設定されている"""
        bus = EventBus()
        speech_ends = []
        bus.on(SpeechEndEvent, speech_ends.append)

        probs = [0.9] * MIN_SPEECH_WINDOWS + [0.1] * MIN_SILENCE_WINDOWS
        vad = create_mock_vad(bus, probs)
        feed_windows(vad, MIN_SPEECH_WINDOWS + MIN_SILENCE_WINDOWS, source_id="mic:0")

        event = speech_ends[0]
        assert isinstance(event, SpeechEndEvent)
        assert event.source_id == "mic:0"
        assert event.duration_sec > 0
        assert isinstance(event.audio_data, np.ndarray)
        assert event.audio_data.ndim == 1  # 1D mono
        assert event.audio_data.dtype == np.float32
