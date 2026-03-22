"""Silero VAD v5 による音声区間検出プロセッサ"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from pyflexaudio.types import AudioChunk, SpeechStartEvent, SpeechEndEvent
from pyflexaudio.events import EventBus

logger = logging.getLogger("pyflexaudio.processors.vad")

__all__ = ["SileroVADProcessor"]


def _default_model_path() -> str:
    """同梱 VAD モデルのパスを返す"""
    return str(Path(__file__).parent / "models" / "silero_vad.onnx")


class SileroVADProcessor:
    """Silero VAD v5 による音声区間検出プロセッサ"""

    WINDOW_SIZE = 512       # 16kHz で 32ms
    SAMPLE_RATE = 16000     # 必須サンプルレート
    THRESHOLD = 0.5         # 発話確率閾値（デフォルト）
    MIN_SILENCE_MS = 500    # 無音区間の最小長（ms）
    MIN_SPEECH_MS = 250     # 発話区間の最小長（ms）

    def __init__(
        self,
        event_bus: EventBus,
        model_path: str | None = None,
        threshold: float = 0.5,
        min_silence_ms: int = 500,
        min_speech_ms: int = 250,
    ):
        self._event_bus = event_bus
        self._threshold = threshold
        self._min_silence_samples = int(min_silence_ms * self.SAMPLE_RATE / 1000)
        self._min_speech_samples = int(min_speech_ms * self.SAMPLE_RATE / 1000)

        # onnxruntime 遅延 import
        try:
            import onnxruntime
        except ImportError:
            raise ImportError(
                "onnxruntime is required for VAD. "
                "Install it with: pip install pyflexaudio[vad]"
            )

        path = model_path or _default_model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"VAD model not found: {path}")

        self._session = onnxruntime.InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
        )

        # Silero VAD v5 の内部状態
        self._state = np.zeros((2, 1, 128), dtype=np.float32)  # 統合 LSTM state
        self._sr = np.array([self.SAMPLE_RATE], dtype=np.int64)

        # チャンク境界をまたぐバッファ
        self._buffer = np.array([], dtype=np.float32)

        # 発話状態管理
        self._is_speech = False
        self._speech_start_time: float | None = None
        self._speech_samples = 0      # 連続発話サンプル数
        self._silence_samples = 0     # 連続無音サンプル数

        # 発話区間の音声データ蓄積用
        # 最大60秒分（16kHz * 60 = 960,000 サンプル）
        self._speech_audio: list[np.ndarray] = []
        self._speech_audio_samples = 0

        # ソースID（最後に処理したチャンクから取得）
        self._last_source_id = ""

    def process(self, chunk: AudioChunk) -> AudioChunk:
        """チャンクを処理。データは変更せず、VAD 結果は EventBus 経由で通知"""
        self._last_source_id = chunk.source_id

        # 入力データを1Dに（mono前提: channels=1, shape=(frames, 1)）
        data = chunk.data.squeeze()  # shape=(frames,)

        # バッファに追加
        self._buffer = np.concatenate([self._buffer, data])

        # 512サンプルずつ推論
        while len(self._buffer) >= self.WINDOW_SIZE:
            window = self._buffer[:self.WINDOW_SIZE]
            self._buffer = self._buffer[self.WINDOW_SIZE:]
            self._process_window(window, chunk.timestamp)

        return chunk  # データは変更しない

    def reset(self) -> None:
        """内部状態をリセット"""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._buffer = np.array([], dtype=np.float32)
        self._is_speech = False
        self._speech_start_time = None
        self._speech_samples = 0
        self._silence_samples = 0
        self._speech_audio.clear()
        self._speech_audio_samples = 0

    def _process_window(self, window: np.ndarray, timestamp: float) -> None:
        """512サンプルの窓を推論"""
        # 推論
        input_tensor = window.reshape(1, -1).astype(np.float32)
        output, new_state = self._session.run(
            ["output", "stateN"],
            {
                "input": input_tensor,
                "state": self._state,
                "sr": self._sr,
            },
        )
        self._state = new_state
        probability = float(output[0][0])

        if probability >= self._threshold:
            # 発話検出
            self._silence_samples = 0
            self._speech_samples += self.WINDOW_SIZE

            if not self._is_speech and self._speech_samples >= self._min_speech_samples:
                # 発話開始
                self._is_speech = True
                self._speech_start_time = timestamp
                self._event_bus.emit(SpeechStartEvent(
                    timestamp=timestamp,
                    source_id=self._last_source_id,
                ))
                logger.debug("Speech started at %.3f", timestamp)

            # 発話中の音声データを蓄積
            if self._is_speech:
                self._speech_audio.append(window.copy())
                self._speech_audio_samples += self.WINDOW_SIZE

        else:
            # 無音検出
            self._silence_samples += self.WINDOW_SIZE

            if self._is_speech:
                # 発話中の音声データも蓄積（無音部分も含める）
                self._speech_audio.append(window.copy())
                self._speech_audio_samples += self.WINDOW_SIZE

                if self._silence_samples >= self._min_silence_samples:
                    # 発話終了
                    duration = self._speech_audio_samples / self.SAMPLE_RATE

                    # 蓄積した音声データを結合してコピー
                    if self._speech_audio:
                        audio_data = np.concatenate(self._speech_audio).copy()
                    else:
                        audio_data = np.array([], dtype=np.float32)

                    self._event_bus.emit(SpeechEndEvent(
                        timestamp=timestamp,
                        duration_sec=duration,
                        audio_data=audio_data,  # float32, 16kHz, mono, 1D
                        source_id=self._last_source_id,
                    ))
                    logger.debug("Speech ended at %.3f (%.1fs)", timestamp, duration)

                    # 状態リセット
                    self._is_speech = False
                    self._speech_start_time = None
                    self._speech_audio.clear()
                    self._speech_audio_samples = 0
            else:
                # 非発話中
                self._speech_samples = 0
