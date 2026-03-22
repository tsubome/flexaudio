from __future__ import annotations

import logging
import queue
import threading

from pyflexaudio.types import AudioChunk

logger = logging.getLogger("pyflexaudio.sources.mic")


__all__ = ["MicrophoneSource"]


class MicrophoneSource:
    """マイク入力ソース。OS に応じたバックエンドを使用"""

    def __init__(
        self,
        device_index: int | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
        blocksize: int = 1024,
    ):
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._channels = channels
        self._blocksize = blocksize
        self._backend = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def source_id(self) -> str:
        idx = self._device_index if self._device_index is not None else "default"
        return f"microphone:{idx}"

    def open(self, data_queue: queue.Queue[AudioChunk | None], stop_event: threading.Event) -> None:
        if self._is_open:
            return
        self._backend = self._create_backend()
        self._backend.open(data_queue, stop_event)
        self._is_open = True

    def close(self) -> None:
        if not self._is_open:
            return
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        self._is_open = False

    def _create_backend(self):
        """バックエンドを選択して作成"""
        # プライマリ: sounddevice
        try:
            from pyflexaudio.sources._backends.mic_sounddevice import SounddeviceMicBackend
            return SounddeviceMicBackend(
                device_index=self._device_index,
                sample_rate=self._sample_rate,
                channels=self._channels,
                blocksize=self._blocksize,
            )
        except ImportError:
            logger.info("sounddevice not available, trying miniaudio fallback")
        except Exception as e:
            # PortAudioError やその他の初期化エラー時もフォールバック
            logger.info("sounddevice backend failed (%s: %s), trying miniaudio fallback", type(e).__name__, e)

        # フォールバック: miniaudio
        try:
            from pyflexaudio.sources._backends.mic_miniaudio import MiniaudioMicBackend
            return MiniaudioMicBackend(
                device_index=self._device_index,
                sample_rate=self._sample_rate,
                channels=self._channels,
            )
        except ImportError:
            pass
        except Exception as e:
            logger.info("miniaudio backend failed (%s: %s)", type(e).__name__, e)

        raise RuntimeError("No microphone backend available. Install sounddevice or miniaudio.")
