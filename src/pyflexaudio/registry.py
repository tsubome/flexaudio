from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger("pyflexaudio.registry")

__all__ = ["StreamRegistry"]


class StreamRegistry:
    """1デバイス1ストリームの排他管理"""

    def __init__(self):
        self._lock = threading.Lock()
        self._streams: dict[str, object] = {}  # device_key -> AudioSource

    def acquire(self, device_key: str, factory: Callable[[], object]) -> object:
        """デバイスのストリームを取得。未登録なら factory で作成。

        Args:
            device_key: デバイス識別子。形式: "{source_type}:{device_index_or_pid}"
                例: "microphone:0", "system_audio:default", "process_audio:12345"
            factory: ストリーム作成用のファクトリ関数

        Returns:
            AudioSource インスタンス

        Raises:
            RuntimeError: 同一デバイスが既に使用中の場合
        """
        with self._lock:
            if device_key in self._streams:
                raise RuntimeError(
                    f"Device '{device_key}' is already in use. "
                    "Close the existing stream first."
                )
            source = factory()
            self._streams[device_key] = source
            logger.debug("Acquired stream: %s", device_key)
            return source

    def release(self, device_key: str) -> None:
        """デバイスのストリームを解放。close() は呼ばない（呼び出し側の責務）。

        Args:
            device_key: デバイス識別子
        """
        with self._lock:
            if device_key in self._streams:
                del self._streams[device_key]
                logger.debug("Released stream: %s", device_key)

    def is_acquired(self, device_key: str) -> bool:
        """デバイスが使用中か"""
        with self._lock:
            return device_key in self._streams

    def get(self, device_key: str) -> object | None:
        """使用中のストリームを取得"""
        with self._lock:
            return self._streams.get(device_key)

    def clear(self) -> None:
        """全登録を解除"""
        with self._lock:
            self._streams.clear()
            logger.debug("Registry cleared")
