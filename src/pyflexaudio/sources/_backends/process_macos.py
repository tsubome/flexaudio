"""macOS process audio capture via ScreenCaptureKit."""

from __future__ import annotations

import logging
import queue
import threading

from pyflexaudio.types import AudioChunk  # noqa: F401 — re-exported for type checkers
from pyflexaudio.sources._backends.system_macos import (
    _SCKBaseBackend,
    _StreamOutputDelegate,  # noqa: F401 — shared delegate
)

logger = logging.getLogger("pyflexaudio.sources.process_macos")

__all__ = ["SCKProcessAudioBackend"]


class SCKProcessAudioBackend(_SCKBaseBackend):
    """ScreenCaptureKit によるプロセス別音声キャプチャ（macOS）

    Parameters
    ----------
    pid:
        対象プロセスの PID。
    mode:
        ``"include"`` — 対象プロセスの音声のみキャプチャ。
        ``"exclude"`` — 対象プロセスの音声を除いた全システム音声をキャプチャ。
    """

    def __init__(self, pid: int, mode: str = "include") -> None:
        super().__init__()
        if mode not in ("include", "exclude"):
            raise ValueError(f"mode must be 'include' or 'exclude', got {mode!r}")
        self._pid = pid
        self._mode = mode

    @property
    def source_id(self) -> str:
        return f"process_audio:{self._pid}"

    def _build_content_filter(self, content):
        """PID と mode に応じた SCContentFilter を構築する。"""
        from ScreenCaptureKit import SCContentFilter  # type: ignore[import]

        # 全ディスプレイ（include/exclude どちらのモードでも使用）
        displays = content.displays()
        if not displays:
            raise RuntimeError("No displays found")
        display = displays[0]

        # 対象アプリを PID で検索
        apps = content.applications()
        target_app = None
        for app in apps:
            if app.processID() == self._pid:
                target_app = app
                break

        if target_app is None:
            raise RuntimeError(
                f"Process {self._pid} not found in shareable content. "
                "The process may not have any capturable audio."
            )

        if self._mode == "include":
            # 対象プロセスのみキャプチャ
            # ウィンドウが存在する場合は initWithDesktopIndependentWindow_ も使えるが、
            # ウィンドウレスプロセス（バックグラウンドオーディオ等）を考慮し
            # initWithDisplay_includingApplications_exceptingWindows_ を優先する。
            content_filter = (
                SCContentFilter.alloc()
                .initWithDisplay_includingApplications_exceptingWindows_(
                    display,
                    [target_app],  # includingApplications
                    [],            # exceptingWindows
                )
            )
        else:
            # 対象プロセス以外の全音声をキャプチャ
            content_filter = (
                SCContentFilter.alloc()
                .initWithDisplay_excludingApplications_exceptingWindows_(
                    display,
                    [target_app],  # excludingApplications
                    [],            # exceptingWindows
                )
            )

        return content_filter
