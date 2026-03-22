"""
Tests for Source dispatch and permissions.

Covers:
  A. AudioSource Protocol conformance
  B. MicrophoneSource dispatch
  C. SystemAudioSource dispatch
  D. ProcessAudioSource dispatch
  E. permissions.py behaviour
"""
from __future__ import annotations

import queue
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

from pyflexaudio.sources.base import AudioSource
from pyflexaudio.sources.mic import MicrophoneSource
from pyflexaudio.sources.system import SystemAudioSource
from pyflexaudio.sources.process import ProcessAudioSource
from pyflexaudio.permissions import (
    PermissionStatus,
    check_microphone_permission,
    request_microphone_permission,
    check_screen_recording_permission,
    request_screen_recording_permission,
    _map_av_status,
)


# ---------------------------------------------------------------------------
# A. AudioSource Protocol conformance
# ---------------------------------------------------------------------------

class TestAudioSourceProtocol:
    def test_microphone_source_is_audio_source(self):
        assert isinstance(MicrophoneSource(), AudioSource)

    def test_system_audio_source_is_audio_source(self):
        assert isinstance(SystemAudioSource(), AudioSource)

    def test_process_audio_source_is_audio_source(self):
        assert isinstance(ProcessAudioSource(pid=1234), AudioSource)


# ---------------------------------------------------------------------------
# B. MicrophoneSource dispatch
# ---------------------------------------------------------------------------

class TestMicrophoneSourceDispatch:
    def test_no_backend_raises_runtime_error(self):
        """sounddevice も miniaudio も使えない場合は RuntimeError"""
        # 両バックエンドモジュールを None にして ImportError を誘発する
        backend_sd = "pyflexaudio.sources._backends.mic_sounddevice"
        backend_ma = "pyflexaudio.sources._backends.mic_miniaudio"
        with patch.dict(sys.modules, {backend_sd: None, backend_ma: None}):
            src = MicrophoneSource()
            with pytest.raises(RuntimeError, match="No microphone backend available"):
                src._create_backend()

    def test_source_id_with_device_index(self):
        src = MicrophoneSource(device_index=2)
        assert src.source_id == "microphone:2"

    def test_source_id_default(self):
        src = MicrophoneSource()
        assert src.source_id == "microphone:default"

    def test_open_idempotent(self):
        """2 回 open() しても例外が出ない（_is_open ガード）"""
        src = MicrophoneSource()
        mock_backend = MagicMock()
        with patch.object(src, "_create_backend", return_value=mock_backend):
            q: queue.Queue = queue.Queue()
            ev = threading.Event()
            src.open(q, ev)
            src.open(q, ev)  # 2 回目は無視される
        assert src.is_open is True
        mock_backend.open.assert_called_once()

    def test_close_idempotent(self):
        """2 回 close() しても例外が出ない"""
        src = MicrophoneSource()
        mock_backend = MagicMock()
        with patch.object(src, "_create_backend", return_value=mock_backend):
            q: queue.Queue = queue.Queue()
            ev = threading.Event()
            src.open(q, ev)
        src.close()
        src.close()  # 2 回目は無視される
        assert src.is_open is False


# ---------------------------------------------------------------------------
# C. SystemAudioSource dispatch
# ---------------------------------------------------------------------------

class TestSystemAudioSourceDispatch:
    def test_source_id_with_device_index(self):
        src = SystemAudioSource(device_index=3)
        assert src.source_id == "system_audio:3"

    def test_source_id_default(self):
        src = SystemAudioSource()
        assert src.source_id == "system_audio:default"

    def test_open_idempotent(self):
        """2 回 open() しても例外が出ない"""
        src = SystemAudioSource()
        mock_backend = MagicMock()
        with patch.object(src, "_create_backend", return_value=mock_backend):
            q: queue.Queue = queue.Queue()
            ev = threading.Event()
            src.open(q, ev)
            src.open(q, ev)
        assert src.is_open is True
        mock_backend.open.assert_called_once()

    def test_close_idempotent(self):
        """2 回 close() しても例外が出ない"""
        src = SystemAudioSource()
        mock_backend = MagicMock()
        with patch.object(src, "_create_backend", return_value=mock_backend):
            q: queue.Queue = queue.Queue()
            ev = threading.Event()
            src.open(q, ev)
        src.close()
        src.close()
        assert src.is_open is False

    def test_linux_raises_not_implemented_error(self):
        """IS_WINDOWS=False, IS_MACOS=False のとき NotImplementedError"""
        with patch("pyflexaudio._platform.IS_WINDOWS", False), \
             patch("pyflexaudio._platform.IS_MACOS", False):
            src = SystemAudioSource()
            with pytest.raises(NotImplementedError):
                src._create_backend()


# ---------------------------------------------------------------------------
# D. ProcessAudioSource dispatch
# ---------------------------------------------------------------------------

class TestProcessAudioSourceDispatch:
    def test_source_id_format(self):
        src = ProcessAudioSource(pid=4242)
        assert src.source_id == "process_audio:4242"

    def test_mode_include_stored(self):
        src = ProcessAudioSource(pid=100, mode="include")
        assert src._mode == "include"

    def test_mode_exclude_stored(self):
        src = ProcessAudioSource(pid=100, mode="exclude")
        assert src._mode == "exclude"

    def test_open_idempotent(self):
        """2 回 open() しても例外が出ない"""
        src = ProcessAudioSource(pid=999)
        mock_backend = MagicMock()
        with patch.object(src, "_create_backend", return_value=mock_backend):
            q: queue.Queue = queue.Queue()
            ev = threading.Event()
            src.open(q, ev)
            src.open(q, ev)
        assert src.is_open is True
        mock_backend.open.assert_called_once()

    def test_close_idempotent(self):
        """2 回 close() しても例外が出ない"""
        src = ProcessAudioSource(pid=999)
        mock_backend = MagicMock()
        with patch.object(src, "_create_backend", return_value=mock_backend):
            q: queue.Queue = queue.Queue()
            ev = threading.Event()
            src.open(q, ev)
        src.close()
        src.close()
        assert src.is_open is False

    def test_linux_raises_not_implemented_error(self):
        """IS_WINDOWS=False, IS_MACOS=False のとき NotImplementedError"""
        with patch("pyflexaudio._platform.IS_WINDOWS", False), \
             patch("pyflexaudio._platform.IS_MACOS", False):
            src = ProcessAudioSource(pid=1)
            with pytest.raises(NotImplementedError):
                src._create_backend()


# ---------------------------------------------------------------------------
# E. permissions.py
# ---------------------------------------------------------------------------

class TestPermissions:
    # --- check_microphone_permission ---

    def test_check_microphone_permission_non_macos_returns_granted(self):
        with patch("pyflexaudio.permissions.IS_MACOS", False):
            assert check_microphone_permission() == PermissionStatus.GRANTED

    def test_check_microphone_permission_avfoundation_import_error_returns_granted(self):
        """macOS 扱いだが AVFoundation がない → GRANTED フォールバック"""
        with patch("pyflexaudio.permissions.IS_MACOS", True), \
             patch.dict(sys.modules, {"AVFoundation": None}):
            result = check_microphone_permission()
            assert result == PermissionStatus.GRANTED

    # --- request_microphone_permission ---

    def test_request_microphone_permission_non_macos_returns_true(self):
        with patch("pyflexaudio.permissions.IS_MACOS", False):
            assert request_microphone_permission() is True

    # --- check_screen_recording_permission ---

    def test_check_screen_recording_permission_non_macos_returns_granted(self):
        with patch("pyflexaudio.permissions.IS_MACOS", False):
            assert check_screen_recording_permission() == PermissionStatus.GRANTED

    def test_check_screen_recording_permission_quartz_import_error_returns_granted(self):
        """macOS 扱いだが Quartz がない → GRANTED フォールバック"""
        with patch("pyflexaudio.permissions.IS_MACOS", True), \
             patch.dict(sys.modules, {"Quartz": None}):
            result = check_screen_recording_permission()
            assert result == PermissionStatus.GRANTED

    # --- request_screen_recording_permission ---

    def test_request_screen_recording_permission_non_macos_returns_true(self):
        with patch("pyflexaudio.permissions.IS_MACOS", False):
            assert request_screen_recording_permission() is True

    # --- _map_av_status ---

    def test_map_av_status_0_not_determined(self):
        assert _map_av_status(0) == PermissionStatus.NOT_DETERMINED

    def test_map_av_status_1_restricted(self):
        assert _map_av_status(1) == PermissionStatus.RESTRICTED

    def test_map_av_status_2_denied(self):
        assert _map_av_status(2) == PermissionStatus.DENIED

    def test_map_av_status_3_granted(self):
        assert _map_av_status(3) == PermissionStatus.GRANTED

    def test_map_av_status_out_of_range_returns_denied(self):
        assert _map_av_status(99) == PermissionStatus.DENIED
