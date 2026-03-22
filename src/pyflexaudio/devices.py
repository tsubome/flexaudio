from __future__ import annotations

import logging

from pyflexaudio.types import DeviceInfo, AudioProcess

logger = logging.getLogger("pyflexaudio.devices")

__all__ = ["list_input_devices", "list_output_devices", "list_loopback_devices", "list_audio_processes"]


def list_input_devices() -> list[DeviceInfo]:
    """入力デバイス一覧を返す"""
    try:
        import sounddevice as sd
    except ImportError:
        logger.warning("sounddevice not available")
        return []

    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append(DeviceInfo(
                index=i,
                name=dev["name"],
                host_api=sd.query_hostapis(dev["hostapi"])["name"],
                max_input_channels=int(dev["max_input_channels"]),
                default_sample_rate=int(dev["default_samplerate"]),
                is_loopback=False,  # sounddevice doesn't expose loopback info
            ))
    return devices


def list_output_devices() -> list[DeviceInfo]:
    """出力デバイス一覧を返す"""
    try:
        import sounddevice as sd
    except ImportError:
        logger.warning("sounddevice not available")
        return []

    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0:
            devices.append(DeviceInfo(
                index=i,
                name=dev["name"],
                host_api=sd.query_hostapis(dev["hostapi"])["name"],
                max_input_channels=int(dev["max_input_channels"]),
                default_sample_rate=int(dev["default_samplerate"]),
                is_loopback=False,
            ))
    return devices


def list_loopback_devices() -> list[DeviceInfo]:
    """WASAPI ループバックデバイス一覧を返す（Windows のみ）"""
    from pyflexaudio._platform import IS_WINDOWS
    if not IS_WINDOWS:
        return []

    try:
        import pyaudiowpatch as pwa
    except ImportError:
        logger.warning("pyaudiowpatch not available")
        return []

    devices = []
    p = pwa.PyAudio()
    try:
        for loopback in p.get_loopback_device_info_generator():
            devices.append(DeviceInfo(
                index=int(loopback["index"]),
                name=loopback["name"],
                host_api="WASAPI",
                max_input_channels=int(loopback["maxInputChannels"]),
                default_sample_rate=int(loopback["defaultSampleRate"]),
                is_loopback=True,
            ))
    finally:
        p.terminate()
    return devices


def list_audio_processes() -> list[AudioProcess]:
    """音声を出力しているプロセス一覧を返す（OS 依存）"""
    from pyflexaudio._platform import IS_WINDOWS, IS_MACOS

    if IS_WINDOWS:
        return _list_audio_processes_windows()
    elif IS_MACOS:
        return _list_audio_processes_macos()
    return []


def _list_audio_processes_windows() -> list[AudioProcess]:
    """Windows: 音声セッションからプロセス一覧を取得"""
    # Phase 1 では簡易実装（全プロセスではなく、DLL 側の機能に依存）
    return []


def _list_audio_processes_macos() -> list[AudioProcess]:
    """macOS: Core Audio HAL から音声出力プロセスを取得"""
    try:
        from AppKit import NSWorkspace

        processes = []
        workspace = NSWorkspace.sharedWorkspace()
        for app in workspace.runningApplications():
            if app.isActive() or app.activationPolicy() == 0:  # Regular app
                processes.append(AudioProcess(
                    pid=app.processIdentifier(),
                    name=app.localizedName() or "",
                    window_title=app.localizedName() or "",
                ))
        return processes
    except ImportError:
        logger.warning("AppKit not available")
        return []
