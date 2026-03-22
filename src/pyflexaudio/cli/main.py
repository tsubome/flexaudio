from __future__ import annotations

import argparse
import sys

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pyflexaudio",
        description="pyflexaudio — Flexible cross-platform audio capture library",
    )
    subparsers = parser.add_subparsers(dest="command")

    # devices コマンド
    subparsers.add_parser("devices", help="List available audio devices")

    # check コマンド
    subparsers.add_parser("check", help="Check environment and dependencies")

    args = parser.parse_args(argv)

    if args.command == "devices":
        return _cmd_devices()
    elif args.command == "check":
        return _cmd_check()
    else:
        parser.print_help()
        return 0


def _cmd_devices() -> int:
    """デバイス一覧を表示"""
    from pyflexaudio.devices import list_input_devices, list_output_devices, list_loopback_devices

    print("=== Input Devices ===")
    inputs = list_input_devices()
    if inputs:
        for dev in inputs:
            print(f"  [{dev.index}] {dev.name}")
            print(f"      Host API: {dev.host_api}, Channels: {dev.max_input_channels}, Rate: {dev.default_sample_rate}")
    else:
        print("  (none found)")

    print()
    print("=== Output Devices ===")
    outputs = list_output_devices()
    if outputs:
        for dev in outputs:
            print(f"  [{dev.index}] {dev.name}")
            print(f"      Host API: {dev.host_api}, Rate: {dev.default_sample_rate}")
    else:
        print("  (none found)")

    print()
    print("=== Loopback Devices (Windows only) ===")
    loopbacks = list_loopback_devices()
    if loopbacks:
        for dev in loopbacks:
            print(f"  [{dev.index}] {dev.name}")
            print(f"      Channels: {dev.max_input_channels}, Rate: {dev.default_sample_rate}")
    else:
        print("  (none found)")

    return 0


def _cmd_check() -> int:
    """環境診断"""
    import platform
    from pyflexaudio._version import __version__
    from pyflexaudio._platform import IS_WINDOWS, IS_MACOS, IS_LINUX

    print(f"pyflexaudio v{__version__}")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print()

    # 依存ライブラリチェック
    print("=== Dependencies ===")
    _check_dep("numpy")
    _check_dep("sounddevice")
    _check_dep("soxr")
    _check_dep("onnxruntime", optional=True, extra="vad")
    _check_dep("miniaudio", optional=True, extra="mic-fallback")

    if IS_WINDOWS:
        _check_dep("pyaudiowpatch", optional=True, extra="win-system")
    if IS_MACOS:
        _check_dep("ScreenCaptureKit", optional=True, extra="mac")
        _check_dep("AVFoundation", optional=True, extra="mac")

    print()

    # 権限チェック（macOS のみ）
    if IS_MACOS:
        print("=== Permissions (macOS) ===")
        from pyflexaudio.permissions import check_microphone_permission, check_screen_recording_permission
        mic = check_microphone_permission()
        screen = check_screen_recording_permission()
        print(f"  Microphone: {mic.value}")
        print(f"  Screen Recording: {screen.value}")
        print()

    # VAD モデル
    print("=== VAD Model ===")
    from pathlib import Path
    model_path = Path(__file__).parent.parent / "processors" / "models" / "silero_vad.onnx"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  Found: {model_path} ({size_mb:.1f} MB)")
    else:
        print(f"  Not found: {model_path}")

    return 0


def _check_dep(name: str, optional: bool = False, extra: str = "") -> None:
    """依存ライブラリの存在をチェック"""
    try:
        __import__(name)
        print(f"  [OK] {name}")
    except ImportError:
        if optional:
            hint = f" (install with: pip install pyflexaudio[{extra}])" if extra else ""
            print(f"  [--] {name} (optional){hint}")
        else:
            print(f"  [!!] {name} (MISSING)")


if __name__ == "__main__":
    sys.exit(main())
