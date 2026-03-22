from __future__ import annotations

import enum
import logging

from pyflexaudio._platform import IS_MACOS, IS_WINDOWS, IS_LINUX

logger = logging.getLogger("pyflexaudio.permissions")

__all__ = ["PermissionStatus", "check_microphone_permission", "request_microphone_permission",
           "check_screen_recording_permission", "request_screen_recording_permission"]


class PermissionStatus(enum.Enum):
    GRANTED = "granted"
    DENIED = "denied"
    NOT_DETERMINED = "not_determined"
    RESTRICTED = "restricted"


def check_microphone_permission() -> PermissionStatus:
    """マイク権限の状態を確認"""
    if not IS_MACOS:
        return PermissionStatus.GRANTED  # Windows/Linux では常に GRANTED

    try:
        from AVFoundation import AVCaptureDevice

        # AVAuthorizationStatus: 0=NotDetermined, 1=Restricted, 2=Denied, 3=Authorized
        status = AVCaptureDevice.authorizationStatusForMediaType_("soun")  # kCMMediaType_Audio
        return _map_av_status(status)
    except ImportError:
        logger.warning("AVFoundation not available")
        return PermissionStatus.GRANTED  # フォールバック


def request_microphone_permission() -> bool:
    """マイク権限を要求。許可された場合 True を返す"""
    if not IS_MACOS:
        return True

    try:
        import threading
        from AVFoundation import AVCaptureDevice

        result_event = threading.Event()
        granted = [False]

        def handler(was_granted):
            granted[0] = was_granted
            result_event.set()

        AVCaptureDevice.requestAccessForMediaType_completionHandler_("soun", handler)
        result_event.wait(timeout=30.0)
        return granted[0]
    except ImportError:
        return True


def check_screen_recording_permission() -> PermissionStatus:
    """画面収録権限の状態を確認（macOS 10.15+）"""
    if not IS_MACOS:
        return PermissionStatus.GRANTED

    try:
        from Quartz import CGPreflightScreenCaptureAccess

        if CGPreflightScreenCaptureAccess():
            return PermissionStatus.GRANTED
        return PermissionStatus.DENIED
    except ImportError:
        logger.warning("Quartz not available")
        return PermissionStatus.GRANTED


def request_screen_recording_permission() -> bool:
    """画面収録権限を要求（macOS 10.15+）。許可された場合 True を返す"""
    if not IS_MACOS:
        return True

    try:
        from Quartz import CGRequestScreenCaptureAccess
        return bool(CGRequestScreenCaptureAccess())
    except ImportError:
        return True


def _map_av_status(status: int) -> PermissionStatus:
    """AVAuthorizationStatus を PermissionStatus にマッピング"""
    mapping = {
        0: PermissionStatus.NOT_DETERMINED,
        1: PermissionStatus.RESTRICTED,
        2: PermissionStatus.DENIED,
        3: PermissionStatus.GRANTED,
    }
    return mapping.get(status, PermissionStatus.DENIED)
