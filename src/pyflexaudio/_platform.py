import sys

IS_WINDOWS: bool = sys.platform == "win32"
IS_MACOS: bool = sys.platform == "darwin"
IS_LINUX: bool = sys.platform.startswith("linux")

__all__ = ["IS_WINDOWS", "IS_MACOS", "IS_LINUX"]
