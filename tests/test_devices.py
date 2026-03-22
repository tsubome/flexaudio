import sys
from unittest.mock import MagicMock, patch
import pytest
from pyflexaudio.types import DeviceInfo


# --- sounddevice が利用できる場合 ---

def _make_fake_sd():
    """sounddevice モジュールを模倣するモックオブジェクトを返す"""
    fake_sd = MagicMock()
    fake_sd.query_devices.return_value = [
        {
            "name": "Built-in Microphone",
            "hostapi": 0,
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 44100.0,
        },
        {
            "name": "Built-in Output",
            "hostapi": 0,
            "max_input_channels": 0,
            "max_output_channels": 2,
            "default_samplerate": 44100.0,
        },
        {
            "name": "USB Headset",
            "hostapi": 0,
            "max_input_channels": 1,
            "max_output_channels": 1,
            "default_samplerate": 48000.0,
        },
    ]
    fake_sd.query_hostapis.return_value = {"name": "Core Audio"}
    return fake_sd


def test_list_input_devices_returns_list_of_device_info():
    fake_sd = _make_fake_sd()
    with patch.dict(sys.modules, {"sounddevice": fake_sd}):
        from pyflexaudio import devices as dev_module
        import importlib
        importlib.reload(dev_module)
        result = dev_module.list_input_devices()

    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, DeviceInfo)


def test_list_input_devices_only_includes_input_capable():
    fake_sd = _make_fake_sd()
    with patch.dict(sys.modules, {"sounddevice": fake_sd}):
        from pyflexaudio import devices as dev_module
        import importlib
        importlib.reload(dev_module)
        result = dev_module.list_input_devices()

    # "Built-in Output" は max_input_channels == 0 なので含まれない
    names = [d.name for d in result]
    assert "Built-in Microphone" in names
    assert "USB Headset" in names
    assert "Built-in Output" not in names


def test_list_input_devices_device_info_fields():
    fake_sd = _make_fake_sd()
    with patch.dict(sys.modules, {"sounddevice": fake_sd}):
        from pyflexaudio import devices as dev_module
        import importlib
        importlib.reload(dev_module)
        result = dev_module.list_input_devices()

    assert len(result) >= 1
    dev = result[0]
    assert isinstance(dev.index, int)
    assert isinstance(dev.name, str)
    assert isinstance(dev.host_api, str)
    assert isinstance(dev.max_input_channels, int)
    assert dev.max_input_channels > 0
    assert isinstance(dev.default_sample_rate, int)
    assert isinstance(dev.is_loopback, bool)


def test_list_output_devices_returns_list_of_device_info():
    fake_sd = _make_fake_sd()
    with patch.dict(sys.modules, {"sounddevice": fake_sd}):
        from pyflexaudio import devices as dev_module
        import importlib
        importlib.reload(dev_module)
        result = dev_module.list_output_devices()

    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, DeviceInfo)


def test_list_output_devices_only_includes_output_capable():
    fake_sd = _make_fake_sd()
    with patch.dict(sys.modules, {"sounddevice": fake_sd}):
        from pyflexaudio import devices as dev_module
        import importlib
        importlib.reload(dev_module)
        result = dev_module.list_output_devices()

    names = [d.name for d in result]
    assert "Built-in Output" in names
    assert "USB Headset" in names
    assert "Built-in Microphone" not in names


# --- sounddevice がない場合 ---

def test_list_input_devices_without_sounddevice_returns_empty_list():
    # sounddevice を sys.modules から除いて ImportError を再現
    modules_backup = sys.modules.copy()
    sys.modules.pop("sounddevice", None)
    sys.modules["sounddevice"] = None  # type: ignore[assignment]

    try:
        from pyflexaudio import devices as dev_module
        import importlib
        importlib.reload(dev_module)
        result = dev_module.list_input_devices()
        assert result == []
    finally:
        # 元に戻す
        if "sounddevice" in modules_backup:
            sys.modules["sounddevice"] = modules_backup["sounddevice"]
        else:
            sys.modules.pop("sounddevice", None)


def test_list_output_devices_without_sounddevice_returns_empty_list():
    modules_backup = sys.modules.copy()
    sys.modules.pop("sounddevice", None)
    sys.modules["sounddevice"] = None  # type: ignore[assignment]

    try:
        from pyflexaudio import devices as dev_module
        import importlib
        importlib.reload(dev_module)
        result = dev_module.list_output_devices()
        assert result == []
    finally:
        if "sounddevice" in modules_backup:
            sys.modules["sounddevice"] = modules_backup["sounddevice"]
        else:
            sys.modules.pop("sounddevice", None)


def test_list_input_devices_import_error_mock():
    """ImportError を直接モックして空リストが返ることを確認"""
    with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: (
        (_ for _ in ()).throw(ImportError("No module named 'sounddevice'"))
        if name == "sounddevice"
        else __import__(name, *args, **kwargs)
    )):
        pass  # builtins.__import__ 差し替えは副作用が大きいため別方法を使う

    # patch を使ったシンプルな方法
    with patch.dict(sys.modules, {"sounddevice": None}):  # type: ignore[dict-item]
        from pyflexaudio import devices as dev_module
        import importlib
        importlib.reload(dev_module)
        result = dev_module.list_input_devices()
    assert result == []
