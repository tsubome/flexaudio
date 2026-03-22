import sys
from unittest.mock import patch, MagicMock
import pytest
from pyflexaudio.cli.main import main


# --- 引数なし / help ---

def test_main_no_args_returns_zero():
    result = main([])
    assert result == 0


def test_main_help_returns_zero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_main_none_argv_returns_zero(monkeypatch):
    # sys.argv を上書きして引数なし相当にする
    monkeypatch.setattr(sys, "argv", ["pyflexaudio"])
    result = main([])
    assert result == 0


# --- devices コマンド ---

def test_main_devices_returns_zero(capsys):
    result = main(["devices"])
    assert result == 0


def test_main_devices_prints_headers(capsys):
    main(["devices"])
    captured = capsys.readouterr()
    assert "Input Devices" in captured.out
    assert "Output Devices" in captured.out


def test_main_devices_no_exception():
    # sounddevice が利用できない環境でも例外を起こさない
    result = main(["devices"])
    assert isinstance(result, int)


# --- check コマンド ---

def test_main_check_returns_zero(capsys):
    result = main(["check"])
    assert result == 0


def test_main_check_prints_version(capsys):
    main(["check"])
    captured = capsys.readouterr()
    assert "pyflexaudio" in captured.out


def test_main_check_prints_python_version(capsys):
    main(["check"])
    captured = capsys.readouterr()
    assert "Python" in captured.out


def test_main_check_prints_dependencies(capsys):
    main(["check"])
    captured = capsys.readouterr()
    assert "Dependencies" in captured.out


def test_main_check_no_exception():
    result = main(["check"])
    assert isinstance(result, int)


# --- 未知コマンド ---

def test_main_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit) as exc_info:
        main(["unknowncommand"])
    assert exc_info.value.code != 0


# --- 戻り値の型 ---

def test_main_returns_int_for_devices():
    result = main(["devices"])
    assert isinstance(result, int)


def test_main_returns_int_for_check():
    result = main(["check"])
    assert isinstance(result, int)
