from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from ocrmypdf._vendor.locro import _dll
from ocrmypdf import aih_calamari_plugin as plugin


@pytest.fixture
def version_library(monkeypatch, tmp_path):
    calls = []
    def version(major, minor):
        major._obj.value = 148
        minor._obj.value = 7
    def load(path):
        calls.append(path)
        return SimpleNamespace(GetLibraryVersion=version)
    monkeypatch.setattr(_dll.ctypes, "CDLL", load)
    monkeypatch.setattr(_dll, "find_screen_ai_dir", lambda: tmp_path)
    return calls, tmp_path


def test_native_version_query_keeps_model_state_and_uses_active_library(monkeypatch, version_library):
    calls, base = version_library
    active = base / "active"
    monkeypatch.setattr(_dll, "_model_dir", active)
    cache = {"model": b"active model"}
    monkeypatch.setattr(_dll, "_file_cache", cache)
    assert _dll.get_library_version() == (148, 7)
    assert calls == [str(active / _dll.LIB_NAME)]
    assert _dll._model_dir == active and _dll._file_cache is cache
    assert cache == {"model": b"active model"}


def test_engine_creator_does_not_construct_second_ocr_pipeline(monkeypatch, version_library):
    monkeypatch.setattr(_dll, "_model_dir", None)
    def forbidden(*args, **kwargs):
        raise AssertionError("version metadata must not initialize OCR")
    monkeypatch.setattr(plugin, "ChromeScreenAIBackend", forbidden)
    monkeypatch.setattr(_dll.ScreenAIDll, "init_ocr", forbidden)
    for _ in range(3):
        assert plugin.ChromeScreenAIOcrEngine.version() == "148.7"
        assert plugin.ChromeScreenAIOcrEngine.creator_tag(None) == "Chrome ScreenAI OCR 148.7"


def test_quiet_native_logging_keeps_original_stderr_for_fatal_errors(monkeypatch):
    stderr = io.StringIO()
    monkeypatch.setattr(_dll.sys, "stderr", stderr)
    monkeypatch.setattr(_dll, "_native_suppressed", False)
    monkeypatch.setenv("GLOG_minloglevel", "1")
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "1")
    monkeypatch.setenv("ABSL_MIN_LOG_LEVEL", "1")
    def forbidden(*args, **kwargs):
        raise AssertionError("must not discard native stderr")
    with monkeypatch.context() as scoped:
        scoped.setattr(_dll.os, "dup2", forbidden)
        _dll._suppress_native_stderr()
    assert _dll.sys.stderr is stderr
    assert _dll.os.environ["GLOG_minloglevel"] == "1"
