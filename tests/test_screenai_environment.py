from __future__ import annotations

import json

import pytest

from ocrmypdf.aih_chrome_screenai import scan_screenai_environment
from ocrmypdf._vendor.locro import _dll, _download, _platform


def _component(path, version="148.7"):
    path.mkdir(parents=True)
    (path / _platform.LIB_NAME).write_bytes(b"library")
    (path / "model.bin").write_bytes(b"model")
    (path / "files_list_ocr.txt").write_text("model.bin\n", encoding="utf-8")
    (path / "manifest.json").write_text(json.dumps({"version": version}), encoding="utf-8")
    return path


@pytest.fixture
def model_roots(tmp_path, monkeypatch):
    chrome = tmp_path / "chrome"
    local = tmp_path / "local"
    monkeypatch.setattr(_dll, "chrome_component_bases", lambda: [chrome])
    monkeypatch.setattr(_platform, "chrome_component_bases", lambda: [chrome])
    monkeypatch.setattr(_platform, "default_model_dir", lambda: local)
    monkeypatch.setattr(_download, "get_default_model_dir", lambda: local)
    def no_load(*args, **kwargs):
        raise AssertionError("scan must not load native libraries")
    monkeypatch.setattr(_dll.ctypes, "CDLL", no_load)
    return chrome, local


def test_scan_selects_same_component_as_runtime_and_checks_catalog(model_roots):
    chrome, local = model_roots
    _component(local / "149.0", "149.0")
    selected = _component(chrome / "148.7")
    status = scan_screenai_environment()
    assert status["directory"] == str(_dll.find_screen_ai_dir().resolve()) == str(selected.resolve())
    assert status["version"] == "148.7"
    assert status["model_file_count"] == 1
    assert status["missing"] == [] and status["error"] is None


def test_scan_finds_local_model_after_install_without_process_restart(model_roots):
    _, local = model_roots
    assert scan_screenai_environment()["error"]
    selected = _component(local / "148.7")
    assert scan_screenai_environment()["directory"] == str(selected.resolve())


def test_scan_reports_empty_and_missing_model_files(model_roots):
    _, local = model_roots
    selected = _component(local / "148.7")
    (selected / "model.bin").write_bytes(b"")
    (selected / "files_list_ocr.txt").write_text("model.bin\nmissing.bin\n", encoding="utf-8")
    status = scan_screenai_environment()
    assert status["missing"] == ["model.bin", "missing.bin"]


@pytest.mark.parametrize("catalog", ["", "../outside.bin"])
def test_scan_reports_invalid_catalog(model_roots, catalog):
    _, local = model_roots
    selected = _component(local / "148.7")
    (selected / "files_list_ocr.txt").write_text(catalog, encoding="utf-8")
    assert scan_screenai_environment()["error"]
