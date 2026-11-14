# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ocrmypdf import aih_line_detection as ld


def _usable_probe(**updates):
    result = {
        "python": None,
        "source_dir": None,
        "paddleocr_version": "3.7.0",
        "paddle_version": "3.3.1",
        "cuda_compiled": False,
        "gpu_count": 0,
        "device_available": {"cpu": True, "gpu": False},
        "gpu_error": None,
        "error": None,
    }
    result.update(updates)
    return result


def test_explicit_missing_python_reports_error_without_fallback(tmp_path):
    with pytest.raises(ld.PaddleEnvironmentError, match="Python executable does not exist"):
        ld.discover_paddle_env(str(tmp_path / "missing-python.exe"))


def test_explicit_unusable_environment_reports_probe_error(tmp_path, monkeypatch):
    python = tmp_path / "python.exe"
    python.touch()
    monkeypatch.delenv("AIH_PADDLE_PYTHON", raising=False)
    monkeypatch.delenv("AIH_PADDLE_SOURCE_DIR", raising=False)
    monkeypatch.setattr(
        ld,
        "probe_paddle_environment",
        lambda *_args: _usable_probe(error="PaddleOCR import failed: missing dependency"),
    )

    with pytest.raises(ld.PaddleEnvironmentError, match="missing dependency"):
        ld.discover_paddle_env(str(python), str(tmp_path))


def test_environment_variable_python_is_authoritative(tmp_path, monkeypatch):
    python = tmp_path / "python.exe"
    python.touch()
    monkeypatch.setenv("AIH_PADDLE_PYTHON", str(python))
    monkeypatch.delenv("AIH_PADDLE_SOURCE_DIR", raising=False)
    monkeypatch.setattr(ld, "probe_paddle_environment", lambda *_args: _usable_probe())

    found = ld.discover_paddle_env()
    assert found is not None
    assert found.python == python
    assert found.source == "environment"


def test_project_paddle_venv_is_first_candidate(tmp_path):
    candidates = ld._venv_python_candidates(tmp_path)
    assert ".venv-paddle" in candidates[0].parts


def test_public_probe_returns_json_error_for_missing_python(tmp_path):
    result = ld.probe_paddle_environment(tmp_path / "missing-python.exe", tmp_path)

    assert result["paddleocr_version"] is None
    assert result["paddle_version"] is None
    assert result["cuda_compiled"] is None
    assert result["gpu_count"] is None
    assert result["device_available"] == {"cpu": False, "gpu": False}
    assert "does not exist" in result["error"]
    json.dumps(result)


def test_probe_timeout_returns_error_and_refresh_clears_probe_cache(tmp_path, monkeypatch):
    python = tmp_path / "python.exe"
    python.touch()
    ld.refresh_paddle_environment_cache()

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("python", 30)

    monkeypatch.setattr(ld.subprocess, "run", timeout)
    result = ld.probe_paddle_environment(python, tmp_path)
    assert "probe failed" in result["error"].lower()
    assert ld._probe_paddle_environment_cached.cache_info().currsize == 1

    ld.refresh_paddle_environment_cache()
    assert ld._probe_paddle_environment_cached.cache_info().currsize == 0


def test_probe_extracts_json_fields_without_model_construction(tmp_path, monkeypatch):
    python = tmp_path / "python.exe"
    python.touch()
    payload = _usable_probe(
        paddleocr_version="3.7.0",
        paddle_version="3.3.1",
        cuda_compiled=True,
        gpu_count=1,
        device_available={"cpu": True, "gpu": True},
    )

    def run(_cmd, **kwargs):
        assert kwargs["timeout"] == 30
        assert "TextDetection(**" not in _cmd[-1]
        assert "model.predict" not in _cmd[-1]
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(ld.subprocess, "run", run)
    ld.refresh_paddle_environment_cache()
    result = ld.probe_paddle_environment(python, tmp_path)
    assert result["paddleocr_version"] == "3.7.0"
    assert result["paddle_version"] == "3.3.1"
    assert result["cuda_compiled"] is True
    assert result["gpu_count"] == 1
    assert result["device_available"]["gpu"] is True
    assert result["error"] is None
