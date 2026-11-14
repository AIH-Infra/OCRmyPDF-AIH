# SPDX-License-Identifier: MPL-2.0
"""Web UI renders without errors and exposes the hidden-text-layer switch."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Never touch the real ~/.aih-ocrmypdf-webui.json."""
    monkeypatch.setenv("AIH_OCRMYPDF_CONFIG", str(tmp_path / "default-cfg.json"))


WEBUI = Path(__file__).resolve().parents[1] / "src" / "ocrmypdf" / "webui.py"


def _box(at, label):
    return next(c for c in at.sidebar.checkbox if "隐形文字层" in c.label and label in c.label)


def test_renders_and_switch_follows_mode():
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    assert not at.exception
    box = _box(at, "移除")
    assert box.value is True and box.disabled is False  # default mode is redo

    mode = next(s for s in at.sidebar.selectbox if s.label == "处理模式")
    mode.set_value("skip").run()
    assert _box(at, "移除").disabled is True  # not applicable to skip
    mode.set_value("default").run()
    assert _box(at, "移除").disabled is False


def test_project_identity():
    import tomllib

    root = WEBUI.parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["name"] == "aih-ocrmypdf"
    assert "aih-ocrmypdf-webui" in data["project"]["scripts"]
    assert "AIH-OCRmyPDF" in (root / "README.md").read_text(encoding="utf-8")


def test_output_suffix_and_dir_are_persisted(tmp_path, monkeypatch):
    import json

    cfg = tmp_path / "cfg.json"
    monkeypatch.setenv("AIH_OCRMYPDF_CONFIG", str(cfg))
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    suffix = next(t for t in at.sidebar.text_input if t.label == "输出文件名后缀")
    outdir = next(t for t in at.sidebar.text_input if t.label == "输出目录")
    suffix.set_value("_已识别").run()
    next(t for t in at.sidebar.text_input if t.label == "输出目录").set_value(str(tmp_path / "out")).run()
    saved = json.loads(cfg.read_text(encoding="utf-8"))
    assert saved["output_suffix"] == "_已识别" and saved["output_dir"] == str(tmp_path / "out")

    again = AppTest.from_file(str(WEBUI), default_timeout=90).run()  # "next launch"
    assert next(t for t in again.sidebar.text_input if t.label == "输出文件名后缀").value == "_已识别"
    assert next(t for t in again.sidebar.text_input if t.label == "输出目录").value == str(tmp_path / "out")
    assert outdir is not None
