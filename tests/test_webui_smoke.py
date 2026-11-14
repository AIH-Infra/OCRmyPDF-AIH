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
    from ocrmypdf import aih_chrome_screenai

    monkeypatch.setattr(aih_chrome_screenai, "scan_screenai_environment", lambda: {
        "directory": str(tmp_path / "ScreenAI"), "version": "148.7", "library": "screenai",
        "model_file_count": 1, "missing": [], "error": None, "search_roots": [],
    })


WEBUI = Path(__file__).resolve().parents[1] / "src" / "ocrmypdf" / "webui.py"


def _box(at, label):
    return next(c for c in at.checkbox if "隐形文字层" in c.label and label in c.label)


def test_renders_and_switch_follows_mode():
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    assert not at.exception
    at.selectbox(key="backend").set_value("Tesseract").run()
    box = _box(at, "移除")
    assert box.value is True and box.disabled is False  # default mode is redo

    mode = next(s for s in at.selectbox if s.label == "处理模式")
    mode.set_value("skip").run()
    assert _box(at, "移除").disabled is True  # not applicable to skip
    mode.set_value("default").run()
    assert _box(at, "移除").disabled is False


def test_project_identity():
    import tomllib

    root = WEBUI.parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["name"] == "aih-ocr"
    assert "aih-ocr-webui" in data["project"]["scripts"]
    assert "AIH-OCR" in (root / "README.md").read_text(encoding="utf-8")


def test_output_suffix_and_dir_are_persisted(tmp_path, monkeypatch):
    import json

    cfg = tmp_path / "cfg.json"
    monkeypatch.setenv("AIH_OCRMYPDF_CONFIG", str(cfg))
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    suffix = next(t for t in at.text_input if t.label == "输出文件名后缀")
    outdir = next(t for t in at.text_input if t.label == "输出目录")
    suffix.set_value("_已识别").run()
    next(t for t in at.text_input if t.label == "输出目录").set_value(str(tmp_path / "out")).run()
    assert not cfg.exists()  # edits remain unsaved until the explicit action
    next(b for b in at.button if b.label == "保存配置").click().run()
    assert not at.exception
    saved = json.loads(cfg.read_text(encoding="utf-8"))
    assert saved["output_suffix"] == "_已识别" and saved["output_dir"] == str(tmp_path / "out")

    again = AppTest.from_file(str(WEBUI), default_timeout=90).run()  # "next launch"
    assert next(t for t in again.text_input if t.label == "输出文件名后缀").value == "_已识别"
    assert next(t for t in again.text_input if t.label == "输出目录").value == str(tmp_path / "out")
    assert outdir is not None


def test_redo_disables_incompatible_preprocessing():
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    assert next(c for c in at.checkbox if c.label == "最终输出也清理").disabled
    assert next(c for c in at.checkbox if c.label == "去除背景").disabled
    next(s for s in at.selectbox if s.label == "处理模式").set_value("force").run()
    assert not next(c for c in at.checkbox if c.label == "去除背景").disabled


@pytest.mark.parametrize("backend", [
    "Tesseract", "PaddleOCR（本地全链路）", "Calamari（HTTP服务）",
    "Chrome ScreenAI（本地）", "无（仅 PDF/A 转换）",
])
def test_backend_configuration_renders(backend, monkeypatch):
    import requests

    def offline(*args, **kwargs):
        raise requests.ConnectionError("test service offline")

    monkeypatch.setattr(requests, "get", offline)
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    next(s for s in at.selectbox if s.label == "OCR 后端").set_value(backend).run()
    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["识别", "文档", "输出", "性能"]


def test_saved_result_still_offers_download(tmp_path):
    output = tmp_path / "saved.pdf"
    output.write_bytes(b"%PDF-1.7\n%%EOF")
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    state = at.session_state["ocr"]
    state.update(done=True, running=False, status="completed", progress=1.0,
                 progress_text="全部文件处理完成", output_dir=str(tmp_path),
                 outputs=[("saved.pdf", str(output))])
    at.run()
    assert not at.exception
    assert len(at.get("download_button")) == 1


def test_invalid_config_reports_error(tmp_path, monkeypatch):
    cfg = tmp_path / "broken.json"
    cfg.write_text("{invalid", encoding="utf-8")
    monkeypatch.setenv("AIH_OCRMYPDF_CONFIG", str(cfg))
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    assert not at.exception
    assert any("配置读取失败" in e.value for e in at.error)
    assert cfg.read_text(encoding="utf-8") == "{invalid"


def test_custom_performance_and_recognition_restore(tmp_path, monkeypatch):
    cfg = tmp_path / "saved.json"
    monkeypatch.setenv("AIH_OCRMYPDF_CONFIG", str(cfg))
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    at.selectbox(key="backend").set_value("Tesseract").run()
    next(r for r in at.radio if r.label == "性能档位").set_value("高级自定义").run()
    next(s for s in at.slider if s.label == "页面并行数（每文件）").set_value(1).run()
    next(b for b in at.button if b.label == "保存配置").click().run()
    again = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    assert not again.exception
    assert again.selectbox(key="backend").value == "Tesseract"
    assert next(r for r in again.radio if r.label == "性能档位").value == "高级自定义"
    assert next(s for s in again.slider if s.label == "页面并行数（每文件）").value == 1


def test_tiers_default_middle_and_save_high(tmp_path, monkeypatch):
    cfg = tmp_path / "tiers.json"
    monkeypatch.setenv("AIH_OCRMYPDF_CONFIG", str(cfg))
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    tier = next(r for r in at.radio if r.label == "性能档位")
    assert tier.options == ["低", "中", "高", "高级自定义"]
    assert tier.value == "中"
    tier.set_value("高").run()
    next(b for b in at.button if b.label == "保存配置").click().run()
    again = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    assert not again.exception
    assert next(r for r in again.radio if r.label == "性能档位").value == "高"


@pytest.mark.parametrize("old,expected", [("自动推荐", "中"), ("自定义", "高级自定义")])
def test_old_performance_settings_migrate(tmp_path, monkeypatch, old, expected):
    import json
    cfg = tmp_path / "old.json"
    cfg.write_text(json.dumps({"settings": {"backend": "Tesseract",
        "performance_Tesseract_mode": old, "performance_Tesseract_jobs": 1}}), encoding="utf-8")
    monkeypatch.setenv("AIH_OCRMYPDF_CONFIG", str(cfg))
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    assert not at.exception
    assert next(r for r in at.radio if r.label == "性能档位").value == expected
    if expected == "高级自定义":
        assert next(s for s in at.slider if s.label == "页面并行数（每文件）").value == 1


def test_calamari_only_detects_paddle_for_paddle_line_detector(tmp_path, monkeypatch):
    import json
    import requests
    from ocrmypdf import aih_line_detection
    cfg = tmp_path / "calamari.json"
    cfg.write_text(json.dumps({"settings": {"backend": "Calamari（HTTP服务）",
        "calamari_line_detector": "tesseract"}}), encoding="utf-8")
    monkeypatch.setenv("AIH_OCRMYPDF_CONFIG", str(cfg))
    def offline(*args, **kwargs):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(requests, "get", offline)
    calls = []
    monkeypatch.setattr(aih_line_detection, "discover_paddle_env", lambda: calls.append(True))
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    assert not at.exception and calls == []
    assert not any(b.label == "重新检测 Paddle 环境" for b in at.button)
    at.selectbox(key="calamari_line_detector").set_value("paddle_det").run()
    assert not at.exception and calls
    assert any(b.key == "refresh_calamari_paddle" for b in at.button)
    at.selectbox(key="calamari_line_detector").set_value("chrome_screenai").run()
    before = len(calls)
    at.run()
    assert len(calls) == before
    assert not any(b.label == "重新检测 Paddle 环境" for b in at.button)


def test_custom_tesseract_on_single_core_has_no_invalid_slider(monkeypatch):
    from ocrmypdf import aih_webui_performance
    monkeypatch.setattr(aih_webui_performance, "detect_hardware", lambda: {
        "cpu_logical_cores": 1, "memory_available_bytes": 2 * 1024**3, "gpus": []})
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    at.selectbox(key="backend").set_value("Tesseract").run()
    next(r for r in at.radio if r.label == "性能档位").set_value("高级自定义").run()
    assert not at.exception
    assert any("当前设备上限" in c.value for c in at.caption)


def test_screenai_scan_button_refreshes_file_status(monkeypatch):
    from ocrmypdf import aih_chrome_screenai
    calls = []
    def scan():
        calls.append(True)
        return {"directory": "C:/models/148.7", "version": "148.7", "library": "chrome_screen_ai.dll",
            "model_file_count": 12, "missing": [], "error": None, "search_roots": ["C:/models"]}
    monkeypatch.setattr(aih_chrome_screenai, "scan_screenai_environment", scan)
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    assert not at.exception and len(calls) == 1
    assert any("148.7" in item.value and "12" in item.value for item in at.success)
    at.button(key="scan_screenai").click().run()
    assert not at.exception and len(calls) == 2
    at.selectbox(key="backend").set_value("Tesseract").run()
    assert not any(b.label == "重新扫描 ScreenAI" for b in at.button)


def test_screenai_defaults_to_sanitization_preserving_body():
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    assert not at.exception
    assert at.selectbox(key="chrome_screenai_text_cleanup").value == "sanitize"
    box = _box(at, "移除")
    assert box.value is True and box.disabled


@pytest.mark.parametrize("legacy", ["mode_default", "clear_all"])
def test_legacy_screenai_cleanup_migrates_to_preserving_body(tmp_path, monkeypatch, legacy):
    import json
    cfg = tmp_path / "cleanup.json"
    cfg.write_text(json.dumps({"settings": {"chrome_screenai_text_cleanup": legacy}}), encoding="utf-8")
    monkeypatch.setenv("AIH_OCRMYPDF_CONFIG", str(cfg))
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    assert not at.exception
    assert at.selectbox(key="chrome_screenai_text_cleanup").value == "sanitize"


def test_screenai_concurrency_visible_in_all_tiers_and_persisted(tmp_path, monkeypatch):
    cfg = tmp_path / "concurrency.json"
    monkeypatch.setenv("AIH_OCRMYPDF_CONFIG", str(cfg))
    key = "performance_Chrome ScreenAI_chrome_concurrency_mode"
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    assert at.radio(key=key).value == "page_parallel"
    assert len([r for r in at.radio if r.label == "并发策略"]) == 1
    at.radio(key=key).set_value("file_parallel").run()
    for tier in ("低", "中", "高", "高级自定义"):
        next(r for r in at.radio if r.label == "性能档位").set_value(tier).run()
        assert not at.exception and at.radio(key=key).value == "file_parallel"
    assert any(s.label == "并行文件数" for s in at.slider)
    next(b for b in at.button if b.label == "保存配置").click().run()
    again = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    assert not again.exception and again.radio(key=key).value == "file_parallel"
    again.radio(key=key).set_value("page_parallel").run()
    assert any(s.label == "单文件并行页段数" for s in again.slider)
    assert not any(s.label == "并行文件数" for s in again.slider)


def test_old_native_direct_custom_setting_keeps_one_worker(tmp_path, monkeypatch):
    import json
    cfg = tmp_path / "old-native.json"
    cfg.write_text(json.dumps({"settings": {
        "performance_Chrome ScreenAI_level": "高级自定义",
        "performance_Chrome ScreenAI_chrome_concurrency_mode": "native_direct"}}), encoding="utf-8")
    monkeypatch.setenv("AIH_OCRMYPDF_CONFIG", str(cfg))
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    assert not at.exception
    assert at.radio(key="performance_Chrome ScreenAI_chrome_concurrency_mode").value == "page_parallel"
    assert next(s for s in at.slider if s.label == "单文件并行页段数").value == 1


@pytest.mark.parametrize("available,expected", [(False, "Tesseract"), (True, "Chrome ScreenAI（本地）")])
def test_fresh_install_backend_follows_screenai_files(monkeypatch, available, expected):
    import ocrmypdf.aih_chrome_screenai as screenai

    monkeypatch.setattr(screenai, "scan_screenai_environment", lambda: {
        "directory": None, "version": None, "library": "screenai",
        "model_file_count": 1 if available else 0, "missing": [],
        "error": None if available else "未发现 ScreenAI 组件", "search_roots": [],
    })
    at = AppTest.from_file(str(WEBUI), default_timeout=90).run()
    assert not at.exception
    assert at.selectbox(key="backend").value == expected
