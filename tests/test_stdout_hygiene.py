# SPDX-License-Identifier: MPL-2.0
"""Guards for the rule "stdout carries only the PDF" (``ocrmypdf in.pdf -``)."""

from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "src" / "ocrmypdf"

# Modules that run inside the main process and therefore must never write to stdout.
MAIN_PROCESS_MODULES = [
    "aih_calamari_plugin.py",
    "aih_chrome_screenai.py",
    "aih_line_detection.py",
    "aih_text_layer.py",
    "aih_paddle_ocr.py",
    "cli.py",
    "__main__.py",
]
# Worker scripts speak a JSON protocol on stdout to their parent; print() is intended.
WORKER_MODULES = {
    "aih_chrome_chunk_worker.py",
    "aih_chrome_line_worker.py",
    "aih_paddle_line_worker.py",
    "aih_paddle_ocr_worker.py",
    "aih_webui_worker_cli.py",
}


def _stdout_prints(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            and not any(kw.arg == "file" for kw in node.keywords)
        ):
            lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("name", MAIN_PROCESS_MODULES)
def test_no_stdout_print_in_main_process_modules(name):
    assert _stdout_prints(PKG / name) == []


def test_worker_allowlist_is_current():
    """Every aih_* script that prints to stdout must be on the explicit allowlist."""
    offenders = {
        p.name
        for p in PKG.glob("aih_*.py")
        if p.name not in MAIN_PROCESS_MODULES and _stdout_prints(p)
    }
    unexpected = offenders - WORKER_MODULES - {"aih_webui_workers.py"}
    assert not unexpected, f"new stdout printers, classify them: {sorted(unexpected)}"


def test_diag_goes_to_logging_not_stdout(capsys, caplog):
    plugin = importlib.import_module("ocrmypdf.aih_calamari_plugin")
    with caplog.at_level("DEBUG", logger="ocrmypdf.aih"):
        plugin._diag("[DIAG] hello", 42, flush=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert "[DIAG] hello 42" in caplog.text


@pytest.mark.parametrize(
    "module",
    [
        "ocrmypdf.aih_line_detection",
        "ocrmypdf.aih_paddle_ocr_worker",
        "ocrmypdf.aih_paddle_line_worker",
    ],
)
class TestPaddleDeviceFallback:
    @staticmethod
    def _fake_paddle(monkeypatch, cuda: bool):
        fake = types.ModuleType("paddle")
        fake.device = types.SimpleNamespace(is_compiled_with_cuda=lambda: cuda)
        monkeypatch.setitem(sys.modules, "paddle", fake)

    def test_cpu_build_downgrades_gpu_request(self, module, monkeypatch, capsys):
        self._fake_paddle(monkeypatch, cuda=False)
        resolve = importlib.import_module(module)._resolve_paddle_device
        assert resolve("gpu:0") == "cpu"
        assert resolve("cuda:1") == "cpu"
        captured = capsys.readouterr()
        assert captured.out == ""  # warning must not pollute stdout / JSON protocol
        assert "no CUDA" in captured.err

    def test_cuda_build_keeps_device(self, module, monkeypatch):
        self._fake_paddle(monkeypatch, cuda=True)
        resolve = importlib.import_module(module)._resolve_paddle_device
        assert resolve("gpu:0") == "gpu:0"

    def test_non_gpu_devices_pass_through(self, module, monkeypatch):
        self._fake_paddle(monkeypatch, cuda=False)
        resolve = importlib.import_module(module)._resolve_paddle_device
        assert resolve("cpu") == "cpu"
        assert resolve("auto") == "auto"
        assert resolve("") == ""
