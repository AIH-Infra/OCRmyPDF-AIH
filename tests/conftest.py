# SPDX-License-Identifier: MPL-2.0
"""Shared fixtures for the AIH-OCR CLI tests.

The suite is split in two layers:

* fast, in-process tests that need no OCR engine (option parsing, source hygiene);
* subprocess tests that run the real ``python -m ocrmypdf`` command. Those that need
  an external engine are marked ``tesseract`` / ``chrome`` / ``paddle`` and are
  skipped automatically when the engine is not available on the machine.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
PAGE_TEXT = "Hello OCR test page"


@dataclass
class CliResult:
    returncode: int
    stdout: bytes
    stderr: str

    @property
    def stdout_text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")


def run_cli(*args: str | Path, timeout: int = 300) -> CliResult:
    """Run ``python -m ocrmypdf`` in a fresh interpreter and capture raw output."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_DIR), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(SRC_DIR)]
    )
    proc = subprocess.run(
        [sys.executable, "-m", "ocrmypdf", *map(str, args)],
        capture_output=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    return CliResult(
        proc.returncode, proc.stdout, proc.stderr.decode("utf-8", errors="replace")
    )


def pdf_text(path: Path | bytes) -> str:
    """Return the concatenated text layer of all pages."""
    import fitz

    doc = fitz.open(stream=path, filetype="pdf") if isinstance(path, bytes) else fitz.open(path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    """A 2-page image-only PDF (text rendered into a bitmap, no text layer)."""
    import fitz

    text_doc = fitz.open()
    for i in (1, 2):
        page = text_doc.new_page()
        page.insert_text((72, 100), f"{PAGE_TEXT} {i}. The quick brown fox.", fontsize=24)
    out = fitz.open()
    for page in text_doc:
        pix = page.get_pixmap(dpi=200)
        new = out.new_page(width=595, height=842)
        new.insert_image(new.rect, pixmap=pix)
    path = tmp_path_factory.mktemp("input") / "scan.pdf"
    out.save(path)
    return path


def _has_tesseract() -> bool:
    return bool(
        shutil.which("tesseract")
        or Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe").exists()
    )


@pytest.fixture(scope="session")
def tesseract_available() -> bool:
    return _has_tesseract()


@pytest.fixture(autouse=True)
def _skip_unavailable(request):
    if request.node.get_closest_marker("tesseract") and not _has_tesseract():
        pytest.skip("tesseract executable not found")
    if request.node.get_closest_marker("chrome"):
        request.getfixturevalue("chrome_ok")
    if request.node.get_closest_marker("paddle"):
        request.getfixturevalue("paddle_python")


@pytest.fixture(scope="session")
def chrome_ok(sample_pdf, tmp_path_factory) -> bool:
    """Probe once whether Chrome ScreenAI works here; skip dependants otherwise."""
    out = tmp_path_factory.mktemp("chrome_probe") / "out.pdf"
    result = run_cli("--ocr-engine", "chrome_screenai", "-l", "eng", sample_pdf, out)
    if result.returncode != 0 or PAGE_TEXT not in pdf_text(out):
        pytest.skip("Chrome ScreenAI is not usable on this machine")
    return True


@pytest.fixture(scope="session")
def paddle_python() -> str:
    """Python executable of an environment that can run PaddleOCR, else skip."""
    from ocrmypdf.aih_line_detection import discover_paddle_env

    env = discover_paddle_env()
    if env is None:
        pytest.skip("no PaddleOCR environment found")
    return str(env.python)
