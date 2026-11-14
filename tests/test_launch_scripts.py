# SPDX-License-Identifier: MPL-2.0
"""Static guards for the install/start scripts (found during acceptance testing)."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BAT_FILES = sorted(ROOT.glob("*.bat"))
UNIX_SCRIPTS = [ROOT / n for n in ("install.sh", "start.sh", "install.command", "start.command")]


def test_bat_files_found():
    assert {p.name for p in BAT_FILES} >= {"install.bat", "start.bat"}


@pytest.mark.parametrize("path", BAT_FILES, ids=lambda p: p.name)
def test_bat_files_use_crlf(path):
    """LF-only batch files can break goto / call :label on Windows."""
    data = path.read_bytes()
    assert data.count(b"\n") == data.count(b"\r\n"), f"{path.name} has bare LF line endings"


def test_gitattributes_keeps_bat_as_crlf():
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.bat text eol=crlf" in text


@pytest.mark.parametrize("path", UNIX_SCRIPTS, ids=lambda p: p.name)
def test_unix_scripts_use_lf(path):
    assert b"\r" not in path.read_bytes()


def test_project_tessdata_is_seeded_with_configs():
    """install.bat creates .tessdata and start.bat uses it as TESSDATA_PREFIX.

    Tesseract also reads configs/ (hocr, txt, ...) from there; a .tessdata holding only
    *.traineddata made every Tesseract OCR fail with "Can't open hocr" (exit code 15).
    """
    install = (ROOT / "install.bat").read_text(encoding="utf-8")
    start = (ROOT / "start.bat").read_text(encoding="utf-8")
    assert ":seed_tessdata_support" in install and "\\configs" in install
    assert "osd.traineddata" in install
    # start.bat must never point TESSDATA_PREFIX at a directory without configs\
    assert 'if exist "%CD%\\.tessdata\\configs" set "TESSDATA_PREFIX=%CD%\\.tessdata"' in start


def test_install_dependency_checks_reach_completion():
    install = (ROOT / "install.bat").read_text(encoding="utf-8")
    main = install.split("call :ensure_tesseract\n", 1)[1].split("\n:ensure_tesseract", 1)[0]
    assert "goto :eof" not in main
    assert main.count("call :seed_tessdata_support") == 1
    assert "call :report_tesseract_languages" in main
    assert "call :ensure_ghostscript" in main
    labels = [line for line in install.splitlines() if line.startswith(":") and not line.startswith("::")]
    assert len(labels) == len(set(labels))
