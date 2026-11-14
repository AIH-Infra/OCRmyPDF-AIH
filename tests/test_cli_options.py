# SPDX-License-Identifier: MPL-2.0
"""Option parsing and AIH plugin auto-loading (fast, no OCR engine needed)."""

from __future__ import annotations

import pytest

from ocrmypdf import cli
from ocrmypdf.cli import (
    AIH_OCR_ENGINES,
    _already_has_aih_plugin,
    _wants_aih_plugin,
    get_options_and_plugins,
    get_parser,
)


def _engine_choices() -> list[str]:
    parser = get_parser()
    action = next(a for a in parser._actions if "--ocr-engine" in a.option_strings)
    return list(action.choices)


def test_all_aih_engines_are_valid_choices():
    choices = _engine_choices()
    for engine in ("auto", "tesseract", "none", *AIH_OCR_ENGINES):
        assert engine in choices
    assert AIH_OCR_ENGINES == ["calamari", "paddle_ocr", "chrome_screenai"]


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["in.pdf", "out.pdf"], False),
        (["--ocr-engine", "tesseract", "in.pdf", "out.pdf"], False),
        (["--ocr-engine", "none", "in.pdf", "out.pdf"], False),
        (["--ocr-engine", "calamari", "in.pdf", "out.pdf"], True),
        (["--ocr-engine", "paddle_ocr", "in.pdf", "out.pdf"], True),
        (["--ocr-engine=chrome_screenai", "in.pdf", "out.pdf"], True),
        (["--calamari-url", "http://x", "in.pdf", "out.pdf"], True),
        (["--paddle-ocr-lang", "ch", "in.pdf", "out.pdf"], True),
        (["--chrome-screenai-light", "in.pdf", "out.pdf"], True),
        (["--help"], True),
        (["-h"], True),
        # --ocr-engine as the very last token has no value: nothing to infer
        (["in.pdf", "out.pdf", "--ocr-engine"], False),
    ],
)
def test_wants_aih_plugin(argv, expected):
    assert _wants_aih_plugin(argv) is expected


def test_already_has_aih_plugin_detects_path_and_module_name():
    assert _already_has_aih_plugin([str(cli._AIH_PLUGIN_PATH)])
    assert _already_has_aih_plugin([str(cli._AIH_PLUGIN_PATH.parent / ".." / "ocrmypdf" / cli._AIH_PLUGIN_PATH.name)])
    assert _already_has_aih_plugin(["ocrmypdf.aih_calamari_plugin"])
    assert not _already_has_aih_plugin([])
    assert not _already_has_aih_plugin(["some_other_plugin"])


def test_plugin_file_exists():
    assert cli._AIH_PLUGIN_PATH.is_file()


def test_paddle_engine_loads_plugin_options():
    options, _pm = get_options_and_plugins(
        ["--ocr-engine", "paddle_ocr", "--paddle-ocr-lang", "ch", "in.pdf", "out.pdf"]
    )
    assert options.ocr_engine == "paddle_ocr"
    assert options.extra_attrs["paddle_ocr_lang"] == "ch"
    assert options.extra_attrs["paddle_ocr_version"]


def test_explicit_plugin_is_not_registered_twice():
    # Passing the plugin by hand AND selecting an AIH engine must not raise
    # pluggy's "already registered" error.
    options, _pm = get_options_and_plugins(
        [
            "--plugin",
            str(cli._AIH_PLUGIN_PATH),
            "--ocr-engine",
            "chrome_screenai",
            "in.pdf",
            "out.pdf",
        ]
    )
    assert options.ocr_engine == "chrome_screenai"


def test_plain_tesseract_run_does_not_load_aih_options():
    options, _pm = get_options_and_plugins(["in.pdf", "out.pdf"])
    assert options.ocr_engine == "auto"
    assert "calamari_url" not in options.extra_attrs


def test_unknown_engine_is_rejected():
    with pytest.raises(SystemExit) as exc:
        get_options_and_plugins(["--ocr-engine", "nope", "in.pdf", "out.pdf"])
    assert exc.value.code == 2
