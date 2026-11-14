# SPDX-License-Identifier: MPL-2.0
"""End-to-end tests that run the real ``python -m ocrmypdf`` command."""

from __future__ import annotations

import pytest

from conftest import PAGE_TEXT, pdf_text, run_cli


class TestErrorsAndHelp:
    """No OCR engine needed."""

    def test_missing_input_exits_2(self, tmp_path):
        r = run_cli(tmp_path / "nope.pdf", tmp_path / "out.pdf")
        assert r.returncode == 2
        assert "File not found" in r.stderr

    def test_unknown_engine_exits_2(self, sample_pdf, tmp_path):
        r = run_cli("--ocr-engine", "nope", sample_pdf, tmp_path / "out.pdf")
        assert r.returncode == 2
        assert "invalid choice" in r.stderr

    def test_help_lists_aih_engines_and_options(self):
        r = run_cli("--help")
        assert r.returncode == 0
        out = r.stdout_text
        assert "paddle_ocr" in out and "chrome_screenai" in out
        assert "--calamari-url" in out
        assert "--paddle-ocr-lang" in out
        assert "--chrome-screenai-light" in out
        assert "[DEBUG]" not in out

    def test_version(self):
        r = run_cli("--version")
        assert r.returncode == 0
        assert r.stdout_text.strip().startswith("0.9")

    def test_unreachable_calamari_reports_clearly(self, sample_pdf, tmp_path):
        r = run_cli(
            "--ocr-engine", "calamari",
            "--calamari-url", "http://127.0.0.1:9",
            "-l", "eng",
            sample_pdf, tmp_path / "out.pdf",
        )  # fmt: skip
        assert r.returncode == 3  # missing dependency
        assert "Cannot reach the Calamari service" in r.stderr
        assert "http://127.0.0.1:9" in r.stderr
        assert not (tmp_path / "out.pdf").exists()


@pytest.mark.tesseract
class TestTesseract:
    def test_basic_run_creates_text_layer(self, sample_pdf, tmp_path):
        out = tmp_path / "out.pdf"
        r = run_cli("-l", "eng", sample_pdf, out)
        assert r.returncode == 0, r.stderr
        assert PAGE_TEXT in pdf_text(out)

    def test_pdf_on_stdout_is_clean(self, sample_pdf):
        r = run_cli("-l", "eng", sample_pdf, "-")
        assert r.returncode == 0, r.stderr
        assert r.stdout.startswith(b"%PDF-")
        assert PAGE_TEXT in pdf_text(r.stdout)

    def test_sidecar_to_file(self, sample_pdf, tmp_path):
        side = tmp_path / "side.txt"
        r = run_cli("-l", "eng", "--sidecar", side, sample_pdf, tmp_path / "out.pdf")
        assert r.returncode == 0, r.stderr
        assert PAGE_TEXT in side.read_text(encoding="utf-8")

    def test_default_run_prints_no_debug_noise(self, sample_pdf, tmp_path):
        r = run_cli("-l", "eng", sample_pdf, tmp_path / "out.pdf")
        assert "[DEBUG]" not in r.stderr and "[DIAG]" not in r.stderr


@pytest.mark.chrome
class TestChromeScreenAI:
    def test_auto_loads_plugin_and_ocrs(self, sample_pdf, tmp_path):
        out = tmp_path / "out.pdf"
        # No --plugin: choosing the engine must be enough.
        r = run_cli("--ocr-engine", "chrome_screenai", "-l", "eng", sample_pdf, out)
        assert r.returncode == 0, r.stderr
        assert PAGE_TEXT in pdf_text(out)

    def test_stdout_pipe_is_a_clean_pdf(self, sample_pdf):
        r = run_cli("--ocr-engine", "chrome_screenai", "-l", "eng", sample_pdf, "-")
        assert r.returncode == 0, r.stderr
        assert r.stdout.startswith(b"%PDF-"), r.stdout[:80]
        assert PAGE_TEXT in pdf_text(r.stdout)

    def test_explicit_plugin_still_works(self, sample_pdf, tmp_path):
        from ocrmypdf.cli import _AIH_PLUGIN_PATH

        r = run_cli(
            "--plugin", _AIH_PLUGIN_PATH,
            "--ocr-engine", "chrome_screenai",
            "-l", "eng",
            sample_pdf, tmp_path / "out.pdf",
        )  # fmt: skip
        assert r.returncode == 0, r.stderr

    @pytest.mark.parametrize("args, expected", [([], 8), (["-j", "3"], 3)])
    def test_page_worker_count(self, sample_pdf, tmp_path, args, expected):
        r = run_cli(
            "--ocr-engine", "chrome_screenai", "-l", "eng", *args,
            sample_pdf, tmp_path / "out.pdf",
        )  # fmt: skip
        assert r.returncode == 0, r.stderr
        assert f"using {expected} page worker(s)" in r.stderr

    def test_default_run_prints_no_debug_noise(self, sample_pdf, tmp_path):
        r = run_cli(
            "--ocr-engine", "chrome_screenai", "-l", "eng",
            sample_pdf, tmp_path / "out.pdf",
        )  # fmt: skip
        assert "[DEBUG]" not in r.stderr and "[DIAG]" not in r.stderr

    def test_verbose_shows_diagnostics_on_stderr_only(self, sample_pdf):
        r = run_cli(
            "-v", "1", "--ocr-engine", "chrome_screenai", "-l", "eng", sample_pdf, "-"
        )
        assert r.returncode == 0, r.stderr
        assert "[DEBUG]" in r.stderr
        assert r.stdout.startswith(b"%PDF-")


@pytest.mark.paddle
@pytest.mark.slow
class TestPaddleOcr:
    def test_cpu_run(self, sample_pdf, tmp_path, paddle_python):
        out = tmp_path / "out.pdf"
        r = run_cli(
            "--ocr-engine", "paddle_ocr",
            "--paddle-ocr-python", paddle_python,
            "--paddle-ocr-device", "cpu",
            "-l", "eng",
            sample_pdf, out,
        )  # fmt: skip
        assert r.returncode == 0, r.stderr
        assert PAGE_TEXT in pdf_text(out)

    def test_gpu_request_runs_on_any_build(self, sample_pdf, tmp_path, paddle_python):
        """gpu:0 must work on CUDA builds and fall back to CPU on CPU-only builds."""
        out = tmp_path / "out.pdf"
        r = run_cli(
            "--ocr-engine", "paddle_ocr",
            "--paddle-ocr-python", paddle_python,
            "--paddle-ocr-device", "gpu:0",
            "-l", "eng",
            sample_pdf, out,
        )  # fmt: skip
        assert r.returncode == 0, r.stderr
        assert PAGE_TEXT in pdf_text(out)

    def test_missing_paddle_env_gives_install_hint(self, sample_pdf, tmp_path):
        r = run_cli(
            "--ocr-engine", "paddle_ocr",
            "--paddle-ocr-python", tmp_path / "no-such-python.exe",
            "--paddle-ocr-source-dir", tmp_path,
            "-l", "eng",
            sample_pdf, tmp_path / "out.pdf",
        )  # fmt: skip
        # Either a real fallback environment is found (then it succeeds) or the
        # message must tell the user how to install the optional extra.
        if r.returncode != 0:
            assert "aih-ocrmypdf[paddle]" in r.stderr
