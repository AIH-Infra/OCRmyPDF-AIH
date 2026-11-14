# SPDX-License-Identifier: MPL-2.0
"""scripts/install_paddle.py: device -> build recommendation and safe defaults."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install_paddle.py"


@pytest.fixture(scope="module")
def ip():
    spec = importlib.util.spec_from_file_location("install_paddle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["install_paddle"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "nvidia, system, kind, tag",
    [
        (("RTX 5060 Ti", 13.0), "Windows", "gpu", "cu130"),
        (("RTX 5060 Ti", 13.2), "Linux", "gpu", "cu130"),
        (("A100", 12.9), "Linux", "gpu", "cu129"),
        (("RTX 4090", 12.8), "Windows", "gpu", "cu126"),
        (("RTX 3060", 12.6), "Windows", "gpu", "cu126"),
        (("GTX 1080", 11.8), "Windows", "cpu", ""),  # driver too old for any build
        (None, "Windows", "cpu", ""),
        (None, "Linux", "cpu", ""),
        (("x", 13.0), "Darwin", "cpu", ""),  # macOS never gets CUDA
    ],
)
def test_plan_follows_the_machine(ip, nvidia, system, kind, tag):
    plan = ip.make_plan(nvidia, system)
    assert plan.kind == kind
    assert (tag in plan.index_url) if tag else (plan.index_url == "")


def test_gpu_install_uses_paddle_index_and_cpu_does_not(ip):
    gpu = ip.make_plan(("RTX 5060 Ti", 13.0), "Windows")
    cpu = ip.make_plan(None, "Windows")
    gpu_cmd = ip.pip_commands(gpu)[-1]
    cpu_cmd = ip.pip_commands(cpu)[-1]
    assert "paddlepaddle-gpu>=3.3,<4" in gpu_cmd and "--extra-index-url" in gpu_cmd
    assert "paddlepaddle>=3.3,<4" in cpu_cmd and "--extra-index-url" not in cpu_cmd
    assert any(arg.startswith("paddleocr") for arg in gpu_cmd + cpu_cmd)


def test_cli_discovers_the_venv_made_by_the_installer(tmp_path, monkeypatch):
    from ocrmypdf import aih_line_detection as ld

    (tmp_path / ".venv-paddle").mkdir()
    monkeypatch.chdir(tmp_path)
    assert tmp_path in ld._project_paddle_roots()
    py_candidates = ld._venv_python_candidates(tmp_path)
    assert any(".venv-paddle" in str(p) for p in py_candidates)


def test_installs_into_separate_venv_not_the_main_one(ip):
    assert ip.VENV_DIR.name == ".venv-paddle"
    first = ip.pip_commands(ip.make_plan(None, "Linux"))[0]
    assert first[1:3] == ["-m", "venv"] and first[-1] == str(ip.VENV_DIR)


def test_non_interactive_default_installs_nothing():
    """Piped stdin + default mode must be a silent-ish no-op, never a download."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input="",
        capture_output=True,
        text=True,
        timeout=60,
        env={**__import__("os").environ, "AIH_INSTALL_PADDLE": "ask"},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "skipped" in proc.stdout.lower() or "already present" in proc.stdout.lower()


def test_skip_mode_is_silent():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "skip"],
        capture_output=True, text=True, timeout=30, check=False,
    )  # fmt: skip
    assert proc.returncode == 0 and proc.stdout == ""


@pytest.mark.parametrize(
    "banner, expected",
    [
        ("| NVIDIA-SMI 551.61   Driver Version: 551.61   CUDA Version: 12.4 |", 12.4),
        ("| NVIDIA-SMI 610.88   KMD Version: 610.88   CUDA UMD Version: 13.3 |", 13.3),
        ("| NVIDIA-SMI 999.99   some future banner without a version |", None),
    ],
)
def test_cuda_version_is_parsed_from_old_and_new_drivers(ip, monkeypatch, banner, expected):
    class Proc:
        stdout = (banner + "\nRTX Test\n").encode()

    monkeypatch.setattr(ip.shutil, "which", lambda _name: "nvidia-smi")
    monkeypatch.setattr(ip.subprocess, "run", lambda *a, **k: Proc())
    found = ip.detect_nvidia()
    assert found is not None and found[1] == expected


def test_unreadable_cuda_version_falls_back_to_cpu_with_a_hint(ip):
    plan = ip.make_plan(("RTX Test", None), "Windows")
    assert plan.kind == "cpu" and "--mode gpu" in plan.reason


def test_detects_the_real_gpu_when_nvidia_smi_exists(ip):
    if not __import__("shutil").which("nvidia-smi"):
        pytest.skip("no nvidia-smi on this machine")
    found = ip.detect_nvidia()
    assert found is not None and found[0]


def test_dry_run_survives_non_utf8_nvidia_smi_output(ip, monkeypatch):
    """Regression: on Chinese Windows nvidia-smi bytes are not valid GBK."""
    sample = b"| NVIDIA-SMI 610.88   CUDA Version: 13.0 |\n\xfe\xfe box drawing \xff\n"

    class Proc:
        stdout = sample

    monkeypatch.setattr(ip.shutil, "which", lambda _name: "nvidia-smi")
    monkeypatch.setattr(ip.subprocess, "run", lambda *a, **k: Proc())
    name, cuda = ip.detect_nvidia()
    assert cuda == 13.0 and name
