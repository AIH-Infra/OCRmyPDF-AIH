#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Optional PaddleOCR installer (called from install.bat / install.sh / install.command).

PaddleOCR is NOT part of the default install: Chrome ScreenAI stays the main engine.
This script inspects the machine, recommends a CPU or CUDA build of PaddlePaddle and,
only if the user agrees, installs it into a separate virtual environment
(``.venv-paddle`` next to pyproject.toml). The main environment is never touched, so
numpy/opencv versions used by the rest of the project are not disturbed.
``--ocr-engine paddle_ocr`` finds ``.venv-paddle`` automatically.

Standard library only, so it runs in any Python 3.11+ before anything else is set up.

Usage:
    python scripts/install_paddle.py                 # ask (default; non-interactive: skip)
    python scripts/install_paddle.py --mode auto     # install the recommended build
    python scripts/install_paddle.py --mode cpu|gpu  # force a build
    python scripts/install_paddle.py --mode skip     # do nothing
    python scripts/install_paddle.py --dry-run       # show the plan and commands only

The default mode can also be set with the environment variable AIH_INSTALL_PADDLE
(ask | auto | cpu | gpu | skip).
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv-paddle"

PADDLEOCR_SPEC = "paddleocr>=3.7,<3.8"
CPU_SPEC = "paddlepaddle>=3.3,<4"
GPU_SPEC = "paddlepaddle-gpu>=3.3,<4"
# PyPI only carries an old paddlepaddle-gpu (2.6.x); CUDA builds live on Paddle's own
# index, one per CUDA version. Highest first. RTX 50-series (Blackwell) needs >= 12.8.
GPU_INDEXES = [(13.0, "cu130"), (12.9, "cu129"), (12.6, "cu126")]
INDEX_URL = "https://www.paddlepaddle.org.cn/packages/stable/{tag}/"


@dataclass
class Plan:
    kind: str  # "gpu" or "cpu"
    reason: str
    gpu_name: str = ""
    index_url: str = ""

    def packages(self) -> list[str]:
        return [GPU_SPEC if self.kind == "gpu" else CPU_SPEC, PADDLEOCR_SPEC]


def _run_text(cmd: list[str]) -> str:
    """Run a command and return stdout as text.

    nvidia-smi prints box-drawing bytes that are not valid in the Windows console code
    page (e.g. GBK), so never rely on the locale to decode.
    """
    proc = subprocess.run(cmd, capture_output=True, timeout=20, check=False)
    return (proc.stdout or b"").decode("utf-8", errors="replace")


def detect_nvidia() -> tuple[str, float | None] | None:
    """Return (gpu name, max CUDA version supported by the driver) or None.

    The version is None when nvidia-smi works but its output has no CUDA version.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        text = _run_text([exe])
        name_out = _run_text([exe, "--query-gpu=name", "--format=csv,noheader"])
    except (OSError, subprocess.SubprocessError):
        return None
    # Older drivers print "CUDA Version: 12.4"; newer ones "CUDA UMD Version: 13.3".
    match = re.search(r"CUDA\s+(?:\w+\s+)?Version:\s*(\d+\.\d+)", text)
    name = name_out.strip().splitlines()[0].strip() if name_out.strip() else ""
    if not match and not name:
        return None  # nvidia-smi ran but reported no usable GPU
    return name or "NVIDIA GPU", (float(match.group(1)) if match else None)


def make_plan(
    nvidia: tuple[str, float | None] | None, system: str | None = None
) -> Plan:
    system = system or platform.system()
    if system == "Darwin":
        return Plan("cpu", "macOS has no CUDA; the CPU build is used")
    if nvidia is None:
        return Plan(
            "cpu", "no NVIDIA GPU / nvidia-smi found (AMD and Intel GPUs are not supported)"
        )
    name, cuda = nvidia
    if cuda is None:
        return Plan(
            "cpu",
            f"{name}: could not read the driver's CUDA version from nvidia-smi; "
            "choose the GPU build manually with --mode gpu if you know it is >= 12.6",
            gpu_name=name,
        )
    for minimum, tag in GPU_INDEXES:
        if cuda >= minimum:
            return Plan(
                "gpu",
                f"{name}, driver supports CUDA {cuda}",
                gpu_name=name,
                index_url=INDEX_URL.format(tag=tag),
            )
    return Plan(
        "cpu",
        f"{name}: driver only supports CUDA {cuda} (< 12.6); update the NVIDIA driver "
        "to use the GPU build",
        gpu_name=name,
    )


def venv_python() -> Path:
    sub = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return VENV_DIR.joinpath(*sub)


def paddle_ready() -> bool:
    py = venv_python()
    if not py.exists():
        return False
    probe = subprocess.run(
        [str(py), "-c", "from paddleocr import PaddleOCR; import paddle"],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


def pip_commands(plan: Plan) -> list[list[str]]:
    py = str(venv_python())
    install = [py, "-m", "pip", "install", *plan.packages()]
    if plan.index_url:
        install += ["--extra-index-url", plan.index_url]
    return [
        [sys.executable, "-m", "venv", str(VENV_DIR)],
        [py, "-m", "pip", "install", "--upgrade", "pip", "-q"],
        install,
    ]


def verify(plan: Plan) -> bool:
    code = (
        "import paddle, paddleocr;"
        "print('PaddlePaddle', paddle.__version__, '| PaddleOCR', paddleocr.__version__,"
        " '| CUDA build:', paddle.device.is_compiled_with_cuda())"
    )
    probe = subprocess.run(
        [str(venv_python()), "-c", code], capture_output=True, text=True, check=False
    )
    if probe.returncode != 0:
        print("[Warning] PaddleOCR was installed but cannot be imported:")
        print((probe.stderr or probe.stdout).strip()[-600:])
        return False
    line = probe.stdout.strip().splitlines()[-1]
    print(f"[OK] {line}")
    if plan.kind == "gpu" and "CUDA build: False" in line:
        print("[Warning] The CUDA build was requested but this PaddlePaddle has no CUDA.")
        print("          It will still run, on the CPU.")
    return True


def describe(plan: Plan) -> None:
    label = "GPU (CUDA)" if plan.kind == "gpu" else "CPU"
    print(f"  Recommended build : {label}")
    print(f"  Because           : {plan.reason}")
    print(f"  Packages          : {' '.join(plan.packages())}")
    if plan.index_url:
        print(f"  Extra index       : {plan.index_url}")
    print(f"  Target            : {VENV_DIR}  (separate from the main .venv)")


def ask(plan: Plan) -> Plan | None:
    """Interactive choice. Returns the plan to install, or None to skip."""
    print()
    print("Optional: PaddleOCR engine (--ocr-engine paddle_ocr)")
    print("Chrome ScreenAI remains the main engine; this is only an extra.")
    print("The download can be several hundred MB (CPU) to a few GB (GPU).")
    describe(plan)
    if plan.kind == "gpu":
        prompt = "Install the GPU build? [y = GPU / c = CPU instead / N = skip]: "
    else:
        prompt = "Install the CPU build? [y / N = skip]: "
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        answer = ""
    if answer == "y":
        return plan
    if answer == "c" and plan.kind == "gpu":
        return Plan("cpu", "CPU build chosen by the user")
    return None


def run(cmds: list[list[str]]) -> bool:
    for cmd in cmds:
        print("  $", " ".join(cmd))
        if subprocess.run(cmd, check=False).returncode != 0:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--mode",
        choices=["ask", "auto", "cpu", "gpu", "skip"],
        default=os.environ.get("AIH_INSTALL_PADDLE", "ask").lower(),
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    parser.add_argument("--force", action="store_true", help="reinstall if present")
    args = parser.parse_args(argv)

    if args.mode == "skip":
        return 0
    if args.mode not in {"ask", "auto", "cpu", "gpu"}:
        print(f"[Warning] Unknown AIH_INSTALL_PADDLE value {args.mode!r}; skipping.")
        return 0

    if not args.force and paddle_ready():
        print(f"[Found] PaddleOCR environment already present: {VENV_DIR}")
        return 0

    recommended = make_plan(detect_nvidia())
    if args.mode == "ask":
        if args.dry_run:
            print()
            describe(recommended)
            for cmd in pip_commands(recommended):
                print("  $", " ".join(cmd))
            return 0
        if not sys.stdin.isatty():
            print(
                "[Info] Optional PaddleOCR skipped (non-interactive). "
                "Run: python scripts/install_paddle.py"
            )
            return 0
        plan = ask(recommended)
    elif args.mode == "auto":
        plan = recommended
    elif args.mode == "cpu":
        plan = Plan("cpu", "CPU build requested")
    else:  # gpu
        plan = recommended if recommended.kind == "gpu" else None
        if plan is None:
            print(f"[Error] No usable CUDA GPU: {recommended.reason}")
            return 1

    if plan is None:
        print("[Info] PaddleOCR skipped. Install later with: python scripts/install_paddle.py")
        return 0

    if args.dry_run:
        describe(plan)
        for cmd in pip_commands(plan):
            print("  $", " ".join(cmd))
        return 0

    print(f"[Info] Installing PaddleOCR ({plan.kind.upper()}) into {VENV_DIR.name} ...")
    if not run(pip_commands(plan)):
        print("[Warning] PaddleOCR installation failed. The rest of the project is unaffected.")
        if plan.kind == "gpu":
            print("          Try the CPU build: python scripts/install_paddle.py --mode cpu")
        return 1
    if not verify(plan):
        return 1
    print("[OK] Use it with:  ocrmypdf --ocr-engine paddle_ocr -l eng in.pdf out.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
