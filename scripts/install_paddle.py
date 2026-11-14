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
# index, one per CUDA version. Highest first. The official Blackwell build is cu129.
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
    if re.search(r"\bRTX\s*50\d{2}\b", name, re.IGNORECASE):
        if cuda >= 12.9:
            return Plan(
                "gpu",
                f"{name}, Blackwell requires the official CUDA 12.9 build; driver supports CUDA {cuda}",
                gpu_name=name,
                index_url=INDEX_URL.format(tag="cu129"),
            )
        return Plan(
            "cpu",
            f"{name} is Blackwell; the official Paddle GPU wheel requires CUDA 12.9 or newer, "
            f"but the driver supports CUDA {cuda}. Update the NVIDIA driver for GPU install.",
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


def probe_environment(python: Path, source_dir: Path | None = None) -> dict[str, object]:
    """Inspect package imports and Paddle device support without loading a model."""
    if not python.exists():
        return {
            "python": str(python),
            "source_dir": str(source_dir) if source_dir else None,
            "paddleocr_version": None,
            "paddle_version": None,
            "cuda_compiled": None,
            "gpu_count": None,
            "device_available": {"cpu": False, "gpu": False},
            "gpu_error": None,
            "error": f"Python executable does not exist: {python}",
        }
    code = r'''import importlib.metadata, json
result = {
    "paddleocr_version": None,
    "paddle_version": None,
    "cuda_compiled": None,
    "gpu_count": None,
    "device_available": {"cpu": False, "gpu": False},
    "gpu_error": None,
    "error": None,
}
errors = []
gpu_error = None
try:
    import paddle
    result["paddle_version"] = getattr(paddle, "__version__", None)
    result["cuda_compiled"] = bool(paddle.device.is_compiled_with_cuda())
    result["device_available"]["cpu"] = True
    if result["cuda_compiled"]:
        try:
            result["gpu_count"] = int(paddle.device.cuda.device_count())
            result["device_available"]["gpu"] = result["gpu_count"] > 0
            if result["gpu_count"] == 0:
                gpu_error = "CUDA build found but no GPU device is available"
        except Exception as exc:
            gpu_error = "GPU device probe failed: " + str(exc)
    else:
        result["gpu_count"] = 0
except Exception as exc:
    errors.append("Paddle import failed: " + str(exc))
try:
    import paddleocr
    from paddleocr import PaddleOCR, TextDetection
    result["paddleocr_version"] = getattr(paddleocr, "__version__", None)
    if not result["paddleocr_version"]:
        result["paddleocr_version"] = importlib.metadata.version("paddleocr")
except Exception as exc:
    errors.append("PaddleOCR import failed: " + str(exc))
result["error"] = "; ".join(errors) or None
result["gpu_error"] = gpu_error
print(json.dumps(result, ensure_ascii=False))'''
    env = os.environ.copy()
    if source_dir is not None and source_dir.exists():
        env["PYTHONPATH"] = os.pathsep.join(
            [str(source_dir), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
    try:
        probe = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            env=env,
            cwd=str(source_dir) if source_dir is not None and source_dir.exists() else None,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "python": str(python),
            "source_dir": str(source_dir) if source_dir else None,
            "paddleocr_version": None,
            "paddle_version": None,
            "cuda_compiled": None,
            "gpu_count": None,
            "device_available": {"cpu": False, "gpu": False},
            "gpu_error": None,
            "error": f"Paddle environment probe failed: {exc}",
        }
    if probe.returncode != 0:
        details = (probe.stderr or probe.stdout or "probe process failed").strip()[-1000:]
        return {
            "python": str(python),
            "source_dir": str(source_dir) if source_dir else None,
            "paddleocr_version": None,
            "paddle_version": None,
            "cuda_compiled": None,
            "gpu_count": None,
            "device_available": {"cpu": False, "gpu": False},
            "gpu_error": None,
            "error": details,
        }
    try:
        result = __import__("json").loads(probe.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        return {
            "python": str(python),
            "source_dir": str(source_dir) if source_dir else None,
            "paddleocr_version": None,
            "paddle_version": None,
            "cuda_compiled": None,
            "gpu_count": None,
            "device_available": {"cpu": False, "gpu": False},
            "gpu_error": None,
            "error": f"Paddle environment probe returned invalid JSON: {exc}",
        }
    if not isinstance(result, dict):
        return {
            "python": str(python),
            "source_dir": str(source_dir) if source_dir else None,
            "paddleocr_version": None,
            "paddle_version": None,
            "cuda_compiled": None,
            "gpu_count": None,
            "device_available": {"cpu": False, "gpu": False},
            "gpu_error": None,
            "error": "Paddle environment probe returned an invalid result",
        }
    result["python"] = str(python)
    result["source_dir"] = str(source_dir) if source_dir else None
    return result


def pip_commands(plan: Plan) -> list[list[str]]:
    py = str(venv_python())
    commands = [
        [sys.executable, "-m", "venv", str(VENV_DIR)],
        [py, "-m", "pip", "install", "--upgrade", "pip", "-q"],
        # The explicit install/update action may switch the private venv between
        # mutually exclusive Paddle distributions. Other environments are untouched.
        [py, "-m", "pip", "uninstall", "-y", "paddlepaddle", "paddlepaddle-gpu"],
    ]
    if plan.kind == "gpu":
        commands.extend(
            [
                [
                    py,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    GPU_SPEC,
                    "--index-url",
                    plan.index_url,
                ],
                [py, "-m", "pip", "install", "--upgrade", PADDLEOCR_SPEC],
            ]
        )
    else:
        commands.append(
            [py, "-m", "pip", "install", "--upgrade", CPU_SPEC, PADDLEOCR_SPEC]
        )
    return commands


def verify(plan: Plan) -> bool:
    status = probe_environment(venv_python(), ROOT)
    if status["error"]:
        print("[Error] PaddleOCR environment verification failed:")
        print(str(status["error"])[-1000:])
        return False
    if not _versions_match(status):
        print(
            "[Error] Installed package versions do not match the requested ranges: "
            f"PaddlePaddle {status['paddle_version']} / PaddleOCR {status['paddleocr_version']}"
        )
        return False
    print(
        "[OK] PaddlePaddle",
        status["paddle_version"],
        "| PaddleOCR",
        status["paddleocr_version"],
        "| CUDA build:",
        status["cuda_compiled"],
        "| GPU count:",
        status["gpu_count"],
    )
    if plan.kind == "gpu" and not status["device_available"]["gpu"]:
        print(
            "[Error] The requested GPU build cannot access a CUDA device:",
            status.get("gpu_error") or "no CUDA device is available",
        )
        return False
    if plan.kind == "cpu" and status["cuda_compiled"] is not False:
        print("[Error] CPU installation verification found a CUDA Paddle build.")
        return False
    return True


def environment_matches_plan(status: dict[str, object], plan: Plan) -> bool:
    if status.get("error") is not None:
        return False
    devices = status.get("device_available") or {}
    if not _versions_match(status):
        return False
    if plan.kind == "gpu":
        return (
            status.get("cuda_compiled") is True
            and bool(devices.get("gpu"))
            and int(status.get("gpu_count") or 0) > 0
        )
    return bool(devices.get("cpu")) and status.get("cuda_compiled") is False


def environment_matches_mode(
    status: dict[str, object], plan: Plan, mode: str
) -> bool:
    if mode == "auto" and _environment_has_usable_gpu(status):
        return True
    return environment_matches_plan(status, plan)


def _environment_has_usable_gpu(status: dict[str, object]) -> bool:
    devices = status.get("device_available") or {}
    return (
        status.get("error") is None
        and _versions_match(status)
        and status.get("cuda_compiled") is True
        and bool(devices.get("gpu"))
        and int(status.get("gpu_count") or 0) > 0
    )


def _versions_match(status: dict[str, object]) -> bool:
    paddle_version = _version_tuple(status.get("paddle_version"))
    paddleocr_version = _version_tuple(status.get("paddleocr_version"))
    return not (
        paddle_version is None
        or paddle_version < (3, 3)
        or paddle_version[0] >= 4
        or paddleocr_version is None
        or paddleocr_version < (3, 7)
        or paddleocr_version >= (3, 8)
    )


def _version_tuple(value: object) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    match = re.match(r"^\s*(\d+)\.(\d+)", value)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def describe(plan: Plan) -> None:
    label = "GPU (CUDA)" if plan.kind == "gpu" else "CPU"
    print(f"  Recommended build : {label}")
    print(f"  Because           : {plan.reason}")
    print(f"  Packages          : {' '.join(plan.packages())}")
    if plan.index_url:
        print(f"  Index             : {plan.index_url}")
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
        print("  $", _format_command(cmd))
        if subprocess.run(cmd, check=False).returncode != 0:
            return False
    return True


def _format_command(cmd: list[str]) -> str:
    """Print commands with package specifiers safe to paste into a shell."""
    return " ".join(
        f'"{arg}"' if any(char in arg for char in "<>|&") else arg
        for arg in cmd
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--mode",
        choices=["ask", "auto", "cpu", "gpu", "skip"],
        default=os.environ.get("AIH_INSTALL_PADDLE", "ask").lower(),
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    parser.add_argument(
        "--force",
        "--upgrade",
        action="store_true",
        dest="force",
        help="reinstall or upgrade the requested build if present",
    )
    args = parser.parse_args(argv)

    if args.mode == "skip":
        return 0
    if args.mode not in {"ask", "auto", "cpu", "gpu"}:
        print(f"[Warning] Unknown AIH_INSTALL_PADDLE value {args.mode!r}; skipping.")
        return 0

    recommended = make_plan(detect_nvidia())
    if args.mode == "ask":
        if args.dry_run:
            print()
            describe(recommended)
            for cmd in pip_commands(recommended):
                print("  $", _format_command(cmd))
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
            print("  $", _format_command(cmd))
        return 0

    if not args.force and venv_python().exists():
        current = probe_environment(venv_python(), ROOT)
        current_is_requested = environment_matches_mode(current, plan, args.mode)
        if current_is_requested:
            print(
                "[Found] Reusing a compatible PaddleOCR environment: "
                f"PaddlePaddle {current['paddle_version']} / PaddleOCR "
                f"{current['paddleocr_version']} at {VENV_DIR}"
            )
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
