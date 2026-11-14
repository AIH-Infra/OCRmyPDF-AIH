"""Small, explainable hardware-based defaults for the Web UI."""

from __future__ import annotations

import csv
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from typing import Any


_GIB = 1024**3


def _as_positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def _sysconf_bytes(name: str) -> int | None:
    try:
        value = os.sysconf(name)
    except (AttributeError, OSError, ValueError):
        return None
    return _as_positive_int(value)


def _linux_memory() -> tuple[int | None, int | None]:
    page_size = _sysconf_bytes("SC_PAGE_SIZE")
    total_pages = _sysconf_bytes("SC_PHYS_PAGES")
    available_pages = _sysconf_bytes("SC_AVPHYS_PAGES")
    total = page_size * total_pages if page_size and total_pages else None
    available = page_size * available_pages if page_size and available_pages else None

    try:
        with open("/proc/meminfo", encoding="ascii") as memory_info:
            for line in memory_info:
                if line.startswith("MemAvailable:"):
                    match = re.search(r"(\d+)\s+kB", line)
                    if match:
                        available = int(match.group(1)) * 1024
                    break
    except OSError:
        pass
    return total, available


def _windows_memory() -> tuple[int | None, int | None]:
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return _as_positive_int(status.ullTotalPhys), _as_positive_int(status.ullAvailPhys)
    except (AttributeError, OSError, ImportError):
        pass
    return None, None


def _run_system_query(*args: str) -> str | None:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _macos_memory() -> tuple[int | None, int | None]:
    total = None
    for name in ("SC_PHYS_PAGES",):
        page_size = _sysconf_bytes("SC_PAGE_SIZE")
        page_count = _sysconf_bytes(name)
        if page_size and page_count:
            total = page_size * page_count
            break
    if total is None:
        output = _run_system_query("sysctl", "-n", "hw.memsize")
        if output:
            total = _as_positive_int(output.strip())

    page_size = _sysconf_bytes("SC_PAGE_SIZE")
    available_pages = _sysconf_bytes("SC_AVPHYS_PAGES")
    available = page_size * available_pages if page_size and available_pages else None
    return total, available


def _psutil_memory() -> tuple[int | None, int | None]:
    try:
        import psutil  # type: ignore[import-not-found]

        memory = psutil.virtual_memory()
    except (ImportError, AttributeError, OSError):
        return None, None
    return _as_positive_int(memory.total), _as_positive_int(memory.available)


def _memory_info() -> tuple[int | None, int | None]:
    system = platform.system()
    if system == "Windows":
        total, available = _windows_memory()
    elif system == "Linux":
        total, available = _linux_memory()
    elif system == "Darwin":
        total, available = _macos_memory()
    else:
        total, available = None, None

    if total is None or available is None:
        psutil_total, psutil_available = _psutil_memory()
        total = total if total is not None else psutil_total
        available = available if available is not None else psutil_available
    return total, available


def _detect_nvidia_gpus() -> list[dict[str, Any]] | None:
    """Read NVIDIA device metadata from nvidia-smi; never initializes a GPU."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None

    query = _run_system_query(
        executable,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    )
    if query is None:
        return None
    banner = _run_system_query(executable)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", banner or "", re.IGNORECASE)
    cuda_version = cuda_match.group(1) if cuda_match else None

    gpus: list[dict[str, Any]] = []
    for row in csv.reader(query.splitlines(), skipinitialspace=True):
        if len(row) < 3:
            continue
        memory_mib = _as_positive_int(row[1])
        gpus.append(
            {
                "name": row[0].strip() or None,
                "vram_bytes": memory_mib * 1024**2 if memory_mib else None,
                "driver_version": row[2].strip() or None,
                "cuda_supported": bool(cuda_version) if cuda_version else None,
                "cuda_version": cuda_version,
            }
        )
    return gpus


def detect_hardware() -> dict[str, Any]:
    """Return local hardware facts; unavailable measurements are ``None``."""
    try:
        logical_cores = _as_positive_int(os.cpu_count())
    except (OSError, TypeError):
        logical_cores = None

    physical_cores = None
    try:
        import psutil  # type: ignore[import-not-found]

        physical_cores = _as_positive_int(psutil.cpu_count(logical=False))
    except (ImportError, AttributeError, OSError):
        pass

    total_memory, available_memory = _memory_info()
    return {
        "cpu_logical_cores": logical_cores,
        "cpu_physical_cores": physical_cores,
        "memory_total_bytes": total_memory,
        "memory_available_bytes": available_memory,
        "is_32bit": sys.maxsize <= 2**32,
        "gpus": _detect_nvidia_gpus(),
    }


def _backend_kind(backend: Any) -> str:
    name = str(backend or "").strip().casefold()
    if name.startswith("chrome screenai") or name.startswith("screenai"):
        return "screenai"
    if name.startswith("paddleocr") or name.startswith("paddle"):
        return "paddle"
    if name.startswith("calamari"):
        return "calamari"
    if name.startswith("tesseract"):
        return "tesseract"
    if name.startswith("无") or name.startswith("none") or name.startswith("pdf/a"):
        return "conversion"
    return "unknown"


def _is_gpu_device(device: Any) -> bool:
    value = str(device or "").strip().casefold()
    return value.startswith(("gpu", "cuda"))


def _memory_worker_limit(
    available_bytes: int | None,
    is_32bit: bool | None,
    worker_memory_bytes: int,
) -> tuple[int, str | None]:
    if is_32bit is True:
        return 1, "32 位系统采用单任务保守设置"
    if available_bytes is None:
        return 1, "可用内存未知，建议单任务运行"
    if available_bytes < 2 * _GIB:
        return 1, "可用内存低于 2 GiB，建议单任务运行"
    return max(1, available_bytes // worker_memory_bytes), None


def recommend_performance(
    backend: str,
    hardware: Mapping[str, Any] | None,
    *,
    device: str = "auto",
    file_count: int = 0,
    level: str = "medium",
    tesseract_oem: int = 1,
    language_count: int = 1,
    screenai_light: bool = False,
    screenai_concurrency: str = "page_parallel",
) -> dict[str, Any]:
    """Recommend conservative worker defaults from supplied hardware facts.

    ``device`` should be the effective Paddle device resolved by the caller.
    An ``auto`` value is deliberately treated as CPU because host NVIDIA
    metadata alone does not establish that the installed Paddle environment
    can use CUDA.
    """
    facts = hardware if isinstance(hardware, Mapping) else {}
    backend_name = _backend_kind(backend)
    cpu_cores = _as_positive_int(facts.get("cpu_logical_cores"))
    available_bytes = _as_positive_int(facts.get("memory_available_bytes"))
    is_32bit = facts.get("is_32bit") if isinstance(facts.get("is_32bit"), bool) else None
    try:
        selected_files = max(1, int(file_count))
    except (TypeError, ValueError, OverflowError):
        selected_files = 1

    cpu_fraction, memory_fraction = {
        "low": (0.25, 0.4), "medium": (0.5, 0.6), "high": (0.85, 0.8),
    }[level]
    reasons: list[str] = [{"low": "低档：减少资源占用", "medium": "中档：平衡并行处理与系统余量", "high": "高档：提高资源预算"}[level]]
    if cpu_cores is None:
        cpu_budget = 1
        reasons.append("CPU 逻辑核数未知，建议仅运行 1 项并行任务")
    else:
        cpu_budget = max(1, min(cpu_cores - 1, int(cpu_cores * cpu_fraction)))
        reasons.append(f"按 {cpu_cores} 个逻辑核保留系统余量，CPU 并行预算为 {cpu_budget}")

    if backend_name == "screenai":
        per_worker_memory = (2 if screenai_light else 3) * _GIB
        reasons.append("ScreenAI 轻量模式" if screenai_light else "ScreenAI 标准模式")
    elif backend_name == "paddle":
        per_worker_memory = 2 * _GIB
    elif backend_name == "tesseract":
        model_load = min(4, max(1, language_count)) * (2 if tesseract_oem == 2 else 1)
        per_worker_memory = int(1.5 * _GIB * model_load)
        reasons.append(f"按 OEM {tesseract_oem}、{language_count} 种语言估算模型内存预算")
    else:
        per_worker_memory = _GIB
    memory_budget, memory_note = _memory_worker_limit(
        int(available_bytes * memory_fraction) if available_bytes is not None else None, is_32bit, per_worker_memory
    )
    if memory_note:
        reasons.append(memory_note)
    elif available_bytes is not None:
        reasons.append(
            f"使用可用内存的 {memory_fraction:.0%} 规划，每项任务估算 {per_worker_memory / _GIB:g} GiB"
        )
    worker_budget = max(1, min(cpu_budget, memory_budget))

    jobs = 1
    file_concurrency = 1
    use_threads = False
    chrome_mode = None
    chrome_page_workers = None
    chrome_file_concurrency = None
    paddle_cpu_threads = None
    paddle_rec_batch_size = None
    preflight_workers = 1
    preflight_chunk = 5

    if backend_name == "tesseract":
        file_concurrency = min(selected_files, worker_budget)
        jobs = max(1, worker_budget // file_concurrency)
        if selected_files > 1:
            reasons.append(
                f"Tesseract 按 {selected_files} 个输入分配文件并发和页并行，并行任务总数不超过 {worker_budget}"
            )
        else:
            reasons.append("Tesseract 单文件优先使用页并行")
    elif backend_name == "screenai":
        use_threads = True
        if screenai_concurrency == "file_parallel":
            chrome_mode = "file_parallel"
            chrome_file_concurrency = min(24, selected_files, worker_budget)
            file_concurrency = chrome_file_concurrency
            reasons.append(
                f"ScreenAI 按 {selected_files} 个输入采用整本并行，每本内部顺序处理"
            )
        else:
            chrome_mode = "page_parallel"
            chrome_page_workers = min(24, worker_budget)
            jobs = chrome_page_workers
            reasons.append("ScreenAI 优先单文件内部页段并行，文件逐个处理，文件并发固定为 1")
    elif backend_name == "paddle":
        if _is_gpu_device(device):
            jobs = 1
            file_concurrency = 1
            paddle_cpu_threads = min(cpu_budget, {"low": 1, "medium": 2, "high": 4}[level])
            gpus = facts.get("gpus") or []
            index_text = str(device).partition(":")[2]
            gpu_index = int(index_text) if index_text.isdecimal() else 0
            vram = _as_positive_int(gpus[gpu_index].get("vram_bytes")) if gpu_index < len(gpus) else None
            batch_limit = 4 if vram is None or vram < 6 * _GIB else 16 if vram < 12 * _GIB else 32
            if available_bytes is None or available_bytes < 4 * _GIB:
                batch_limit = min(batch_limit, 4)
            paddle_rec_batch_size = min({"low": 4, "medium": 16, "high": 32}[level], batch_limit)
            reasons.append((f"GPU 总显存 {vram / _GIB:g} GiB" if vram else "GPU 显存读数未知") + f"，识别批量预算上限 {batch_limit}")
            reasons.append("设备已确认为 GPU，建议单文件、单页处理以减少 GPU 资源争用")
        else:
            jobs = min({"low": 1, "medium": 2, "high": 4}[level], worker_budget)
            paddle_cpu_threads = max(1, min({"low": 2, "medium": 4, "high": 8}[level], cpu_budget // jobs))
            if available_bytes is None or available_bytes < 4 * _GIB or is_32bit is True:
                paddle_rec_batch_size = 2
            elif available_bytes < 8 * _GIB:
                paddle_rec_batch_size = 4
            else:
                paddle_rec_batch_size = {"low": 2, "medium": 4, "high": 8}[level]
            paddle_rec_batch_size = min(paddle_rec_batch_size, {"low": 2, "medium": 4, "high": 8}[level])
            reasons.append(
                "按 CPU 设备规划；"
                f"每页线程数为 {paddle_cpu_threads}，识别批大小为 {paddle_rec_batch_size}"
            )
    elif backend_name == "calamari":
        preflight_workers = min(10, worker_budget)
        preflight_chunk = min({"low": 5, "medium": 10, "high": 20}[level], max(5, memory_budget * 5))
        reasons.append("Calamari 服务建议每次处理一个文件和一个页面，减少服务端资源争用")
    elif backend_name == "conversion":
        jobs = min(4, worker_budget)
        reasons.append("PDF 转换任务采用受 CPU 与内存共同限制的并行度")
    else:
        reasons.append("处理方式未识别，建议单任务保守运行")

    if available_bytes is None:
        reasons.append("可用内存读数未知")
    if cpu_cores is None:
        reasons.append("CPU 核数读数未知")

    return {
        "jobs": int(jobs),
        "file_concurrency": int(file_concurrency),
        "use_threads": bool(use_threads),
        "chrome_concurrency_mode": chrome_mode,
        "chrome_page_workers": chrome_page_workers,
        "chrome_file_concurrency": chrome_file_concurrency,
        "paddle_ocr_cpu_threads": paddle_cpu_threads,
        "paddle_ocr_rec_batch_size": paddle_rec_batch_size,
        "preflight_workers": preflight_workers,
        "preflight_chunk": preflight_chunk,
        "reason": "；".join(reasons),
    }
