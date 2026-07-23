# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from ocrmypdf.aih_file_lock import file_job_lock


def ocr_one_file_worker(
    file_data: bytes,
    fname: str,
    out_name: str,
    ocr_kwargs: dict,
    backend: str,
    save_to_disk: bool,
    output_dir: str,
) -> dict:
    """Process one uploaded file in a child process.

    Keep this outside webui.py so Windows spawned workers do not import the
    Streamlit page module.
    """
    import builtins
    import ocrmypdf

    worker_state = {
        "page_current": 0,
        "page_total": 0,
        "page_desc": "",
    }
    builtins._ocrmypdf_page_progress_state = worker_state

    start_time = time.time()
    stem = Path(fname).stem
    safe_suffix = f"_{stem[:40]}.pdf"
    tmp_in_path = tmp_out_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=safe_suffix
        ) as tmp_in, tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as tmp_out:
            tmp_in.write(file_data)
            tmp_in.flush()
            tmp_in_path, tmp_out_path = tmp_in.name, tmp_out.name

        plugins = [str(Path(__file__).parent / "aih_progress_plugin.py")]
        if backend in ("Calamari（HTTP服务）", "PaddleOCR（本地全链路）", "Chrome ScreenAI（本地）"):
            plugin_path = Path(__file__).parent / "aih_calamari_plugin.py"
            if plugin_path.exists():
                plugins.append(str(plugin_path))
            else:
                raise FileNotFoundError(f"AIH OCR 插件未找到：{plugin_path}")

        kw = {**ocr_kwargs, "progress_bar": True}
        result = ocrmypdf.ocr(
            tmp_in_path,
            tmp_out_path,
            plugins=plugins,
            **kw,
        )

        elapsed = time.time() - start_time
        elapsed_str = f"{elapsed:.1f}秒"
        out_data = b""
        if Path(tmp_out_path).exists():
            out_data = Path(tmp_out_path).read_bytes()

        payload = {
            "fname": fname,
            "out_name": out_name,
            "elapsed": elapsed,
            "output": None,
            "error": None,
            "log": None,
        }

        if result == ocrmypdf.ExitCode.ok:
            payload["output"] = (out_name, out_data)
            payload["log"] = ("success", fname, f"{fname} -> {out_name} (耗时: {elapsed_str})")
        elif result == ocrmypdf.ExitCode.pdfa_conversion_failed:
            if len(out_data) > 0:
                payload["output"] = (out_name, out_data)
            selected_output_type = kw.get("output_type")
            if selected_output_type == "auto":
                msg = f"{fname}：未能生成 PDF/A，已自动回退为普通 PDF (耗时: {elapsed_str})"
            else:
                msg = f"{fname}：PDF/A 转换失败，输出可能不符合标准 (耗时: {elapsed_str})"
            payload["log"] = ("warning", fname, msg)
        elif result == ocrmypdf.ExitCode.already_done_ocr:
            payload["log"] = ("info", fname, f"{fname}：已包含文字层，跳过 (耗时: {elapsed_str})")
        elif result == ocrmypdf.ExitCode.encrypted_pdf:
            payload["error"] = (fname, "PDF 已加密")
            payload["log"] = ("error", fname, f"{fname}：PDF 已加密，请先解密 (耗时: {elapsed_str})")
        else:
            payload["error"] = (fname, f"退出码：{result.name}")
            payload["log"] = ("error", fname, f"{fname}：处理失败（{result.name}）(耗时: {elapsed_str})")

        if payload["output"] and save_to_disk and output_dir:
            output_path = Path(output_dir) / out_name
            with file_job_lock(str(output_path), label=out_name):
                output_path.write_bytes(out_data)
        return payload
    except Exception as e:
        elapsed = time.time() - start_time
        elapsed_str = f"{elapsed:.1f}秒"
        return {
            "fname": fname,
            "out_name": out_name,
            "elapsed": elapsed,
            "output": None,
            "error": (fname, str(e)),
            "log": ("error", fname, f"{fname}：{e} (耗时: {elapsed_str})"),
        }
    finally:
        for p in (tmp_in_path, tmp_out_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


def ocr_one_file_path_worker(
    input_path: str,
    output_path: str,
    fname: str,
    out_name: str,
    ocr_kwargs: dict,
    backend: str,
) -> dict:
    """Process one file path in an external worker process."""
    import builtins
    import ocrmypdf

    worker_state = {
        "page_current": 0,
        "page_total": 0,
        "page_desc": "",
    }
    builtins._ocrmypdf_page_progress_state = worker_state

    start_time = time.time()
    try:
        plugins = [str(Path(__file__).parent / "aih_progress_plugin.py")]
        if backend in ("Calamari（HTTP服务）", "PaddleOCR（本地全链路）", "Chrome ScreenAI（本地）"):
            plugin_path = Path(__file__).parent / "aih_calamari_plugin.py"
            if plugin_path.exists():
                plugins.append(str(plugin_path))
            else:
                raise FileNotFoundError(f"AIH OCR 插件未找到：{plugin_path}")

        kw = {**ocr_kwargs, "progress_bar": True}
        with file_job_lock(input_path, output_path, label=fname):
            result = ocrmypdf.ocr(
                input_path,
                output_path,
                plugins=plugins,
                **kw,
            )

        elapsed = time.time() - start_time
        elapsed_str = f"{elapsed:.1f}秒"
        output_exists = Path(output_path).exists() and Path(output_path).stat().st_size > 0

        payload = {
            "fname": fname,
            "out_name": out_name,
            "elapsed": elapsed,
            "output_path": output_path if output_exists else None,
            "error": None,
            "log": None,
        }

        if result == ocrmypdf.ExitCode.ok:
            payload["log"] = ("success", fname, f"{fname} -> {out_name} (耗时: {elapsed_str})")
        elif result == ocrmypdf.ExitCode.pdfa_conversion_failed:
            selected_output_type = kw.get("output_type")
            if selected_output_type == "auto":
                msg = f"{fname}：未能生成 PDF/A，已自动回退为普通 PDF (耗时: {elapsed_str})"
            else:
                msg = f"{fname}：PDF/A 转换失败，输出可能不符合标准 (耗时: {elapsed_str})"
            payload["log"] = ("warning", fname, msg)
        elif result == ocrmypdf.ExitCode.already_done_ocr:
            payload["output_path"] = None
            payload["log"] = ("info", fname, f"{fname}：已包含文字层，跳过 (耗时: {elapsed_str})")
        elif result == ocrmypdf.ExitCode.encrypted_pdf:
            payload["output_path"] = None
            payload["error"] = (fname, "PDF 已加密")
            payload["log"] = ("error", fname, f"{fname}：PDF 已加密，请先解密 (耗时: {elapsed_str})")
        else:
            payload["output_path"] = None
            payload["error"] = (fname, f"退出码：{result.name}")
            payload["log"] = ("error", fname, f"{fname}：处理失败（{result.name}）(耗时: {elapsed_str})")

        return payload
    except Exception as e:
        elapsed = time.time() - start_time
        elapsed_str = f"{elapsed:.1f}秒"
        return {
            "fname": fname,
            "out_name": out_name,
            "elapsed": elapsed,
            "output_path": None,
            "error": (fname, str(e)),
            "log": ("error", fname, f"{fname}：{e} (耗时: {elapsed_str})"),
        }
