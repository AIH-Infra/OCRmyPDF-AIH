# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import base64
import html
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocrmypdf.aih_line_detection import discover_paddle_env


@dataclass(frozen=True)
class PaddleOcrSettings:
    ocr_version: str = "PP-OCRv6"
    lang: str = "en"
    det_model_name: str = ""
    rec_model_name: str = ""
    det_model_dir: str = ""
    rec_model_dir: str = ""
    device: str = "gpu:0"
    engine: str = ""
    enable_mkldnn: bool | None = False
    cpu_threads: int | None = None
    enable_hpi: bool | None = None
    rec_batch_size: int | None = 16
    min_confidence: float = 0.0
    return_word_box: bool = False
    paddle_python: str = ""
    paddle_source_dir: str = ""


def run_paddle_ocr_page(
    image_path: Path,
    *,
    page_number: int = 0,
    settings: PaddleOcrSettings,
) -> dict[str, Any]:
    image_path = Path(image_path).resolve()
    paddle_env = discover_paddle_env(settings.paddle_python, settings.paddle_source_dir)
    if paddle_env is None:
        raise RuntimeError(
            "PaddleOCR environment was not found. Configure Paddle Python/source dir "
            "or install an external PaddleOCR environment."
        )

    src_dir = str(Path(__file__).parents[1])
    pythonpath_parts = [src_dir, str(paddle_env.source_dir)]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    job = {
        "image_path": str(image_path),
        "page_number": page_number,
        "ocr_version": settings.ocr_version,
        "lang": settings.lang,
        "det_model_name": settings.det_model_name,
        "rec_model_name": settings.rec_model_name,
        "det_model_dir": settings.det_model_dir,
        "rec_model_dir": settings.rec_model_dir,
        "device": settings.device,
        "engine": settings.engine,
        "enable_mkldnn": settings.enable_mkldnn,
        "cpu_threads": settings.cpu_threads,
        "enable_hpi": settings.enable_hpi,
        "rec_batch_size": settings.rec_batch_size,
        "min_confidence": settings.min_confidence,
        "return_word_box": settings.return_word_box,
    }
    token = base64.b64encode(json.dumps(job, ensure_ascii=False).encode("utf-8")).decode("ascii")
    worker_path = Path(__file__).with_name("aih_paddle_ocr_worker.py")
    result = subprocess.run(
        [
            str(paddle_env.python),
            "-c",
            "import runpy, sys; sys.argv = [sys.argv[1], sys.argv[2]]; "
            "runpy.run_path(sys.argv[0], run_name=\"__main__\")",
            str(worker_path),
            token,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
        cwd=str(paddle_env.source_dir),
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    payload = None
    if stdout:
        try:
            payload = json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError as exc:
            if result.returncode == 0:
                raise RuntimeError(f"PaddleOCR worker returned invalid JSON: {stdout!r}") from exc
    if result.returncode != 0:
        if isinstance(payload, dict) and not payload.get("ok"):
            raise RuntimeError(payload.get("error") or "PaddleOCR worker failed")
        raise RuntimeError(stderr or stdout or "PaddleOCR worker failed")
    if payload is None:
        raise RuntimeError(f"PaddleOCR worker returned no JSON: stdout={stdout!r}, stderr={stderr!r}")
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error") or "PaddleOCR worker failed")
    return payload


def write_paddle_hocr(
    result: dict[str, Any],
    output_hocr: Path,
    output_text: Path,
) -> None:
    width = int(result.get("width") or 1)
    height = int(result.get("height") or 1)
    dpi = result.get("dpi") or [300, 300]
    dpi_x = int(dpi[0] or 300) if isinstance(dpi, (list, tuple)) and dpi else 300
    dpi_y = int(dpi[1] or dpi_x) if isinstance(dpi, (list, tuple)) and len(dpi) > 1 else dpi_x
    lines = [line for line in result.get("lines") or [] if isinstance(line, dict)]

    line_spans = []
    text_lines = []
    for index, line in enumerate(lines, start=1):
        text = str(line.get("text") or "")
        bbox = _normalise_bbox(line.get("bbox"), width, height)
        if bbox is None:
            continue
        text_lines.append(text)
        x0, y0, x1, y1 = bbox
        words = text.split()
        if not words:
            escaped = html.escape(text)
            line_spans.append(
                f"<span class='ocr_line' id='line_1_{index}' title='bbox {x0} {y0} {x1} {y1}'>"
                f"<span class='ocrx_word' id='word_1_{index}_1' title='bbox {x0} {y0} {x1} {y1}'>{escaped}</span>"
                "</span>"
            )
            continue
        word_spans = []
        total_chars = max(1, sum(len(word) for word in words))
        char_offset = 0
        for word_index, word in enumerate(words, start=1):
            wx0 = x0 + int((x1 - x0) * char_offset / total_chars)
            char_offset += len(word)
            wx1 = x0 + int((x1 - x0) * char_offset / total_chars)
            if wx1 <= wx0:
                wx1 = min(x1, wx0 + 1)
            escaped = html.escape(word)
            word_spans.append(
                f"<span class='ocrx_word' id='word_1_{index}_{word_index}' "
                f"title='bbox {wx0} {y0} {wx1} {y1}'>{escaped}</span>"
            )
        line_spans.append(
            f"<span class='ocr_line' id='line_1_{index}' title='bbox {x0} {y0} {x1} {y1}'>"
            f"{' '.join(word_spans)}</span>"
        )

    hocr_content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\"\n"
        '    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">\n'
        "<head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\"/>"
        "<meta name='ocr-system' content='PaddleOCR'/></head>\n"
        "<body>\n"
        f"<div class='ocr_page' id='page_1' title='bbox 0 0 {width} {height}; scan_res {dpi_x} {dpi_y}'>\n"
        f"<div class='ocr_carea' id='block_1_1' title='bbox 0 0 {width} {height}'>\n"
        f"<p class='ocr_par' id='par_1_1' title='bbox 0 0 {width} {height}'>\n"
        f"{chr(10).join(line_spans)}\n"
        "</p></div></div>\n"
        "</body></html>"
    )
    output_hocr.write_text(hocr_content, encoding="utf-8")
    output_text.write_text("\n".join(text_lines), encoding="utf-8")


def _normalise_bbox(value: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)
