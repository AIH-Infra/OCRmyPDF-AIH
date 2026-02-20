# SPDX-License-Identifier: MPL-2.0

"""OCRmyPDF Calamari 插件 — 通过 REST API 调用 Calamari Docker 容器.

Calamari API 规范 (v5.1):
    GET  /health        → {"status":"healthy", "cached_models":[...], "version":"5.1.0"}
    GET  /models        → {"models":["gt4histocr", ...]}
    POST /ocr           → file + model → {"text":"...", "status":"success"}
    POST /ocr/batch     → files + model + mode → {"results":[...], "binding_method":"..."}

两种工作模式:
    1. server_segmentation (默认): 发送整页图片到 /ocr，服务端自动行分割 + 批量推理
    2. client_segmentation: 客户端行分割 → 发送行图片到 /ocr/batch → 按索引重排

用法:
    ocrmypdf --plugin misc/ocrmypdf_calamari.py --ocr-engine calamari \
             --calamari-url http://GPU_SERVER:11800 \
             --calamari-model gt4histocr \
             -l gt4histocr input.pdf output.pdf
"""

from __future__ import annotations

import html
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from subprocess import PIPE, run as subprocess_run
from typing import TYPE_CHECKING, Optional

import requests
from ocrmypdf import hookimpl
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence

if TYPE_CHECKING:
    from collections.abc import Set

    from ocrmypdf._options import OcrOptions
    from ocrmypdf.pdfinfo import PdfInfo

log = logging.getLogger(__name__)

# Ghostscript 可执行文件
_GS = "gswin64c" if os.name == "nt" else "gs"

# ---------------------------------------------------------------------------
# 全局配置（通过 CLI 参数或前端传入）
# ---------------------------------------------------------------------------
_config = {
    "url": "http://localhost:11800",
    "model": "gt4histocr",
    "timeout": 300,
    "batch_mode": "indexed",  # indexed | sequential
    "retry": 2,
    "segmentation": "server",  # server | client
    "preflight": False,  # 是否启用预飞批量 OCR
    "preflight_chunk": 10,  # 每批处理的页数
    "preflight_dpi": 300,  # 预渲染 DPI
}

# ---------------------------------------------------------------------------
# 预飞缓存：validate() 中预 OCR 的结果写入磁盘文件，generate_hocr() 读取
# 使用磁盘而非内存，因为 Windows 的 ProcessPoolExecutor 用 spawn 而非 fork
# ---------------------------------------------------------------------------
_preflight_cache_dir: Path | None = None


def _preflight_cache_path(pageno: int) -> Path | None:
    """返回某页的预飞缓存文件路径（0-indexed）."""
    if _preflight_cache_dir is None:
        return None
    return _preflight_cache_dir / f"page_{pageno:06d}.txt"


def _preflight_cache_write(pageno: int, text: str):
    """写入预飞缓存到磁盘."""
    p = _preflight_cache_path(pageno)
    if p:
        p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 客户端行分割（参考 AIH-Contexture 方案）
# ---------------------------------------------------------------------------

def _detect_line_positions(image_path: Path, min_line_height: int = 15) -> list[tuple[int, int, int, int]]:
    """检测图片中文本行的 bbox，返回 [(x0, y0, x1, y1), ...]."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h, w = img.shape
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 水平投影 → 行 y 范围
    h_proj = np.sum(binary, axis=1) / 255.0
    threshold = w * 0.02
    in_line = h_proj > threshold

    y_ranges = []
    start = None
    for i in range(len(in_line)):
        if in_line[i] and start is None:
            start = i
        elif not in_line[i] and start is not None:
            if i - start >= min_line_height:
                y_ranges.append((start, i))
            start = None
    if start is not None and len(in_line) - start >= min_line_height:
        y_ranges.append((start, len(in_line)))

    # 对每行做垂直投影 → x 范围
    lines = []
    for y0, y1 in y_ranges:
        row_strip = binary[y0:y1, :]
        v_proj = np.sum(row_strip, axis=0) / 255.0
        line_h = y1 - y0
        # 阈值：列中至少 5% 的像素有墨迹才算文字，过滤噪点
        x_threshold = max(line_h * 0.05, 2)
        cols = np.where(v_proj > x_threshold)[0]
        if len(cols) > 0:
            x0, x1 = int(cols[0]), int(cols[-1]) + 1
        else:
            x0, x1 = 0, w
        lines.append((x0, y0, x1, y1))

    return lines


def _segment_lines_client(image_path: Path, min_line_height: int = 15, padding: int = 5) -> list[bytes]:
    """客户端行分割：投影法切分页面为行图片，返回 PNG bytes 列表."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        log.warning("cv2/numpy not available, skipping client segmentation")
        return [image_path.read_bytes()]

    img = cv2.imread(str(image_path))
    if img is None:
        return [image_path.read_bytes()]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h_proj = np.sum(binary, axis=1) / 255.0
    threshold = w * 0.02
    in_line = h_proj > threshold

    lines = []
    start = None
    for i in range(len(in_line)):
        if in_line[i] and start is None:
            start = i
        elif not in_line[i] and start is not None:
            if i - start >= min_line_height:
                lines.append((start, i))
            start = None
    if start is not None and len(in_line) - start >= min_line_height:
        lines.append((start, len(in_line)))

    if not lines:
        return [image_path.read_bytes()]

    line_images = []
    for y0, y1 in lines:
        y0p = max(0, y0 - padding)
        y1p = min(h, y1 + padding)
        crop = img[y0p:y1p, 0:w]
        _, buf = cv2.imencode('.png', crop)
        line_images.append(buf.tobytes())

    log.info("Client segmentation: %d lines from %s", len(line_images), image_path.name)
    return line_images


# ---------------------------------------------------------------------------
# 预飞批量 OCR（在 validate() 中执行，跨页合并行图片）
# ---------------------------------------------------------------------------

def _gs_render_page(pdf_path: Path, pageno: int, output_path: Path, dpi: int = 300):
    """用 Ghostscript 渲染单页 PDF 为 PNG."""
    args = [
        _GS, "-dSAFER", "-dBATCH", "-dNOPAUSE",
        "-dInterpolateControl=-1",
        "-sDEVICE=png16m",
        f"-dFirstPage={pageno}",
        f"-dLastPage={pageno}",
        f"-r{dpi}x{dpi}",
        "-o", str(output_path),
        "-sstdout=%stderr",
        "-dAutoRotatePages=/None",
        "-f", str(pdf_path),
    ]
    subprocess_run(args, stdout=PIPE, stderr=PIPE, check=True)


def _preflight_render_and_segment(
    pdf_path: Path, page_numbers: list[int], dpi: int, tmpdir: Path
) -> dict[int, list[bytes]]:
    """渲染多页并分割行图片. 返回 {pageno: [line_png_bytes, ...]}."""
    result: dict[int, list[bytes]] = {}
    for pageno in page_numbers:
        out_png = tmpdir / f"page_{pageno:06d}.png"
        try:
            _gs_render_page(pdf_path, pageno, out_png, dpi)
            lines = _segment_lines_client(out_png)
            result[pageno] = lines
            log.debug("Preflight page %d: %d lines", pageno, len(lines))
        except Exception as e:
            log.warning("Preflight render failed for page %d: %s", pageno, e)
            result[pageno] = []  # generate_hocr 会降级到正常 OCR
        finally:
            if out_png.exists():
                out_png.unlink()
    return result


def _preflight_ocr_chunk(
    client: "CalamariClient",
    pages_lines: dict[int, list[bytes]],
    model: str,
) -> dict[int, str]:
    """对一个 chunk 的所有行图片做一次批量 OCR，返回 {pageno: text}.

    文件名编码: image_PPPPPP_LLLLLL.png (P=页码, L=行号)
    这样可以从返回结果中还原页码和行号。
    """
    all_files = []
    page_line_counts: dict[int, int] = {}

    for pageno in sorted(pages_lines.keys()):
        lines = pages_lines[pageno]
        if not lines:
            continue
        page_line_counts[pageno] = len(lines)
        for line_idx, img_bytes in enumerate(lines):
            filename = f"image_{pageno:06d}_{line_idx:06d}.png"
            all_files.append(("files", (filename, img_bytes, "image/png")))

    if not all_files:
        return {}

    total_lines = len(all_files)
    log.info(
        "Preflight batch: %d pages, %d lines total",
        len(page_line_counts), total_lines,
    )

    # 发送到 /ocr/batch
    try:
        resp = client._session.post(
            f"{client.base_url}/ocr/batch",
            files=all_files,
            data={"model": model, "mode": "indexed"},
            timeout=client.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        log.warning("Preflight batch request failed: %s", e)
        return {}

    # 解析结果：从 filename 还原 pageno + line_idx
    page_texts: dict[int, dict[int, str]] = {}
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        filename = item.get("filename", "")
        m = re.search(r"image_(\d+)_(\d+)", filename)
        if not m:
            continue
        pg = int(m.group(1))
        ln = int(m.group(2))
        page_texts.setdefault(pg, {})[ln] = text

    # 按行号排序拼接
    result: dict[int, str] = {}
    for pageno in sorted(page_texts.keys()):
        lines_dict = page_texts[pageno]
        ordered = [lines_dict.get(i, "") for i in range(max(lines_dict.keys()) + 1)]
        result[pageno] = "\n".join(t for t in ordered if t)

    elapsed = payload.get("elapsed_seconds", 0)
    log.info(
        "Preflight batch done: %d lines in %.1fs (%.1f lines/s)",
        total_lines, elapsed, total_lines / max(elapsed, 0.01),
    )
    return result


def _run_preflight(pdfinfo: "PdfInfo", options: "OcrOptions"):
    """预飞主流程：渲染 → 分割 → 分块批量 OCR → 写入磁盘缓存."""
    global _preflight_cache_dir

    # 创建持久缓存目录（不用 TemporaryDirectory 的 context manager，
    # 因为 worker 进程需要在之后读取；OCRmyPDF 结束时会清理 work_folder）
    _preflight_cache_dir = Path(tempfile.mkdtemp(prefix="calamari_preflight_"))
    log.info("Preflight cache dir: %s", _preflight_cache_dir)

    pdf_path = Path(options.input_file)
    total_pages = len(pdfinfo)
    chunk_size = _config["preflight_chunk"]
    dpi = _config["preflight_dpi"]
    model = _config["model"]
    client = _get_client()

    log.info(
        "Preflight OCR: %d pages, chunk=%d, dpi=%d, model=%s",
        total_pages, chunk_size, dpi, model,
    )

    all_page_numbers = list(range(1, total_pages + 1))  # GS 1-indexed
    cached = 0

    # 渲染用临时目录（用完即删）
    with tempfile.TemporaryDirectory(prefix="calamari_render_") as render_tmp:
        render_path = Path(render_tmp)

        for chunk_start in range(0, len(all_page_numbers), chunk_size):
            chunk_pages = all_page_numbers[chunk_start : chunk_start + chunk_size]
            log.info("Preflight chunk: pages %d-%d", chunk_pages[0], chunk_pages[-1])

            pages_lines = _preflight_render_and_segment(
                pdf_path, chunk_pages, dpi, render_path
            )
            chunk_results = _preflight_ocr_chunk(client, pages_lines, model)

            # 写入磁盘缓存 (GS 1-indexed → 0-indexed)
            for gs_pageno, text in chunk_results.items():
                _preflight_cache_write(gs_pageno - 1, text)
                cached += 1

    # 把缓存目录路径存到 _config，worker 进程通过 check_options 重建时不会丢失
    # 但 Windows spawn 模式下 _config 也会丢失，所以还需要通过 options 传递
    # 通过环境变量传递缓存目录给 worker 进程（Windows spawn 兼容）
    os.environ["CALAMARI_PREFLIGHT_CACHE"] = str(_preflight_cache_dir)
    log.info("Preflight complete: %d/%d pages cached", cached, total_pages)


# ---------------------------------------------------------------------------
# API 客户端（参考 AIH-Contexture CalamariService 模式）
# ---------------------------------------------------------------------------

class CalamariClient:
    """Calamari REST API 客户端，带重试、批量、索引重排和降级逻辑."""

    def __init__(self, base_url: str, timeout: int = 300, retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self._session = requests.Session()

    def health(self) -> dict | None:
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def models(self) -> list[str]:
        try:
            resp = self._session.get(f"{self.base_url}/models", timeout=5)
            resp.raise_for_status()
            return resp.json().get("models", [])
        except Exception:
            return []

    def ocr_single(self, image_path: Path, model: str) -> str:
        """单张图片 OCR（发送到 /ocr，服务端自动行分割）."""
        last_err = None
        for attempt in range(1, self.retries + 1):
            try:
                with open(image_path, "rb") as f:
                    resp = self._session.post(
                        f"{self.base_url}/ocr",
                        files={"file": (image_path.name, f, "image/png")},
                        data={"model": model},
                        timeout=self.timeout,
                    )
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") == "success":
                    return data.get("text", "")
                raise RuntimeError(f"Calamari returned status={data.get('status')}")
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    log.warning("Calamari OCR attempt %d/%d failed: %s", attempt, self.retries, e)
                    time.sleep(1)
        raise RuntimeError(f"Calamari OCR failed after {self.retries} attempts: {last_err}")

    def ocr_page_client_seg(self, image_path: Path, model: str) -> str:
        """客户端行分割 + /ocr/batch 批量发送 + 索引重排（AIH 模式）."""
        line_images = _segment_lines_client(image_path)
        if len(line_images) <= 1:
            # 只有一行或分割失败，直接用 /ocr
            return self.ocr_single(image_path, model)

        log.info("Sending %d line images to /ocr/batch", len(line_images))
        return self._batch_lines_with_ordering(line_images, model)

    def _batch_lines_with_ordering(self, line_images: list[bytes], model: str) -> str:
        """发送行图片到 /ocr/batch，用文件名索引保证返回顺序."""
        files = []
        for i, img_bytes in enumerate(line_images):
            filename = f"image_{i:06d}.png"
            files.append(("files", (filename, img_bytes, "image/png")))

        for attempt in range(1, self.retries + 2):
            try:
                resp = self._session.post(
                    f"{self.base_url}/ocr/batch",
                    files=files,
                    data={"model": model, "mode": "indexed"},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                payload = resp.json()

                texts = self._parse_and_reorder(payload, len(line_images))
                log.info(
                    "Batch OCR: %d lines, binding=%s, %.1fs",
                    len(texts), payload.get("binding_method", "?"),
                    payload.get("elapsed_seconds", 0),
                )
                return "\n".join(t for t in texts if t.strip())

            except Exception as e:
                log.warning("Batch attempt %d failed: %s", attempt, e)
                if attempt <= self.retries:
                    time.sleep(1)
                    continue
                # 降级：逐张发送
                log.warning("Falling back to sequential for %d lines", len(line_images))
                return self._sequential_lines(line_images, model)

        return ""

    def _parse_and_reorder(self, payload: dict, expected_count: int) -> list[str]:
        """从批量响应中按索引重排结果（参考 AIH _parse_and_reorder_results）."""
        ordered = [""] * expected_count

        results_list = payload.get("results", [])
        if not isinstance(results_list, list):
            raise ValueError("Invalid batch response: results not a list")

        filled = 0
        for item in results_list:
            if not isinstance(item, dict):
                continue
            text = (item.get("text") or "").strip()
            idx = item.get("index")

            # 尝试从 filename 提取索引
            if idx is None:
                filename = item.get("filename", "")
                m = re.search(r"image_(\d+)", filename)
                if m:
                    idx = int(m.group(1))

            if isinstance(idx, int) and 0 <= idx < expected_count:
                ordered[idx] = text
                filled += 1

        if filled < expected_count * 0.5:
            raise ValueError(f"Too few results mapped: {filled}/{expected_count}")

        return ordered

    def _sequential_lines(self, line_images: list[bytes], model: str) -> str:
        """降级：逐行发送到 /ocr."""
        texts = []
        for i, img_bytes in enumerate(line_images):
            try:
                resp = self._session.post(
                    f"{self.base_url}/ocr",
                    files={"file": (f"line_{i:04d}.png", img_bytes, "image/png")},
                    data={"model": model},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                t = (data.get("text") or "").strip()
                if t:
                    texts.append(t)
            except Exception as e:
                log.warning("Sequential line %d failed: %s", i, e)
        return "\n".join(texts)

    def ocr_batch(
        self, image_paths: list[Path], model: str, mode: str = "indexed"
    ) -> list[dict]:
        """批量 OCR（兼容旧接口）."""
        files = []
        for i, p in enumerate(image_paths):
            filename = f"image_{i:06d}.png"
            files.append(("files", (filename, open(p, "rb"), "image/png")))
        try:
            resp = self._session.post(
                f"{self.base_url}/ocr/batch",
                files=files,
                data={"model": model, "mode": mode},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            log.info(
                "Calamari batch: %d images, binding=%s, %.1fs",
                data.get("count", 0),
                data.get("binding_method", "?"),
                data.get("elapsed_seconds", 0),
            )
            return data.get("results", [])
        except Exception as e:
            log.warning("Calamari batch failed: %s, falling back to sequential", e)
            results = []
            for i, p in enumerate(image_paths):
                text = self.ocr_single(p, model)
                results.append({"filename": p.name, "text": text, "index": i})
            return results
        finally:
            for _, (_, fobj, _) in files:
                fobj.close()


# 全局客户端实例（延迟初始化）
_client: CalamariClient | None = None


def _get_client() -> CalamariClient:
    global _client
    if _client is None:
        _client = CalamariClient(
            base_url=_config["url"],
            timeout=_config["timeout"],
            retries=_config["retry"],
        )
    return _client


# ---------------------------------------------------------------------------
# OcrEngine 实现
# ---------------------------------------------------------------------------

class CalamariOcrEngine(OcrEngine):

    @staticmethod
    def version() -> str:
        info = _get_client().health()
        if info:
            return info.get("version", "5.0.0")
        return "5.0.0 (offline)"

    @staticmethod
    def creator_tag(options: OcrOptions) -> str:
        return f"Calamari OCR {CalamariOcrEngine.version()} via REST API"

    def __str__(self) -> str:
        return f"Calamari OCR {self.version()}"

    @staticmethod
    def languages(options: OcrOptions) -> Set[str]:
        models = _get_client().models()
        return set(models) if models else {_config["model"]}

    @staticmethod
    def get_orientation(
        input_file: Path, options: OcrOptions
    ) -> OrientationConfidence:
        # Calamari 不支持方向检测
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def generate_hocr(
        input_file: Path,
        output_hocr: Path,
        output_text: Path,
        options: OcrOptions,
    ) -> None:
        """调用 Calamari API，生成 hOCR 输出.

        优先从预飞缓存读取；缓存未命中时降级到正常 per-page OCR。
        """
        model = _config["model"]
        if options.languages:
            model = options.languages[0]

        # ---- 尝试从预飞缓存读取 ----
        # input_file 路径格式: .../000001_ocr.png (1-indexed)
        # 用环境变量传递缓存目录（跨进程安全，Windows spawn 兼容）
        text = None
        cache_dir = os.environ.get("CALAMARI_PREFLIGHT_CACHE")
        if cache_dir:
            m = re.match(r"(\d+)_", input_file.name)
            if m:
                pageno = int(m.group(1)) - 1
                cache_file = Path(cache_dir) / f"page_{pageno:06d}.txt"
                if cache_file.exists():
                    text = cache_file.read_text(encoding="utf-8")
                    log.debug("Preflight cache hit: page %d (%d chars)", pageno, len(text))

        # ---- 缓存未命中：正常 OCR ----
        if text is None:
            client = _get_client()
            seg_mode = _config.get("segmentation", "server")
            if seg_mode == "client":
                text = client.ocr_page_client_seg(input_file, model)
            else:
                text = client.ocr_single(input_file, model)

        # 获取图片尺寸和 DPI 用于 hOCR bbox
        from PIL import Image

        with Image.open(input_file) as img:
            w, h = img.size
            dpi_info = img.info.get("dpi", (300, 300))
            dpi_x = int(round(dpi_info[0])) if dpi_info[0] else 300
            dpi_y = int(round(dpi_info[1])) if dpi_info[1] else 300

        log.info(
            "hOCR diag: image=%s size=%dx%d dpi=(%d,%d) raw_dpi=%r",
            input_file.name, w, h, dpi_x, dpi_y, dpi_info,
        )

        # 生成 hOCR — 用投影法检测实际行位置
        text_lines = [ln for ln in text.splitlines() if ln.strip()]
        if not text_lines:
            text_lines = [text] if text.strip() else [""]

        # 检测图片中实际的文本行 y 坐标
        detected = _detect_line_positions(input_file)
        if detected:
            log.info(
                "hOCR diag: detected %d lines, first=%s last=%s",
                len(detected), detected[0], detected[-1],
            )

        # 将 OCR 文本行映射到检测到的行位置 (x0, y0, x1, y1)
        if detected and len(detected) >= len(text_lines):
            positions = detected[:len(text_lines)]
        elif detected:
            positions = detected
        else:
            # 检测失败：回退到均匀估算
            est_h = max(h // max(len(text_lines) + 2, 1), 20)
            margin = max(int(h * 0.03), 10)
            positions = [(0, margin + i * est_h, w, min(margin + (i + 1) * est_h, h))
                         for i in range(len(text_lines))]

        line_spans = []
        for i, line_text in enumerate(text_lines):
            if i < len(positions):
                x0, y0, x1, y1 = positions[i]
            else:
                x0, y0, x1, y1 = positions[-1]
            escaped = html.escape(line_text)
            line_spans.append(
                f"<span class='ocr_line' title='bbox {x0} {y0} {x1} {y1}'>"
                f"<span class='ocrx_word' title='bbox {x0} {y0} {x1} {y1}'>"
                f"{escaped}</span></span>"
            )

        hocr_body = "\n".join(line_spans)
        hocr_content = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\"\n"
            '    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n'
            '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">\n'
            "<head><meta http-equiv=\"Content-Type\" "
            'content="text/html; charset=utf-8"/></head>\n'
            "<body>\n"
            f"<div class='ocr_page' title='bbox 0 0 {w} {h}; scan_res {dpi_x} {dpi_y}'>\n"
            f"<div class='ocr_carea' title='bbox 0 0 {w} {h}'>\n"
            f"<p class='ocr_par' title='bbox 0 0 {w} {h}'>\n"
            f"{hocr_body}\n"
            "</p></div></div>\n"
            "</body></html>"
        )

        output_hocr.write_text(hocr_content, encoding="utf-8")
        output_text.write_text(text, encoding="utf-8")

    @staticmethod
    def generate_pdf(
        input_file: Path,
        output_pdf: Path,
        output_text: Path,
        options: OcrOptions,
    ) -> None:
        raise NotImplementedError("Use hOCR rendering path")


# ---------------------------------------------------------------------------
# pluggy hooks
# ---------------------------------------------------------------------------

@hookimpl
def add_options(parser):
    calamari_group = parser.add_argument_group("Calamari", "Calamari OCR 远程引擎选项")
    calamari_group.add_argument(
        "--calamari-url",
        default="http://localhost:11800",
        help="Calamari OCR API 地址（默认 http://localhost:11800）",
    )
    calamari_group.add_argument(
        "--calamari-model",
        default="gt4histocr",
        help="Calamari 模型名称（默认 gt4histocr）",
    )
    calamari_group.add_argument(
        "--calamari-timeout",
        type=int,
        default=300,
        help="Calamari 请求超时秒数（默认 300）",
    )
    calamari_group.add_argument(
        "--calamari-batch-mode",
        choices=["indexed", "sequential"],
        default="indexed",
        help="Calamari 批量模式：indexed（快）或 sequential（稳）",
    )
    calamari_group.add_argument(
        "--calamari-segmentation",
        choices=["server", "client"],
        default="server",
        help="行分割位置：server（服务端，默认）或 client（客户端分割+批量发送，更快）",
    )
    calamari_group.add_argument(
        "--calamari-preflight",
        action="store_true",
        default=False,
        help="启用预飞批量 OCR：在处理前预渲染所有页面并跨页合并行图片批量推理",
    )
    calamari_group.add_argument(
        "--calamari-preflight-chunk",
        type=int,
        default=10,
        help="预飞模式每批处理的页数（默认 10，约 200-400 行/批）",
    )
    calamari_group.add_argument(
        "--calamari-preflight-dpi",
        type=int,
        default=300,
        help="预飞渲染 DPI（默认 300）",
    )


@hookimpl
def check_options(options):
    global _client

    url = getattr(options, "calamari_url", None) or _config["url"]
    model = getattr(options, "calamari_model", None) or _config["model"]
    timeout = getattr(options, "calamari_timeout", None) or _config["timeout"]
    batch_mode = getattr(options, "calamari_batch_mode", None) or _config["batch_mode"]
    segmentation = getattr(options, "calamari_segmentation", None) or _config["segmentation"]
    preflight = getattr(options, "calamari_preflight", False)
    preflight_chunk = getattr(options, "calamari_preflight_chunk", None) or _config["preflight_chunk"]
    preflight_dpi = getattr(options, "calamari_preflight_dpi", None) or _config["preflight_dpi"]

    _config["url"] = url
    _config["model"] = model
    _config["timeout"] = timeout
    _config["batch_mode"] = batch_mode
    _config["segmentation"] = segmentation
    _config["preflight"] = preflight
    _config["preflight_chunk"] = preflight_chunk
    _config["preflight_dpi"] = preflight_dpi

    # 预飞模式强制使用 client 分割（需要客户端切行才能跨页合并）
    if preflight:
        _config["segmentation"] = "client"
        log.info("Preflight mode enabled: forcing client segmentation, chunk=%d", preflight_chunk)

    # 重建客户端
    _client = CalamariClient(base_url=url, timeout=timeout)

    # 仅在选择 calamari 引擎时检查连通性
    engine = getattr(options, "ocr_engine", "auto")
    if engine in ("calamari",):
        info = _client.health()
        if info is None:
            log.warning(
                "Calamari 服务不可达: %s — OCR 可能失败。"
                "请确认容器已启动且端口映射正确。",
                url,
            )
        else:
            log.info(
                "Calamari 已连接: %s, version=%s, models=%s",
                url,
                info.get("version"),
                info.get("cached_models"),
            )


@hookimpl
def validate(pdfinfo: "PdfInfo", options: "OcrOptions"):
    """预飞批量 OCR：在 worker 进程启动前，预渲染+分割+批量推理所有页面.

    此 hook 在主进程中执行，可以修改全局状态。
    预飞结果缓存到 _preflight_cache，generate_hocr() 直接读取。
    """
    engine = getattr(options, "ocr_engine", "auto")
    if engine not in ("calamari",):
        return
    if not _config.get("preflight"):
        return

    try:
        _run_preflight(pdfinfo, options)
    except Exception as e:
        log.warning(
            "Preflight OCR failed: %s — falling back to per-page OCR", e
        )


@hookimpl
def get_ocr_engine(options):
    engine = getattr(options, "ocr_engine", "auto") if options else "auto"
    if engine in ("calamari",):
        return CalamariOcrEngine()
    return None
