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
    ocrmypdf --plugin src/ocrmypdf/aih_calamari_plugin.py --ocr-engine calamari \
             --calamari-url http://GPU_SERVER:11800 \
             --calamari-model gt4histocr \
             -l gt4histocr input.pdf output.pdf
"""

from __future__ import annotations

# 在导入任何模块之前，先禁用警告和日志
import base64
import json
import warnings
warnings.filterwarnings('ignore')

import html
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

# 配置日志级别（必须在导入streamlit相关模块之前）
logging.getLogger('streamlit').setLevel(logging.CRITICAL)
logging.getLogger('streamlit.runtime').setLevel(logging.CRITICAL)
logging.getLogger('streamlit.runtime.scriptrunner_utils').setLevel(logging.CRITICAL)
logging.getLogger('streamlit.runtime.state').setLevel(logging.CRITICAL)
logging.getLogger('streamlit.runtime.caching').setLevel(logging.CRITICAL)
from subprocess import PIPE, run as subprocess_run
from typing import TYPE_CHECKING, NamedTuple, Optional

import requests
from ocrmypdf import hookimpl
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence
from ocrmypdf.aih_chrome_screenai import ChromeScreenAIBackend
from ocrmypdf.aih_line_detection import detect_lines_paddle_det
from ocrmypdf.aih_paddle_ocr import (
    PaddleOcrSettings,
    run_paddle_ocr_page,
    write_paddle_hocr,
)

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
    "timeout": 600,  # 增加到600秒以支持大批次
    "batch_mode": "indexed",  # indexed | sequential
    "retry": 2,
    "segmentation": "client",  # server | client - 默认client以使用批量端点
    "preflight": False,  # 是否启用预飞批量 OCR
    "preflight_chunk": 10,  # 每批处理的页数
    "preflight_dpi": 300,  # 预渲染 DPI
    "preflight_workers": 1,  # 渲染并行度
    "preflight_batch_lines": 512,  # 每个 Calamari multipart 子批最多行数
    "preflight_batch_mb": 64,  # 每个 Calamari multipart 子批最多图像字节数
    "line_detector": "paddle_det",  # paddle_det | chrome_screenai | tesseract | projection_debug
    "chrome_screenai_light": False,
    "chrome_screenai_min_confidence": 0.0,
    "chrome_screenai_filter_types": "printed",
    "chrome_screenai_text_layer_mode": "native_direct",
    "chrome_screenai_force_horizontal": True,
    "chrome_screenai_text_cleanup": "mode_default",
    "chrome_screenai_rasterize_dpi": 300,
    "chrome_screenai_rasterize_quality": 88,
    "paddle_det_model": "PP-OCRv5_server_det",
    "paddle_det_device": "gpu:0",
    "paddle_det_min_confidence": 0.0,
    "paddle_det_python": "",
    "paddle_det_source_dir": "",
    "paddle_ocr_version": "PP-OCRv6",
    "paddle_ocr_lang": "en",
    "paddle_ocr_det_model": "",
    "paddle_ocr_rec_model": "",
    "paddle_ocr_det_model_dir": "",
    "paddle_ocr_rec_model_dir": "",
    "paddle_ocr_device": "gpu:0",
    "paddle_ocr_engine": "",
    "paddle_ocr_enable_mkldnn": False,
    "paddle_ocr_cpu_threads": None,
    "paddle_ocr_enable_hpi": None,
    "paddle_ocr_rec_batch_size": 16,
    "paddle_ocr_min_confidence": 0.0,
    "paddle_ocr_return_word_box": False,
    "paddle_ocr_python": "",
    "paddle_ocr_source_dir": "",
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


def _preflight_cache_write_hocr(pageno: int, hocr_content: str, dpi: int):
    """写入 hOCR 缓存到磁盘，同时记录 DPI."""
    cache_dir = os.environ.get("CALAMARI_PREFLIGHT_CACHE")
    if not cache_dir:
        return

    cache_path = Path(cache_dir)
    hocr_path = cache_path / f"page_{pageno:06d}.hocr"
    dpi_path = cache_path / f"page_{pageno:06d}_dpi.txt"

    hocr_path.write_text(hocr_content, encoding="utf-8")
    dpi_path.write_text(str(dpi), encoding="utf-8")


def _preflight_cache_write_lineboxes(pageno: int, line_boxes, dpi: int, detector: str):
    """写入外部行检测器行框缓存到磁盘，同时记录源 DPI。"""
    cache_dir = os.environ.get("CALAMARI_PREFLIGHT_CACHE")
    if not cache_dir:
        return

    cache_path = Path(cache_dir)
    boxes_path = cache_path / f"page_{pageno:06d}.{detector}_lineboxes.json"

    payload = {
        "dpi": int(dpi),
        "lines": [
            {
                "bbox": [int(v) for v in line_box.bbox],
                "text": str(getattr(line_box, "text", "") or ""),
                "words": [
                    {
                        "text": str(word_text or ""),
                        "bbox": [int(v) for v in word_bbox] if word_bbox else None,
                    }
                    for word_text, word_bbox in (getattr(line_box, "words", None) or [])
                ],
            }
            for line_box in (line_boxes or [])
            if getattr(line_box, "bbox", None)
        ],
        "bboxes": [
            [int(v) for v in line_box.bbox]
            for line_box in (line_boxes or [])
            if getattr(line_box, "bbox", None)
        ],
    }
    boxes_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _preflight_cache_read_hocr(pageno: int) -> tuple[str, int] | None:
    """读取缓存的 hOCR 和 DPI."""
    cache_dir = os.environ.get("CALAMARI_PREFLIGHT_CACHE")
    if not cache_dir:
        return None

    cache_path = Path(cache_dir)
    hocr_path = cache_path / f"page_{pageno:06d}.hocr"
    dpi_path = cache_path / f"page_{pageno:06d}_dpi.txt"

    if hocr_path.exists() and dpi_path.exists():
        hocr = hocr_path.read_text(encoding="utf-8")
        dpi = int(dpi_path.read_text(encoding="utf-8").strip())
        return (hocr, dpi)
    return None


def _preflight_cache_read_lineboxes(
    pageno: int,
    detector: str,
) -> tuple[list["_ChromeLineBox"], int] | None:
    """读取外部行检测器行框缓存和源 DPI。"""
    cache_dir = os.environ.get("CALAMARI_PREFLIGHT_CACHE")
    if not cache_dir:
        return None

    cache_path = Path(cache_dir)
    boxes_path = cache_path / f"page_{pageno:06d}.{detector}_lineboxes.json"
    if not boxes_path.exists():
        return None

    payload = json.loads(boxes_path.read_text(encoding="utf-8"))
    dpi = int(payload.get("dpi") or 300)
    lines = payload.get("lines") or []
    line_boxes = []
    if isinstance(lines, list):
        for item in lines:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            words = []
            for word in item.get("words") or []:
                if not isinstance(word, dict):
                    continue
                word_bbox = word.get("bbox")
                words.append(
                    (
                        str(word.get("text") or ""),
                        tuple(int(v) for v in word_bbox)
                        if isinstance(word_bbox, (list, tuple)) and len(word_bbox) == 4
                        else None,
                    )
                )
            line_boxes.append(
                _ChromeLineBox(
                    tuple(int(v) for v in bbox),
                    str(item.get("text") or ""),
                    tuple(words),
                )
            )
    if not line_boxes:
        bboxes = payload.get("bboxes") or []
        line_boxes = [
            _ChromeLineBox(tuple(int(v) for v in bbox))
            for bbox in bboxes
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4
        ]
    return (line_boxes, dpi)


def _get_image_dpi(image_path: Path) -> int:
    """从图像 EXIF 获取 DPI，默认 400."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            dpi_info = img.info.get("dpi", (400, 400))
            return int(round(dpi_info[0])) if dpi_info[0] else 400
    except Exception:
        return 400


def _scale_hocr_dpi(hocr_content: str, source_dpi: int, target_dpi: int) -> str:
    """缩放 hOCR 中的所有 bbox 坐标以匹配目标 DPI."""
    if source_dpi == target_dpi:
        return hocr_content

    from bs4 import BeautifulSoup
    import re

    scale = target_dpi / source_dpi
    soup = BeautifulSoup(hocr_content, 'html.parser')

    # 查找所有包含 bbox 的元素
    for elem in soup.find_all(attrs={"title": True}):
        title = elem.get('title', '')

        # 缩放 bbox
        def scale_bbox(match):
            x0, y0, x1, y1 = map(int, match.groups())
            x0_new = int(round(x0 * scale))
            y0_new = int(round(y0 * scale))
            x1_new = int(round(x1 * scale))
            y1_new = int(round(y1 * scale))
            return f'bbox {x0_new} {y0_new} {x1_new} {y1_new}'

        new_title = re.sub(
            r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',
            scale_bbox,
            title
        )

        # 缩放 baseline intercept
        def scale_baseline(match):
            slope = float(match.group(1))
            intercept = float(match.group(2))
            intercept_new = intercept * scale
            return f'baseline {slope} {intercept_new}'

        new_title = re.sub(
            r'baseline\s+([-\d.]+)\s+([-\d.]+)',
            scale_baseline,
            new_title
        )

        # 更新 scan_res（如果存在）
        new_title = re.sub(
            r'scan_res\s+\d+\s+\d+',
            f'scan_res {target_dpi} {target_dpi}',
            new_title
        )

        elem['title'] = new_title

    return str(soup)


def _scale_lineboxes_dpi(line_boxes, source_dpi: int, target_dpi: int):
    """按 DPI 缩放缓存的 Chrome ScreenAI 行框."""
    if source_dpi == target_dpi:
        return list(line_boxes)

    scale = target_dpi / source_dpi
    scaled = []
    for line_box in line_boxes:
        x0, y0, x1, y1 = line_box.bbox
        words = []
        for word_text, word_bbox in (getattr(line_box, "words", None) or []):
            if word_bbox:
                wx0, wy0, wx1, wy1 = word_bbox
                words.append(
                    (
                        word_text,
                        (
                            int(round(wx0 * scale)),
                            int(round(wy0 * scale)),
                            int(round(wx1 * scale)),
                            int(round(wy1 * scale)),
                        ),
                    )
                )
            else:
                words.append((word_text, None))
        scaled.append(
            _ChromeLineBox(
                (
                    int(round(x0 * scale)),
                    int(round(y0 * scale)),
                    int(round(x1 * scale)),
                    int(round(y1 * scale)),
                ),
                getattr(line_box, "text", ""),
                tuple(words),
            )
        )
    return scaled


def _scale_lineboxes_to_image(line_boxes, image_path: Path):
    """Scale line boxes into the target OCR image pixel space when needed."""
    with Image.open(image_path) as img:
        width, height = img.size

    valid_boxes = [
        line_box.bbox
        for line_box in (line_boxes or [])
        if getattr(line_box, "bbox", None)
    ]
    if not valid_boxes or width <= 0 or height <= 0:
        return list(line_boxes or [])

    max_x = max(bbox[2] for bbox in valid_boxes)
    max_y = max(bbox[3] for bbox in valid_boxes)
    if max_x <= width * 1.10 and max_y <= height * 1.10:
        return list(line_boxes)

    sx = width / max_x if max_x else 1.0
    sy = height / max_y if max_y else 1.0
    scaled = []
    for line_box in line_boxes:
        x0, y0, x1, y1 = line_box.bbox
        words = []
        for word_text, word_bbox in (getattr(line_box, "words", None) or []):
            if word_bbox:
                wx0, wy0, wx1, wy1 = word_bbox
                words.append(
                    (
                        word_text,
                        (
                            max(0, min(width, int(round(wx0 * sx)))),
                            max(0, min(height, int(round(wy0 * sy)))),
                            max(0, min(width, int(round(wx1 * sx)))),
                            max(0, min(height, int(round(wy1 * sy)))),
                        ),
                    )
                )
            else:
                words.append((word_text, None))
        scaled.append(
            _ChromeLineBox(
                (
                    max(0, min(width, int(round(x0 * sx)))),
                    max(0, min(height, int(round(y0 * sy)))),
                    max(0, min(width, int(round(x1 * sx)))),
                    max(0, min(height, int(round(y1 * sy)))),
                ),
                getattr(line_box, "text", ""),
                tuple(words),
            )
        )
    return scaled


def _split_line_text_to_word_spans(
    line_text: str,
    bbox: tuple[int, int, int, int],
    *,
    word_id_prefix: str,
) -> str:
    words = line_text.split()
    if not words:
        return ""

    x0, y0, x1, y1 = bbox
    width = max(1, x1 - x0)
    total_chars = sum(len(word) for word in words) + max(0, len(words) - 1)
    total_chars = total_chars or len(words)
    char_offset = 0
    spans = []
    for word_index, word in enumerate(words):
        wx0 = x0 + int(round(width * char_offset / total_chars))
        char_offset += len(word)
        wx1 = x0 + int(round(width * char_offset / total_chars))
        if word_index == len(words) - 1:
            wx1 = x1
        if wx1 <= wx0:
            wx1 = min(x1, wx0 + 1)
        escaped = html.escape(word)
        spans.append(
            f"<span class='ocrx_word' id='{word_id_prefix}_{word_index:03d}' "
            f"title='bbox {wx0} {y0} {wx1} {y1}'>{escaped}</span>"
        )
        char_offset += 1
    return " ".join(spans)


def _build_word_spans_from_explicit_words(
    words: list[tuple[str, tuple[int, int, int, int] | None]],
    *,
    fallback_text: str,
    line_bbox: tuple[int, int, int, int],
    word_id_prefix: str,
) -> str:
    fallback_words = fallback_text.split()
    explicit_bboxes = [word_bbox for _, word_bbox in words if word_bbox]
    if not fallback_words or not explicit_bboxes:
        return _split_line_text_to_word_spans(
            fallback_text,
            line_bbox,
            word_id_prefix=word_id_prefix,
        )

    if len(explicit_bboxes) != len(fallback_words):
        return _split_line_text_to_word_spans(
            fallback_text,
            line_bbox,
            word_id_prefix=word_id_prefix,
        )

    spans = []
    x0, y0, x1, y1 = line_bbox
    for word_index, (word_text, word_bbox) in enumerate(zip(fallback_words, explicit_bboxes)):
        wx0, wy0, wx1, wy1 = word_bbox
        wx0 = max(x0, min(x1, int(wx0)))
        wy0 = max(y0, min(y1, int(wy0)))
        wx1 = max(x0, min(x1, int(wx1)))
        wy1 = max(y0, min(y1, int(wy1)))
        if wx1 <= wx0 or wy1 <= wy0:
            continue
        escaped = html.escape(word_text)
        spans.append(
            f"<span class='ocrx_word' id='{word_id_prefix}_{word_index:03d}' "
            f"title='bbox {wx0} {wy0} {wx1} {wy1}'>{escaped}</span>"
        )
    return " ".join(spans)


# ---------------------------------------------------------------------------
# Tesseract 行分割集成
# ---------------------------------------------------------------------------

def _check_tesseract_available() -> bool:
    """检查 Tesseract 是否可用."""
    try:
        print(f"[DEBUG] PATH in check: {os.environ.get('PATH', 'NOT SET')[:200]}", flush=True)
        result = subprocess_run(['tesseract', '--version'], capture_output=True, timeout=5, env=os.environ)
        print(f"[DEBUG] Tesseract check returncode: {result.returncode}", flush=True)
        return result.returncode == 0
    except Exception as e:
        print(f"[DEBUG] Tesseract check exception: {e}", flush=True)
        return False


def _get_tesseract_hocr(image_path: Path, language: str = "deu") -> str:
    """调用 Tesseract 生成 hOCR，用于获取准确的行 bbox.

    Args:
        image_path: 图像文件路径
        language: Tesseract语言代码（会自动从Calamari模型名映射）
    """
    # Tesseract行分割算法独立于语言包，统一使用deu（普通德语）
    # 我们只需要bbox，不需要Tesseract的识别结果
    tess_lang = "deu"

    with tempfile.NamedTemporaryFile(suffix='', delete=False) as f:
        output_base = Path(f.name)

    try:
        subprocess_run(
            ['tesseract', str(image_path), str(output_base), '-l', tess_lang, '--psm', '1', 'hocr'],
            check=True,
            capture_output=True,
            timeout=60,
            env=os.environ
        )
        hocr_path = output_base.with_suffix('.hocr')
        return hocr_path.read_text(encoding='utf-8')
    finally:
        hocr_path = output_base.with_suffix('.hocr')
        if hocr_path.exists():
            hocr_path.unlink()


def _parse_line_bboxes_from_hocr(hocr_content: str) -> list[tuple[int, int, int, int]]:
    """从 hOCR 中提取行 bbox，遵循 Tesseract 的阅读顺序."""
    from bs4 import BeautifulSoup
    import re

    soup = BeautifulSoup(hocr_content, 'html.parser')
    careas = soup.find_all('div', class_='ocr_carea')

    bboxes = []
    if careas:
        # 检查是否有 reading order 属性
        carea_data = []
        for carea in careas:
            title = carea.get('title', '')
            # 提取 reading order
            order_match = re.search(r'x_reading_order\s+(\d+)', title)
            reading_order = int(order_match.group(1)) if order_match else None

            lines = carea.find_all('span', class_='ocr_line')
            line_bboxes = []
            for line in lines:
                line_title = line.get('title', '')
                bbox_match = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', line_title)
                if bbox_match:
                    line_bboxes.append(tuple(map(int, bbox_match.groups())))

            if line_bboxes:
                carea_data.append((reading_order, line_bboxes))

        # 如果有 reading order，按此排序；否则保持原顺序
        if any(order is not None for order, _ in carea_data):
            carea_data.sort(key=lambda c: c[0] if c[0] is not None else 999999)

        for _, line_bboxes in carea_data:
            bboxes.extend(line_bboxes)
    else:
        lines = soup.find_all('span', class_='ocr_line')
        for line in lines:
            title = line.get('title', '')
            bbox_match = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', title)
            if bbox_match:
                bboxes.append(tuple(map(int, bbox_match.groups())))

    return bboxes


def _crop_line_images_by_bbox(image_path: Path, bboxes: list[tuple[int, int, int, int]], padding: int = 5, binarize: bool = False) -> list[bytes]:
    """根据 bbox 切割行图像，返回 PNG bytes 列表."""
    try:
        import cv2
    except ImportError:
        log.warning("cv2 not available for bbox cropping")
        return []

    img = cv2.imread(str(image_path))
    if img is None:
        return []

    h, w = img.shape[:2]
    line_images = []

    for x0, y0, x1, y1 in bboxes:
        y0 = max(0, y0 - padding)
        y1 = min(h, y1 + padding)
        x0 = max(0, x0 - padding)
        x1 = min(w, x1 + padding)

        line_img = img[y0:y1, x0:x1]

        if binarize:
            if len(line_img.shape) == 3:
                gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = line_img
            _, line_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        success, buffer = cv2.imencode('.png', line_img)
        if success:
            line_images.append(buffer.tobytes())

    return line_images


def _crop_line_images_by_lineboxes(image_path: Path, line_boxes, padding: int = 5, binarize: bool = False) -> list[bytes]:
    """根据 LineBox 列表裁剪行图像。"""
    bboxes = [line_box.bbox for line_box in line_boxes]
    return _crop_line_images_by_bbox(
        image_path,
        bboxes,
        padding=padding,
        binarize=binarize,
    )


def _replace_hocr_line_texts(hocr_content: str, calamari_texts: list[str]) -> str:
    """替换 Tesseract hOCR 中的文本内容为 Calamari 识别结果，保留 bbox."""
    from bs4 import BeautifulSoup
    import re

    soup = BeautifulSoup(hocr_content, 'html.parser')
    careas = soup.find_all('div', class_='ocr_carea')

    if careas:
        # 检查是否有 reading order 属性
        carea_data = []
        for carea in careas:
            title = carea.get('title', '')
            # 提取 reading order
            order_match = re.search(r'x_reading_order\s+(\d+)', title)
            reading_order = int(order_match.group(1)) if order_match else None

            lines = carea.find_all('span', class_='ocr_line')
            if lines:
                carea_data.append((reading_order, lines))

        # 如果有 reading order，按此排序；否则保持原顺序
        if any(order is not None for order, _ in carea_data):
            carea_data.sort(key=lambda c: c[0] if c[0] is not None else 999999)

        ocr_lines = []
        for _, lines in carea_data:
            ocr_lines.extend(lines)
        else:
            ocr_lines = soup.find_all('span', class_='ocr_line')
    else:
        ocr_lines = soup.find_all('span', class_='ocr_line')

    print(f"[DEBUG] hOCR has {len(ocr_lines)} lines, Calamari has {len(calamari_texts)} texts", flush=True)

    for i, line_span in enumerate(ocr_lines):
        if i < len(calamari_texts):
            # 提取该行的 bbox
            title = line_span.get('title', '')
            bbox_match = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', title)

            # 清空该行内容
            line_span.clear()

            # 创建单个 ocrx_word 包含整行文本
            if bbox_match:
                x0, y0, x1, y1 = bbox_match.groups()
                word_tag = soup.new_tag('span', attrs={
                    'class': 'ocrx_word',
                    'title': f'bbox {x0} {y0} {x1} {y1}'
                })
                word_tag.string = calamari_texts[i]
                line_span.append(word_tag)
            else:
                # 没有 bbox，直接设置文本
                line_span.string = calamari_texts[i]

    return str(soup)


def _build_simple_hocr_from_lineboxes(
    image_path: Path,
    line_boxes,
    texts: list[str],
    *,
    dpi_x: int,
    dpi_y: int,
) -> str:
    """根据统一的 LineBox 列表和 OCR 文本构建最小 hOCR。"""
    with Image.open(image_path) as img:
        width, height = img.size

    page_lines = []
    for index, line_box in enumerate(line_boxes):
        x0, y0, x1, y1 = line_box.bbox
        x0 = max(0, min(width, int(x0)))
        y0 = max(0, min(height, int(y0)))
        x1 = max(0, min(width, int(x1)))
        y1 = max(0, min(height, int(y1)))
        if x1 <= x0 or y1 <= y0:
            continue
        line_text = texts[index] if index < len(texts) else ""
        word_spans = _build_word_spans_from_explicit_words(
            list(getattr(line_box, "words", ()) or ()),
            fallback_text=line_text,
            line_bbox=(x0, y0, x1, y1),
            word_id_prefix=f"word_{index:06d}",
        )
        if not word_spans:
            continue
        page_lines.append(
            f"      <span class='ocr_line' id='line_{index:06d}' "
            f"title='bbox {x0} {y0} {x1} {y1}; scan_res {dpi_x} {dpi_y}'>"
            f"{word_spans}"
            f"</span>"
        )

    body = "\n".join(page_lines)
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\"\n"
        "    \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\n"
        "<html xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"en\" lang=\"en\">\n"
        "<head>\n"
        "  <title>OCRmyPDF-AIH Calamari hOCR</title>\n"
        "  <meta http-equiv='Content-Type' content='text/html;charset=utf-8'/>\n"
        "  <meta name='ocr-system' content='OCRmyPDF-AIH Calamari'/>\n"
        "</head>\n"
        "<body>\n"
        f"  <div class='ocr_page' id='page_1' title='bbox 0 0 {width} {height}; scan_res {dpi_x} {dpi_y}'>\n"
        "    <p class='ocr_par' id='par_1'>\n"
        f"{body}\n"
        "    </p>\n"
        "  </div>\n"
        "</body>\n"
        "</html>\n"
    )


def _write_empty_hocr_page(image_path: Path, output_hocr: Path, output_text: Path, reason: str) -> None:
    """Write a valid empty hOCR page for blank/no-text pages."""
    with Image.open(image_path) as img:
        dpi_info = img.info.get("dpi", (300, 300))
        dpi_x = int(round(dpi_info[0])) if dpi_info[0] else 300
        dpi_y = int(round(dpi_info[1])) if dpi_info[1] else 300
    final_hocr = _build_simple_hocr_from_lineboxes(
        image_path,
        [],
        [],
        dpi_x=dpi_x,
        dpi_y=dpi_y,
    )
    output_hocr.write_text(final_hocr, encoding="utf-8")
    output_text.write_text("", encoding="utf-8")
    print(f"[DIAG] Empty hOCR page written: {reason}", flush=True)


class _ChromeLineBox(NamedTuple):
    bbox: tuple[int, int, int, int]
    text: str = ""
    words: tuple[tuple[str, tuple[int, int, int, int] | None], ...] = ()


def _detect_lines_with_chrome_screenai(
    image_path: Path,
    *,
    page_number: int = 0,
):
    payload = {
        "image_path": str(image_path),
        "page_number": int(page_number),
        "light_mode": bool(_config.get("chrome_screenai_light", False)),
        "force_horizontal": bool(_config.get("chrome_screenai_force_horizontal", True)),
        "filter_printed_text": _config.get("chrome_screenai_filter_types", "printed") != "all",
        "min_confidence": float(_config.get("chrome_screenai_min_confidence", 0.0) or 0.0),
    }
    token = base64.b64encode(
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    env = os.environ.copy()
    src_dir = str(Path(__file__).parents[1])
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_dir if not existing_pythonpath else f"{src_dir}{os.pathsep}{existing_pythonpath}"
    )
    result = subprocess_run(
        [sys.executable, "-m", "ocrmypdf.aih_chrome_line_worker", token],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=src_dir,
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        raise RuntimeError(
            stderr or stdout or f"Chrome ScreenAI line worker failed for {image_path.name}"
        )

    try:
        payload_out = json.loads(stdout.splitlines()[-1] if stdout else "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Chrome ScreenAI line worker returned invalid JSON for {image_path.name}: {stdout!r}"
        ) from exc

    if not payload_out.get("ok"):
        raise RuntimeError(
            payload_out.get("error") or f"Chrome ScreenAI line worker failed for {image_path.name}"
        )

    if isinstance(payload_out.get("lines"), list):
        line_boxes = []
        for item in payload_out.get("lines", []):
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            words = []
            for word in item.get("words") or []:
                if not isinstance(word, dict):
                    continue
                word_bbox = word.get("bbox")
                words.append(
                    (
                        str(word.get("text") or ""),
                        tuple(int(v) for v in word_bbox)
                        if isinstance(word_bbox, (list, tuple)) and len(word_bbox) == 4
                        else None,
                    )
                )
            line_boxes.append(
                _ChromeLineBox(
                    tuple(int(v) for v in bbox),
                    str(item.get("text") or ""),
                    tuple(words),
                )
            )
        return line_boxes

    return [
        _ChromeLineBox(tuple(int(v) for v in bbox))
        for bbox in payload_out.get("bboxes", [])
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4
    ]


def _detect_lines_with_paddle_det(
    image_path: Path,
    *,
    page_number: int = 0,
):
    return detect_lines_paddle_det(
        image_path,
        page_number=page_number,
        model_name=str(_config.get("paddle_det_model", "PP-OCRv5_server_det") or "PP-OCRv5_server_det"),
        device=str(_config.get("paddle_det_device", "gpu:0") or "gpu:0"),
        min_confidence=float(_config.get("paddle_det_min_confidence", 0.0) or 0.0),
        paddle_python=str(_config.get("paddle_det_python", "") or "") or None,
        paddle_source_dir=str(_config.get("paddle_det_source_dir", "") or "") or None,
    )


# ---------------------------------------------------------------------------
# 客户端行分割（参考 AIH-Contexture 方案）
# ---------------------------------------------------------------------------

def _detect_line_positions(image_path: Path) -> list[tuple[int, int, int, int]]:
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

    # 轻度水平膨胀（连接同行字符，但不跨行）
    kernel_width = max(w // 300, 3)  # 减少膨胀避免行间连接
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    dilated = cv2.dilate(binary, kernel)

    # 水平投影 → 行 y 范围
    h_proj = np.sum(dilated, axis=1) / 255.0

    # 阈值（降低以更容易检测行间隙）
    threshold = w * 0.003

    in_line = h_proj > threshold

    # 自适应最小行高
    min_line_height = max(h // 250, 8)

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

    # 对每行做垂直投影 → x 范围，并过滤假行
    lines = []
    for y0, y1 in y_ranges:
        row_strip = binary[y0:y1, :]
        v_proj = np.sum(row_strip, axis=0) / 255.0
        line_h = y1 - y0

        # 检测实际文本宽度
        text_cols = np.where(v_proj > line_h * 0.1)[0]

        if len(text_cols) == 0:
            # 没有文本，跳过
            continue

        x0, x1 = int(text_cols[0]), int(text_cols[-1]) + 1
        text_width = x1 - x0
        width_ratio = text_width / w

        # 过滤宽度异常的行
        if width_ratio > 0.98:  # 横跨整页的横线
            continue
        if width_ratio < 0.03:  # 很窄的装饰线
            continue

        # 过滤长宽比异常的行
        aspect_ratio = text_width / line_h
        if aspect_ratio > 500:  # 极扁（横线）
            continue
        if aspect_ratio < 0.5:  # 极窄（装饰线）
            continue

        lines.append((x0, y0, x1, y1))

    return lines


def _detect_word_positions(image_path: Path, y0: int, y1: int, min_word_width: int = 5, min_gap: int = 3) -> list[tuple[int, int]]:
    """检测一行中每个词的 x 坐标范围，返回 [(x0, x1), ...]."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h, w = img.shape
    if y0 < 0 or y1 > h or y0 >= y1:
        return []

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_strip = binary[y0:y1, :]

    # 垂直投影
    v_proj = np.sum(row_strip, axis=0) / 255.0
    line_h = y1 - y0
    threshold = max(line_h * 0.1, 2)
    in_word = v_proj > threshold

    # 检测词边界
    words = []
    start = None
    for i in range(len(in_word)):
        if in_word[i] and start is None:
            start = i
        elif not in_word[i] and start is not None:
            if i - start >= min_word_width:
                words.append((start, i))
            start = None
    if start is not None and len(in_word) - start >= min_word_width:
        words.append((start, len(in_word)))

    # 合并间距太小的词（可能是字符间距）
    if len(words) > 1:
        merged = [words[0]]
        for i in range(1, len(words)):
            prev_x1 = merged[-1][1]
            curr_x0, curr_x1 = words[i]
            if curr_x0 - prev_x1 < min_gap:
                merged[-1] = (merged[-1][0], curr_x1)
            else:
                merged.append((curr_x0, curr_x1))
        words = merged

    return words


def _segment_lines_client(image_path: Path, padding: int = 5) -> list[bytes]:
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

    # 轻度水平膨胀（连接同行字符，但不跨行）
    kernel_width = max(w // 300, 3)  # 减少膨胀避免行间连接
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    dilated = cv2.dilate(binary, kernel)

    # 在膨胀后的图像上做投影
    h_proj = np.sum(dilated, axis=1) / 255.0

    # 阈值（降低以更容易检测行间隙）
    threshold = w * 0.003

    in_line = h_proj > threshold

    # 自适应最小行高
    min_line_height = max(h // 250, 8)

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

    # 过滤假行（横线、装饰线等）
    filtered_lines = []
    for y0, y1 in lines:
        # 检测实际文本宽度
        row_strip = binary[y0:y1, :]
        v_proj = np.sum(row_strip, axis=0) / 255.0
        line_h = y1 - y0

        # 找到有文本的列
        text_cols = np.where(v_proj > line_h * 0.1)[0]

        if len(text_cols) == 0:
            # 没有文本，跳过
            continue

        text_width = text_cols[-1] - text_cols[0] + 1
        width_ratio = text_width / w

        # 过滤宽度异常的行
        if width_ratio > 0.98:  # 横跨整页的横线
            continue
        if width_ratio < 0.03:  # 很窄的装饰线
            continue

        # 过滤长宽比异常的行
        aspect_ratio = text_width / line_h
        if aspect_ratio > 500:  # 极扁（横线）
            continue
        if aspect_ratio < 0.5:  # 极窄（装饰线）
            continue

        filtered_lines.append((y0, y1))

    lines = filtered_lines

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

def _gs_render_page(pdf_path: Path, pageno: int, output_path: Path, dpi: int = 400):
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


def _detect_skew_by_projection(image_path: Path, angle_range: float = 10.0) -> float:
    """使用投影法检测图像倾斜角度（适用于 Fraktur 等 OSD 无法识别的字体）。

    Args:
        image_path: 图像路径
        angle_range: 检测角度范围（±degrees）

    Returns:
        float: 检测到的倾斜角度
    """
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0

    # 二值化
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 在角度范围内搜索
    angles = np.arange(-angle_range, angle_range + 0.5, 0.5)
    variances = []

    h, w = binary.shape
    center = (w // 2, h // 2)

    for angle in angles:
        # 旋转图像
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_LINEAR)

        # 计算水平投影方差
        projection = np.sum(rotated, axis=1)
        variance = np.var(projection)
        variances.append(variance)

    # 找到方差最大的角度
    best_idx = np.argmax(variances)
    best_angle = angles[best_idx]

    return float(best_angle)


def _detect_and_rotate_if_needed(image_path: Path) -> dict:
    """使用 Tesseract OSD 检测页面方向并纠偏，返回角度信息。

    Returns:
        dict: {'rotate': 90, 'deskew': 2.5, 'confidence': 8.5}
    """
    print(f"[DESKEW] Detecting: {image_path.name}", flush=True)
    try:
        result = subprocess_run(
            ['tesseract', str(image_path), 'stdout', '--psm', '0', '-l', 'osd'],
            capture_output=True, text=True, timeout=10
        )

        if result.returncode != 0:
            if 'osd.traineddata' in result.stderr.lower():
                print(f"[OSD] Warning: OSD data file not found", flush=True)
            return {'rotate': 0, 'deskew': 0, 'confidence': 0}

        # 解析OSD输出
        rotate_angle = 0
        orientation_degrees = 0.0
        confidence = 0.0

        for line in result.stdout.split('\n'):
            if 'Rotate:' in line:
                try:
                    rotate_angle = int(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif 'Orientation in degrees:' in line:
                try:
                    orientation_degrees = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif 'Orientation confidence:' in line:
                try:
                    confidence = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass

        # 输出原始检测结果
        print(f"[DESKEW] OSD raw: rotate={rotate_angle}, orient={orientation_degrees:.2f}°, conf={confidence:.2f}", flush=True)

        # 如果 OSD 未检测到倾斜，使用投影法补充检测（适用于 Fraktur 等字体）
        if abs(orientation_degrees) < 0.5 and confidence >= 1.0:
            projection_angle = _detect_skew_by_projection(image_path)
            if abs(projection_angle) >= 0.5:
                print(f"[DESKEW] Projection detected: {projection_angle:.2f}°", flush=True)
                orientation_degrees = projection_angle
                confidence = 10.0  # 投影法置信度设为固定值

        # 置信度检查（降低阈值以适应 Fraktur 等历史字体）
        if confidence < 1.0:
            if rotate_angle != 0 or abs(orientation_degrees) > 0.5:
                print(f"[DESKEW] Low confidence ({confidence:.2f}), skipping", flush=True)
            return {'rotate': 0, 'deskew': 0, 'confidence': confidence}

        # 角度范围限制
        if abs(orientation_degrees) > 10:
            print(f"[DESKEW] Angle too large ({orientation_degrees:.2f}°), skipping", flush=True)
            orientation_degrees = 0
        if abs(orientation_degrees) < 0.5:
            orientation_degrees = 0

        # 如果不需要任何旋转，直接返回
        if rotate_angle == 0 and orientation_degrees == 0:
            return {'rotate': 0, 'deskew': 0, 'confidence': confidence}

        # 应用旋转和纠偏
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[DESKEW] Failed to read image: {image_path}", flush=True)
            return {'rotate': 0, 'deskew': 0, 'confidence': 0}

        h, w = img.shape[:2]
        modified = False

        # 先应用90°旋转
        if rotate_angle == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            modified = True
        elif rotate_angle == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
            modified = True
        elif rotate_angle == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            modified = True

        # 再应用小角度纠偏
        if abs(orientation_degrees) > 0.5:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, orientation_degrees, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
            modified = True

        # 保存修改后的图像
        if modified:
            cv2.imwrite(str(image_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        return {
            'rotate': rotate_angle,
            'deskew': orientation_degrees,
            'confidence': confidence
        }

    except FileNotFoundError:
        print(f"[DESKEW] Tesseract not found in PATH", flush=True)
        return {'rotate': 0, 'deskew': 0, 'confidence': 0}
    except Exception as e:
        print(f"[DESKEW] Unexpected error: {e}", flush=True)
        return {'rotate': 0, 'deskew': 0, 'confidence': 0}


def _save_deskew_angle(pageno: int, deskew_info: dict):
    """保存纠偏角度到缓存."""
    if _preflight_cache_dir:
        import json
        cache_file = _preflight_cache_dir / f"page_{pageno:06d}_deskew.json"
        cache_file.write_text(json.dumps(deskew_info), encoding='utf-8')


def _read_cached_deskew_angle(input_file: Path) -> dict | None:
    """从缓存读取纠偏角度."""
    import json
    cache_dir = os.environ.get("CALAMARI_PREFLIGHT_CACHE")
    if cache_dir:
        m = re.match(r"(\d+)_", input_file.name)
        if m:
            pageno = int(m.group(1)) - 1
            cache_file = Path(cache_dir) / f"page_{pageno:06d}_deskew.json"
            if cache_file.exists():
                return json.loads(cache_file.read_text(encoding='utf-8'))
    return None


def _apply_deskew(image_path: Path, deskew_info: dict) -> Path:
    """应用已知角度的纠偏，返回临时文件路径."""
    try:
        import cv2
        import tempfile

        img = cv2.imread(str(image_path))
        if img is None:
            log.warning(f"Failed to read image: {image_path}")
            return image_path

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        modified = False

        # 先90°旋转
        if deskew_info['rotate'] == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            modified = True
        elif deskew_info['rotate'] == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
            modified = True
        elif deskew_info['rotate'] == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            modified = True

        # 再小角度纠偏
        if abs(deskew_info['deskew']) > 0.5:
            M = cv2.getRotationMatrix2D(center, deskew_info['deskew'], 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
            modified = True

        # 如果没有修改，直接返回原图
        if not modified:
            return image_path

        # 保存到临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        success = cv2.imwrite(temp_file.name, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if not success:
            log.warning(f"Failed to write deskewed image")
            return image_path

        return Path(temp_file.name)
    except Exception as e:
        log.warning(f"Apply deskew failed: {e}")
        return image_path


def _preflight_render_and_segment(
    pdf_path: Path,
    page_numbers: list[int],
    dpi: int,
    tmpdir: Path,
    workers: int = 1,
    use_tesseract: bool = False,
    binarize: bool = True,
    line_detector: str = "projection_debug",
) -> dict[int, list[bytes]]:
    """渲染多页并分割行图片. 返回 {pageno: [line_png_bytes, ...]}."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    lineboxes_cache: dict[int, list[_ChromeLineBox]] = {}

    def _render_one_page(
        pageno: int,
    ) -> tuple[int, list[bytes], str | None, list[_ChromeLineBox] | None]:
        """渲染单页并切行."""
        out_png = tmpdir / f"page_{pageno:06d}.png"
        hocr_content = None
        detected_line_boxes = None
        try:
            _gs_render_page(pdf_path, pageno, out_png, dpi)

            # OSD 检测和纠偏
            if _config.get("deskew_enabled", False):
                deskew_info = _detect_and_rotate_if_needed(out_png)
            else:
                deskew_info = {'rotate': 0, 'deskew': 0, 'confidence': 0}

            # 保存角度到缓存
            _save_deskew_angle(pageno - 1, deskew_info)  # GS 1-indexed → 0-indexed

            if deskew_info['rotate'] != 0 or deskew_info['deskew'] != 0:
                print(f"[DESKEW] Page {pageno}: rotate={deskew_info['rotate']}°, "
                      f"deskew={deskew_info['deskew']:.2f}°, conf={deskew_info['confidence']:.2f}", flush=True)

            # 使用 Tesseract / Chrome ScreenAI / 投影法 分割
            if use_tesseract and _check_tesseract_available():
                try:
                    tesseract_hocr = _get_tesseract_hocr(out_png, "deu")
                    hocr_content = tesseract_hocr  # 保存 hOCR
                    line_bboxes = _parse_line_bboxes_from_hocr(tesseract_hocr)
                    if line_bboxes:
                        lines = _crop_line_images_by_bbox(out_png, line_bboxes, binarize=binarize)
                        print(f"[DIAG] Preflight page {pageno}: Tesseract detected {len(line_bboxes)} lines, cropped {len(lines)}", flush=True)
                    else:
                        lines = _segment_lines_client(out_png)
                        print(f"[DIAG] Preflight page {pageno}: Tesseract failed, using projection: {len(lines)} lines", flush=True)
                except Exception as e:
                    print(f"[DIAG] Preflight page {pageno}: Tesseract error ({e}), using projection", flush=True)
                    lines = _segment_lines_client(out_png)
            elif line_detector == "chrome_screenai":
                line_boxes = _detect_lines_with_chrome_screenai(out_png)
                detected_line_boxes = line_boxes
                lines = _crop_line_images_by_lineboxes(
                    out_png,
                    line_boxes,
                    binarize=binarize,
                )
                print(
                    f"[DIAG] Preflight page {pageno}: Chrome ScreenAI detected {len(line_boxes)} lines, cropped {len(lines)}",
                    flush=True,
                )
            elif line_detector == "paddle_det":
                line_boxes = _detect_lines_with_paddle_det(out_png, page_number=pageno - 1)
                detected_line_boxes = line_boxes
                lines = _crop_line_images_by_lineboxes(
                    out_png,
                    line_boxes,
                    binarize=binarize,
                )
                print(
                    f"[DIAG] Preflight page {pageno}: PaddleOCR detected {len(line_boxes)} lines, cropped {len(lines)}",
                    flush=True,
                )
            else:
                lines = _segment_lines_client(out_png)

            log.debug("Preflight page %d: %d lines", pageno, len(lines))
            return (pageno, lines, hocr_content, detected_line_boxes)
        except Exception as e:
            if line_detector in ("chrome_screenai", "paddle_det"):
                raise
            log.warning("Preflight render failed for page %d: %s", pageno, e)
            return (pageno, [], None, None)
        finally:
            if out_png.exists():
                out_png.unlink()

    result: dict[int, list[bytes]] = {}

    if workers > 1:
        # 并行渲染
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_render_one_page, pn): pn for pn in page_numbers}
            for future in as_completed(futures):
                pageno, lines, hocr, detected_line_boxes = future.result()
                result[pageno] = lines
                # 缓存 hOCR（如果有）
                if hocr:
                    _preflight_cache_write_hocr(pageno - 1, hocr, dpi)  # GS 1-indexed → 0-indexed
                if detected_line_boxes is not None:
                    lineboxes_cache[pageno] = detected_line_boxes
    else:
        # 串行渲染
        for pageno in page_numbers:
            _, lines, hocr, detected_line_boxes = _render_one_page(pageno)
            result[pageno] = lines
            # 缓存 hOCR（如果有）
            if hocr:
                _preflight_cache_write_hocr(pageno - 1, hocr, dpi)  # GS 1-indexed → 0-indexed
            if detected_line_boxes is not None:
                lineboxes_cache[pageno] = detected_line_boxes

    if line_detector in ("chrome_screenai", "paddle_det"):
        for pageno, detected_line_boxes in lineboxes_cache.items():
            _preflight_cache_write_lineboxes(
                pageno - 1,
                detected_line_boxes,
                dpi,
                line_detector,
            )

    return result


def _preflight_ocr_chunk(
    client: "CalamariClient",
    pages_lines: dict[int, list[bytes]],
    model: str,
) -> dict[int, str]:
    """对一个 chunk 的行图片做有界批量 OCR，返回 {pageno: text}.

    文件名编码: image_PPPPPP_LLLLLL.png (P=页码, L=行号)
    这样可以从返回结果中还原页码和行号。
    """
    batch_files = []
    batch_bytes = 0
    page_line_counts: dict[int, int] = {}
    page_texts: dict[int, dict[int, str]] = {}
    batch_count = 0
    total_elapsed = 0.0
    max_batch_lines = max(1, int(_config.get("preflight_batch_lines", 512) or 512))
    max_batch_bytes = max(1, int(_config.get("preflight_batch_mb", 64) or 64)) * 1024 * 1024

    def flush_batch() -> bool:
        nonlocal batch_files, batch_bytes, batch_count, total_elapsed
        if not batch_files:
            return True
        batch_count += 1
        current_files = batch_files
        current_bytes = batch_bytes
        batch_files = []
        batch_bytes = 0

        log.info(
            "Preflight batch part %d: %d lines, %.1f MB",
            batch_count,
            len(current_files),
            current_bytes / (1024 * 1024),
        )
        try:
            resp = client._session.post(
                f"{client.base_url}/ocr/batch",
                files=current_files,
                data={"model": model, "mode": "indexed"},
                timeout=client.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            log.warning("Preflight batch part %d request failed: %s", batch_count, e)
            return False

        total_elapsed += float(payload.get("elapsed_seconds", 0) or 0)
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
        return True

    for pageno in sorted(pages_lines.keys()):
        lines = pages_lines[pageno]
        if not lines:
            continue
        page_line_counts[pageno] = len(lines)
        for line_idx, img_bytes in enumerate(lines):
            filename = f"image_{pageno:06d}_{line_idx:06d}.png"
            if (
                batch_files
                and (
                    len(batch_files) >= max_batch_lines
                    or batch_bytes + len(img_bytes) > max_batch_bytes
                )
                and not flush_batch()
            ):
                return {}
            batch_files.append(("files", (filename, img_bytes, "image/png")))
            batch_bytes += len(img_bytes)

    if not batch_files and not page_line_counts:
        return {}

    total_lines = sum(page_line_counts.values())
    log.info(
        "Preflight batch: %d pages, %d lines total, limit=%d lines/%.1f MB",
        len(page_line_counts), total_lines, max_batch_lines, max_batch_bytes / (1024 * 1024),
    )

    if not flush_batch():
        return {}

    # 按行号排序拼接（保留空行以保持索引对齐）
    result: dict[int, str] = {}
    for pageno in sorted(page_texts.keys()):
        lines_dict = page_texts[pageno]
        ordered = [lines_dict.get(i, "") for i in range(max(lines_dict.keys()) + 1)]
        result[pageno] = "\n".join(ordered)

    log.info(
        "Preflight batch done: %d lines in %d part(s), %.1fs (%.1f lines/s)",
        total_lines, batch_count, total_elapsed, total_lines / max(total_elapsed, 0.01),
    )
    return result


def _run_preflight(pdfinfo: "PdfInfo", options: "OcrOptions"):
    """预飞主流程：渲染 → 分割 → 分块批量 OCR → 写入磁盘缓存."""
    global _preflight_cache_dir

    # 限制OpenBLAS线程数，避免内存分配错误
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # 创建持久缓存目录（不用 TemporaryDirectory 的 context manager，
    # 因为 worker 进程需要在之后读取；OCRmyPDF 结束时会清理 work_folder）
    _preflight_cache_dir = Path(tempfile.mkdtemp(prefix="calamari_preflight_"))
    log.info("Preflight cache dir: %s", _preflight_cache_dir)

    # 立即设置环境变量，供 _preflight_cache_write_hocr 使用
    os.environ["CALAMARI_PREFLIGHT_CACHE"] = str(_preflight_cache_dir)

    pdf_path = Path(options.input_file)
    total_pages = len(pdfinfo)
    chunk_size = _config["preflight_chunk"]
    dpi = _config["preflight_dpi"]
    workers = _config["preflight_workers"]
    model = _config["model"]
    use_tesseract = _get_option(options, "calamari_use_tesseract_segmentation", False)
    binarize_lines = _get_option(options, "calamari_binarize_lines", True)
    line_detector = str(_config.get("line_detector", "projection_debug"))
    client = _get_client()

    print(f"[DIAG] Preflight: use_tesseract_segmentation={use_tesseract}, binarize_lines={binarize_lines}", flush=True)
    print(f"[DIAG] Preflight: line_detector={line_detector}", flush=True)

    log.info(
        "Preflight OCR: %d pages, chunk=%d, dpi=%d, workers=%d, model=%s, tesseract=%s, line_detector=%s",
        total_pages, chunk_size, dpi, workers, model, use_tesseract, line_detector,
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
                pdf_path,
                chunk_pages,
                dpi,
                render_path,
                workers,
                use_tesseract,
                binarize_lines,
                line_detector,
            )
            if line_detector in ("chrome_screenai", "paddle_det"):
                missing_linebox_pages = [
                    pageno for pageno in chunk_pages
                    if not (
                        _preflight_cache_dir
                        / f"page_{pageno - 1:06d}.{line_detector}_lineboxes.json"
                    ).exists()
                ]
                if missing_linebox_pages:
                    raise RuntimeError(
                        f"{line_detector} linebox cache missing for pages "
                        f"{missing_linebox_pages[:10]}"
                    )
            chunk_results = _preflight_ocr_chunk(client, pages_lines, model)

            # 写入磁盘缓存 (GS 1-indexed → 0-indexed)
            for gs_pageno, text in chunk_results.items():
                _preflight_cache_write(gs_pageno - 1, text)
                cached += 1
            for gs_pageno in chunk_pages:
                if gs_pageno not in chunk_results and not pages_lines.get(gs_pageno):
                    _preflight_cache_write(gs_pageno - 1, "")
                    cached += 1

    # 把缓存目录路径存到 _config，worker 进程通过 check_options 重建时不会丢失
    # 但 Windows spawn 模式下 _config 也会丢失，所以还需要通过 options 传递
    # 环境变量已在创建缓存目录时设置（line 1038）
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

        # 强制使用批量端点，即使只有1行（/ocr/batch 可以正确处理单行）
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
                return "\n".join(texts)

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
                texts.append(t)
            except Exception as e:
                log.warning("Sequential line %d failed: %s", i, e)
                texts.append("")
        return "\n".join(texts)

    def ocr_batch(
        self, image_paths: list[Path] | list[bytes], model: str, mode: str = "indexed"
    ) -> list[dict]:
        """批量 OCR（支持文件路径或字节数据）."""
        files = []
        for i, p in enumerate(image_paths):
            filename = f"image_{i:06d}.png"
            if isinstance(p, bytes):
                files.append(("files", (filename, p, "image/png")))
            else:
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
                if isinstance(p, bytes):
                    text = self._sequential_lines([p], model)[0:]
                else:
                    text = self.ocr_single(p, model)
                results.append({"filename": f"image_{i:06d}.png", "text": text, "index": i})
            return results
        finally:
            for _, (_, fobj, _) in files:
                if hasattr(fobj, 'close'):
                    fobj.close()

    def ocr_batch_ordered_texts(
        self, image_paths: list[Path] | list[bytes], model: str, mode: str = "indexed"
    ) -> list[str]:
        """批量 OCR 并按提交索引返回文本。"""
        results = self.ocr_batch(image_paths, model, mode=mode)
        payload = {"results": results}
        return self._parse_and_reorder(payload, len(image_paths))


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
        """使用 Tesseract OSD 检测页面方向并返回旋转角度"""
        print(f"[OSD] Detecting orientation for {input_file.name}", flush=True)

        try:
            # 调用 Tesseract OSD (Orientation and Script Detection)
            result = subprocess_run(
                ['tesseract', str(input_file), 'stdout', '--psm', '0', '-l', 'osd'],
                capture_output=True,
                text=True,
                timeout=10,
                env=os.environ
            )

            # 检查命令是否成功
            if result.returncode != 0:
                print(f"[OSD] Tesseract OSD failed with code {result.returncode}", flush=True)
                if result.stderr:
                    print(f"[OSD] Error: {result.stderr.strip()}", flush=True)
                return OrientationConfidence(angle=0, confidence=0.0)

            # 解析输出获取旋转角度和置信度
            angle = 0
            confidence = 0.0

            for line in result.stdout.split('\n'):
                if 'Rotate:' in line:
                    try:
                        angle = int(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'Orientation confidence:' in line:
                    try:
                        confidence = float(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass

            print(f"[OSD] Detected: angle={angle}°, confidence={confidence:.2f}", flush=True)

            # 置信度阈值检查：低于 5.0 认为不可靠
            if confidence < 5.0:
                print(f"[OSD] Confidence too low ({confidence:.2f} < 5.0), using angle=0", flush=True)
                return OrientationConfidence(angle=0, confidence=0.0)

            print(f"[OSD] Applying rotation: {angle}°", flush=True)
            return OrientationConfidence(angle=angle, confidence=confidence)

        except FileNotFoundError:
            print("[OSD] Tesseract not found, fallback to angle=0", flush=True)
            return OrientationConfidence(angle=0, confidence=0.0)
        except Exception as e:
            print(f"[OSD] Failed: {e}, fallback to angle=0", flush=True)
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
        print(f"[DIAG] generate_hocr called: output_hocr={output_hocr}, output_text={output_text}", flush=True)

        # ===== 纠偏处理 =====
        temp_file_to_cleanup = None
        original_input_file = input_file  # 保存原始文件名用于缓存匹配
        cached_angle = _read_cached_deskew_angle(input_file)

        if cached_angle and (cached_angle['rotate'] != 0 or cached_angle['deskew'] != 0):
            # 有缓存角度，应用纠偏
            print(f"[DESKEW] Applying cached angle: {cached_angle}", flush=True)
            deskewed_file = _apply_deskew(input_file, cached_angle)
            if deskewed_file != input_file:
                temp_file_to_cleanup = deskewed_file
                input_file = deskewed_file
        elif _config.get("preflight") and _config.get("deskew_enabled"):
            # Preflight模式但缓存丢失 - 异常情况
            log.warning(f"Preflight deskew cache missing for {input_file.name}")
        elif _config.get("deskew_enabled"):
            # 非Preflight模式，检测并纠偏
            deskew_info = _detect_and_rotate_if_needed(input_file)
            if deskew_info['rotate'] != 0 or deskew_info['deskew'] != 0:
                print(f"[DESKEW] Detected angle: {deskew_info}", flush=True)
                deskewed_file = _apply_deskew(input_file, deskew_info)
                if deskewed_file != input_file:
                    temp_file_to_cleanup = deskewed_file
                    input_file = deskewed_file

        # ===== 原有逻辑开始 =====
        model = _config["model"]
        if options.languages:
            model = options.languages[0]

        # 初始化 Tesseract 行分割变量
        tesseract_line_bboxes = None
        tesseract_hocr_full = None

        # ---- 尝试从预飞缓存读取 ----
        # input_file 路径格式: .../000001_ocr.png (1-indexed)
        # 用环境变量传递缓存目录（跨进程安全，Windows spawn 兼容）
        text = None
        use_tesseract_seg = _get_option(options, "calamari_use_tesseract_segmentation", False)
        binarize_lines = _get_option(options, "calamari_binarize_lines", True)
        line_detector = str(_config.get("line_detector", "projection_debug"))

        cache_dir = os.environ.get("CALAMARI_PREFLIGHT_CACHE")
        if cache_dir:
            m = re.match(r"(\d+)_", original_input_file.name)
            if m:
                pageno = int(m.group(1)) - 1
                cache_file = Path(cache_dir) / f"page_{pageno:06d}.txt"
                if cache_file.exists():
                    text = cache_file.read_text(encoding="utf-8")
                    print(f"[DIAG] Preflight cache hit: page {pageno}", flush=True)
                    print(f"[DIAG] use_tesseract_seg={use_tesseract_seg}, tesseract_available={_check_tesseract_available()}", flush=True)

                    if line_detector == "paddle_det":
                        try:
                            cached_lineboxes_data = _preflight_cache_read_lineboxes(pageno, "paddle_det")
                            if not cached_lineboxes_data:
                                raise RuntimeError(f"PaddleOCR linebox cache missing for page {pageno}")
                            paddle_line_boxes, source_dpi = cached_lineboxes_data
                            if not paddle_line_boxes:
                                _write_empty_hocr_page(
                                    input_file,
                                    output_hocr,
                                    output_text,
                                    f"PaddleOCR cache has no line boxes for page {pageno}",
                                )
                                return
                            target_dpi = _get_image_dpi(input_file)
                            if source_dpi != target_dpi:
                                paddle_line_boxes = _scale_lineboxes_dpi(
                                    paddle_line_boxes,
                                    source_dpi,
                                    target_dpi,
                                )
                            paddle_line_boxes = _scale_lineboxes_to_image(
                                paddle_line_boxes,
                                input_file,
                            )
                            texts = text.splitlines()
                            print(
                                f"[DIAG] Preflight PaddleOCR cache hit: {len(paddle_line_boxes)} boxes, {len(texts)} texts",
                                flush=True,
                            )
                            final_hocr = _build_simple_hocr_from_lineboxes(
                                input_file,
                                paddle_line_boxes,
                                texts,
                                dpi_x=target_dpi,
                                dpi_y=target_dpi,
                            )
                            output_hocr.write_text(final_hocr, encoding="utf-8")
                            output_text.write_text("\n".join(texts), encoding="utf-8")
                            print("[SUCCESS] Preflight: Generated hOCR from PaddleOCR line boxes", flush=True)
                            return
                        except Exception as e:
                            raise RuntimeError(
                                f"Preflight PaddleOCR hOCR failed for page {pageno}: {e}"
                            ) from e

                    if line_detector == "chrome_screenai":
                        try:
                            cached_lineboxes_data = _preflight_cache_read_lineboxes(pageno, "chrome_screenai")
                            if cached_lineboxes_data:
                                chrome_line_boxes, source_dpi = cached_lineboxes_data
                                target_dpi = _get_image_dpi(input_file)
                                if source_dpi != target_dpi:
                                    chrome_line_boxes = _scale_lineboxes_dpi(
                                        chrome_line_boxes,
                                        source_dpi,
                                        target_dpi,
                                    )
                                chrome_line_boxes = _scale_lineboxes_to_image(
                                    chrome_line_boxes,
                                    input_file,
                                )
                                texts = text.splitlines()
                                print(
                                    f"[DIAG] Preflight Chrome cache hit: {len(chrome_line_boxes)} boxes, {len(texts)} texts",
                                    flush=True,
                                )
                                final_hocr = _build_simple_hocr_from_lineboxes(
                                    input_file,
                                    chrome_line_boxes,
                                    texts,
                                    dpi_x=target_dpi,
                                    dpi_y=target_dpi,
                                )
                                output_hocr.write_text(final_hocr, encoding="utf-8")
                                output_text.write_text("\n".join(texts), encoding="utf-8")
                                print("[SUCCESS] Preflight: Generated hOCR from cached Chrome ScreenAI line boxes", flush=True)
                                return
                            raise RuntimeError(
                                f"Preflight Chrome linebox cache missing for page {pageno}"
                            )
                        except Exception as e:
                            raise RuntimeError(
                                f"Preflight Chrome hOCR cache failed for page {pageno}: {e}"
                            ) from e

                    # Preflight模式下也使用Tesseract hOCR替换方案
                    if use_tesseract_seg and _check_tesseract_available():
                        try:
                            # 尝试读取缓存的 hOCR
                            cached_hocr_data = _preflight_cache_read_hocr(pageno)

                            if cached_hocr_data:
                                cached_hocr, source_dpi = cached_hocr_data
                                print(f"[DIAG] Using cached hOCR from Preflight (source DPI: {source_dpi})", flush=True)

                                # 获取目标 DPI
                                target_dpi = _get_image_dpi(input_file)
                                print(f"[DIAG] Target image DPI: {target_dpi}", flush=True)

                                # 如果 DPI 不同，缩放 bbox
                                if source_dpi != target_dpi:
                                    print(f"[DIAG] Scaling hOCR bbox from {source_dpi} to {target_dpi} DPI", flush=True)
                                    cached_hocr = _scale_hocr_dpi(cached_hocr, source_dpi, target_dpi)

                                tesseract_hocr_full = cached_hocr
                            else:
                                # Fallback：重新调用 Tesseract
                                print("[DIAG] Cached hOCR not found, calling Tesseract", flush=True)
                                language = options.languages[0] if options.languages else "deu"
                                tesseract_hocr_full = _get_tesseract_hocr(input_file, language)

                            # 将缓存文本按行分割
                            calamari_texts = text.splitlines()
                            print(f"[DIAG] Preflight: Replacing {len(calamari_texts)} texts in hOCR", flush=True)
                            print(f"[DIAG] First 3 texts: {calamari_texts[:3]}", flush=True)

                            # 替换hOCR中的文本
                            final_hocr = _replace_hocr_line_texts(tesseract_hocr_full, calamari_texts)
                            output_hocr.write_text(final_hocr, encoding="utf-8")
                            output_text.write_text("\n".join(calamari_texts), encoding="utf-8")
                            print(f"[SUCCESS] Preflight: Generated hOCR preserving all lines", flush=True)
                            return
                        except Exception as e:
                            print(f"[DIAG] Preflight Tesseract hOCR failed: {e}, using fallback", flush=True)

        # ---- 缓存未命中：正常 OCR ----
        if text is None:
            client = _get_client()
            seg_mode = _config.get("segmentation", "server")
            use_tesseract_seg = _get_option(options, "calamari_use_tesseract_segmentation", False)
            if line_detector == "tesseract":
                use_tesseract_seg = True

            # 使用 Tesseract 行分割模式
            # 暂时禁用，使用fallback路径
            if False and use_tesseract_seg and seg_mode == "client":
                print("[DIAG] Using Tesseract line segmentation mode", flush=True)

                # 检查 Tesseract 可用性
                if not _check_tesseract_available():
                    print("[DIAG] Tesseract not available, falling back", flush=True)
                    use_tesseract_seg = False

                if use_tesseract_seg:
                    try:
                        # 获取语言参数
                        language = options.languages[0] if options.languages else "deu"

                        # 调用 Tesseract 获取完整 hOCR
                        tesseract_hocr_full = _get_tesseract_hocr(input_file, language)

                        # 解析提取行 bbox
                        line_bboxes = _parse_line_bboxes_from_hocr(tesseract_hocr_full)

                        if line_bboxes:
                            print(f"[DIAG] Tesseract detected {len(line_bboxes)} lines", flush=True)

                            # 根据 bbox 切割行图像
                            line_images = _crop_line_images_by_bbox(input_file, line_bboxes, binarize=binarize_lines)
                            print(f"[DIAG] Cropped {len(line_images)} line images (binarize={binarize_lines})", flush=True)

                            if line_images and len(line_images) == len(line_bboxes):
                                # 发送给 Calamari 批量识别
                                results = client.ocr_batch(line_images, model, mode=_config["batch_mode"])
                                print(f"[DIAG] Calamari returned {len(results)} results", flush=True)

                                # 提取文本（保留所有结果，包括空字符串）
                                calamari_texts = [r.get("text", "") for r in results]
                                print(f"[DIAG] Replacing {len(calamari_texts)} texts in Tesseract hOCR", flush=True)

                                # 直接替换 Tesseract hOCR 中的文本，保留所有行结构
                                final_hocr = _replace_hocr_line_texts(tesseract_hocr_full, calamari_texts)

                                # 直接写入输出并返回
                                output_hocr.write_text(final_hocr, encoding="utf-8")
                                output_text.write_text("\n".join(calamari_texts), encoding="utf-8")
                                print(f"[SUCCESS] Generated hOCR with all {len(line_bboxes)} lines preserved", flush=True)
                                return
                            else:
                                print(f"[DIAG] Image crop mismatch: {len(line_images)} != {len(line_bboxes)}, falling back", flush=True)
                                use_tesseract_seg = False
                        else:
                            print("[DIAG] No lines detected by Tesseract, falling back", flush=True)
                            use_tesseract_seg = False
                    except Exception as e:
                        print(f"[DIAG] Tesseract segmentation failed: {e}, falling back", flush=True)
                        use_tesseract_seg = False

            # 降级到原有方法
            if not use_tesseract_seg:
                if seg_mode == "client" and line_detector in ("chrome_screenai", "paddle_det"):
                    if line_detector == "chrome_screenai":
                        line_boxes = _detect_lines_with_chrome_screenai(input_file)
                        label = "Chrome ScreenAI"
                        line_boxes = _scale_lineboxes_to_image(
                            line_boxes,
                            input_file,
                        )
                    else:
                        line_boxes = _detect_lines_with_paddle_det(input_file)
                        label = "PaddleOCR"
                    if not line_boxes:
                        _write_empty_hocr_page(
                            input_file,
                            output_hocr,
                            output_text,
                            f"{label} returned no line boxes",
                        )
                        return
                    print(f"[DIAG] {label} detected {len(line_boxes)} lines", flush=True)
                    line_images = _crop_line_images_by_lineboxes(
                        input_file,
                        line_boxes,
                        binarize=binarize_lines,
                    )
                    print(f"[DIAG] Cropped {len(line_images)} {label} line images", flush=True)
                    texts = client.ocr_batch_ordered_texts(
                        line_images,
                        model,
                        mode=_config["batch_mode"],
                    )
                    with Image.open(input_file) as img:
                        dpi_info = img.info.get("dpi", (300, 300))
                        dpi_x = int(round(dpi_info[0])) if dpi_info[0] else 300
                        dpi_y = int(round(dpi_info[1])) if dpi_info[1] else 300
                    final_hocr = _build_simple_hocr_from_lineboxes(
                        input_file,
                        line_boxes,
                        texts,
                        dpi_x=dpi_x,
                        dpi_y=dpi_y,
                    )
                    output_hocr.write_text(final_hocr, encoding="utf-8")
                    output_text.write_text("\n".join(texts), encoding="utf-8")
                    print(f"[SUCCESS] Generated hOCR from {label} line boxes", flush=True)
                    return
                if seg_mode == "client":
                    text = client.ocr_page_client_seg(input_file, model)
                else:
                    text = client.ocr_single(input_file, model)

        # 确保 text 已被赋值（防止某些路径下 text 仍为 None）
        if text is None:
            client = _get_client()
            model = _config["model"]
            if options.languages:
                model = options.languages[0]
            text = client.ocr_single(input_file, model)

        # 获取图片尺寸和 DPI 用于 hOCR bbox
        with Image.open(input_file) as img:
            w, h = img.size
            dpi_info = img.info.get("dpi", (300, 300))
            dpi_x = int(round(dpi_info[0])) if dpi_info[0] else 300
            dpi_y = int(round(dpi_info[1])) if dpi_info[1] else 300

        log.info(
            "hOCR diag: image=%s size=%dx%d dpi=(%d,%d) raw_dpi=%r",
            input_file.name, w, h, dpi_x, dpi_y, dpi_info,
        )

        # 生成 hOCR — 检测实际行位置
        text_lines = text.splitlines()
        non_empty_text_lines = sum(1 for ln in text_lines if ln.strip())
        log.info(
            "[DIAG] After splitlines preserving empty lines: total=%d non_empty=%d text length=%d",
            len(text_lines),
            non_empty_text_lines,
            len(text),
        )
        if not text_lines:
            text_lines = [text] if text.strip() else [""]

        # 检测图片中实际的文本行坐标
        use_tesseract_seg = _get_option(options, "calamari_use_tesseract_segmentation", False)
        detected = None

        # 如果启用了Tesseract但bbox为None（preflight缓存情况），获取bbox
        if use_tesseract_seg and tesseract_line_bboxes is None:
            print(f"[DIAG] Preflight mode: fetching Tesseract bboxes", flush=True)
            if _check_tesseract_available():
                try:
                    language = options.languages[0] if options.languages else "deu"
                    tesseract_hocr = _get_tesseract_hocr(input_file, language)
                    line_bboxes = _parse_line_bboxes_from_hocr(tesseract_hocr)
                    if line_bboxes:
                        tesseract_line_bboxes = line_bboxes
                        print(f"[DIAG] Tesseract detected {len(line_bboxes)} lines (preflight mode)", flush=True)
                except Exception as e:
                    print(f"[DIAG] Tesseract bbox extraction failed: {e}", flush=True)

        # 诊断日志
        log.info("[DIAG] use_tesseract_seg=%s, tesseract_line_bboxes is not None=%s",
                 use_tesseract_seg, tesseract_line_bboxes is not None)
        if tesseract_line_bboxes is not None:
            log.info("[DIAG] tesseract_line_bboxes exists with %d lines", len(tesseract_line_bboxes))

        # 如果使用了 Tesseract 分割且已有 bbox，直接使用
        if use_tesseract_seg and tesseract_line_bboxes is not None:
            detected = tesseract_line_bboxes
            log.info("[SUCCESS] Using Tesseract line bboxes: %d lines", len(detected))
        else:
            line_detector = str(_config.get("line_detector", "projection_debug"))
            if line_detector == "chrome_screenai":
                try:
                    chrome_line_boxes = _detect_lines_with_chrome_screenai(input_file)
                    detected = [tuple(line_box.bbox) for line_box in chrome_line_boxes]
                    log.info("[SUCCESS] Using Chrome ScreenAI line bboxes: %d lines", len(detected))
                except Exception as e:
                    log.warning("Chrome ScreenAI line detection failed in hOCR mapping: %s", e)
                    detected = None
            elif line_detector == "paddle_det":
                try:
                    paddle_line_boxes = _detect_lines_with_paddle_det(input_file)
                    detected = [tuple(line_box.bbox) for line_box in paddle_line_boxes]
                    log.info("[SUCCESS] Using PaddleOCR line bboxes: %d lines", len(detected))
                except Exception as e:
                    raise RuntimeError(f"PaddleOCR line detection failed in hOCR mapping: {e}") from e
            if detected is None:
                # 使用投影法检测
                log.warning("[FALLBACK] Using projection method instead of Tesseract!")
                detected = _detect_line_positions(input_file)
            if detected:
                log.info(
                    "hOCR diag: detected %d lines, first=%s last=%s",
                    len(detected), detected[0], detected[-1],
                )

        # 将 OCR 文本行映射到检测到的行位置 (x0, y0, x1, y1)
        print(f"[DIAG] Mapping: detected={len(detected) if detected else 0} bboxes, text_lines={len(text_lines)} lines", flush=True)

        if detected and len(detected) >= len(text_lines):
            positions = detected[:len(text_lines)]
            print(f"[DIAG] Case 1: Using first {len(positions)} bboxes (detected >= text_lines)", flush=True)
        elif detected:
            positions = detected
            print(f"[DIAG] Case 2: Using all {len(positions)} bboxes (detected < text_lines)", flush=True)
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

            # 分词并生成词级别 bbox
            words = line_text.split()
            if not words:
                continue

            # 检测词的 x 坐标
            word_x_ranges = _detect_word_positions(input_file, y0, y1)

            # 匹配词和坐标
            word_spans = []
            if word_x_ranges and abs(len(word_x_ranges) - len(words)) <= 2:
                # 数量接近，尝试匹配
                for j, word in enumerate(words):
                    if j < len(word_x_ranges):
                        wx0, wx1 = word_x_ranges[j]
                    else:
                        # 降级：基于字符数加权分配
                        total_chars = sum(len(w) for w in words)
                        if total_chars > 0:
                            char_offset = sum(len(words[k]) for k in range(j))
                            wx0 = x0 + int((x1 - x0) * char_offset / total_chars)
                            wx1 = x0 + int((x1 - x0) * (char_offset + len(word)) / total_chars)
                        else:
                            # 均匀分配
                            wx0 = x0 + int((x1 - x0) * j / len(words))
                            wx1 = x0 + int((x1 - x0) * (j + 1) / len(words))
                    escaped = html.escape(word)
                    word_spans.append(
                        f"<span class='ocrx_word' title='bbox {wx0} {y0} {wx1} {y1}'>{escaped}</span>"
                    )
            else:
                # 数量差异大，用基于字符数的加权分配
                total_chars = sum(len(w) for w in words)
                if total_chars == 0:
                    # 所有词都是空字符串，均匀分配
                    total_chars = len(words)
                    char_offset = 0
                    for j, word in enumerate(words):
                        wx0 = x0 + int((x1 - x0) * j / total_chars)
                        wx1 = x0 + int((x1 - x0) * (j + 1) / total_chars)
                        escaped = html.escape(word)
                        word_spans.append(
                            f"<span class='ocrx_word' title='bbox {wx0} {y0} {wx1} {y1}'>{escaped}</span>"
                        )
                else:
                    char_offset = 0
                    for j, word in enumerate(words):
                        wx0 = x0 + int((x1 - x0) * char_offset / total_chars)
                        wx1 = x0 + int((x1 - x0) * (char_offset + len(word)) / total_chars)
                        char_offset += len(word)
                        escaped = html.escape(word)
                        word_spans.append(
                            f"<span class='ocrx_word' title='bbox {wx0} {y0} {wx1} {y1}'>{escaped}</span>"
                        )

            line_spans.append(
                f"<span class='ocr_line' title='bbox {x0} {y0} {x1} {y1}'>"
                f"{' '.join(word_spans)}</span>"
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

        # 清理临时纠偏文件
        if temp_file_to_cleanup and temp_file_to_cleanup.exists():
            temp_file_to_cleanup.unlink(missing_ok=True)

    @staticmethod
    def generate_pdf(
        input_file: Path,
        output_pdf: Path,
        output_text: Path,
        options: OcrOptions,
    ) -> None:
        raise NotImplementedError("Use hOCR rendering path")


class ChromeScreenAIOcrEngine(OcrEngine):
    @staticmethod
    def version() -> str:
        try:
            backend = ChromeScreenAIBackend(
                light_mode=bool(_config.get("chrome_screenai_light", False)),
                force_horizontal=bool(_config.get("chrome_screenai_force_horizontal", True)),
            )
            version = backend._get_screen_ai().version
            return f"{version[0]}.{version[1]}"
        except Exception:
            return "unavailable"

    @staticmethod
    def creator_tag(options: OcrOptions) -> str:
        return f"Chrome ScreenAI OCR {ChromeScreenAIOcrEngine.version()}"

    def __str__(self) -> str:
        return f"Chrome ScreenAI OCR {self.version()}"

    @staticmethod
    def languages(options: OcrOptions) -> Set[str]:
        return {"eng"}

    @staticmethod
    def get_orientation(
        input_file: Path, options: OcrOptions
    ) -> OrientationConfidence:
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def supports_generate_ocr() -> bool:
        return True

    @staticmethod
    def generate_ocr(
        input_file: Path,
        options: OcrOptions,
        page_number: int = 0,
    ):
        backend = ChromeScreenAIBackend(
            light_mode=bool(_config.get("chrome_screenai_light", False)),
            force_horizontal=bool(_config.get("chrome_screenai_force_horizontal", True)),
        )
        filter_types = str(_config.get("chrome_screenai_filter_types", "printed"))
        min_confidence = float(_config.get("chrome_screenai_min_confidence", 0.0) or 0.0)
        return backend.generate_ocr_tree(
            input_file,
            page_number=page_number,
            filter_printed_text=(filter_types != "all"),
            min_confidence=min_confidence,
        )

    @staticmethod
    def generate_hocr(
        input_file: Path,
        output_hocr: Path,
        output_text: Path,
        options: OcrOptions,
    ) -> None:
        raise NotImplementedError("Chrome ScreenAI uses the direct generate_ocr path")

    @staticmethod
    def generate_pdf(
        input_file: Path,
        output_pdf: Path,
        output_text: Path,
        options: OcrOptions,
    ) -> None:
        raise NotImplementedError("Use fpdf2/direct OCR rendering path")


class PaddleOcrEngine(OcrEngine):
    @staticmethod
    def version() -> str:
        return str(_config.get("paddle_ocr_version", "PP-OCRv6") or "PP-OCRv6")

    @staticmethod
    def creator_tag(options: OcrOptions) -> str:
        return f"PaddleOCR {PaddleOcrEngine.version()}"

    def __str__(self) -> str:
        return f"PaddleOCR {self.version()}"

    @staticmethod
    def languages(options: OcrOptions) -> Set[str]:
        lang = str(_config.get("paddle_ocr_lang", "en") or "en")
        aliases = {lang, "eng", "en"}
        if lang == "ch":
            aliases.update({"chi_sim", "chi_tra"})
        if lang == "japan":
            aliases.add("jpn")
        return aliases

    @staticmethod
    def get_orientation(
        input_file: Path, options: OcrOptions
    ) -> OrientationConfidence:
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def generate_hocr(
        input_file: Path,
        output_hocr: Path,
        output_text: Path,
        options: OcrOptions,
    ) -> None:
        settings = PaddleOcrSettings(
            ocr_version=str(_config.get("paddle_ocr_version", "PP-OCRv6") or "PP-OCRv6"),
            lang=str(_config.get("paddle_ocr_lang", "en") or "en"),
            det_model_name=str(_config.get("paddle_ocr_det_model", "") or ""),
            rec_model_name=str(_config.get("paddle_ocr_rec_model", "") or ""),
            det_model_dir=str(_config.get("paddle_ocr_det_model_dir", "") or ""),
            rec_model_dir=str(_config.get("paddle_ocr_rec_model_dir", "") or ""),
            device=str(_config.get("paddle_ocr_device", "gpu:0") or "gpu:0"),
            engine=str(_config.get("paddle_ocr_engine", "") or ""),
            enable_mkldnn=_config.get("paddle_ocr_enable_mkldnn"),
            cpu_threads=_config.get("paddle_ocr_cpu_threads"),
            enable_hpi=_config.get("paddle_ocr_enable_hpi"),
            rec_batch_size=_config.get("paddle_ocr_rec_batch_size"),
            min_confidence=float(_config.get("paddle_ocr_min_confidence", 0.0) or 0.0),
            return_word_box=bool(_config.get("paddle_ocr_return_word_box", False)),
            paddle_python=str(_config.get("paddle_ocr_python", "") or ""),
            paddle_source_dir=str(_config.get("paddle_ocr_source_dir", "") or ""),
        )
        page_number = 0
        m = re.match(r"(\d+)_", input_file.name)
        if m:
            page_number = max(0, int(m.group(1)) - 1)
        result = run_paddle_ocr_page(
            input_file,
            page_number=page_number,
            settings=settings,
        )
        write_paddle_hocr(result, output_hocr, output_text)

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
    calamari_group.add_argument(
        "--calamari-preflight-batch-lines",
        type=int,
        default=512,
        help="预飞模式每个 Calamari 子批最多行数（默认 512）",
    )
    calamari_group.add_argument(
        "--calamari-preflight-batch-mb",
        type=int,
        default=64,
        help="预飞模式每个 Calamari 子批最多图像字节数 MB（默认 64）",
    )
    calamari_group.add_argument(
        "--calamari-use-tesseract-segmentation",
        action="store_true",
        default=False,
        help="使用 Tesseract 进行行分割（更准确，解决行合并问题）",
    )
    calamari_group.add_argument(
        "--calamari-binarize-lines",
        action="store_true",
        default=True,
        help="对切割的行图像进行 Otsu 二值化预处理（改善泛黄背景文档识别质量）",
    )
    calamari_group.add_argument(
        "--calamari-deskew",
        action="store_true",
        default=False,
        help="启用页面纠偏（小角度修正±10°，提高识别准确率）",
    )
    calamari_group.add_argument(
        "--calamari-line-detector",
        choices=["projection_debug", "tesseract", "chrome_screenai", "paddle_det"],
        default="paddle_det",
        help="客户端行检测器：projection_debug（旧投影法）、tesseract（legacy）、chrome_screenai 或 paddle_det。",
    )
    calamari_group.add_argument(
        "--chrome-screenai-light",
        action="store_true",
        default=False,
        help="Chrome ScreenAI 使用轻量模式。",
    )
    calamari_group.add_argument(
        "--chrome-screenai-min-confidence",
        type=float,
        default=0.0,
        help="Chrome ScreenAI 行检测最低置信度阈值。",
    )
    calamari_group.add_argument(
        "--chrome-screenai-filter-types",
        choices=["printed", "all"],
        default="printed",
        help="Chrome ScreenAI 检测结果过滤：printed 仅保留印刷文本，all 保留全部。",
    )
    calamari_group.add_argument(
        "--chrome-screenai-text-layer-mode",
        choices=["native_direct", "proxy_match"],
        default="native_direct",
        help="Chrome ScreenAI 文本层模式：native_direct 完全直通；proxy_match 由 OCRmyPDF-AIH 代理匹配文本层。",
    )
    calamari_group.add_argument(
        "--chrome-screenai-force-horizontal",
        action="store_true",
        default=True,
        help="Chrome ScreenAI 结果强制按横排书页坐标匹配：从上到下、同一行从左到右（默认启用）。",
    )
    calamari_group.add_argument(
        "--chrome-screenai-text-cleanup",
        choices=["mode_default", "clear_all", "rasterize"],
        default="mode_default",
        help="Chrome OCR 前文字层处理：默认按 mode；clear_all 轻量删除全部文字；rasterize 显式栅格化。",
    )
    calamari_group.add_argument(
        "--chrome-screenai-rasterize-dpi",
        type=int,
        default=300,
        help="Chrome 显式栅格化清理的 DPI。",
    )
    calamari_group.add_argument(
        "--chrome-screenai-rasterize-quality",
        type=int,
        default=88,
        help="Chrome 显式栅格化清理的 JPEG 质量（1-100）。",
    )
    calamari_group.add_argument(
        "--calamari-paddle-model",
        default="PP-OCRv5_server_det",
        help="PaddleOCR 文本检测模型名称。",
    )
    calamari_group.add_argument(
        "--calamari-paddle-device",
        default="gpu:0",
        help="PaddleOCR 推理设备，例如 gpu:0、cpu。",
    )
    calamari_group.add_argument(
        "--calamari-paddle-python",
        default="",
        help="外部 PaddleOCR Python 解释器路径；留空则自动发现。",
    )
    calamari_group.add_argument(
        "--calamari-paddle-source-dir",
        default="",
        help="外部 PaddleOCR 源码目录，用于 PYTHONPATH；留空则自动发现。",
    )
    calamari_group.add_argument(
        "--calamari-paddle-min-confidence",
        type=float,
        default=0.0,
        help="PaddleOCR 行检测最低置信度阈值。",
    )
    calamari_group.add_argument(
        "--paddle-ocr-version",
        choices=["auto", "PP-OCRv6", "PP-OCRv5"],
        default="PP-OCRv6",
        help="PaddleOCR 全链路 OCR 版本基线。",
    )
    calamari_group.add_argument(
        "--paddle-ocr-lang",
        default="en",
        help="PaddleOCR 全链路识别语言，例如 en、ch、japan。",
    )
    calamari_group.add_argument(
        "--paddle-ocr-det-model",
        default="",
        help="PaddleOCR 全链路检测模型名称；留空按版本自动选择。",
    )
    calamari_group.add_argument(
        "--paddle-ocr-rec-model",
        default="",
        help="PaddleOCR 全链路识别模型名称；留空按版本自动选择。",
    )
    calamari_group.add_argument(
        "--paddle-ocr-det-model-dir",
        default="",
        help="PaddleOCR 全链路检测模型目录。",
    )
    calamari_group.add_argument(
        "--paddle-ocr-rec-model-dir",
        default="",
        help="PaddleOCR 全链路识别模型目录。",
    )
    calamari_group.add_argument(
        "--paddle-ocr-device",
        default="gpu:0",
        help="PaddleOCR 全链路推理设备，例如 auto、gpu:0、cpu。",
    )
    calamari_group.add_argument(
        "--paddle-ocr-engine",
        default="",
        help="PaddleOCR 推理引擎，例如 paddle、paddle_static；留空用默认。",
    )
    calamari_group.add_argument(
        "--paddle-ocr-enable-mkldnn",
        action="store_true",
        default=False,
        help="PaddleOCR CPU 模式启用 MKL-DNN。",
    )
    calamari_group.add_argument(
        "--paddle-ocr-cpu-threads",
        type=int,
        default=None,
        help="PaddleOCR CPU 推理线程数。",
    )
    calamari_group.add_argument(
        "--paddle-ocr-enable-hpi",
        action="store_true",
        default=None,
        help="PaddleOCR 启用高性能推理。",
    )
    calamari_group.add_argument(
        "--paddle-ocr-rec-batch-size",
        type=int,
        default=16,
        help="PaddleOCR 识别 batch size。",
    )
    calamari_group.add_argument(
        "--paddle-ocr-min-confidence",
        type=float,
        default=0.0,
        help="PaddleOCR 全链路最低识别置信度。",
    )
    calamari_group.add_argument(
        "--paddle-ocr-return-word-box",
        action="store_true",
        default=False,
        help="请求 PaddleOCR 返回词级框（若当前版本支持）。",
    )
    calamari_group.add_argument(
        "--paddle-ocr-python",
        default="",
        help="PaddleOCR 全链路外部 Python 解释器路径；留空则自动发现。",
    )
    calamari_group.add_argument(
        "--paddle-ocr-source-dir",
        default="",
        help="PaddleOCR 全链路源码目录，用于 PYTHONPATH；留空则自动发现。",
    )


def _get_option(options, name, default=None):
    """从 options 或 extra_attrs 中获取参数值.

    插件参数未在 OcrOptions 中注册时会被存储在 extra_attrs 字典中。
    """
    value = getattr(options, name, None)
    if value is None:
        value = getattr(options, 'extra_attrs', {}).get(name, default)
    return value


@hookimpl
def check_options(options):
    global _client

    url = _get_option(options, "calamari_url") or _config["url"]
    model = _get_option(options, "calamari_model") or _config["model"]
    timeout = _get_option(options, "calamari_timeout") or _config["timeout"]
    batch_mode = _get_option(options, "calamari_batch_mode") or _config["batch_mode"]
    segmentation = _get_option(options, "calamari_segmentation") or _config["segmentation"]
    preflight = _get_option(options, "calamari_preflight", False)
    preflight_chunk = _get_option(options, "calamari_preflight_chunk") or _config["preflight_chunk"]
    preflight_dpi = _get_option(options, "calamari_preflight_dpi") or _config["preflight_dpi"]
    preflight_workers = _get_option(options, "calamari_preflight_workers") or _config["preflight_workers"]
    preflight_batch_lines = _get_option(options, "calamari_preflight_batch_lines") or _config["preflight_batch_lines"]
    preflight_batch_mb = _get_option(options, "calamari_preflight_batch_mb") or _config["preflight_batch_mb"]
    use_tesseract_seg = _get_option(options, "calamari_use_tesseract_segmentation", False)
    deskew_enabled = _get_option(options, "calamari_deskew", False)
    line_detector = _get_option(options, "calamari_line_detector") or _config["line_detector"]
    chrome_screenai_light = _get_option(options, "chrome_screenai_light", False)
    chrome_screenai_min_confidence = _get_option(options, "chrome_screenai_min_confidence", 0.0)
    chrome_screenai_filter_types = _get_option(options, "chrome_screenai_filter_types", "printed")
    chrome_screenai_text_layer_mode = _get_option(options, "chrome_screenai_text_layer_mode", "native_direct")
    chrome_screenai_force_horizontal = _get_option(options, "chrome_screenai_force_horizontal", True)
    chrome_screenai_text_cleanup = _get_option(options, "chrome_screenai_text_cleanup", "mode_default")
    chrome_screenai_rasterize_dpi = _get_option(options, "chrome_screenai_rasterize_dpi", 300)
    chrome_screenai_rasterize_quality = _get_option(options, "chrome_screenai_rasterize_quality", 88)
    paddle_det_model = _get_option(options, "calamari_paddle_model", "PP-OCRv5_server_det")
    paddle_det_device = _get_option(options, "calamari_paddle_device", "gpu:0")
    paddle_det_min_confidence = _get_option(options, "calamari_paddle_min_confidence", 0.0)
    paddle_det_python = _get_option(options, "calamari_paddle_python", "")
    paddle_det_source_dir = _get_option(options, "calamari_paddle_source_dir", "")
    paddle_ocr_version = _get_option(options, "paddle_ocr_version", "PP-OCRv6")
    paddle_ocr_lang = _get_option(options, "paddle_ocr_lang", "en")
    paddle_ocr_det_model = _get_option(options, "paddle_ocr_det_model", "")
    paddle_ocr_rec_model = _get_option(options, "paddle_ocr_rec_model", "")
    paddle_ocr_det_model_dir = _get_option(options, "paddle_ocr_det_model_dir", "")
    paddle_ocr_rec_model_dir = _get_option(options, "paddle_ocr_rec_model_dir", "")
    paddle_ocr_device = _get_option(options, "paddle_ocr_device", "gpu:0")
    paddle_ocr_engine = _get_option(options, "paddle_ocr_engine", "")
    paddle_ocr_enable_mkldnn = _get_option(options, "paddle_ocr_enable_mkldnn", False)
    paddle_ocr_cpu_threads = _get_option(options, "paddle_ocr_cpu_threads", None)
    paddle_ocr_enable_hpi = _get_option(options, "paddle_ocr_enable_hpi", None)
    paddle_ocr_rec_batch_size = _get_option(options, "paddle_ocr_rec_batch_size", 16)
    paddle_ocr_min_confidence = _get_option(options, "paddle_ocr_min_confidence", 0.0)
    paddle_ocr_return_word_box = _get_option(options, "paddle_ocr_return_word_box", False)
    paddle_ocr_python = _get_option(options, "paddle_ocr_python", "")
    paddle_ocr_source_dir = _get_option(options, "paddle_ocr_source_dir", "")

    print(f"[DEBUG] Deskew params: deskew_enabled={deskew_enabled}, use_tesseract_seg={use_tesseract_seg}", flush=True)
    log.info(
        "[DEBUG] Preflight params: preflight=%s, chunk=%s, dpi=%s, workers=%s, batch_lines=%s, batch_mb=%s",
        preflight,
        preflight_chunk,
        preflight_dpi,
        preflight_workers,
        preflight_batch_lines,
        preflight_batch_mb,
    )
    log.info("[DEBUG] Deskew params: deskew_enabled=%s, use_tesseract_seg=%s", deskew_enabled, use_tesseract_seg)
    log.info("[DEBUG] extra_attrs=%s", getattr(options, 'extra_attrs', {}))

    _config["url"] = url
    _config["model"] = model
    _config["timeout"] = timeout
    _config["batch_mode"] = batch_mode
    _config["segmentation"] = segmentation
    _config["preflight"] = preflight
    _config["preflight_chunk"] = preflight_chunk
    _config["preflight_dpi"] = preflight_dpi
    _config["preflight_workers"] = preflight_workers
    _config["preflight_batch_lines"] = max(1, int(preflight_batch_lines or 512))
    _config["preflight_batch_mb"] = max(1, int(preflight_batch_mb or 64))
    _config["use_tesseract_segmentation"] = use_tesseract_seg
    _config["deskew_enabled"] = deskew_enabled
    _config["line_detector"] = line_detector
    _config["chrome_screenai_light"] = bool(chrome_screenai_light)
    _config["chrome_screenai_min_confidence"] = float(chrome_screenai_min_confidence or 0.0)
    _config["chrome_screenai_filter_types"] = str(chrome_screenai_filter_types or "printed")
    _config["chrome_screenai_text_layer_mode"] = str(chrome_screenai_text_layer_mode or "native_direct")
    _config["chrome_screenai_force_horizontal"] = bool(chrome_screenai_force_horizontal)
    _config["chrome_screenai_text_cleanup"] = str(chrome_screenai_text_cleanup or "mode_default")
    _config["chrome_screenai_rasterize_dpi"] = max(72, int(chrome_screenai_rasterize_dpi or 300))
    _config["chrome_screenai_rasterize_quality"] = max(1, min(100, int(chrome_screenai_rasterize_quality or 88)))
    _config["paddle_det_model"] = str(paddle_det_model or "PP-OCRv5_server_det")
    _config["paddle_det_device"] = str(paddle_det_device or "gpu:0")
    _config["paddle_det_min_confidence"] = float(paddle_det_min_confidence or 0.0)
    _config["paddle_det_python"] = str(paddle_det_python or "")
    _config["paddle_det_source_dir"] = str(paddle_det_source_dir or "")
    _config["paddle_ocr_version"] = str(paddle_ocr_version or "PP-OCRv6")
    _config["paddle_ocr_lang"] = str(paddle_ocr_lang or "en")
    _config["paddle_ocr_det_model"] = str(paddle_ocr_det_model or "")
    _config["paddle_ocr_rec_model"] = str(paddle_ocr_rec_model or "")
    _config["paddle_ocr_det_model_dir"] = str(paddle_ocr_det_model_dir or "")
    _config["paddle_ocr_rec_model_dir"] = str(paddle_ocr_rec_model_dir or "")
    _config["paddle_ocr_device"] = str(paddle_ocr_device or "gpu:0")
    _config["paddle_ocr_engine"] = str(paddle_ocr_engine or "")
    _config["paddle_ocr_enable_mkldnn"] = paddle_ocr_enable_mkldnn
    _config["paddle_ocr_cpu_threads"] = (
        int(paddle_ocr_cpu_threads) if paddle_ocr_cpu_threads else None
    )
    _config["paddle_ocr_enable_hpi"] = paddle_ocr_enable_hpi
    _config["paddle_ocr_rec_batch_size"] = max(1, int(paddle_ocr_rec_batch_size or 16))
    _config["paddle_ocr_min_confidence"] = float(paddle_ocr_min_confidence or 0.0)
    _config["paddle_ocr_return_word_box"] = bool(paddle_ocr_return_word_box)
    _config["paddle_ocr_python"] = str(paddle_ocr_python or "")
    _config["paddle_ocr_source_dir"] = str(paddle_ocr_source_dir or "")

    # 预飞模式强制使用 client 分割（需要客户端切行才能跨页合并）
    if preflight:
        _config["segmentation"] = "client"
        log.info("Preflight mode enabled: forcing client segmentation, chunk=%d", preflight_chunk)

    if deskew_enabled:
        log.info("Page deskewing enabled")

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
        if str(_config.get("line_detector", "")) == "chrome_screenai":
            raise
        log.warning(
            "Preflight OCR failed: %s — falling back to per-page OCR", e
        )


@hookimpl
def get_ocr_engine(options):
    engine = getattr(options, "ocr_engine", "auto") if options else "auto"
    if engine in ("calamari",):
        return CalamariOcrEngine()
    if engine in ("chrome_screenai",):
        return ChromeScreenAIOcrEngine()
    if engine in ("paddle_ocr",):
        return PaddleOcrEngine()
    return None
