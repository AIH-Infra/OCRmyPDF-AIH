# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import base64
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        _print_json({"ok": False, "error": "missing job token"})
        return 2
    try:
        job = json.loads(base64.b64decode(argv[0]).decode("utf-8"))
        payload = _run_paddle_ocr(job)
        _print_json({"ok": True, **payload})
        return 0
    except Exception as exc:
        _print_json({"ok": False, "error": str(exc)})
        return 1


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def _run_paddle_ocr(job: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(job["image_path"])
    page_number = int(job.get("page_number", 0) or 0)
    min_confidence = float(job.get("min_confidence", 0.0) or 0.0)

    with Image.open(image_path) as img:
        width, height = img.size
        dpi_info = img.info.get("dpi", (300, 300))
        dpi_x = int(round(dpi_info[0])) if dpi_info and dpi_info[0] else 300
        dpi_y = int(round(dpi_info[1])) if dpi_info and dpi_info[1] else 300

    model = _get_paddle_ocr_model(
        ocr_version=str(job.get("ocr_version") or "PP-OCRv6"),
        lang=str(job.get("lang") or "en"),
        det_model_name=str(job.get("det_model_name") or ""),
        rec_model_name=str(job.get("rec_model_name") or ""),
        det_model_dir=str(job.get("det_model_dir") or ""),
        rec_model_dir=str(job.get("rec_model_dir") or ""),
        device=str(job.get("device") or "gpu:0"),
        engine=str(job.get("engine") or ""),
        enable_mkldnn=_to_optional_bool(job.get("enable_mkldnn")),
        cpu_threads=_to_optional_int(job.get("cpu_threads")),
        enable_hpi=_to_optional_bool(job.get("enable_hpi")),
        rec_batch_size=_to_optional_int(job.get("rec_batch_size")),
    )

    predict_kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "return_word_box": bool(job.get("return_word_box", False)),
    }
    try:
        results = model.predict(str(image_path), **predict_kwargs)
    except TypeError:
        predict_kwargs.pop("return_word_box", None)
        try:
            results = model.predict(str(image_path), **predict_kwargs)
        except TypeError:
            results = model.predict(str(image_path))

    lines = _extract_ocr_lines(results, width, height, min_confidence)
    return {
        "page": page_number,
        "width": width,
        "height": height,
        "dpi": [dpi_x, dpi_y],
        "lines": lines,
    }


def _resolve_paddle_device(device: str) -> str:
    """Fall back to CPU when a GPU device is requested but Paddle has no CUDA.

    The optional ``paddle`` extra installs the CPU build of PaddlePaddle, while the
    defaults historically assume ``gpu:0``.
    """
    if not device or not device.lower().startswith(("gpu", "cuda")):
        return device
    try:
        import paddle

        if paddle.device.is_compiled_with_cuda():
            return device
    except Exception:  # pragma: no cover - depends on local optional install
        return device
    print(f"PaddlePaddle has no CUDA support; using CPU instead of {device!r}", file=sys.stderr)
    return "cpu"


@lru_cache(maxsize=4)
def _get_paddle_ocr_model(
    *,
    ocr_version: str,
    lang: str,
    det_model_name: str,
    rec_model_name: str,
    det_model_dir: str,
    rec_model_dir: str,
    device: str,
    engine: str,
    enable_mkldnn: bool | None,
    cpu_threads: int | None,
    enable_hpi: bool | None,
    rec_batch_size: int | None,
):
    from paddleocr import PaddleOCR

    kwargs: dict[str, Any] = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    if ocr_version and ocr_version.lower() != "auto":
        kwargs["ocr_version"] = ocr_version
    if lang:
        kwargs["lang"] = lang
    if det_model_name:
        kwargs["text_detection_model_name"] = det_model_name
    if rec_model_name:
        kwargs["text_recognition_model_name"] = rec_model_name
    if det_model_dir:
        kwargs["text_detection_model_dir"] = det_model_dir
    if rec_model_dir:
        kwargs["text_recognition_model_dir"] = rec_model_dir
    if rec_batch_size:
        kwargs["text_recognition_batch_size"] = rec_batch_size
    if device and device.lower() != "auto":
        kwargs["device"] = _resolve_paddle_device(device)
    if engine:
        kwargs["engine"] = engine
    if enable_mkldnn is not None:
        kwargs["enable_mkldnn"] = enable_mkldnn
    if cpu_threads:
        kwargs["cpu_threads"] = cpu_threads
    if enable_hpi is not None:
        kwargs["enable_hpi"] = enable_hpi

    try:
        return PaddleOCR(**kwargs)
    except TypeError:
        # Older PaddleOCR versions may not support the newer preprocessor flags.
        for key in (
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "use_textline_orientation",
            "engine",
            "enable_hpi",
        ):
            kwargs.pop(key, None)
        try:
            return PaddleOCR(**kwargs)
        except TypeError:
            minimal = {}
            if lang:
                minimal["lang"] = lang
            if device and device.lower() != "auto":
                minimal["device"] = device
            try:
                return PaddleOCR(**minimal)
            except TypeError:
                minimal.pop("device", None)
                return PaddleOCR(**minimal)


def _extract_ocr_lines(
    results: Any,
    width: int,
    height: int,
    min_confidence: float,
) -> list[dict[str, Any]]:
    payload = _first_result_payload(results)
    rec_texts = list(_get_field(payload, "rec_texts") or [])
    rec_scores = _to_float_list(_get_field(payload, "rec_scores") or [])
    rec_polys = list(_get_field(payload, "rec_polys") or [])
    if not rec_polys:
        rec_polys = list(_get_field(payload, "dt_polys") or [])

    lines: list[dict[str, Any]] = []
    total = min(len(rec_texts), len(rec_polys))
    for index in range(total):
        text = str(rec_texts[index] or "")
        score = rec_scores[index] if index < len(rec_scores) else 1.0
        if score < min_confidence:
            continue
        poly = _normalise_poly(rec_polys[index])
        bbox = _poly_to_bbox(poly, width, height)
        if bbox is None:
            continue
        lines.append(
            {
                "index": len(lines),
                "text": text,
                "confidence": float(score),
                "poly": [[float(x), float(y)] for x, y in poly],
                "bbox": list(bbox),
            }
        )

    lines.sort(key=lambda item: ((item["bbox"][1] + item["bbox"][3]) / 2, item["bbox"][0]))
    for index, item in enumerate(lines):
        item["index"] = index
    return lines


def _first_result_payload(results: Any) -> Any:
    for item in _as_iterable_results(results):
        if isinstance(item, dict) and isinstance(item.get("res"), dict):
            return item["res"]
        res_attr = getattr(item, "res", None)
        if isinstance(res_attr, dict):
            return res_attr
        if hasattr(item, "to_dict"):
            try:
                data = item.to_dict()
                if isinstance(data, dict):
                    return data.get("res", data)
            except Exception:
                pass
        if isinstance(item, dict):
            return item
    return {}


def _as_iterable_results(results: Any) -> list[Any]:
    if results is None:
        return []
    if isinstance(results, (list, tuple)):
        return list(results)
    if hasattr(results, "__iter__") and not isinstance(results, (str, bytes, dict)):
        try:
            return list(results)
        except TypeError:
            return [results]
    return [results]


def _get_field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    if hasattr(obj, name):
        return getattr(obj, name)
    if hasattr(obj, "json"):
        try:
            data = obj.json
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data, dict):
                return data.get(name)
        except Exception:
            pass
    return None


def _normalise_poly(poly: Any) -> list[tuple[float, float]]:
    if hasattr(poly, "tolist"):
        poly = poly.tolist()
    if isinstance(poly, dict):
        for key in ("points", "poly", "dt_poly", "bbox", "coordinate"):
            if key in poly:
                poly = poly[key]
                break
    if not isinstance(poly, (list, tuple)):
        return []
    if len(poly) == 4 and all(isinstance(v, (int, float)) for v in poly):
        x0, y0, x1, y1 = [float(v) for v in poly]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    points = []
    for point in poly:
        if hasattr(point, "tolist"):
            point = point.tolist()
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
    return points


def _poly_to_bbox(
    poly: list[tuple[float, float]],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if not poly:
        return None
    xs = [point[0] for point in poly]
    ys = [point[1] for point in poly]
    x0 = max(0, min(width, int(round(min(xs)))))
    y0 = max(0, min(height, int(round(min(ys)))))
    x1 = max(0, min(width, int(round(max(xs)))))
    y1 = max(0, min(height, int(round(max(ys)))))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _to_float_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            result.append(0.0)
    return result


def _to_optional_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
