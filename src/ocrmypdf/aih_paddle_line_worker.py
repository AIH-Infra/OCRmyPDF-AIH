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
        boxes = _detect_lines_paddle_det_inprocess(
            Path(job["image_path"]),
            page_number=int(job.get("page_number", 0) or 0),
            model_name=str(job.get("model_name") or "PP-OCRv5_server_det"),
            model_dir=str(job.get("model_dir") or "") or None,
            device=str(job.get("device") or "gpu:0"),
            min_confidence=float(job.get("min_confidence", 0.0) or 0.0),
        )
        _print_json({"ok": True, "lines": boxes})
        return 0
    except Exception as exc:
        _print_json({"ok": False, "error": str(exc)})
        return 1


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def _detect_lines_paddle_det_inprocess(
    image_path: Path,
    *,
    page_number: int = 0,
    model_name: str = "PP-OCRv5_server_det",
    model_dir: str | None = None,
    device: str = "gpu:0",
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        width, height = img.size

    model = _get_paddle_text_detector(model_name, model_dir or "", device)
    results = model.predict(str(image_path))
    polys = _extract_dt_polys(results)
    boxes: list[tuple[tuple[int, int, int, int], float]] = []
    for poly in polys:
        bbox = _poly_to_bbox(poly, width, height)
        if bbox is None:
            continue
        score = _extract_score(poly)
        if score < min_confidence:
            continue
        x0, y0, x1, y1 = bbox
        if (x1 - x0) < max(4, width * 0.005):
            continue
        if (y1 - y0) < max(4, height * 0.002):
            continue
        boxes.append((bbox, score))

    boxes = _nms_boxes(boxes)
    score_by_bbox = {bbox: score for bbox, score in boxes}
    ordered_boxes = _sort_boxes_reading_order([bbox for bbox, _ in boxes])
    return [
        {
            "page": page_number,
            "line": index,
            "bbox": list(bbox),
            "poly": [list(point) for point in _poly_from_bbox(bbox)],
            "confidence": float(score_by_bbox.get(bbox, 0.0)),
            "source": "paddle_det",
        }
        for index, bbox in enumerate(ordered_boxes)
    ]


@lru_cache(maxsize=4)
def _get_paddle_text_detector(model_name: str, model_dir: str, device: str):
    from paddleocr import TextDetection

    kwargs = {"model_name": model_name, "device": device}
    if model_dir:
        kwargs["model_dir"] = model_dir
    try:
        return TextDetection(**kwargs)
    except TypeError:
        kwargs.pop("device", None)
        try:
            return TextDetection(**kwargs)
        except TypeError:
            return TextDetection()


def _extract_dt_polys(results: Any) -> list[Any]:
    polys: list[Any] = []
    for item in _as_iterable_results(results):
        payloads = [item]
        if isinstance(item, dict) and isinstance(item.get("res"), dict):
            payloads.append(item["res"])
        res_attr = getattr(item, "res", None)
        if isinstance(res_attr, dict):
            payloads.append(res_attr)
        for payload in payloads:
            value = _get_field(payload, "dt_polys")
            if value is not None:
                polys.extend(list(value))
            value = _get_field(payload, "rec_polys")
            if value is not None and not polys:
                polys.extend(list(value))
    return polys


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
    if hasattr(obj, "to_dict"):
        try:
            data = obj.to_dict()
            if isinstance(data, dict):
                return data.get(name)
        except Exception:
            pass
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


def _poly_to_bbox(poly: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    points = _normalise_poly(poly)
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0 = max(0, min(width, int(round(min(xs)))))
    y0 = max(0, min(height, int(round(min(ys)))))
    x1 = max(0, min(width, int(round(max(xs)))))
    y1 = max(0, min(height, int(round(max(ys)))))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


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


def _extract_score(poly: Any) -> float:
    if isinstance(poly, dict):
        for key in ("score", "confidence", "det_score"):
            if key in poly:
                try:
                    return float(poly[key])
                except (TypeError, ValueError):
                    pass
    return 1.0


def _sort_boxes_reading_order(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    heights = sorted(max(1, b[3] - b[1]) for b in boxes)
    median_h = heights[len(heights) // 2]
    row_tol = max(8, int(median_h * 0.65))
    rows: list[list[tuple[int, int, int, int]]] = []
    for box in sorted(boxes, key=lambda b: ((b[1] + b[3]) / 2, b[0])):
        cy = (box[1] + box[3]) / 2
        for row in rows:
            row_cy = sum((b[1] + b[3]) / 2 for b in row) / len(row)
            if abs(cy - row_cy) <= row_tol:
                row.append(box)
                break
        else:
            rows.append([box])
    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b[0]))
    return ordered


def _nms_boxes(
    boxes: list[tuple[tuple[int, int, int, int], float]],
    iou_threshold: float = 0.35,
) -> list[tuple[tuple[int, int, int, int], float]]:
    kept: list[tuple[tuple[int, int, int, int], float]] = []
    for bbox, score in sorted(boxes, key=lambda item: item[1], reverse=True):
        if all(_iou(bbox, other) < iou_threshold for other, _ in kept):
            kept.append((bbox, score))
    return kept


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return inter / max(1, area_a + area_b - inter)


def _poly_from_bbox(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = bbox
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


if __name__ == "__main__":
    raise SystemExit(main())
