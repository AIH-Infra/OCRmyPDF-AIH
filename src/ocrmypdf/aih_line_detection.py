# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ocrmypdf.aih_data_structures import DetectorSource, LineBox, poly_from_bbox


_DEFAULT_PADDLE_MODEL = "PP-OCRv5_server_det"
_DEFAULT_PADDLE_DEVICE = "gpu:0"
_DEFAULT_PADDLE_SOURCE_DIR: Path | None = None
_DEFAULT_PADDLE_PYTHON: Path | None = None


@dataclass(frozen=True)
class PaddleEnv:
    python: Path
    source_dir: Path
    source: str


class PaddleEnvironmentError(RuntimeError):
    """An explicitly configured PaddleOCR environment is invalid or unusable."""


def detect_lines_paddle_det(
    image_path: Path,
    *,
    page_number: int = 0,
    model_name: str = _DEFAULT_PADDLE_MODEL,
    model_dir: str | None = None,
    device: str = _DEFAULT_PADDLE_DEVICE,
    min_confidence: float = 0.0,
    paddle_python: str | None = None,
    paddle_source_dir: str | None = None,
) -> list[LineBox]:
    """Detect text lines with PaddleOCR TextDetection.

    Prefer an external PaddleOCR venv when available. This keeps AIH-OCR's
    own environment clean while letting the detector use the already-installed
    Paddle GPU stack and cached PP-OCRv5 model.
    """
    image_path = Path(image_path)
    paddle_env = discover_paddle_env(paddle_python, paddle_source_dir)
    if paddle_env is not None:
        return _detect_lines_paddle_det_external(
            image_path,
            page_number=page_number,
            model_name=model_name,
            model_dir=model_dir,
            device=device,
            min_confidence=min_confidence,
            paddle_python=paddle_env.python,
            paddle_source_dir=paddle_env.source_dir,
        )
    return _detect_lines_paddle_det_inprocess(
        image_path,
        page_number=page_number,
        model_name=model_name,
        model_dir=model_dir,
        device=device,
        min_confidence=min_confidence,
    )


def discover_paddle_env(
    paddle_python: str | None = None,
    paddle_source_dir: str | None = None,
) -> PaddleEnv | None:
    """Find a PaddleOCR Python environment across local installs.

    Priority:
    1. explicit UI/CLI paths
    2. AIH_PADDLE_PYTHON / AIH_PADDLE_SOURCE_DIR
    3. common PaddleOCR source trees next to this project
    4. common virtualenv names under those source trees
    5. PATH/current Python if they can import paddleocr
    """
    explicit_python = _path_or_none(paddle_python)
    explicit_source = _path_or_none(paddle_source_dir)
    env_python = _path_or_none(os.environ.get("AIH_PADDLE_PYTHON", ""))
    env_source = _path_or_none(os.environ.get("AIH_PADDLE_SOURCE_DIR", ""))

    if explicit_python or explicit_source:
        return _discover_configured_paddle_env(
            explicit_python, explicit_source, "explicit"
        )
    if env_python or env_source:
        return _discover_configured_paddle_env(env_python, env_source, "environment")
    return _discover_paddle_env_cached()


def refresh_paddle_environment_cache() -> None:
    """Clear cached probes so the UI can pick up an environment install or change."""
    _discover_paddle_env_cached.cache_clear()
    _probe_paddle_environment_cached.cache_clear()


def _discover_configured_paddle_env(
    python_path: Path | None,
    source_dir: Path | None,
    source: str,
) -> PaddleEnv:
    if python_path is not None and not python_path.is_file():
        raise PaddleEnvironmentError(
            f"Configured PaddleOCR Python executable does not exist: {python_path}. "
            "Set --paddle-ocr-python (or AIH_PADDLE_PYTHON) to an existing Python executable. "
            "To create an environment, run python scripts/install_paddle.py or "
            'pip install "aih-ocr[paddle]" (CPU).'
        )
    if source_dir is not None and not source_dir.is_dir():
        raise PaddleEnvironmentError(
            f"Configured PaddleOCR source directory does not exist: {source_dir}"
        )

    if python_path is None and source_dir is not None:
        for candidate in _venv_python_candidates(source_dir):
            if candidate.is_file():
                python_path = candidate
                break
    if python_path is None:
        raise PaddleEnvironmentError(
            f"{source.title()} PaddleOCR path was set, but no Python executable was found."
        )

    source_dir = source_dir or _source_dir_from_python(python_path) or python_path.parent
    if not source_dir.is_dir():
        raise PaddleEnvironmentError(
            f"Configured PaddleOCR source directory does not exist: {source_dir}"
        )
    status = probe_paddle_environment(python_path, source_dir)
    if status["error"]:
        raise PaddleEnvironmentError(
            f"Configured PaddleOCR environment at {python_path} is unusable: "
            f"{status['error']}"
        )
    return PaddleEnv(python=python_path, source_dir=source_dir, source=source)


@lru_cache(maxsize=1)
def _discover_paddle_env_cached() -> PaddleEnv | None:
    source_candidates = _unique_paths(
        [
            _DEFAULT_PADDLE_SOURCE_DIR,
            *_project_paddle_roots(),
            *_nearby_paddle_source_dirs(),
        ]
    )
    candidate_pairs: list[tuple[Path, Path, str]] = []
    for source_dir in source_candidates:
        for python_path in _venv_python_candidates(source_dir):
            candidate_pairs.append((python_path, source_dir, f"source:{source_dir}"))

    for python_path in _unique_paths([_DEFAULT_PADDLE_PYTHON]):
        source_dir = _source_dir_from_python(python_path) or python_path.parent
        candidate_pairs.append((python_path, source_dir, f"python:{python_path}"))

    path_python = shutil.which("python")
    if path_python:
        candidate_pairs.append((Path(path_python), Path.cwd(), "PATH"))
    candidate_pairs.append((Path(sys.executable), Path.cwd(), "current-python"))

    seen: set[tuple[str, str]] = set()
    for python_path, source_dir, source in candidate_pairs:
        python_path = python_path.expanduser()
        source_dir = source_dir.expanduser()
        key = (str(python_path).casefold(), str(source_dir).casefold())
        if key in seen:
            continue
        seen.add(key)
        if not python_path.is_file() or not source_dir.is_dir():
            continue
        status = probe_paddle_environment(python_path, source_dir)
        if status["error"] is None:
            return PaddleEnv(python=python_path, source_dir=source_dir, source=source)
    return None


def _path_or_none(value: str | None) -> Path | None:
    if not value:
        return None
    value = str(value).strip().strip('"')
    if not value:
        return None
    return Path(value)


def _unique_paths(paths: list[Path | None]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if path is None:
            continue
        key = str(path.expanduser()).casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(path.expanduser())
    return result


def _project_paddle_roots() -> list[Path]:
    """Project folders holding the PaddleOCR venv made by scripts/install_paddle.py.

    The installer creates ``.venv-paddle`` next to ``pyproject.toml`` so that the
    optional Paddle stack never touches the main environment.
    """
    roots = [Path(__file__).resolve().parents[2], Path.cwd()]
    names = (".venv-paddle", ".venv-paddle-gpu", ".venv-paddle-cpu")
    return [root for root in _unique_paths(roots) if any((root / name).is_dir() for name in names)]


def _nearby_paddle_source_dirs() -> list[Path]:
    roots = []
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        roots.append(parent)
        roots.append(parent.parent if parent.parent != parent else parent)
    names = [
        "PaddleOCR-3.7.0",
        "PaddleOCR-3.6.0",
        "PaddleOCR-3.5.0",
        "PaddleOCR",
        "paddleocr",
    ]
    candidates: list[Path] = []
    for root in _unique_paths(roots):
        for name in names:
            candidates.extend(
                [
                    root / name,
                    root / "aih-contexture-reference" / name,
                    root / "reference" / name,
                    root / "third_party" / name,
                    root / "vendor" / name,
                ]
            )
    return [path for path in _unique_paths(candidates) if path.exists()]


def _venv_python_candidates(source_dir: Path) -> list[Path]:
    venv_names = [
        ".venv-paddle",
        ".venv-paddle-gpu",
        ".venv-paddle-cpu",
        ".venv",
        "venv",
        "env",
    ]
    rels = [
        ("Scripts", "python.exe"),
        ("bin", "python"),
        ("bin", "python3"),
    ]
    candidates = []
    for venv_name in venv_names:
        for folder, exe in rels:
            candidates.append(source_dir / venv_name / folder / exe)
    return candidates


def _source_dir_from_python(python_path: Path) -> Path | None:
    for parent in python_path.parents:
        if parent.name in {
            ".venv-paddle-gpu",
            ".venv-paddle-cpu",
            ".venv-paddle",
            ".venv",
            "venv",
            "env",
        }:
            return parent.parent
    return None


def probe_paddle_environment(
    python: str | Path,
    source_dir: str | Path | None,
) -> dict[str, Any]:
    """Probe PaddleOCR imports and devices without constructing a model.

    The returned mapping is JSON-serializable. GPU availability is based on Paddle's
    CUDA build plus its device count; no model is loaded and no inference is run.
    """
    python_path = Path(python).expanduser()
    source_path = (
        Path(source_dir).expanduser()
        if source_dir is not None
        else (_source_dir_from_python(python_path) or python_path.parent)
    )
    if not python_path.is_file():
        status = _empty_paddle_probe(f"Python executable does not exist: {python_path}")
    elif not source_path.is_dir():
        status = _empty_paddle_probe(
            f"PaddleOCR source directory does not exist: {source_path}"
        )
    else:
        status = _probe_paddle_environment_cached(
            str(python_path.resolve()), str(source_path.resolve())
        )
    status = dict(status)
    status["python"] = str(python_path)
    status["source_dir"] = str(source_path)
    status["device_available"] = dict(status["device_available"])
    return status


def _empty_paddle_probe(error: str) -> dict[str, Any]:
    return {
        "python": None,
        "source_dir": None,
        "paddleocr_version": None,
        "paddle_version": None,
        "cuda_compiled": None,
        "gpu_count": None,
        "device_available": {"cpu": False, "gpu": False},
        "gpu_error": None,
        "error": error,
    }


@lru_cache(maxsize=64)
def _probe_paddle_environment_cached(
    python_path: str, source_dir: str
) -> dict[str, Any]:
    env = os.environ.copy()
    pythonpath_parts = []
    pythonpath_parts.append(source_dir)
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    code = r'''import importlib.metadata, json
result = {
    "python": None,
    "source_dir": None,
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
    try:
        result = subprocess.run(
            [
                python_path,
                "-c",
                code,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            env=env,
            cwd=source_dir,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        status = _empty_paddle_probe(f"Paddle environment probe failed: {exc}")
    else:
        try:
            status = json.loads((result.stdout or "").strip().splitlines()[-1])
            if not isinstance(status, dict):
                raise ValueError("probe result is not an object")
            if result.returncode != 0 and not status.get("error"):
                status["error"] = (result.stderr or "probe process failed").strip()[-1000:]
        except (IndexError, ValueError, json.JSONDecodeError) as exc:
            details = (result.stderr or result.stdout or "").strip()[-1000:]
            status = _empty_paddle_probe(
                f"Paddle environment probe returned invalid JSON: {exc}; {details}"
            )
    status["python"] = python_path
    status["source_dir"] = source_dir
    return status


def _detect_lines_paddle_det_external(
    image_path: Path,
    *,
    page_number: int,
    model_name: str,
    model_dir: str | None,
    device: str,
    min_confidence: float,
    paddle_python: Path,
    paddle_source_dir: Path,
) -> list[LineBox]:
    src_dir = str(Path(__file__).parents[1])
    pythonpath_parts = [src_dir, str(paddle_source_dir)]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    job = {
        "image_path": str(image_path),
        "page_number": page_number,
        "model_name": model_name,
        "model_dir": model_dir or "",
        "device": device,
        "min_confidence": min_confidence,
    }
    token = base64.b64encode(json.dumps(job, ensure_ascii=False).encode("utf-8")).decode("ascii")
    result = subprocess.run(
        [str(paddle_python), "-c", "import runpy, sys; sys.argv = [sys.argv[1], sys.argv[2]]; runpy.run_path(sys.argv[0], run_name=\"__main__\")", str(Path(__file__).with_name("aih_paddle_line_worker.py")), token],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
        cwd=str(paddle_source_dir),
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    payload = None
    if stdout:
        try:
            payload = json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError as exc:
            if result.returncode == 0:
                raise RuntimeError(f"PaddleOCR line worker returned invalid JSON: {stdout!r}") from exc
    if result.returncode != 0:
        if isinstance(payload, dict) and not payload.get("ok"):
            raise RuntimeError(payload.get("error") or "PaddleOCR line worker failed")
        raise RuntimeError(stderr or stdout or "PaddleOCR line worker failed")
    if payload is None:
        raise RuntimeError(f"PaddleOCR line worker returned no JSON: stdout={stdout!r}, stderr={stderr!r}")
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error") or "PaddleOCR line worker failed")
    boxes = []
    for item in payload.get("lines") or []:
        bbox = item.get("bbox") if isinstance(item, dict) else None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        bbox_tuple = tuple(int(v) for v in bbox)
        boxes.append(
            LineBox(
                page=int(item.get("page", page_number)),
                line=int(item.get("line", len(boxes))),
                bbox=bbox_tuple,
                poly=poly_from_bbox(bbox_tuple),
                confidence=float(item.get("confidence", 0.0) or 0.0),
                source=DetectorSource.PADDLE_DET.value,
            )
        )
    return boxes


def _detect_lines_paddle_det_inprocess(
    image_path: Path,
    *,
    page_number: int = 0,
    model_name: str = _DEFAULT_PADDLE_MODEL,
    model_dir: str | None = None,
    device: str = _DEFAULT_PADDLE_DEVICE,
    min_confidence: float = 0.0,
) -> list[LineBox]:
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        width, height = img.size

    model = _get_paddle_text_detector(model_name, model_dir or "", device)
    results = model.predict(str(image_path))
    polys = _extract_dt_polys(results)
    boxes = []
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
        LineBox(
            page=page_number,
            line=index,
            bbox=bbox,
            poly=poly_from_bbox(bbox),
            confidence=float(score_by_bbox.get(bbox, 0.0)),
            source=DetectorSource.PADDLE_DET.value,
        )
        for index, bbox in enumerate(ordered_boxes)
    ]


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
def _get_paddle_text_detector(model_name: str, model_dir: str, device: str):
    try:
        from paddleocr import TextDetection
    except Exception as exc:  # pragma: no cover - depends on local optional install
        raise RuntimeError(
            "PaddleOCR is required for paddle_det line detection. "
            "Install paddlepaddle and paddleocr or configure an external PaddleOCR venv."
        ) from exc

    kwargs = {"model_name": model_name, "device": _resolve_paddle_device(device)}
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


def write_line_debug(image_path: Path, boxes: list[LineBox], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        overlay = img.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for line_box in boxes:
        draw.rectangle(line_box.bbox, outline=(0, 180, 80), width=3)
        draw.text((line_box.bbox[0], max(0, line_box.bbox[1] - 14)), str(line_box.line), fill=(255, 0, 0))
    overlay.save(out_dir / f"{image_path.stem}.paddle_overlay.png")
    payload = [
        {
            "page": box.page,
            "line": box.line,
            "bbox": list(box.bbox),
            "confidence": box.confidence,
            "source": box.source,
        }
        for box in boxes
    ]
    (out_dir / f"{image_path.stem}.paddle_lineboxes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )



