# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import builtins
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from ocrmypdf.aih_data_structures import (
    ContentType,
    DetectorSource,
    LineBox,
    TextDirection,
    bbox_from_xywh,
    poly_from_bbox,
)
from ocrmypdf.hocrtransform import Baseline, BoundingBox, OcrClass, OcrElement

_PROGRESS_STATE_KEY = "_ocrmypdf_page_progress_state"


class ChromeScreenAIBackend:
    """Thin adapter around the local locro ScreenAI wrapper."""

    def __init__(self, *, light_mode: bool = False, force_horizontal: bool = True):
        self.light_mode = light_mode
        self.force_horizontal = force_horizontal
        self._screen_ai = None
        self._thread_local = threading.local()
        self._screen_ai_init_lock = threading.Lock()

    def _create_screen_ai(self):
        with self._screen_ai_init_lock:
            try:
                from ocrmypdf._vendor.locro import ScreenAI
            except Exception:
                _ensure_locro_import_path()
                from locro import ScreenAI
            return ScreenAI(light_mode=self.light_mode)

    def _get_screen_ai(self):
        if self._screen_ai is None:
            self._screen_ai = self._create_screen_ai()
        return self._screen_ai

    def _get_thread_screen_ai(self):
        screen_ai = getattr(self._thread_local, "screen_ai", None)
        if screen_ai is None:
            screen_ai = self._create_screen_ai()
            self._thread_local.screen_ai = screen_ai
        return screen_ai

    def detect_lines(
        self,
        image_path: Path,
        *,
        page_number: int = 0,
        filter_printed_text: bool = True,
        min_confidence: float = 0.0,
        force_horizontal: bool | None = None,
    ) -> list[LineBox]:
        if force_horizontal is None:
            force_horizontal = self.force_horizontal
        with Image.open(image_path) as img:
            original_size = img.size
            screen_ai = self._get_screen_ai()
            raw_lines, ocr_size = self._perform_raw_ocr(screen_ai, img)

        sx = original_size[0] / ocr_size[0] if ocr_size[0] else 1.0
        sy = original_size[1] / ocr_size[1] if ocr_size[1] else 1.0

        boxes: list[LineBox] = []
        normalized_lines: list[Any] = []
        for line in raw_lines:
            normalized_lines.extend(self._normalize_raw_line(line))

        for index, line in enumerate(normalized_lines):
            content_type = ContentType.from_value(getattr(line, "content_type", None))
            if filter_printed_text and content_type not in (
                ContentType.PRINTED_TEXT,
                ContentType.UNKNOWN,
            ):
                continue

            confidence = float(getattr(line, "confidence", 0.0) or 0.0)
            if confidence < min_confidence:
                continue

            bbox = bbox_from_xywh(
                getattr(line, "x", 0) * sx,
                getattr(line, "y", 0) * sy,
                getattr(line, "width", 0) * sx,
                getattr(line, "height", 0) * sy,
            )

            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            words = []
            for word in getattr(line, "words", []) or []:
                word_bbox = None
                if getattr(word, "width", 0):
                    word_bbox = bbox_from_xywh(
                        getattr(word, "x", 0) * sx,
                        getattr(word, "y", 0) * sy,
                        getattr(word, "width", 0) * sx,
                        getattr(word, "height", 0) * sy,
                    )
                words.append((getattr(word, "text", ""), word_bbox))

            boxes.append(
                LineBox(
                    page=page_number,
                    line=len(boxes),
                    bbox=bbox,
                    poly=poly_from_bbox(bbox),
                    confidence=confidence,
                    source=DetectorSource.CHROME_SCREENAI.value,
                    text=getattr(line, "text", "") or "",
                    content_type=content_type,
                    direction=(
                        TextDirection.LEFT_TO_RIGHT
                        if force_horizontal
                        else TextDirection.from_value(getattr(line, "direction", None))
                    ),
                    block_id=getattr(line, "block_id", None),
                    paragraph_id=getattr(line, "paragraph_id", None),
                    language=getattr(line, "language", None),
                    words=words,
                )
            )

        if force_horizontal:
            boxes = _sort_line_boxes_top_to_bottom(boxes)
            boxes = [
                LineBox(
                    page=box.page,
                    line=index,
                    bbox=box.bbox,
                    poly=box.poly,
                    confidence=box.confidence,
                    source=box.source,
                    text=box.text,
                    content_type=box.content_type,
                    direction=TextDirection.LEFT_TO_RIGHT,
                    block_id=box.block_id,
                    paragraph_id=box.paragraph_id,
                    language=box.language,
                    words=box.words,
                )
                for index, box in enumerate(boxes)
            ]
        return boxes

    def _normalize_raw_line(self, line: Any) -> list[Any]:
        width = float(getattr(line, "width", 0) or 0)
        height = float(getattr(line, "height", 0) or 0)
        words = list(getattr(line, "words", []) or [])

        if width <= 0 or height <= 0 or len(words) <= 1:
            return [line]

        valid_word_boxes = [
            word for word in words
            if getattr(word, "width", 0) and getattr(word, "height", 0)
        ]
        if len(valid_word_boxes) <= 1:
            return [line]

        word_heights = sorted(float(getattr(word, "height", 0) or 0) for word in valid_word_boxes)
        median_word_height = word_heights[len(word_heights) // 2] if word_heights else 0.0
        if median_word_height <= 0:
            return [line]

        # Paragraph-like blocks show up as very tall boxes relative to their words.
        # In those cases, regroup words by vertical centers into true text lines.
        if height <= median_word_height * 2.4:
            return [line]

        regrouped_words = self._cluster_words_into_lines(valid_word_boxes, median_word_height)
        if len(regrouped_words) <= 1:
            return [line]

        split_lines: list[Any] = []
        for paragraph_index, group in enumerate(regrouped_words):
            line_obj = self._build_line_from_words(
                parent=line,
                words=group,
                paragraph_index=paragraph_index,
            )
            if line_obj is not None:
                split_lines.append(line_obj)
        return split_lines or [line]

    def _cluster_words_into_lines(self, words: list[Any], median_word_height: float) -> list[list[Any]]:
        threshold = max(4.0, median_word_height * 0.65)
        ordered = sorted(
            words,
            key=lambda word: (
                float(getattr(word, "y", 0) or 0) + float(getattr(word, "height", 0) or 0) / 2.0,
                float(getattr(word, "x", 0) or 0),
            ),
        )
        groups: list[list[Any]] = []
        current_group: list[Any] = []
        current_center: float | None = None

        for word in ordered:
            center_y = float(getattr(word, "y", 0) or 0) + float(getattr(word, "height", 0) or 0) / 2.0
            if not current_group:
                current_group = [word]
                current_center = center_y
                continue

            if current_center is not None and abs(center_y - current_center) <= threshold:
                current_group.append(word)
                current_center = (current_center * (len(current_group) - 1) + center_y) / len(current_group)
            else:
                groups.append(sorted(current_group, key=lambda w: float(getattr(w, "x", 0) or 0)))
                current_group = [word]
                current_center = center_y

        if current_group:
            groups.append(sorted(current_group, key=lambda w: float(getattr(w, "x", 0) or 0)))
        return groups

    def _build_line_from_words(
        self,
        *,
        parent: Any,
        words: list[Any],
        paragraph_index: int,
    ) -> Any | None:
        if not words:
            return None

        xs = [float(getattr(word, "x", 0) or 0) for word in words]
        ys = [float(getattr(word, "y", 0) or 0) for word in words]
        rights = [float(getattr(word, "x", 0) or 0) + float(getattr(word, "width", 0) or 0) for word in words]
        bottoms = [float(getattr(word, "y", 0) or 0) + float(getattr(word, "height", 0) or 0) for word in words]
        x0 = min(xs)
        y0 = min(ys)
        x1 = max(rights)
        y1 = max(bottoms)
        if x1 <= x0 or y1 <= y0:
            return None

        text = " ".join((getattr(word, "text", "") or "").strip() for word in words).strip()
        return _CompatLine(
            text=text,
            language=getattr(parent, "language", "") or "",
            block_id=int(getattr(parent, "block_id", 0) or 0),
            paragraph_id=int(getattr(parent, "paragraph_id", 0) or 0) * 1000 + paragraph_index,
            confidence=float(getattr(parent, "confidence", 0.0) or 0.0),
            x=x0,
            y=y0,
            width=max(1.0, x1 - x0),
            height=max(1.0, y1 - y0),
            angle=float(getattr(parent, "angle", 0.0) or 0.0),
            direction=int(getattr(parent, "direction", TextDirection.UNSPECIFIED.value) or 0),
            content_type=int(getattr(parent, "content_type", ContentType.UNKNOWN.value) or 0),
            words=[
                _CompatWord(
                    text=getattr(word, "text", "") or "",
                    x=float(getattr(word, "x", 0) or 0),
                    y=float(getattr(word, "y", 0) or 0),
                    width=float(getattr(word, "width", 0) or 0),
                    height=float(getattr(word, "height", 0) or 0),
                    angle=float(getattr(word, "angle", 0.0) or 0.0),
                )
                for word in words
            ],
        )

    def ocr_to_searchable_pdf(
        self,
        input_pdf: Path,
        output_pdf: Path,
        *,
        pages: list[int] | None = None,
        workers: int = 1,
        collect_result: bool = True,
    ) -> Any:
        workers = max(1, int(workers or 1))
        if workers <= 1:
            screen_ai = self._get_screen_ai()
            progress_state = getattr(builtins, _PROGRESS_STATE_KEY, None)
            if progress_state is not None:
                try:
                    import fitz

                    doc = fitz.open(str(input_pdf))
                    total_pages = len(pages) if pages else len(doc)
                    doc.close()
                except Exception:
                    total_pages = 0
                progress_state["page_current"] = 0
                progress_state["page_total"] = total_pages
                progress_state["page_desc"] = "Chrome ScreenAI OCR"
            return screen_ai.ocr_to_searchable_pdf(
                input_pdf,
                output_pdf,
                pages=pages,
            )

        try:
            import fitz
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for PDF operations: pip install PyMuPDF"
            ) from exc

        progress_state = getattr(builtins, _PROGRESS_STATE_KEY, None)
        with fitz.open(str(input_pdf)) as doc:
            page_set = set(pages) if pages is not None else None
            selected_pages = [
                i + 1 for i in range(len(doc))
                if page_set is None or (i + 1) in page_set
            ]

        total_pages = len(selected_pages)
        if progress_state is not None:
            progress_state["page_current"] = 0
            progress_state["page_total"] = total_pages
            progress_state["page_desc"] = f"Chrome ScreenAI OCR ({workers} workers)"

        if not selected_pages:
            with fitz.open(str(input_pdf)) as doc:
                doc.save(str(output_pdf), deflate=True)
            return type("EmptyOcrResult", (), {"pages": []})()

        return self._ocr_pdf_via_chunk_processes(
            input_pdf=input_pdf,
            output_pdf=output_pdf,
            selected_pages=selected_pages,
            workers=workers,
            collect_result=collect_result,
        )

    def _ocr_pdf_via_chunk_processes(
        self,
        *,
        input_pdf: Path,
        output_pdf: Path,
        selected_pages: list[int],
        workers: int,
        collect_result: bool,
    ):
        import fitz

        if collect_result:
            try:
                from ocrmypdf._vendor.locro.models import (
                    BoundingBox as VendorBoundingBox,
                    OcrBlock,
                    OcrLine,
                    OcrPage,
                    OcrResult,
                    OcrWord,
                )
            except Exception:
                _ensure_locro_import_path()
                from locro.models import (
                    BoundingBox as VendorBoundingBox,
                    OcrBlock,
                    OcrLine,
                    OcrPage,
                    OcrResult,
                    OcrWord,
                )

        progress_state = getattr(builtins, _PROGRESS_STATE_KEY, None)
        max_workers = max(1, min(int(workers or 1), len(selected_pages)))
        chunk_size = max(1, (len(selected_pages) + max_workers - 1) // max_workers)
        chunks = [
            selected_pages[i : i + chunk_size]
            for i in range(0, len(selected_pages), chunk_size)
        ]

        with tempfile.TemporaryDirectory(prefix="ocrmypdf-chrome-chunks-") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            chunk_jobs = []
            with fitz.open(str(input_pdf)) as src_doc:
                for chunk_index, chunk_pages in enumerate(chunks):
                    chunk_input = tmpdir / f"chunk_{chunk_index:03d}_input.pdf"
                    chunk_output = tmpdir / f"chunk_{chunk_index:03d}_output.pdf"
                    chunk_result = tmpdir / f"chunk_{chunk_index:03d}_result.json"
                    chunk_progress = tmpdir / f"chunk_{chunk_index:03d}_progress.json"
                    subdoc = fitz.open()
                    for page_num in chunk_pages:
                        subdoc.insert_pdf(src_doc, from_page=page_num - 1, to_page=page_num - 1)
                    subdoc.save(str(chunk_input), deflate=True)
                    subdoc.close()
                    chunk_jobs.append({
                        "chunk_pages": chunk_pages,
                        "input_pdf": chunk_input,
                        "output_pdf": chunk_output,
                        "result_json": chunk_result,
                        "progress_json": chunk_progress,
                    })

            running: list[dict] = []
            pending = list(chunk_jobs)
            completed_pages = 0

            def stop_running_workers() -> None:
                for item in list(running):
                    proc = item.get("proc")
                    if proc is None or proc.poll() is not None:
                        continue
                    proc.terminate()
                for item in list(running):
                    proc = item.get("proc")
                    if proc is None or proc.poll() is not None:
                        continue
                    try:
                        proc.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.communicate()

            def launch(job: dict):
                job_payload = {
                    "input_pdf": str(job["input_pdf"]),
                    "output_pdf": str(job["output_pdf"]),
                    "result_json": str(job["result_json"]),
                    "progress_json": str(job["progress_json"]),
                    "light_mode": self.light_mode,
                    "collect_result": bool(collect_result),
                }
                token = base64.b64encode(
                    json.dumps(job_payload, ensure_ascii=False).encode("utf-8")
                ).decode("ascii")
                env = os.environ.copy()
                src_dir = str(Path(__file__).parents[1])
                existing_pythonpath = env.get("PYTHONPATH")
                env["PYTHONPATH"] = (
                    src_dir if not existing_pythonpath else f"{src_dir}{os.pathsep}{existing_pythonpath}"
                )
                proc = subprocess.Popen(
                    [sys.executable, "-m", "ocrmypdf.aih_chrome_chunk_worker", token],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    cwd=str(Path(__file__).parents[1]),
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                running.append({"proc": proc, "job": job})

            try:
                while pending and len(running) < max_workers:
                    launch(pending.pop(0))

                while running:
                    for item in list(running):
                        proc = item["proc"]
                        if proc.poll() is None:
                            if progress_state is not None:
                                total_done = completed_pages
                                for running_item in running:
                                    progress_path = Path(running_item["job"]["progress_json"])
                                    if not progress_path.exists():
                                        continue
                                    try:
                                        progress_data = json.loads(progress_path.read_text(encoding="utf-8"))
                                    except Exception:
                                        continue
                                    chunk_page_count = len(running_item["job"]["chunk_pages"])
                                    total_done += max(
                                        0,
                                        min(
                                            int(progress_data.get("page_current", 0) or 0),
                                            chunk_page_count,
                                        ),
                                    )
                                progress_state["page_current"] = min(total_done, len(selected_pages))
                                progress_state["page_total"] = len(selected_pages)
                                progress_state["page_desc"] = f"Chrome ScreenAI OCR ({max_workers} workers)"
                            continue
                        stdout, stderr = proc.communicate()
                        job = item["job"]
                        if proc.returncode != 0:
                            raise RuntimeError(
                                stderr.strip() or stdout.strip() or f"Chrome chunk worker failed: {job['input_pdf']}"
                            )
                        result_meta = json.loads(job["result_json"].read_text(encoding="utf-8"))
                        if not result_meta.get("ok"):
                            raise RuntimeError(result_meta.get("error", "Chrome chunk worker failed"))
                        completed_pages += len(job["chunk_pages"])
                        if progress_state is not None:
                            progress_state["page_current"] = completed_pages
                            progress_state["page_total"] = len(selected_pages)
                            progress_state["page_desc"] = f"Chrome ScreenAI OCR ({max_workers} workers)"
                        running.remove(item)
                        if pending:
                            launch(pending.pop(0))
                        break
                    else:
                        import time
                        time.sleep(0.2)

                ordered_pages: list[Any] = []
                merged_pdf = None
                for job in chunk_jobs:
                    result_meta = json.loads(job["result_json"].read_text(encoding="utf-8"))
                    if collect_result:
                        for page_index, page_dict in enumerate(result_meta.get("result", {}).get("pages", [])):
                            source_page_number = job["chunk_pages"][page_index]
                            blocks = []
                            for block_dict in page_dict.get("blocks", []):
                                lines = []
                                for line_dict in block_dict.get("lines", []):
                                    words = []
                                    for word_dict in line_dict.get("words", []):
                                        word_bbox = word_dict.get("bounding_box")
                                        words.append(
                                            OcrWord(
                                                text=word_dict.get("text", "") or "",
                                                confidence=word_dict.get("confidence"),
                                                bounding_box=VendorBoundingBox(**word_bbox) if word_bbox else None,
                                            )
                                        )
                                    line_bbox = line_dict.get("bounding_box")
                                    lines.append(
                                        OcrLine(
                                            text=line_dict.get("text", "") or "",
                                            words=words,
                                            bounding_box=VendorBoundingBox(**line_bbox) if line_bbox else None,
                                        )
                                    )
                                block_bbox = block_dict.get("bounding_box")
                                blocks.append(
                                    OcrBlock(
                                        block_type=block_dict.get("block_type", "paragraph"),
                                        lines=lines,
                                        bounding_box=VendorBoundingBox(**block_bbox) if block_bbox else None,
                                    )
                                )
                            ordered_pages.append(
                                OcrPage(
                                    page_number=source_page_number,
                                    blocks=blocks,
                                    width=page_dict.get("width"),
                                    height=page_dict.get("height"),
                                )
                            )
                    if merged_pdf is None:
                        import pikepdf

                        merged_pdf = pikepdf.Pdf.new()
                    with pikepdf.open(str(job["output_pdf"])) as chunk_pdf:
                        merged_pdf.pages.extend(chunk_pdf.pages)

                if merged_pdf is not None:
                    merged_pdf.save(str(output_pdf))
                    merged_pdf.close()
                if collect_result:
                    return OcrResult(pages=ordered_pages)
                return type("ChromeOcrResult", (), {"pages": []})()
            except BaseException:
                stop_running_workers()
                raise

    def generate_ocr_tree(
        self,
        image_path: Path,
        *,
        page_number: int = 0,
        filter_printed_text: bool = False,
        min_confidence: float = 0.0,
    ) -> tuple[OcrElement, str]:
        line_boxes = self.detect_lines_external(
            image_path,
            page_number=page_number,
            filter_printed_text=filter_printed_text,
            min_confidence=min_confidence,
        )
        with Image.open(image_path) as img:
            width, height = img.size
            dpi_info = img.info.get("dpi", (300, 300))
            dpi = dpi_info[0] if isinstance(dpi_info, tuple) else dpi_info

        page = OcrElement(
            ocr_class=OcrClass.PAGE,
            bbox=BoundingBox(left=0, top=0, right=width, bottom=height),
            dpi=float(dpi or 300),
            page_number=page_number,
        )

        paragraphs: dict[tuple[int | None, int | None], OcrElement] = {}
        text_lines: list[str] = []
        for line_box in line_boxes:
            key = (line_box.block_id, line_box.paragraph_id)
            paragraph = paragraphs.get(key)
            if paragraph is None:
                paragraph = OcrElement(
                    ocr_class=OcrClass.PARAGRAPH,
                    bbox=_ocr_bbox(line_box.bbox),
                    language=line_box.language,
                )
                paragraphs[key] = paragraph
                page.children.append(paragraph)
            else:
                paragraph.bbox = _merge_bboxes(paragraph.bbox, _ocr_bbox(line_box.bbox))

            line_element = OcrElement(
                ocr_class=OcrClass.LINE,
                bbox=_ocr_bbox(line_box.bbox),
                text="",
                confidence=line_box.confidence,
                direction=_ocr_direction(line_box.direction),
                language=line_box.language,
                baseline=Baseline(0.0, 0.0),
            )
            words = [
                (word_text, word_bbox)
                for word_text, word_bbox in line_box.words
                if word_text and word_bbox is not None
            ]
            if not words and line_box.text:
                words = [(line_box.text, line_box.bbox)]
            for word_text, word_bbox in words:
                line_element.children.append(
                    OcrElement(
                        ocr_class=OcrClass.WORD,
                        bbox=_ocr_bbox(word_bbox),
                        text=word_text,
                        confidence=line_box.confidence,
                        direction=_ocr_direction(line_box.direction),
                        language=line_box.language,
                    )
                )
            paragraph.children.append(line_element)
            if line_box.text:
                text_lines.append(line_box.text)

        return page, "\n".join(text_lines)

    def detect_lines_external(
        self,
        image_path: Path,
        *,
        page_number: int = 0,
        filter_printed_text: bool = True,
        min_confidence: float = 0.0,
        timeout: int = 180,
    ) -> list[LineBox]:
        """Run Chrome ScreenAI line detection in a child process.

        Chrome ScreenAI is a native DLL. Keeping proxy text-layer detection out of
        the Streamlit/OCR parent process prevents native access violations from
        taking down the whole Web UI.
        """
        job = {
            "image_path": str(Path(image_path)),
            "page_number": int(page_number or 0),
            "light_mode": bool(self.light_mode),
            "force_horizontal": bool(self.force_horizontal),
            "filter_printed_text": bool(filter_printed_text),
            "min_confidence": float(min_confidence or 0.0),
        }
        token = base64.b64encode(json.dumps(job).encode("utf-8")).decode("ascii")
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        proc = subprocess.run(
            [sys.executable, "-m", "ocrmypdf.aih_chrome_line_worker", token],
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )

        payload = _load_worker_json(proc.stdout)
        if proc.returncode != 0:
            detail = ""
            if payload and payload.get("error"):
                detail = str(payload["error"])
            elif proc.stderr.strip():
                detail = proc.stderr.strip()
            elif proc.stdout.strip():
                detail = proc.stdout.strip()
            else:
                detail = f"exit code {proc.returncode}"
            raise RuntimeError(f"Chrome ScreenAI worker failed: {detail}")

        if not payload:
            detail = proc.stderr.strip() or proc.stdout.strip() or "empty worker output"
            raise RuntimeError(f"Chrome ScreenAI worker returned invalid JSON: {detail}")
        if not payload.get("ok"):
            raise RuntimeError(f"Chrome ScreenAI worker failed: {payload.get('error', 'unknown error')}")

        line_boxes: list[LineBox] = []
        for index, item in enumerate(payload.get("lines", []) or []):
            bbox = tuple(float(v) for v in item.get("bbox", (0, 0, 0, 0)))
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            words = []
            for word in item.get("words", []) or []:
                word_bbox = word.get("bbox")
                words.append(
                    (
                        word.get("text", "") or "",
                        tuple(float(v) for v in word_bbox) if word_bbox else None,
                    )
                )
            line_boxes.append(
                LineBox(
                    page=page_number,
                    line=index,
                    bbox=bbox,
                    poly=poly_from_bbox(bbox),
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    source=DetectorSource.CHROME_SCREENAI.value,
                    text=item.get("text", "") or "",
                    content_type=ContentType.from_value(item.get("content_type")),
                    direction=TextDirection.from_value(item.get("direction")),
                    block_id=item.get("block_id"),
                    paragraph_id=item.get("paragraph_id"),
                    language=item.get("language"),
                    words=words,
                )
            )
        return line_boxes

    def _perform_raw_ocr(self, screen_ai, img: Image.Image):
        perform_ocr = getattr(screen_ai, "_perform_ocr", None)
        if callable(perform_ocr):
            return perform_ocr(img)

        page = screen_ai.ocr_pil_image(img)
        raw_lines = []
        for block_index, block in enumerate(getattr(page, "blocks", []) or []):
            for paragraph_index, line in enumerate(getattr(block, "lines", []) or []):
                bb = getattr(line, "bounding_box", None)
                if bb is None:
                    continue
                raw_lines.append(
                    _CompatLine(
                        text=getattr(line, "text", "") or "",
                        language=None,
                        block_id=block_index,
                        paragraph_id=paragraph_index,
                        confidence=0.0,
                        x=getattr(bb, "x", 0),
                        y=getattr(bb, "y", 0),
                        width=getattr(bb, "width", 0),
                        height=getattr(bb, "height", 0),
                        angle=getattr(bb, "angle", 0.0),
                        direction=TextDirection.UNSPECIFIED.value,
                        content_type=ContentType.UNKNOWN.value,
                        words=[
                            _CompatWord(
                                text=getattr(word, "text", "") or "",
                                x=getattr(getattr(word, "bounding_box", None), "x", 0),
                                y=getattr(getattr(word, "bounding_box", None), "y", 0),
                                width=getattr(getattr(word, "bounding_box", None), "width", 0),
                                height=getattr(getattr(word, "bounding_box", None), "height", 0),
                                angle=getattr(getattr(word, "bounding_box", None), "angle", 0.0),
                            )
                            for word in getattr(line, "words", []) or []
                        ],
                    )
                )
        return raw_lines, (int(page.width or img.size[0]), int(page.height or img.size[1]))


class _CompatWord:
    def __init__(self, *, text: str, x: float, y: float, width: float, height: float, angle: float):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.angle = angle


class _CompatLine:
    def __init__(
        self,
        *,
        text: str,
        language: str | None,
        block_id: int,
        paragraph_id: int,
        confidence: float,
        x: float,
        y: float,
        width: float,
        height: float,
        angle: float,
        direction: int,
        content_type: int,
        words: list[_CompatWord],
    ):
        self.text = text
        self.language = language or ""
        self.block_id = block_id
        self.paragraph_id = paragraph_id
        self.confidence = confidence
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.angle = angle
        self.direction = direction
        self.content_type = content_type
        self.words = words


def _load_worker_json(output: str) -> dict[str, Any] | None:
    for line in reversed((output or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _ocr_bbox(bbox: tuple[int, int, int, int]) -> BoundingBox:
    x0, y0, x1, y1 = bbox
    return BoundingBox(left=float(x0), top=float(y0), right=float(x1), bottom=float(y1))


def _merge_bboxes(left: BoundingBox | None, right: BoundingBox | None) -> BoundingBox | None:
    if left is None:
        return right
    if right is None:
        return left
    return BoundingBox(
        left=min(left.left, right.left),
        top=min(left.top, right.top),
        right=max(left.right, right.right),
        bottom=max(left.bottom, right.bottom),
    )


def _ocr_direction(direction: TextDirection) -> str | None:
    if direction == TextDirection.RIGHT_TO_LEFT:
        return "rtl"
    if direction == TextDirection.LEFT_TO_RIGHT:
        return "ltr"
    return None


def _sort_line_boxes_top_to_bottom(line_boxes: list[LineBox]) -> list[LineBox]:
    if not line_boxes:
        return []
    heights = sorted(max(1, box.bbox[3] - box.bbox[1]) for box in line_boxes)
    median_h = heights[len(heights) // 2]
    row_tol = max(8, int(median_h * 0.65))
    rows: list[list[LineBox]] = []
    for box in sorted(line_boxes, key=lambda item: ((item.bbox[1] + item.bbox[3]) / 2, item.bbox[0])):
        cy = (box.bbox[1] + box.bbox[3]) / 2
        for row in rows:
            row_cy = sum((item.bbox[1] + item.bbox[3]) / 2 for item in row) / len(row)
            if abs(cy - row_cy) <= row_tol:
                row.append(box)
                break
        else:
            rows.append([box])

    ordered: list[LineBox] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: item.bbox[0]))
    return ordered


def _overlay_line_results(
    page,
    lines: list[Any],
    sx: float,
    sy: float,
) -> None:
    import fitz  # noqa: F811

    for line in lines:
        for word in getattr(line, "words", []) or []:
            if not getattr(word, "text", "") or not getattr(word, "width", 0):
                continue
            x0 = getattr(word, "x", 0) * sx
            y0 = getattr(word, "y", 0) * sy
            x1 = (getattr(word, "x", 0) + getattr(word, "width", 0)) * sx
            y1 = (getattr(word, "y", 0) + getattr(word, "height", 0)) * sy
            _insert_invisible_word(
                page,
                fitz.Rect(x0, y0, x1, y1),
                getattr(word, "text", ""),
            )


def _insert_invisible_word(page, rect, text: str) -> None:
    import fitz  # noqa: F811

    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
        return

    font_size = max(1.0, rect.height * 0.8)
    for _ in range(6):
        spare = page.insert_textbox(
            rect,
            text,
            fontsize=font_size,
            render_mode=3,
            rotate=0,
            align=fitz.TEXT_ALIGN_LEFT,
        )
        if spare >= 0:
            return
        font_size *= 0.82

    page.insert_text(
        fitz.Point(rect.x0, rect.y1 - max(1.0, font_size) * 0.1),
        text,
        fontsize=max(1.0, font_size),
        render_mode=3,
        rotate=0,
    )


def _line_results_to_page(
    lines: list[Any],
    *,
    page_number: int,
    img_size: tuple[int, int],
):
    try:
        from ocrmypdf._vendor.locro.models import (
            BoundingBox as VendorBoundingBox,
            OcrBlock,
            OcrLine,
            OcrPage,
            OcrWord,
        )
    except Exception:
        _ensure_locro_import_path()
        from locro.models import (
            BoundingBox as VendorBoundingBox,
            OcrBlock,
            OcrLine,
            OcrPage,
            OcrWord,
        )

    blocks_map: dict[int, list[Any]] = {}
    for line in lines:
        block_id = int(getattr(line, "block_id", 0) or 0)
        blocks_map.setdefault(block_id, []).append(line)

    blocks = []
    for block_id in sorted(blocks_map):
        ocr_lines = []
        for line in blocks_map[block_id]:
            words = []
            for word in getattr(line, "words", []) or []:
                width = getattr(word, "width", 0)
                bbox = None
                if width:
                    bbox = VendorBoundingBox(
                        x=getattr(word, "x", 0),
                        y=getattr(word, "y", 0),
                        width=width,
                        height=getattr(word, "height", 0),
                        angle=getattr(word, "angle", 0.0),
                    )
                words.append(
                    OcrWord(
                        text=getattr(word, "text", "") or "",
                        confidence=getattr(word, "confidence", None),
                        bounding_box=bbox,
                    )
                )

            line_bbox = None
            if getattr(line, "width", 0):
                line_bbox = VendorBoundingBox(
                    x=getattr(line, "x", 0),
                    y=getattr(line, "y", 0),
                    width=getattr(line, "width", 0),
                    height=getattr(line, "height", 0),
                    angle=getattr(line, "angle", 0.0),
                )

            ocr_lines.append(
                OcrLine(
                    text=getattr(line, "text", "") or "",
                    words=words,
                    bounding_box=line_bbox,
                )
            )
        blocks.append(OcrBlock(block_type="paragraph", lines=ocr_lines))

    return OcrPage(
        page_number=page_number,
        blocks=blocks,
        width=img_size[0],
        height=img_size[1],
    )


def _ensure_locro_import_path() -> None:
    vendored_root = Path(__file__).resolve().parent / "_vendor"
    if vendored_root.exists() and str(vendored_root) not in sys.path:
        sys.path.insert(0, str(vendored_root))
        if _can_import_locro():
            return

    if _can_import_locro():
        return

    candidates = []
    configured = os.environ.get("OCRMYPDF_CHROME_SCREENAI_PYTHONPATH") or os.environ.get(
        "OCRMYPDF_LOCRO_PATH"
    )
    if configured:
        candidates.append(Path(configured))

    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            if _can_import_locro():
                return


def _can_import_locro() -> bool:
    try:
        import locro  # noqa: F401
    except Exception:
        return False
    return True
