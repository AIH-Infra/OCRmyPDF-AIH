# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import logging

from pikepdf import Dictionary, Name, Operator, Pdf, Stream, parse_content_stream
from pikepdf.models import unparse_content_stream

log = logging.getLogger(__name__)

_TEXT_SHOW_OPERATORS = {"Tj", "TJ", "'", '"'}
# Operators that (re)position the text line, so a following show no longer depends on
# the glyph advance of text we removed.
_TEXT_POSITION_OPERATORS = {"Td", "TD", "Tm", "T*"}
_ALPHA_EPSILON = 1e-6
# More than 5 % invisible characters is a hidden layer, not noise (same cut-off as the
# humanities-doc-toolkit pdf_quality.layers module this approach is borrowed from).
INVISIBLE_FRACTION = 0.05
PAGE_KINDS = ("native", "ocr_invisible_only", "mixed_layers", "image_only", "empty")


def classify_page_text(page) -> dict[str, Any]:
    """Classify one PyMuPDF page by what text it carries.

    Uses MuPDF's own rendering semantics (``get_texttrace``: render mode, opacity), which
    is more faithful than guessing from the content stream. Kinds: ``native`` (visible
    text), ``ocr_invisible_only`` (scan with a hidden OCR layer), ``mixed_layers``
    (both), ``image_only`` and ``empty``. ``backed`` tells whether the page has an image
    or form XObject behind the text, i.e. whether removing hidden text is lossless.
    """
    visible = invisible = 0
    for trace in page.get_texttrace():
        count = len(trace.get("chars", ()))
        if trace.get("type") == 0 and float(trace.get("opacity", 1.0)) > 0:
            visible += count
        else:
            invisible += count
    has_images = bool(page.get_images(full=True))
    backed = has_images or bool(page.get_xobjects())
    total = visible + invisible
    if total == 0:
        kind = "image_only" if has_images else "empty"
    elif invisible / total <= INVISIBLE_FRACTION:
        kind = "native"
    elif visible / total <= INVISIBLE_FRACTION:
        kind = "ocr_invisible_only"
    else:
        kind = "mixed_layers"
    return {
        "kind": kind,
        "visible_chars": visible,
        "invisible_chars": invisible,
        "backed": backed,
    }


def classify_text_layers(
    pdf_path: str | Path, pages: Iterable[int] | None = None
) -> dict[int, dict[str, Any]]:
    """Return ``{1-based page number: classify_page_text(...)}`` for the pages."""
    import fitz

    with fitz.open(str(pdf_path)) as doc:
        selected = _normalise_pages(pages, len(doc))
        return {number: classify_page_text(doc[number - 1]) for number in selected}


@dataclass
class _GraphicsState:
    render_mode: int = 0
    fill_alpha: float = 1.0
    stroke_alpha: float = 1.0

    def copy(self) -> "_GraphicsState":
        return _GraphicsState(
            render_mode=self.render_mode,
            fill_alpha=self.fill_alpha,
            stroke_alpha=self.stroke_alpha,
        )


@dataclass
class TextLayerStripReport:
    total_pages: int
    selected_pages: list[int]
    scope: str = "hidden"
    pages_changed: list[int] = field(default_factory=list)
    pages_with_visible_text: list[int] = field(default_factory=list)
    text_show_operations_removed: int = 0
    hidden_show_operations_removed: int = 0
    visible_show_operations_preserved: int = 0
    visible_show_operations_removed: int = 0
    form_xobjects_rewritten: int = 0
    form_xobjects_skipped: int = 0
    # Pages that had hidden text but were deliberately left alone (and why).
    pages_skipped_no_image: list[int] = field(default_factory=list)
    pages_skipped_shift_risk: list[int] = field(default_factory=list)
    # Verification against MuPDF after writing (None when PyMuPDF is unavailable).
    invisible_chars_remaining: int | None = None
    visible_text_preserved: bool | None = None
    page_kinds: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages_skipped_no_image": self.pages_skipped_no_image,
            "pages_skipped_shift_risk": self.pages_skipped_shift_risk,
            "invisible_chars_remaining": self.invisible_chars_remaining,
            "visible_text_preserved": self.visible_text_preserved,
            "total_pages": self.total_pages,
            "selected_pages": self.selected_pages,
            "scope": self.scope,
            "pages_changed": self.pages_changed,
            "pages_with_visible_text": self.pages_with_visible_text,
            "text_show_operations_removed": self.text_show_operations_removed,
            "hidden_show_operations_removed": self.hidden_show_operations_removed,
            "visible_show_operations_preserved": self.visible_show_operations_preserved,
            "visible_show_operations_removed": self.visible_show_operations_removed,
            "form_xobjects_rewritten": self.form_xobjects_rewritten,
            "form_xobjects_skipped": self.form_xobjects_skipped,
        }


@dataclass
class _RewriteStats:
    removed: int = 0
    hidden_removed: int = 0
    visible: int = 0
    visible_removed: int = 0
    forms_rewritten: int = 0
    forms_skipped: int = 0
    shift_skipped: int = 0

    def add(self, other: "_RewriteStats") -> None:
        self.shift_skipped += other.shift_skipped
        self.removed += other.removed
        self.hidden_removed += other.hidden_removed
        self.visible += other.visible
        self.visible_removed += other.visible_removed
        self.forms_rewritten += other.forms_rewritten
        self.forms_skipped += other.forms_skipped


def strip_hidden_text_layer(
    input_pdf: str | Path,
    output_pdf: str | Path,
    *,
    pages: Iterable[int] | None = None,
    scope: str = "hidden",
) -> TextLayerStripReport:
    """Remove clearly invisible PDF text without rasterizing page content.

    With ``scope="hidden"`` (the default) only text that is not painted is removed (render
    mode 3, zero opacity, ...), which is what an OCR layer on a scan looks like. Safeguards:

    * pages are classified with MuPDF first and pages without hidden text are not touched
      (their content streams are not even rewritten);
    * hidden text on a page with nothing behind it (no image / form) is kept, because
      removing it would blank the page;
    * ``'`` and ``"`` are replaced by ``T*`` so the line advance survives;
    * if visible text would shift because it follows removed text in the same text
      object, that stream is left unchanged and the page is reported;
    * the result is re-checked with MuPDF (``report.invisible_chars_remaining`` and
      ``report.visible_text_preserved``).

    ``scope="all"`` is the explicit "clear every PDF text object" option and skips the
    classification and safeguards.
    """
    input_path = Path(input_pdf)
    output_path = Path(output_pdf)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Refusing to overwrite the source PDF")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if scope not in {"hidden", "all"}:
        raise ValueError(f"Unsupported text removal scope: {scope}")

    with Pdf.open(input_path) as pdf:
        total_pages = len(pdf.pages)
        selected_pages = _normalise_pages(pages, total_pages)
        recurse_forms = len(selected_pages) == total_pages
        report = TextLayerStripReport(
            total_pages=total_pages,
            selected_pages=selected_pages,
            scope=scope,
        )
        visited_forms: set[tuple[int, int]] = set()
        before = _try_classify(input_path, selected_pages) if scope == "hidden" else None
        if before is not None:
            report.page_kinds = {n: info["kind"] for n, info in before.items()}

        for page_number in selected_pages:
            if before is not None:
                info = before[page_number]
                if info["invisible_chars"] == 0:
                    continue  # nothing hidden here: leave the page exactly as it is
                if not info["backed"]:
                    # Hidden text is all this page has; removing it would blank it.
                    report.pages_skipped_no_image.append(page_number)
                    continue
            page = pdf.pages[page_number - 1]
            resources = _get_resources(page.obj)
            stats = _rewrite_content_container(
                pdf,
                page,
                resources,
                recurse_forms=recurse_forms,
                visited_forms=visited_forms,
                scope=scope,
            )
            if stats.removed:
                report.pages_changed.append(page_number)
            if stats.shift_skipped:
                report.pages_skipped_shift_risk.append(page_number)
            if stats.visible:
                report.pages_with_visible_text.append(page_number)
            report.text_show_operations_removed += stats.removed
            report.hidden_show_operations_removed += stats.hidden_removed
            report.visible_show_operations_preserved += stats.visible
            report.visible_show_operations_removed += stats.visible_removed
            report.form_xobjects_rewritten += stats.forms_rewritten
            report.form_xobjects_skipped += stats.forms_skipped

        pdf.save(output_path)

    if before is not None and report.pages_changed:
        _verify_strip(output_path, before, report)
    return report


def _try_classify(path: Path, pages: list[int]) -> dict[int, dict[str, Any]] | None:
    try:
        return classify_text_layers(path, pages)
    except ImportError:  # PyMuPDF missing: fall back to touching every selected page
        return None
    except Exception as exc:  # pragma: no cover - unreadable by MuPDF
        log.debug("MuPDF could not classify %s: %s", path, exc)
        return None


def _verify_strip(
    output_path: Path,
    before: dict[int, dict[str, Any]],
    report: TextLayerStripReport,
) -> None:
    """Re-read the written file and record what is really left."""
    after = _try_classify(output_path, report.pages_changed)
    if after is None:
        return
    report.invisible_chars_remaining = sum(i["invisible_chars"] for i in after.values())
    report.visible_text_preserved = all(
        after[n]["visible_chars"] == before[n]["visible_chars"]
        for n in report.pages_changed
    )
    if not report.visible_text_preserved:
        log.warning(
            "Hidden text removal changed the visible text on page(s) %s",
            [
                n
                for n in report.pages_changed
                if after[n]["visible_chars"] != before[n]["visible_chars"]
            ],
        )


def rasterize_pdf_pages(
    input_pdf: str | Path,
    output_pdf: str | Path,
    *,
    pages: Iterable[int] | None = None,
    dpi: int = 300,
    jpeg_quality: int = 88,
) -> dict[str, Any]:
    """Rasterize selected pages with bounded memory; preserve all other pages."""
    import fitz

    input_path = Path(input_pdf)
    output_path = Path(output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with fitz.open(input_path) as source:
        total_pages = len(source)
        selected_pages = _normalise_pages(pages, total_pages)
        selected_set = set(selected_pages)
        metadata = source.metadata
        toc = source.get_toc(simple=False)

        with fitz.open() as target:
            for page_number in range(1, len(source) + 1):
                if page_number not in selected_set:
                    target.insert_pdf(
                        source,
                        from_page=page_number - 1,
                        to_page=page_number - 1,
                        links=True,
                        annots=True,
                    )
                    continue

                page = source[page_number - 1]
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0),
                    alpha=False,
                )
                image_bytes = pixmap.tobytes(
                    "jpeg",
                    jpg_quality=max(1, min(100, int(jpeg_quality))),
                )
                new_page = target.new_page(
                    width=page.rect.width,
                    height=page.rect.height,
                )
                new_page.insert_image(new_page.rect, stream=image_bytes)

            if metadata:
                target.set_metadata(metadata)
            if toc:
                target.set_toc(toc)
            target.save(output_path, garbage=4, deflate=True)

    return {
        "strategy": "rasterize",
        "total_pages": total_pages,
        "selected_pages": selected_pages,
        "pages_processed": len(selected_pages),
        "dpi": int(dpi),
        "jpeg_quality": int(jpeg_quality),
    }


def _normalise_pages(pages: Iterable[int] | None, total_pages: int) -> list[int]:
    if pages is None:
        return list(range(1, total_pages + 1))
    selected = sorted({int(page) for page in pages})
    invalid = [page for page in selected if page < 1 or page > total_pages]
    if invalid:
        raise ValueError(f"Pages out of range: {invalid}")
    if not selected:
        raise ValueError("pages cannot be empty")
    return selected


def _rewrite_content_container(
    pdf: Pdf,
    container,
    resources: Dictionary,
    *,
    recurse_forms: bool,
    visited_forms: set[tuple[int, int]],
    scope: str,
) -> _RewriteStats:
    instructions = list(parse_content_stream(container, ""))
    output = []
    state = _GraphicsState()
    state_stack: list[_GraphicsState] = []
    in_text_object = False
    stats = _RewriteStats()
    pending_advance = False
    shift_risk = False

    for operands, operator in instructions:
        op = str(operator)
        if op == "q":
            state_stack.append(state.copy())
        elif op == "Q":
            if state_stack:
                state = state_stack.pop()
        elif op == "BT":
            in_text_object = True
        elif op == "ET":
            in_text_object = False
        elif op == "Tr" and operands:
            state.render_mode = int(operands[0])
        elif op == "gs" and operands:
            _apply_extgstate(state, resources, operands[0])

        if op in _TEXT_POSITION_OPERATORS or op == "BT":
            pending_advance = False

        if in_text_object and op in _TEXT_SHOW_OPERATORS:
            invisible = _is_invisible_text_state(state)
            if scope == "all" or invisible:
                stats.removed += 1
                if invisible:
                    stats.hidden_removed += 1
                else:
                    stats.visible_removed += 1
                # ' and " also move to the next line; keep that part of their effect.
                if op == "'":
                    output.append(([], Operator("T*")))
                elif op == '"' and len(operands) >= 3:
                    output.append(([operands[0]], Operator("Tw")))
                    output.append(([operands[1]], Operator("Tc")))
                    output.append(([], Operator("T*")))
                else:
                    pending_advance = True  # Tj / TJ moved the text position
                continue
            if op in ("'", '"'):
                pending_advance = False
            elif pending_advance:
                shift_risk = True
            stats.visible += 1
        output.append((operands, operator))

    if stats.removed == 0:
        pass  # nothing to change: leave the stream byte-for-byte alone
    elif shift_risk:
        # Visible text follows removed text inside one text object without being
        # repositioned; dropping the hidden part would move it. Leave this stream as is.
        stats = _RewriteStats(visible=stats.visible, shift_skipped=1)
    else:
        rewritten = unparse_content_stream(output)
        if isinstance(container, Stream):
            container.write(rewritten)
        else:
            container.obj.Contents = Stream(pdf, rewritten)

    xobjects = resources.get(Name.XObject, Dictionary())
    for _name, xobject in xobjects.items():
        if not isinstance(xobject, Stream) or xobject.get(Name.Subtype) != Name.Form:
            continue
        if not recurse_forms:
            stats.forms_skipped += 1
            continue
        objgen = tuple(xobject.objgen)
        if objgen in visited_forms:
            continue
        visited_forms.add(objgen)
        child_resources = _get_resources(xobject, fallback=resources)
        child_stats = _rewrite_content_container(
            pdf,
            xobject,
            child_resources,
            recurse_forms=True,
            visited_forms=visited_forms,
            scope=scope,
        )
        stats.add(child_stats)
        stats.forms_rewritten += 1
    return stats


def _get_resources(obj, fallback: Dictionary | None = None) -> Dictionary:
    resources = obj.get(Name.Resources, None)
    if isinstance(resources, Dictionary):
        return resources
    return fallback if fallback is not None else Dictionary()


def _apply_extgstate(
    state: _GraphicsState,
    resources: Dictionary,
    state_name,
) -> None:
    extgstates = resources.get(Name.ExtGState, Dictionary())
    extgstate = extgstates.get(state_name, None)
    if not isinstance(extgstate, Dictionary):
        return
    if Name("/ca") in extgstate:
        state.fill_alpha = float(extgstate[Name("/ca")])
    if Name("/CA") in extgstate:
        state.stroke_alpha = float(extgstate[Name("/CA")])


def _is_invisible_text_state(state: _GraphicsState) -> bool:
    mode = state.render_mode
    if mode == 3:
        return True
    if mode in (0, 4):
        return state.fill_alpha <= _ALPHA_EPSILON
    if mode in (1, 5):
        return state.stroke_alpha <= _ALPHA_EPSILON
    if mode in (2, 6):
        return (
            state.fill_alpha <= _ALPHA_EPSILON
            and state.stroke_alpha <= _ALPHA_EPSILON
        )
    # Mode 7 can build a clipping path used by later visible graphics.
    return False
