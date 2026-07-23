# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from pikepdf import Dictionary, Name, Pdf, Stream, parse_content_stream
from pikepdf.models import unparse_content_stream

_TEXT_SHOW_OPERATORS = {"Tj", "TJ", "'", '"'}
_ALPHA_EPSILON = 1e-6


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

    def to_dict(self) -> dict[str, Any]:
        return {
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

    def add(self, other: "_RewriteStats") -> None:
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
    """Remove clearly invisible PDF text without rasterizing page content."""
    input_path = Path(input_pdf)
    output_path = Path(output_pdf)
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

        for page_number in selected_pages:
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
            if stats.visible:
                report.pages_with_visible_text.append(page_number)
            report.text_show_operations_removed += stats.removed
            report.hidden_show_operations_removed += stats.hidden_removed
            report.visible_show_operations_preserved += stats.visible
            report.visible_show_operations_removed += stats.visible_removed
            report.form_xobjects_rewritten += stats.forms_rewritten
            report.form_xobjects_skipped += stats.forms_skipped

        pdf.save(output_path)
    return report


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

        if in_text_object and op in _TEXT_SHOW_OPERATORS:
            invisible = _is_invisible_text_state(state)
            if scope == "all" or invisible:
                stats.removed += 1
                if invisible:
                    stats.hidden_removed += 1
                else:
                    stats.visible_removed += 1
                continue
            stats.visible += 1
        output.append((operands, operator))

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
