"""High-level OCR interface -- the main public API."""

from __future__ import annotations

import builtins
import logging
import os
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from PIL import Image

from ._dll import ScreenAIDll, find_screen_ai_dir
from ._protobuf import LineResult, parse_visual_annotation
from .models import BoundingBox, OcrBlock, OcrLine, OcrPage, OcrResult, OcrWord

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
_PROGRESS_STATE_KEY = "_ocrmypdf_page_progress_state"

# ---------------------------------------------------------------------------
# Font fallback for invisible text overlay
# ---------------------------------------------------------------------------

# Character ranges that the base-14 Helvetica font cannot encode.  Each range
# maps to a script key; the key resolves to a system font covering that script
# (see _fallback_font_candidates below).
_SCRIPT_RANGES = [
    (0x0370, 0x03FF, "greek"),      # Greek and Coptic
    (0x0400, 0x052F, "cyrillic"),   # Cyrillic + Cyrillic Supplement
    (0x0590, 0x05FF, "hebrew"),     # Hebrew
    (0x0600, 0x06FF, "arabic"),     # Arabic
    (0x0750, 0x077F, "arabic"),     # Arabic Supplement
    (0x08A0, 0x08FF, "arabic"),     # Arabic Extended-A
    (0x0E00, 0x0E7F, "thai"),       # Thai
    (0x3040, 0x309F, "cjk"),        # Hiragana
    (0x30A0, 0x30FF, "cjk"),        # Katakana
    (0x3400, 0x4DBF, "cjk"),        # CJK Extension A
    (0x4E00, 0x9FFF, "cjk"),        # CJK Unified Ideographs
    (0xAC00, 0xD7AF, "cjk"),        # Hangul Syllables
    (0x3000, 0x303F, "cjk"),        # CJK Symbols and Punctuation
    (0xFB50, 0xFDFF, "arabic"),     # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF, "arabic"),     # Arabic Presentation Forms-B
    (0xFF00, 0xFFEF, "cjk"),        # Halfwidth/Fullwidth forms
]

# One representative character per script, used to verify a candidate font
# really carries the needed glyphs before it gets embedded.
_SCRIPT_PROBE_CHARS = {
    "greek": "Ω",
    "cyrillic": "Я",
    "hebrew": "א",
    "arabic": "ا",
    "thai": "ก",
    "cjk": "中",
}


def _detect_fallback_script(text: str) -> str | None:
    """Return the script key of the first character the base-14 fonts cannot
    encode, or ``None`` when the built-in Latin fonts suffice."""
    for ch in text:
        cp = ord(ch)
        for start, end, script in _SCRIPT_RANGES:
            if start <= cp <= end:
                return script
    return None


def _fallback_font_candidates(script: str) -> list[Path]:
    """System fonts that can render *script*, in priority order."""
    if sys.platform == "win32":
        win_fonts = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "Fonts"
        user_fonts = Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts"
        names = {
            # Segoe UI first: its space glyph reverse-maps to U+0020 in the
            # embedded ToUnicode, while the Arial family yields U+00A0.
            "greek": ["segoeui.ttf", "arial.ttf", "tahoma.ttf", "times.ttf"],
            "cyrillic": ["segoeui.ttf", "arial.ttf", "tahoma.ttf", "times.ttf"],
            "hebrew": ["segoeui.ttf", "arial.ttf", "tahoma.ttf", "times.ttf"],
            "arabic": ["segoeui.ttf", "arial.ttf", "tahoma.ttf", "times.ttf"],
            "thai": ["tahoma.ttf", "leelawadeeui.ttf"],
            "cjk": ["simhei.ttf", "simsun.ttc", "msyh.ttc"],
        }.get(script, [])
        return [win_fonts / name for name in names] + [
            user_fonts / name for name in names
        ]
    if sys.platform == "darwin":
        unicode_font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        return {
            "greek": [unicode_font],
            "cyrillic": [unicode_font],
            "hebrew": [unicode_font, Path("/System/Library/Fonts/Hebrew.ttc")],
            "arabic": [unicode_font, Path("/System/Library/Fonts/GeezaPro.ttc")],
            "thai": [
                unicode_font,
                Path("/System/Library/Fonts/Thonburi.ttc"),
                Path("/System/Library/Fonts/Thonburi.ttf"),
            ],
            "cjk": [
                Path("/System/Library/Fonts/PingFang.ttc"),
                Path("/System/Library/Fonts/STHeiti Light.ttc"),
                Path("/Library/Fonts/Arial Unicode.ttf"),
            ],
        }.get(script, [unicode_font])
    noto_dirs = [Path("/usr/share/fonts"), Path("/usr/share/fonts/truetype/noto")]
    truetype = Path("/usr/share/fonts/truetype")
    if script == "cjk":
        return [
            truetype / "wqy/wqy-zenhei.ttc",
            noto_dirs[1] / "NotoSansCJK-Regular.ttc",
            truetype / "dejavu/DejaVuSans.ttf",
        ]
    noto_names = {
        "arabic": ["NotoSansArabic-Regular", "NotoNaskhArabic-Regular"],
        "hebrew": ["NotoSansHebrew-Regular", "DejaVuSans"],
        "thai": ["NotoSansThai-Regular"],
        "greek": ["NotoSans-Regular", "DejaVuSans"],
        "cyrillic": ["NotoSans-Regular", "DejaVuSans"],
    }.get(script, [])
    candidates = [
        noto_dir / f"{name}.{ext}"
        for noto_dir in noto_dirs
        for name in noto_names
        for ext in ("ttf", "otf")
    ]
    if script == "thai":
        candidates += [
            truetype / "thai/Loma.ttf",
            truetype / "tlwg/Loma.ttf",
            truetype / "thai/Garuda.ttf",
        ]
    if script in ("arabic", "hebrew", "greek", "cyrillic"):
        candidates.append(truetype / "dejavu/DejaVuSans.ttf")
    return candidates


def _probe_font_covers(font_path: Path, script: str) -> bool:
    """Check that *font_path* really has glyphs for *script*."""
    probe = _SCRIPT_PROBE_CHARS.get(script)
    if not probe:
        return True
    try:
        import fitz

        has_glyph = getattr(fitz.Font(fontfile=str(font_path)), "has_glyph", None)
        if callable(has_glyph):
            return all(has_glyph(ord(ch)) for ch in probe)
    except Exception:
        return False
    return True


_fallback_font_cache: dict[str, Path | None] = {}


def _get_fallback_font(script: str) -> Path | None:
    """Resolve and cache a system font able to render *script*.

    The base-14 Helvetica only covers Latin-1, so any script listed in
    _SCRIPT_RANGES would otherwise be replaced by question marks in the
    invisible text overlay.
    """
    if script in _fallback_font_cache:
        return _fallback_font_cache[script]
    resolved: Path | None = None
    for candidate in _fallback_font_candidates(script):
        if candidate.exists() and _probe_font_covers(candidate, script):
            resolved = candidate
            break
    if resolved is None:
        log.warning(
            "No system font found covering script %r; "
            "non-Latin text may be lost in the OCR text layer",
            script,
        )
    _fallback_font_cache[script] = resolved
    return resolved


def _embedded_font_name(font_path: Path) -> str:
    stem = "".join(ch for ch in font_path.stem if ch.isalnum())[:24] or "Fallback"
    return f"AIHFB{stem}{_FONT_ALIAS_SALT}"


# 字体别名盐：每次 ocr_to_searchable_pdf 调用开始时重新随机。PyMuPDF 的
# insert_text 遇到页面已有同名字体资源时会复用旧对象而忽略 fontfile——若
# 输入 PDF 带着上一遍 OCR 的 subset 字体（字形编号已被重排），第二遍按完
# 整字体 cmap 算出的 CID 会全部落空，写成 .notdef（表现为整段文字变
# NUL/乱码，L024 双遍处理即此）。同一次调用内盐不变（同文档共享一次嵌
# 入），跨调用必变（同一进程连续跑两遍也不会撞名）。
_FONT_ALIAS_SALT = os.urandom(3).hex()


def _reroll_font_alias_salt() -> None:
    global _FONT_ALIAS_SALT
    _FONT_ALIAS_SALT = os.urandom(3).hex()



def _update_progress(current: int, total: int, desc: str) -> None:
    state = getattr(builtins, _PROGRESS_STATE_KEY, None)
    if state is None:
        return
    state["page_current"] = current
    state["page_total"] = total
    state["page_desc"] = desc


# ---------------------------------------------------------------------------
# RTL line normalization (Chrome ScreenAI quirk)
# ---------------------------------------------------------------------------

# ScreenAI returns RTL lines (direction == 2) with an unreliable word order
# and, for individual words or whole lines, x coordinates mirrored
# horizontally (x_true = page_width - x - width).  The word *text* itself is
# always correct.  Normalization therefore rebuilds geometry and order from
# the coordinates: each word's frame is chosen to minimise box overlaps
# (mirrored words overlap their neighbours in the given frame), then the
# logical order is reconstructed run-aware: RTL runs read right-to-left with
# embedded LTR runs kept left-to-right.
#
# A residual DLL quirk cannot be detected from the OCR output alone: some
# lines are returned fully mirrored with no overlap in either frame (the
# mirrored layout is the mirror image of the correct one).  Those keep their
# DLL order; they are no worse than without this normalization.

_RTL_CHAR_RANGES = (
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0700, 0x08FF),  # Syriac / Arabic supplements
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)


def _classify_word(text: str) -> str:
    """Classify a word as 'rtl', 'ltr' or 'neutral' (no strong letters)."""
    for ch in text:
        if not ch.isalpha():
            continue
        if any(lo <= ord(ch) <= hi for lo, hi in _RTL_CHAR_RANGES):
            return "rtl"
        return "ltr"
    return "neutral"


def _strong_script_counts(text: str) -> tuple[int, int]:
    """Count (rtl, ltr) strong letters; digits/punctuation don't count."""
    ar = lt = 0
    for ch in text:
        if not ch.isalpha():
            continue
        if any(lo <= ord(ch) <= hi for lo, hi in _RTL_CHAR_RANGES):
            ar += 1
        else:
            lt += 1
    return ar, lt


def _resolve_neutral_words(kinds: list[str], paragraph_rtl: bool) -> list[str]:
    """Give every neutral word the direction of the strong run it belongs to.

    Bidi-style neutral resolution: join the surrounding strong runs when
    they agree, otherwise follow the paragraph base direction; edge
    neutrals join their only neighbour.  Without this, tokens like ':'
    or '13,413' would split runs and land at the wrong end of the line.
    """
    resolved = list(kinds)
    n = len(kinds)
    for i, kind in enumerate(kinds):
        if kind != "neutral":
            continue
        prev_kind = next(
            (kinds[j] for j in range(i - 1, -1, -1) if kinds[j] != "neutral"), None
        )
        next_kind = next(
            (kinds[j] for j in range(i + 1, n) if kinds[j] != "neutral"), None
        )
        if prev_kind is None and next_kind is None:
            target = "rtl" if paragraph_rtl else "ltr"
        elif prev_kind is None:
            target = next_kind
        elif next_kind is None:
            target = prev_kind
        elif prev_kind == next_kind:
            target = prev_kind
        else:
            target = "rtl" if paragraph_rtl else "ltr"
        resolved[i] = target
    return resolved


def _count_overlaps(spans: list[tuple[float, float]]) -> int:
    count = 0
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a, b = spans[i], spans[j]
            if a[0] < b[1] and b[0] < a[1]:
                count += 1
    return count


def _normalize_rtl_line(ln, page_width: int) -> None:
    words = list(ln.words)
    geometry = [w for w in words if w.width > 0]

    # Pass 0: absurdly wide boxes (a short word claiming most of the line)
    # carry one trustworthy edge and a garbage extent.  Re-anchor them to
    # the largest stretch of their claimed span not covered by the
    # reasonably-sized words.
    if geometry:
        widths = sorted(w.width for w in geometry)
        median_w = widths[len(widths) // 2] or 1
        oversized = [w for w in geometry if w.width > 3 * median_w]
        if oversized:
            reliable = [w for w in geometry if w.width <= 3 * median_w]
            for w in oversized:
                span = (w.x, w.x + w.width)
                covered = sorted(
                    (max(r.x, span[0]), min(r.x + r.width, span[1]))
                    for r in reliable
                    if r.x < span[1] and r.x + r.width > span[0]
                )
                merged: list[list[int]] = []
                for lo, hi in covered:
                    if merged and lo <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], hi)
                    else:
                        merged.append([lo, hi])
                gaps: list[tuple[int, int]] = []
                cursor = span[0]
                for lo, hi in merged:
                    if lo > cursor:
                        gaps.append((cursor, lo))
                    cursor = max(cursor, hi)
                if cursor < span[1]:
                    gaps.append((cursor, span[1]))
                best = max(gaps, key=lambda g: g[1] - g[0], default=None)
                if best is not None and (best[1] - best[0]) >= max(8, median_w * 0.5):
                    w.x, w.width = best[0], best[1] - best[0]

    # Pass 1: flip individually-mirrored words until no flip reduces the
    # number of overlapping word boxes.
    if len(geometry) > 1:
        spans = {id(w): (float(w.x), float(w.x + w.width)) for w in geometry}

        def flipped_span(w) -> tuple[float, float]:
            return (float(page_width - w.x - w.width), float(page_width - w.x))

        improved = True
        while improved:
            improved = False
            base = _count_overlaps(list(spans.values()))
            for w in geometry:
                old = spans[id(w)]
                spans[id(w)] = flipped_span(w)
                candidate = _count_overlaps(list(spans.values()))
                if candidate < base:
                    base = candidate
                    improved = True
                else:
                    spans[id(w)] = old
        for w in geometry:
            x0, x1 = spans[id(w)]
            if abs(x0 - w.x) > 0.5:
                w.x = max(0, round(x0))

    # Pass 2: rebuild the logical word order from coordinates.  RTL runs
    # always read right-to-left (descending x) and LTR runs left-to-right;
    # only the run sequence direction depends on the paragraph base
    # direction.  Base direction follows the line's own script majority
    # (strong letters only): a scholarly footnote line that is mostly Latin
    # reads LTR even when an Arabic quote word sits at its left edge, and an
    # Arabic-dominant line reads RTL.  Line-edge script is NOT a reliable
    # cue — ScreenAI places quote continuations at either edge.
    ar_chars = 0
    lt_chars = 0
    for w in words:
        a, l = _strong_script_counts(w.text)
        ar_chars += a
        lt_chars += l
    paragraph_rtl = ar_chars > lt_chars

    visual = sorted(words, key=lambda w: w.x)
    kinds = [_classify_word(w.text) for w in visual]
    resolved = _resolve_neutral_words(kinds, paragraph_rtl)
    runs: list[tuple[str, list]] = []
    for w, kind in zip(visual, resolved):
        if runs and runs[-1][0] == kind:
            runs[-1][1].append(w)
        else:
            runs.append((kind, [w]))
    logical: list = []
    for kind, run in (reversed(runs) if paragraph_rtl else runs):
        logical.extend(reversed(run) if kind == "rtl" else run)
    ln.words = logical
    ln.text = " ".join(w.text for w in logical)


class ScreenAI:
    """Direct interface to Chrome's screen-ai OCR.

    Loads the screen-ai shared library via ctypes -- no browser needed.

    >>> ai = ScreenAI()
    >>> result = ai.ocr("scan.pdf")
    >>> print(result.to_text())
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        *,
        light_mode: bool = False,
    ):
        if model_dir is None:
            model_dir = find_screen_ai_dir()
        self._dll = ScreenAIDll(model_dir)
        if light_mode:
            self._dll.set_light_mode(True)
        if not self._dll.init_ocr():
            raise RuntimeError("Failed to initialize screen-ai OCR pipeline")
        self._max_dim = self._dll.get_max_image_dimension()
        log.info("OCR ready (max dimension: %d)", self._max_dim)

    # -- Public API ---------------------------------------------------------

    @property
    def version(self) -> tuple[int, int]:
        return self._dll.get_version()

    @property
    def max_image_dimension(self) -> int:
        return self._max_dim

    def ocr(
        self,
        file: str | Path,
        *,
        pages: Iterable[int] | None = None,
    ) -> OcrResult:
        """OCR a PDF or image file.

        Args:
            file: Path to a PDF or image.
            pages: For PDFs, 1-based page numbers to OCR.
                   ``None`` means all pages.  Ignored for images.

        Returns:
            Structured :class:`OcrResult`.
        """
        path = Path(file)
        if path.suffix.lower() == ".pdf":
            return self._ocr_pdf(path, pages=pages)
        return self._ocr_image_file(path)

    def ocr_pil_image(self, img: Image.Image) -> OcrPage:
        """OCR a single PIL Image.  Returns one :class:`OcrPage`."""
        lines, size = self._perform_ocr(img)
        return _lines_to_page(lines, page_number=1, img_size=size)

    def ocr_to_searchable_pdf(
        self,
        input_pdf: str | Path,
        output_pdf: str | Path,
        *,
        pages: Iterable[int] | None = None,
    ) -> OcrResult:
        """OCR a PDF and write a searchable copy with invisible text overlay.

        The output PDF looks identical to the input but its text is
        selectable and searchable.

        Args:
            input_pdf: Source (image-only) PDF.
            output_pdf: Destination path for the searchable PDF.
            pages: 1-based page numbers to OCR.  ``None`` = all.

        Returns:
            The :class:`OcrResult` that was overlaid.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for PDF operations: pip install PyMuPDF"
            ) from exc

        doc = fitz.open(str(input_pdf))
        _reroll_font_alias_salt()  # 每次调用独立盐：二次处理同文档不撞旧字体资源名
        page_set = set(pages) if pages is not None else None
        selected_pages = [
            i + 1 for i in range(len(doc))
            if page_set is None or (i + 1) in page_set
        ]
        ocr_pages: list[OcrPage] = []
        scripts_used: set[str] = set()
        _update_progress(0, len(selected_pages), "Chrome ScreenAI OCR")

        for index, page_num in enumerate(selected_pages, start=1):
            i = page_num - 1

            pg = doc[i]
            if pg.rotation:
                pg.remove_rotation()
            scale = min(1.0, self._max_dim / max(pg.rect.width, pg.rect.height))
            dpi = int(72 * scale * 2)
            pix = pg.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            lines, ocr_size = self._perform_ocr(img)
            ocr_page = _lines_to_page(lines, page_number=page_num, img_size=ocr_size)
            ocr_pages.append(ocr_page)

            # Scale factor: OCR pixel coords -> PDF point coords
            sx = pg.rect.width / ocr_size[0]
            sy = pg.rect.height / ocr_size[1]

            scripts_used.update(_overlay_text(pg, lines, sx, sy))

            log.info(
                "Page %d: %d blocks, %d lines (overlaid)",
                page_num,
                len(ocr_page.blocks),
                sum(len(b.lines) for b in ocr_page.blocks),
            )
            _update_progress(index, len(selected_pages), "Chrome ScreenAI OCR")

        if hasattr(doc, "subset_fonts"):
            doc.subset_fonts()
        doc.save(str(output_pdf), deflate=True, garbage=4)
        doc.close()
        if "arabic" in scripts_used:
            try:
                _normalize_arabic_tounicode(Path(output_pdf))
            except Exception:
                log.warning("Arabic ToUnicode normalization failed", exc_info=True)
        log.info("Searchable PDF -> %s", output_pdf)
        return OcrResult(pages=ocr_pages)

    # -- Internals ----------------------------------------------------------

    def _perform_ocr(
        self, img: Image.Image,
    ) -> tuple[list[LineResult], tuple[int, int]]:
        """Resize, convert to BGRA, call library, parse protobuf."""
        w, h = img.size
        if max(w, h) > self._max_dim:
            scale = self._max_dim / max(w, h)
            nw, nh = int(w * scale), int(h * scale)
            log.info("Resizing %dx%d -> %dx%d", w, h, nw, nh)
            img = img.resize((nw, nh), Image.LANCZOS)

        img = img.convert("RGBA")
        width, height = img.size

        raw = self._dll.perform_ocr(img.tobytes("raw", "BGRA"), width, height)
        if raw is None:
            return [], (width, height)
        lines = parse_visual_annotation(raw)
        for ln in lines:
            if ln.direction == 2 and ln.words:
                _normalize_rtl_line(ln, width)
        return lines, (width, height)

    def _ocr_image_file(self, path: Path) -> OcrResult:
        img = Image.open(path)
        lines, size = self._perform_ocr(img)
        page = _lines_to_page(lines, page_number=1, img_size=size)
        return OcrResult(pages=[page])

    def _ocr_pdf(
        self, path: Path, *, pages: Iterable[int] | None = None,
    ) -> OcrResult:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for PDF OCR: pip install PyMuPDF"
            ) from exc

        doc = fitz.open(path)
        page_set = set(pages) if pages is not None else None
        selected_pages = [
            i + 1 for i in range(len(doc))
            if page_set is None or (i + 1) in page_set
        ]
        ocr_pages: list[OcrPage] = []
        _update_progress(0, len(selected_pages), "Chrome ScreenAI OCR")

        for index, page_num in enumerate(selected_pages, start=1):
            i = page_num - 1

            pg = doc[i]
            scale = min(1.0, self._max_dim / max(pg.rect.width, pg.rect.height))
            pix = pg.get_pixmap(dpi=int(72 * scale * 2))
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            lines, size = self._perform_ocr(img)
            ocr_page = _lines_to_page(lines, page_number=page_num, img_size=size)
            ocr_pages.append(ocr_page)
            log.info(
                "Page %d: %d blocks, %d lines", page_num,
                len(ocr_page.blocks),
                sum(len(b.lines) for b in ocr_page.blocks),
            )
            _update_progress(index, len(selected_pages), "Chrome ScreenAI OCR")

        return OcrResult(pages=ocr_pages)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lines_to_page(
    lines: list[LineResult], page_number: int, img_size: tuple[int, int],
) -> OcrPage:
    blocks_map: dict[int, list[LineResult]] = defaultdict(list)
    for ln in lines:
        blocks_map[ln.block_id].append(ln)

    blocks: list[OcrBlock] = []
    for block_id in sorted(blocks_map):
        ocr_lines: list[OcrLine] = []
        for ln in blocks_map[block_id]:
            words = [
                OcrWord(
                    text=w.text,
                    confidence=w.confidence or None,
                    bounding_box=BoundingBox(x=w.x, y=w.y, width=w.width, height=w.height, angle=w.angle)
                    if w.width else None,
                )
                for w in ln.words
            ]
            ocr_lines.append(OcrLine(
                text=ln.text, words=words,
                bounding_box=BoundingBox(x=ln.x, y=ln.y, width=ln.width, height=ln.height, angle=ln.angle)
                if ln.width else None,
            ))
        blocks.append(OcrBlock(block_type="paragraph", lines=ocr_lines))

    return OcrPage(
        page_number=page_number, blocks=blocks,
        width=img_size[0], height=img_size[1],
    )


def _overlay_text(
    page,  # fitz.Page
    lines: list[LineResult],
    sx: float,
    sy: float,
) -> set[str]:
    """Insert invisible text onto a PyMuPDF page for each OCR'd word.

    Returns the set of non-Latin scripts for which a fallback font was used.
    """
    import fitz  # noqa: F811

    scripts_used: set[str] = set()
    for ln in lines:
        for w in ln.words:
            if not w.text or not w.width:
                continue
            # Convert OCR pixel coords to PDF points
            x0 = w.x * sx
            y0 = w.y * sy
            x1 = (w.x + w.width) * sx
            y1 = (w.y + w.height) * sy
            script = _insert_invisible_word(page, fitz.Rect(x0, y0, x1, y1), w.text)
            if script:
                scripts_used.add(script)
    return scripts_used


# Thresholds for conditionally stretching proportional (mostly Latin) words.
_LATIN_STRETCH_THRESHOLD = 1.25
_LATIN_MAX_STRETCH = 2.5


def _insert_invisible_word(page, rect, text: str) -> str | None:
    """Insert invisible horizontal text constrained to the OCR word box.

    Returns the fallback script key used, or ``None`` for plain Latin words.
    """
    import fitz  # noqa: F811

    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
        return None

    script = _detect_fallback_script(text)
    font_path = _get_fallback_font(script) if script else None
    if font_path is not None:
        font = fitz.Font(fontfile=str(font_path))
        font_size = max(1.0, rect.height * 0.95)
        text_width = font.text_length(text, font_size)
        if text_width > 0:
            scale_x = rect.width / text_width
            baseline_y = rect.y1 - font_size * 0.1
            point = fitz.Point(rect.x0, baseline_y)
            page.insert_text(
                point,
                text,
                fontsize=font_size,
                render_mode=3,
                rotate=0,
                morph=(point, fitz.Matrix(scale_x, 1.0)),
                fontfile=str(font_path),
                fontname=_embedded_font_name(font_path),
            )
            return script

    font_size = max(1.0, rect.height * 0.8)
    font = fitz.Font("helv")
    text_width = font.text_length(text, font_size)
    if text_width > 0:
        scale_x = rect.width / text_width
        if scale_x > _LATIN_STRETCH_THRESHOLD and scale_x <= _LATIN_MAX_STRETCH:
            baseline_y = rect.y1 - font_size * 0.1
            point = fitz.Point(rect.x0, baseline_y)
            page.insert_text(
                point,
                text,
                fontsize=font_size,
                fontname="helv",
                render_mode=3,
                rotate=0,
                morph=(point, fitz.Matrix(scale_x, 1.0)),
            )
            return None

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
            return None
        font_size *= 0.82

    page.insert_text(
        fitz.Point(rect.x0, rect.y1 - max(1.0, font_size) * 0.1),
        text,
        fontsize=max(1.0, font_size),
        render_mode=3,
        rotate=0,
    )
    return None


# ---------------------------------------------------------------------------
# Arabic ToUnicode normalization
# ---------------------------------------------------------------------------

# PyMuPDF shapes Arabic text with HarfBuzz before writing it, so the embedded
# font's ToUnicode map points at presentation forms (U+FB50-U+FDFF /
# U+FE70-U+FEFF) instead of the logical code points the OCR engine produced.
# Extracted text then contains presentation forms and plain logical Arabic
# queries stop matching.  Shaping is irrelevant for an invisible layer, so we
# rewrite those ToUnicode destinations to their NFKC equivalents after saving.

_ARABIC_PRESENTATION_RANGES = ((0xFB50, 0xFDFF), (0xFE70, 0xFEFF))


def _normalize_arabic_tounicode(pdf_path: Path) -> bool:
    """Fold Arabic presentation forms in embedded ToUnicode CMaps to logical
    code points.  Returns True when the PDF file was rewritten."""
    import re
    import unicodedata

    try:
        import pikepdf
    except ImportError:
        log.warning("pikepdf unavailable; skipping Arabic ToUnicode normalization")
        return False

    # PyMuPDF emits bfchar entries without the trailing semicolon
    # (``<0041> <0061>``), so entries are parsed line-based.  Edits are
    # surgical: untouched lines keep their original bytes, presentation-form
    # destinations are folded in place, and bfranges that contain folded
    # CIDs are split into sub-ranges (plus bfchar entries appended before
    # ``endcmap``).  Regenerating the whole CMap from parsed entries makes
    # MuPDF reject the mapping, so the original stream must be preserved.
    entry_bfchar_re = re.compile(rb"^<([0-9A-Fa-f]{4})>\s*<([0-9A-Fa-f]{4,})>;?\s*$")
    entry_bfrange_re = re.compile(rb"^<([0-9A-Fa-f]{4})>\s*<([0-9A-Fa-f]{4})>\s*<([0-9A-Fa-f]{4})>\s*$")

    def _is_presentation(cp: int) -> bool:
        return any(lo <= cp <= hi for lo, hi in _ARABIC_PRESENTATION_RANGES)

    def _dst_units(hexbytes: bytes) -> list[int]:
        return [int(hexbytes[i : i + 4], 16) for i in range(0, len(hexbytes), 4)]

    def _fold(units: list[int]) -> list[int] | None:
        text = "".join(chr(u) for u in units)
        if not all(_is_presentation(ord(ch)) for ch in text):
            return None
        folded = unicodedata.normalize("NFKC", text)
        if folded == text:
            return None
        return [ord(ch) for ch in folded]

    def _hex(units: list[int]) -> bytes:
        return "".join(f"{u:04X}" for u in units).encode("ascii")

    def _edit(data: bytes) -> bytes | None:
        out_lines: list[bytes] = []
        folded: dict[int, list[int]] = {}
        changed = False
        for raw in data.split(b"\n"):
            line = raw.strip()
            m = entry_bfchar_re.match(line)
            if m:
                units = _dst_units(m.group(2))
                folded_units = _fold(units)
                if folded_units is not None:
                    suffix = b";" if line.endswith(b";") else b""
                    out_lines.append(b"<" + m.group(1) + b"> <" + _hex(folded_units) + b">" + suffix)
                    changed = True
                    continue
                out_lines.append(raw)
                continue
            m = entry_bfrange_re.match(line)
            if m:
                lo, hi, base = (int(m.group(i), 16) for i in (1, 2, 3))
                line_folded = False
                sub_lo = lo
                for src in range(lo, hi + 1):
                    folded_units = _fold([base + src - lo])
                    if folded_units is None:
                        continue
                    folded[src] = folded_units
                    changed = True
                    line_folded = True
                    if src > sub_lo:
                        sub_dst = base + (sub_lo - lo)
                        out_lines.append(
                            f"<{sub_lo:04X}> <{src - 1:04X}> <".encode("ascii")
                            + f"{sub_dst:04X}".encode("ascii") + b">"
                        )
                    sub_lo = src + 1
                if line_folded:
                    if sub_lo <= hi:
                        sub_dst = base + (sub_lo - lo)
                        out_lines.append(
                            f"<{sub_lo:04X}> <{hi:04X}> <".encode("ascii")
                            + f"{sub_dst:04X}".encode("ascii") + b">"
                        )
                    continue
                out_lines.append(raw)
                continue
            out_lines.append(raw)
        if not changed:
            return None

        if folded:
            items = sorted(folded.items())
            block: list[bytes] = []
            for chunk_start in range(0, len(items), 100):
                chunk = items[chunk_start : chunk_start + 100]
                block.append(str(len(chunk)).encode("ascii") + b" beginbfchar")
                for src, units in chunk:
                    block.append(f"<{src:04X}> <".encode("ascii") + _hex(units) + b">")
                block.append(b"endbfchar")
            for idx in range(len(out_lines) - 1, -1, -1):
                if out_lines[idx].strip() == b"endcmap":
                    out_lines[idx:idx] = block
                    break
        return b"\n".join(out_lines)

    with pikepdf.open(str(pdf_path)) as pdf:
        seen: set = set()
        rewritten = False
        for page in pdf.pages:
            res = page.get("/Resources")
            if res is None:
                continue
            fonts = res.get("/Font")
            if fonts is None:
                continue
            for font in fonts.values():
                tu = font.get("/ToUnicode")
                if tu is None:
                    continue
                try:
                    key = tu.objgen
                except Exception:
                    key = id(tu)
                if key in seen:
                    continue
                seen.add(key)
                new_data = _edit(tu.read_bytes())
                if new_data is not None:
                    tu.write(new_data)
                    rewritten = True
        if not rewritten:
            return False
        tmp = pdf_path.with_suffix(pdf_path.suffix + ".tmp")
        pdf.save(str(tmp))
    os.replace(tmp, pdf_path)
    return True
