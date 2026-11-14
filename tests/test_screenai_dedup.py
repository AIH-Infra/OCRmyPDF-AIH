from __future__ import annotations

from types import SimpleNamespace

import fitz

from ocrmypdf._vendor.locro.ocr import _overlay_text
from ocrmypdf.aih_text_layer import sanitize_pdf_for_screenai


def _word(text, rect):
    rect = fitz.Rect(rect)
    return SimpleNamespace(text=text, x=rect.x0, y=rect.y0, width=rect.width, height=rect.height)


def _write(page, words):
    return _overlay_text(page, [SimpleNamespace(words=words)], 1, 1)


def _counts(page):
    visible = hidden = 0
    for t in page.get_texttrace():
        if t["type"] in (0, 1) and t.get("opacity", 1) > 0:
            visible += len(t["chars"])
        else:
            hidden += len(t["chars"])
    return visible, hidden


def test_visible_body_is_not_overlaid_even_when_ocr_spelling_differs():
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((40, 60), "Native body", fontsize=14)
        boxes = page.get_text("words")
        before = page.get_pixmap().samples
        _write(page, [_word("Natlve", boxes[0][:4]), _word("body", boxes[1][:4])])
        assert _counts(page) == (len("Native body"), 0)
        assert page.get_pixmap().samples == before
        assert page.get_text().count("Native body") == 1


def test_mixed_page_keeps_heading_and_adds_only_uncovered_scanned_words():
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((40, 60), "Heading", fontsize=14)
        heading = page.get_text("words")[0][:4]
        _write(page, [_word("Heading", heading), _word("SCAN", (40, 100, 100, 114))])
        assert page.get_text().count("Heading") == 1
        assert page.get_text().count("SCAN") == 1
        assert _counts(page)[1] == 4


def test_recognizer_duplicate_words_are_not_written_twice():
    with fitz.open() as doc:
        page = doc.new_page()
        _write(page, [_word("SCAN", (40, 100, 100, 114)),
                     _word("scan", (40.2, 100.1, 100.2, 114.1)),
                     _word("SCAN", (140, 100, 200, 114))])
        assert page.get_text().upper().count("SCAN") == 2
        assert _counts(page)[1] == 8


def test_small_visible_fragment_does_not_suppress_an_uncovered_word():
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((40, 110), "A", fontsize=10)
        _write(page, [_word("ALPHABET", (40, 100, 140, 114))])
        assert "ALPHABET" in page.get_text()


def test_processing_twice_does_not_accumulate_hidden_layers(tmp_path):
    source = tmp_path / "source.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((40, 60), "Heading", fontsize=14)
        page.insert_text((40, 110), "OLD OCR", render_mode=3)
        doc.save(source)
    result = source
    for i in range(2):
        cleaned, output = tmp_path / f"clean{i}.pdf", tmp_path / f"ocr{i}.pdf"
        sanitize_pdf_for_screenai(result, cleaned)
        with fitz.open(cleaned) as doc:
            heading = doc[0].get_text("words")[0][:4]
            _write(doc[0], [_word("Heading", heading), _word("SCAN", (40, 100, 100, 114))])
            doc.save(output)
        result = output
        with fitz.open(result) as doc:
            assert doc[0].get_text().count("Heading") == 1
            assert doc[0].get_text().count("SCAN") == 1
            assert "OLD OCR" not in doc[0].get_text()
            assert _counts(doc[0])[1] == 4
