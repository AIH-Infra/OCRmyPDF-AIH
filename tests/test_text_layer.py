# SPDX-License-Identifier: MPL-2.0
"""Hidden OCR text layer removal (aih_text_layer) and the default strip-then-OCR policy."""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from conftest import PAGE_TEXT, pdf_text, run_cli  # noqa: E402
from ocrmypdf.aih_text_layer import (  # noqa: E402
    classify_text_layers,
    strip_hidden_text_layer,
)

TEXT = "The quick brown fox jumps over the lazy dog while the committee reads."


def _scan_page(doc, text=TEXT):
    tmp = fitz.open()
    p = tmp.new_page(width=595, height=842)
    p.insert_text((50, 100), text, fontsize=14)
    pg = doc.new_page(width=595, height=842)
    pg.insert_image(pg.rect, pixmap=p.get_pixmap(dpi=100))
    return pg


def _append(doc, ops: str):
    """Append raw content-stream operators to page 1 (a font resource must exist)."""
    pg = doc[0]
    pg.insert_text((50, 20), "x", fontsize=1)
    font = pg.get_fonts()[0][4]
    xref = pg.get_contents()[0]
    doc.update_stream(xref, doc.xref_stream(xref) + ops.replace("{F}", font).encode())


def _save(doc, path):
    doc.save(path)
    return path


def _trace(path):
    vis = inv = 0
    pos = []
    for page in fitz.open(path):
        for t in page.get_texttrace():
            n = len(t.get("chars", ()))
            if t.get("type") == 0 and float(t.get("opacity", 1)) > 0:
                vis += n
                pos.append((round(t["bbox"][0]), round(t["bbox"][1])))
            else:
                inv += n
    return vis, inv, sorted(pos)


@pytest.fixture
def scan_invisible(tmp_path):
    d = fitz.open()
    _scan_page(d).insert_text((50, 100), TEXT, fontsize=14, render_mode=3)
    return _save(d, tmp_path / "a.pdf")


def test_removes_hidden_layer_from_scan(scan_invisible, tmp_path):
    out = tmp_path / "out.pdf"
    r = strip_hidden_text_layer(scan_invisible, out)
    assert r.pages_changed == [1]
    assert r.invisible_chars_remaining == 0
    assert r.visible_text_preserved is True
    assert _trace(out)[:2] == (0, 0)
    assert len(fitz.open(out)[0].get_images()) == 1  # the scan itself survives


def test_source_is_never_overwritten(scan_invisible):
    with pytest.raises(ValueError):
        strip_hidden_text_layer(scan_invisible, scan_invisible)


def test_visible_text_is_kept(tmp_path):
    d = fitz.open()
    pg = _scan_page(d)
    pg.insert_text((50, 100), TEXT, fontsize=14, render_mode=3)
    pg.insert_text((50, 400), "Native visible heading.", fontsize=14)
    out = tmp_path / "out.pdf"
    r = strip_hidden_text_layer(_save(d, tmp_path / "b.pdf"), out)
    assert r.pages_changed == [1] and r.visible_text_preserved
    vis, inv, _ = _trace(out)
    assert vis == len("Native visible heading.") and inv == 0


def test_hidden_text_without_image_is_kept(tmp_path):
    """Nothing is behind the text: removing it would blank the page."""
    d = fitz.open()
    d.new_page(width=595, height=842).insert_text((50, 100), TEXT, fontsize=14, render_mode=3)
    src = _save(d, tmp_path / "c.pdf")
    out = tmp_path / "out.pdf"
    r = strip_hidden_text_layer(src, out)
    assert r.pages_changed == [] and r.pages_skipped_no_image == [1]
    assert _trace(out)[1] == len(TEXT)


def test_form_xobject_layer_is_removed(tmp_path):
    d = fitz.open()
    pg = _scan_page(d)
    t = fitz.open()
    t.new_page(width=595, height=842).insert_text((50, 100), TEXT, fontsize=14, render_mode=3)
    pg.show_pdf_page(pg.rect, t, 0)
    out = tmp_path / "out.pdf"
    r = strip_hidden_text_layer(_save(d, tmp_path / "d.pdf"), out)
    assert r.pages_changed == [1] and r.invisible_chars_remaining == 0


def test_visible_text_after_hidden_in_same_object_is_not_shifted(tmp_path):
    d = fitz.open()
    _scan_page(d)
    _append(d, "\nBT /{F} 14 Tf 50 300 Td 3 Tr (hidden words ) Tj 0 Tr (shown words) Tj ET\n")
    src = _save(d, tmp_path / "e.pdf")
    out = tmp_path / "out.pdf"
    r = strip_hidden_text_layer(src, out)
    assert r.pages_skipped_shift_risk == [1] and r.pages_changed == []
    assert _trace(out)[2] == _trace(src)[2]


def test_quote_operator_keeps_its_line_advance(tmp_path):
    d = fitz.open()
    _scan_page(d)
    _append(
        d,
        "\nBT /{F} 14 Tf 16 TL 50 500 Td 3 Tr (inv one) Tj (inv two) ' 0 Tr (visible) ' ET\n",
    )
    src = _save(d, tmp_path / "f.pdf")
    out = tmp_path / "out.pdf"
    r = strip_hidden_text_layer(src, out)
    assert r.pages_changed == [1] and r.visible_text_preserved
    before, after = _trace(src), _trace(out)
    assert after[1] == 0
    assert after[2] == before[2]  # same visible text at the same positions


def test_untouched_pages_keep_their_content_stream(tmp_path):
    d = fitz.open()
    _scan_page(d).insert_text((50, 100), TEXT, fontsize=14, render_mode=3)
    pg = d.new_page()
    pg.insert_text((50, 100), "Plain native page", fontsize=12)
    src = _save(d, tmp_path / "g.pdf")
    out = tmp_path / "out.pdf"
    r = strip_hidden_text_layer(src, out)
    assert r.pages_changed == [1]
    assert fitz.open(src)[1].read_contents() == fitz.open(out)[1].read_contents()


def test_classification_kinds(scan_invisible, tmp_path):
    assert classify_text_layers(scan_invisible)[1]["kind"] == "ocr_invisible_only"
    d = fitz.open()
    d.new_page().insert_text((50, 100), "native", fontsize=12)
    assert classify_text_layers(_save(d, tmp_path / "n.pdf"))[1]["kind"] == "native"


def test_scope_all_clears_visible_text_too(tmp_path):
    d = fitz.open()
    d.new_page().insert_text((50, 100), "native", fontsize=12)
    out = tmp_path / "out.pdf"
    strip_hidden_text_layer(_save(d, tmp_path / "n.pdf"), out, scope="all")
    assert _trace(out)[:2] == (0, 0)


def test_cli_help_documents_the_option():
    r = run_cli("--help")
    assert "--hidden-text-layer" in r.stdout_text


@pytest.mark.tesseract
class TestReocr:
    @pytest.fixture
    def ocred(self, sample_pdf, tmp_path):
        out = tmp_path / "ocred.pdf"
        r = run_cli("-l", "eng", "--hidden-text-layer", "keep", sample_pdf, out)
        assert r.returncode == 0, r.stderr
        return out

    def test_default_reocrs_a_previously_ocred_scan(self, ocred, tmp_path):
        out = tmp_path / "again.pdf"
        r = run_cli("-l", "eng", ocred, out)
        assert r.returncode == 0, r.stderr
        assert "hidden text layer" in r.stderr
        assert PAGE_TEXT in pdf_text(out)

    def test_keep_restores_upstream_error(self, ocred, tmp_path):
        r = run_cli("-l", "eng", "--hidden-text-layer", "keep", ocred, tmp_path / "x.pdf")
        assert r.returncode == 6
