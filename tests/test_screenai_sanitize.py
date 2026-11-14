from __future__ import annotations

import io
import shutil
from pathlib import Path
from types import SimpleNamespace

import fitz
import pikepdf
import pytest
from PIL import Image

from ocrmypdf.aih_text_layer import sanitize_pdf_for_screenai


def _source(path, *, pages=1, hidden=True, annotations=True):
    image = io.BytesIO()
    Image.new("RGB", (120, 80), "#dbc9a2").save(image, format="JPEG", quality=90)
    with fitz.open() as doc:
        for index in range(pages):
            page = doc.new_page(width=300, height=400)
            page.insert_image(fitz.Rect(10, 100, 250, 260), stream=image.getvalue())
            page.insert_text((30, 60), f"VISIBLE BODY {index}", fontsize=13)
            if hidden:
                page.insert_text((30, 150), "OLD HIDDEN OCR", render_mode=3)
            if annotations:
                page.add_text_annot((80, 80), "discard note")
                page.add_highlight_annot(fitz.Rect(25, 45, 170, 65))
                page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(20, 300, 150, 320), "uri": "https://example.org"})
                widget = fitz.Widget()
                widget.field_name = f"field{index}"
                widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
                widget.field_value = "old form value"
                widget.rect = fitz.Rect(20, 330, 170, 350)
                page.add_widget(widget)
        doc.set_toc([[1, "Bookmark", 1]])
        doc.save(path)
    return path


def _body_pixels(page):
    # Remove annotations in memory before comparing the body. Even annots=False
    # uses a different transparency render group when a highlight is present.
    for widget in list(page.widgets() or []):
        page.delete_widget(widget)
    while page.first_annot:
        page.delete_annot(page.first_annot)
    page = page.parent.reload_page(page)
    return page.get_pixmap(annots=False).samples


def _images(path):
    with pikepdf.open(path) as pdf:
        return [obj.read_raw_bytes() for obj in pdf.objects
                if isinstance(obj, pikepdf.Stream) and obj.get("/Subtype") == "/Image"]


def test_sanitize_preserves_body_pixels_images_geometry_bookmarks_and_source(tmp_path):
    source = _source(tmp_path / "source.pdf")
    original = source.read_bytes()
    output = tmp_path / "clean.pdf"
    with fitz.open(source) as doc:
        pixels = _body_pixels(doc[0])
        toc = doc.get_toc()
    report = sanitize_pdf_for_screenai(source, output)
    assert report["hidden_operations_removed"] > 0 and report["annotations_removed"] >= 4
    assert report["output_bytes"] <= report["input_bytes"]
    assert source.read_bytes() == original
    assert _images(source) == _images(output)
    with fitz.open(output) as doc:
        assert doc[0].get_pixmap(annots=False).samples == pixels
        assert "VISIBLE BODY 0" in doc[0].get_text()
        assert "OLD HIDDEN OCR" not in doc[0].get_text()
        assert not list(doc[0].annots() or []) and not doc[0].get_links()
        assert not list(doc[0].widgets() or [])
        assert doc.get_toc() == toc
    with pikepdf.open(output) as pdf:
        assert "/AcroForm" not in pdf.Root


def test_selected_pages_leave_other_text_annotations_and_fields_untouched(tmp_path):
    source = _source(tmp_path / "source.pdf", pages=2)
    output = tmp_path / "clean.pdf"
    sanitize_pdf_for_screenai(source, output, pages=[1])
    with fitz.open(output) as doc:
        assert "OLD HIDDEN OCR" not in doc[0].get_text()
        assert "OLD HIDDEN OCR" in doc[1].get_text()
        assert list(doc[1].annots()) and doc[1].get_links()
        assert len(list(doc[1].widgets())) == 1


def test_already_clean_pdf_is_copied_byte_for_byte(tmp_path):
    source = _source(tmp_path / "source.pdf", hidden=False, annotations=False)
    output = tmp_path / "clean.pdf"
    report = sanitize_pdf_for_screenai(source, output)
    assert report["output_bytes"] == report["input_bytes"]
    assert source.read_bytes() == output.read_bytes()


def test_hidden_text_without_background_is_removed(tmp_path):
    source, output = tmp_path / "source.pdf", tmp_path / "clean.pdf"
    with fitz.open() as doc:
        doc.new_page().insert_text((50, 50), "hidden", render_mode=3)
        doc.save(source)
    sanitize_pdf_for_screenai(source, output)
    with fitz.open(output) as doc:
        assert not doc[0].get_text()


def test_shared_form_is_cleaned_per_invocation_and_inherited_opacity(tmp_path):
    source = _source(tmp_path / "base.pdf", hidden=False)
    modified, output = tmp_path / "forms.pdf", tmp_path / "clean.pdf"
    with pikepdf.open(source) as pdf:
        page = pdf.pages[0]
        form = pikepdf.Stream(pdf, b"BT /helv 12 Tf 30 200 Td (SHARED FORM) Tj ET")
        form.Type = pikepdf.Name.XObject
        form.Subtype = pikepdf.Name.Form
        form.BBox = [0, 0, 300, 400]
        form.Resources = pikepdf.Dictionary(Font=page.Resources.Font)
        page.Resources.XObject.Shared = form
        page.Resources.ExtGState = pikepdf.Dictionary(Zero=pikepdf.Dictionary(ca=0, CA=0))
        page.contents_add(b"q /Zero gs /Shared Do Q /Shared Do")
        pdf.save(modified)
    with fitz.open(modified) as doc:
        pixels = _body_pixels(doc[0])
    sanitize_pdf_for_screenai(modified, output)
    with fitz.open(output) as doc:
        assert doc[0].get_text().count("SHARED FORM") == 1
        assert doc[0].get_pixmap(annots=False).samples == pixels


def test_mixed_hidden_advance_reports_error_without_moving_body(tmp_path):
    source = _source(tmp_path / "base.pdf", hidden=False)
    modified = tmp_path / "mixed.pdf"
    with pikepdf.open(source) as pdf:
        pdf.pages[0].contents_add(b"BT /helv 12 Tf 30 200 Td 3 Tr (hidden ) Tj 0 Tr (body) Tj ET")
        pdf.save(modified)
    with pytest.raises(ValueError, match="共用连续字距"):
        sanitize_pdf_for_screenai(modified, tmp_path / "clean.pdf")


def test_pdfa_claim_is_accepted_as_input_and_image_streams_preserved(tmp_path):
    source = _source(tmp_path / "base.pdf")
    archival, output = tmp_path / "archival.pdf", tmp_path / "clean.pdf"
    with pikepdf.open(source) as pdf:
        with pdf.open_metadata(set_pikepdf_as_editor=False) as meta:
            meta["pdfaid:part"] = "2"
            meta["pdfaid:conformance"] = "B"
        pdf.save(archival)
    sanitize_pdf_for_screenai(archival, output)
    assert _images(archival) == _images(output)
    with fitz.open(output) as doc:
        assert "VISIBLE BODY" in doc[0].get_text() and "OLD HIDDEN" not in doc[0].get_text()


@pytest.mark.parametrize("output_type", ["pdf", "auto", "pdfa", "pdfa-1"])
def test_native_output_uses_standard_postprocessing_without_changing_format(tmp_path, monkeypatch, output_type):
    from ocrmypdf._pipelines import ocr
    source = _source(tmp_path / "source.pdf", hidden=False, annotations=False)
    output = tmp_path / "output.pdf"
    calls = []
    options = SimpleNamespace(extra_attrs={}, pages=None, output_type=output_type,
        jobs=1, sidecar=None, input_file=source, output_file=output)
    class FakeBackend:
        def __init__(self, **kwargs):
            pass
        def ocr_to_searchable_pdf(self, input_pdf, target_pdf, **kwargs):
            shutil.copyfile(input_pdf, target_pdf)
    monkeypatch.setattr(ocr, "ChromeScreenAIBackend", FakeBackend)
    monkeypatch.setattr(ocr, "do_get_pdfinfo", lambda *a: object())
    def postprocess(path, context, executor):
        calls.append(context.options.output_type)
        return path, ["validated"]
    monkeypatch.setattr(ocr, "postprocess", postprocess)
    monkeypatch.setattr(ocr, "report_output_pdf", lambda options, original, messages: messages)
    result = ocr._run_chrome_screenai_native_pdf(options=options, work_folder=tmp_path,
        origin_pdf=source, start_input_file=source, plugin_manager=object(), executor=object())
    assert calls == [output_type] and options.output_type == output_type
    assert result == ["validated"] and output.is_file()


def test_sanitize_never_overwrites_source(tmp_path):
    source = _source(tmp_path / "source.pdf")
    with pytest.raises(ValueError, match="overwrite"):
        sanitize_pdf_for_screenai(source, source)


@pytest.mark.parametrize("output_type", ["pdf", "pdfa"])
def test_real_pipeline_sanitizes_input_and_postprocesses_mocked_ocr(tmp_path, monkeypatch, output_type):
    import ocrmypdf
    from ocrmypdf.aih_chrome_screenai import ChromeScreenAIBackend
    source = _source(tmp_path / "source.pdf")
    output = tmp_path / "output.pdf"
    def fake_ocr(self, input_pdf, target_pdf, **kwargs):
        with fitz.open(input_pdf) as doc:
            assert "OLD HIDDEN OCR" not in doc[0].get_text()
            assert not list(doc[0].annots() or []) and not list(doc[0].widgets() or [])
            doc[0].insert_text((30, 280), "NEW OCR LAYER", render_mode=3)
            doc.save(target_pdf)
    monkeypatch.setattr(ChromeScreenAIBackend, "ocr_to_searchable_pdf", fake_ocr)
    plugin = Path(__file__).parents[1] / "src/ocrmypdf/aih_calamari_plugin.py"
    result = ocrmypdf.ocr(source, output, plugins=[str(plugin)], ocr_engine="chrome_screenai",
        output_type=output_type, mode="redo", progress_bar=False, jobs=1, optimize=0,
        chrome_screenai_text_cleanup="sanitize")
    assert int(result) == 0
    with fitz.open(output) as doc:
        assert "VISIBLE BODY" in doc[0].get_text() and "NEW OCR LAYER" in doc[0].get_text()
        assert "OLD HIDDEN OCR" not in doc[0].get_text()
    if output_type == "pdf":
        assert _images(source) == _images(output)
    else:
        from ocrmypdf.pdfa import file_claims_pdfa
        assert file_claims_pdfa(output)["pass"]


def test_selected_shared_form_keeps_unselected_visible_invocation(tmp_path):
    source = _source(tmp_path / "base.pdf", pages=2, hidden=False)
    modified, output = tmp_path / "forms.pdf", tmp_path / "clean.pdf"
    with pikepdf.open(source) as pdf:
        form = pikepdf.Stream(pdf, b"BT /helv 12 Tf 30 200 Td (SHARED) Tj ET")
        form.Type = pikepdf.Name.XObject
        form.Subtype = pikepdf.Name.Form
        form.BBox = [0, 0, 300, 400]
        form.Resources = pikepdf.Dictionary(Font=pdf.pages[0].Resources.Font)
        for index, page in enumerate(pdf.pages):
            page.Resources.XObject.Shared = form
            if index == 0:
                page.Resources.ExtGState = pikepdf.Dictionary(Zero=pikepdf.Dictionary(ca=0, CA=0))
                page.contents_add(b"q /Zero gs /Shared Do Q")
            else:
                page.contents_add(b"/Shared Do")
        pdf.save(modified)
    sanitize_pdf_for_screenai(modified, output, pages=[1])
    with fitz.open(modified) as before, fitz.open(output) as after:
        assert "SHARED" not in after[0].get_text()
        assert "SHARED" in after[1].get_text()
        assert after[1].get_pixmap().samples == before[1].get_pixmap().samples


def test_stroked_visible_text_is_preserved(tmp_path):
    source = _source(tmp_path / "base.pdf")
    modified, output = tmp_path / "stroked.pdf", tmp_path / "clean.pdf"
    with fitz.open(source) as doc:
        doc[0].insert_text((30, 290), "STROKED BODY", render_mode=1)
        doc.save(modified)
    sanitize_pdf_for_screenai(modified, output)
    with fitz.open(output) as doc:
        assert "STROKED BODY" in doc[0].get_text()


def test_output_size_guard_does_not_publish_a_larger_copy(tmp_path, monkeypatch):
    source = _source(tmp_path / "source.pdf")
    output = tmp_path / "clean.pdf"
    save = pikepdf.Pdf.save
    def padded_save(self, path, **kwargs):
        save(self, path, **kwargs)
        with Path(path).open("ab") as handle:
            handle.write(b"\n%" + b"padding" * 2000)
    monkeypatch.setattr(pikepdf.Pdf, "save", padded_save)
    with pytest.raises(ValueError, match="大于原文件"):
        sanitize_pdf_for_screenai(source, output)
    assert not output.exists()


def test_page_inherited_resources_are_preserved(tmp_path):
    source = _source(tmp_path / "base.pdf")
    inherited, output = tmp_path / "inherited.pdf", tmp_path / "clean.pdf"
    with pikepdf.open(source) as pdf:
        page = pdf.pages[0]
        page.obj.Parent.Resources = page.Resources
        del page.obj.Resources
        pdf.save(inherited)
    sanitize_pdf_for_screenai(inherited, output)
    with fitz.open(inherited) as before, fitz.open(output) as after:
        assert _body_pixels(before[0]) == after[0].get_pixmap(annots=False).samples
        assert "VISIBLE BODY" in after[0].get_text()
