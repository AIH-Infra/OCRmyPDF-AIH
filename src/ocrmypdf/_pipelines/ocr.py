# SPDX-FileCopyrightText: 2019-2023 James R. Barlow
# SPDX-FileCopyrightText: 2019 Martin Wind
# SPDX-License-Identifier: MPL-2.0

"""Implements the concurrent and page synchronous parts of the pipeline."""

from __future__ import annotations

import logging
import logging.handlers
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from tempfile import mkdtemp

import PIL

from ocrmypdf.aih_chrome_screenai import ChromeScreenAIBackend
from ocrmypdf.aih_text_layer import rasterize_pdf_pages, sanitize_pdf_for_screenai, strip_hidden_text_layer
from ocrmypdf._concurrent import Executor
from ocrmypdf._graft import OcrGrafter
from ocrmypdf._jobcontext import PageContext, PdfContext
from ocrmypdf._options import OcrOptions, ProcessingMode, _pages_from_ranges
from ocrmypdf._pipeline import (
    copy_final,
    is_ocr_required,
    merge_sidecars,
    ocr_engine_direct,
    ocr_engine_hocr,
    ocr_engine_textonly_pdf,
    triage,
    validate_pdfinfo_options,
)
from ocrmypdf._pipelines._common import (
    PageResult,
    cli_exception_handler,
    do_get_pdfinfo,
    manage_debug_log_handler,
    manage_work_folder,
    postprocess,
    process_page,
    report_output_pdf,
    set_thread_pageno,
    setup_pipeline,
    worker_init,
)
from ocrmypdf._plugin_manager import OcrmypdfPluginManager
from ocrmypdf._progressbar import ProgressBar
from ocrmypdf._validation import (
    check_requested_output_file,
    create_input_file,
)
from ocrmypdf.exceptions import ExitCode
from ocrmypdf.helpers import available_cpu_count
from ocrmypdf.models.ocr_element import OcrElement

log = logging.getLogger(__name__)


def _image_to_ocr_text(
    page_context: PageContext, ocr_image_out: Path
) -> tuple[Path | None, Path, OcrElement | None]:
    """Run OCR engine on image to create OCR PDF and text file."""
    options = page_context.options
    pdf_renderer = options.pdf_renderer

    # fpdf2 is the default renderer (auto resolves to fpdf2)
    if pdf_renderer in ('auto', 'fpdf2'):
        # Use generate_ocr() if the engine supports it, otherwise use hOCR path
        ocr_engine = page_context.plugin_manager.get_ocr_engine(options=options)
        if ocr_engine and ocr_engine.supports_generate_ocr():
            ocr_tree, text_out = ocr_engine_direct(ocr_image_out, page_context)
            return None, text_out, ocr_tree
        ocr_out, text_out = ocr_engine_hocr(ocr_image_out, page_context)
    elif pdf_renderer == 'sandwich':
        ocr_out, text_out = ocr_engine_textonly_pdf(ocr_image_out, page_context)
    else:
        raise NotImplementedError(f"pdf_renderer {pdf_renderer}")
    return ocr_out, text_out, None


def _exec_page_sync(page_context: PageContext) -> PageResult:
    """Execute a pipeline for a single page synchronously."""
    set_thread_pageno(page_context.pageno + 1)

    if not is_ocr_required(page_context):
        return PageResult(pageno=page_context.pageno)

    ocr_image_out, pdf_page_from_image_out, orientation_correction = process_page(
        page_context
    )
    ocr_out, text_out, ocr_tree = _image_to_ocr_text(page_context, ocr_image_out)
    return PageResult(
        pageno=page_context.pageno,
        pdf_page_from_image=pdf_page_from_image_out,
        ocr=ocr_out,
        text=text_out,
        orientation_correction=orientation_correction,
        ocr_tree=ocr_tree,
    )


def exec_concurrent(context: PdfContext, executor: Executor) -> Sequence[str]:
    """Execute the OCR pipeline concurrently."""
    options = context.options
    jobs = options.jobs or available_cpu_count()
    worker_strategy = getattr(options, "extra_attrs", {}).get("_ocr_worker_strategy")
    force_threads = worker_strategy == "threaded"
    force_serial = worker_strategy == "serial"
    max_workers = max(1, min(len(context.pdfinfo), jobs))
    if force_serial:
        max_workers = 1
    if max_workers > 1:
        log.info("Starting processing with %d workers concurrently", max_workers)
    if force_threads:
        log.info("Using threaded OCR workers due to engine-specific worker strategy")
    elif force_serial:
        log.info("Using serial OCR worker execution due to engine-specific worker strategy")

    sidecars: list[Path | None] = [None] * len(context.pdfinfo)
    ocrgraft = OcrGrafter(context)

    def update_page(result: PageResult, pbar: ProgressBar):
        """After OCR is complete for a page, update the PDF."""
        try:
            set_thread_pageno(result.pageno + 1)
            sidecars[result.pageno] = result.text
            pbar.update(0.5)
            ocrgraft.graft_page(
                pageno=result.pageno,
                image=result.pdf_page_from_image,
                ocr_output=result.ocr,
                ocr_tree=result.ocr_tree,
                autorotate_correction=result.orientation_correction,
            )
            pbar.update(0.5)
        finally:
            set_thread_pageno(None)

    executor(
        use_threads=force_threads or options.use_threads,
        max_workers=max_workers,
        progress_kwargs=dict(
            total=len(context.pdfinfo),
            desc='OCR' if options.ocr_engine != 'none' else 'Image processing',
            unit='page',
            disable=not options.progress_bar,
        ),
        worker_initializer=partial(worker_init, PIL.Image.MAX_IMAGE_PIXELS),
        task=_exec_page_sync,
        task_arguments=context.get_page_context_args(),
        task_finished=update_page,
    )

    # Output sidecar text
    if options.sidecar:
        text = merge_sidecars(sidecars, context)
        # Copy text file to destination
        copy_final(text, options.sidecar, options.input_file)

    # Merge layers to one single pdf
    pdf = ocrgraft.finalize()

    messages: Sequence[str] = []
    if options.output_type != 'none':
        # PDF/A and metadata
        log.info("Postprocessing...")
        pdf, messages = postprocess(pdf, context, executor)

        # Copy PDF file to destination
        copy_final(pdf, options.output_file, options.input_file)
    return messages


def _run_chrome_screenai_native_pdf(
    *,
    options: OcrOptions,
    work_folder: Path,
    origin_pdf: Path,
    start_input_file: Path,
    plugin_manager: OcrmypdfPluginManager,
    executor: Executor,
) -> ExitCode:
    extra = getattr(options, "extra_attrs", {})
    requested_pages = options.pages
    if isinstance(requested_pages, str):
        requested_pages = _pages_from_ranges(requested_pages)
    pages = (
        sorted(page_number + 1 for page_number in requested_pages)
        if requested_pages
        else None
    )

    output_pdf = work_folder / "chrome_screenai_native.pdf"
    chrome_input_pdf = origin_pdf

    # Same policy as the Web UI: default 8 page workers, at most 24 (set with -j).
    workers = max(1, min(24, int(options.jobs or 8)))
    log.info("Chrome ScreenAI native PDF path using %d page worker(s)", workers)
    backend = ChromeScreenAIBackend(
        light_mode=bool(extra.get("chrome_screenai_light", False)),
        force_horizontal=bool(extra.get("chrome_screenai_force_horizontal", True)),
    )
    result = backend.ocr_to_searchable_pdf(
        chrome_input_pdf,
        output_pdf,
        pages=pages,
        workers=workers,
        collect_result=bool(options.sidecar),
    )

    if options.sidecar:
        sidecar_text = ""
        if hasattr(result, "to_text"):
            sidecar_text = result.to_text()
        sidecar_file = work_folder / "chrome_screenai_native.txt"
        sidecar_file.write_text(sidecar_text, encoding="utf-8")
        copy_final(sidecar_file, options.sidecar, options.input_file)

    pdfinfo = do_get_pdfinfo(chrome_input_pdf, executor, options)
    context = PdfContext(options, work_folder, chrome_input_pdf, pdfinfo, plugin_manager)
    output_pdf, messages = postprocess(output_pdf, context, executor)
    copy_final(output_pdf, options.output_file, options.input_file)
    return report_output_pdf(options, start_input_file, messages)


def _strip_hidden_text_before_ocr(
    options: OcrOptions, work_folder: Path, origin_pdf: Path
) -> Path:
    """Drop a previous OCR's invisible text layer so the pages are OCRed again.

    Default behaviour (``--hidden-text-layer strip``). Only hidden text on pages that
    have an image behind it is removed, visible text is never touched, and the input
    file is not modified. Use ``--hidden-text-layer keep`` for upstream behaviour
    (``--skip-text`` / ``--redo-ocr`` / exit code 6 on already-OCRed input).
    """
    extra = getattr(options, "extra_attrs", {}) or {}
    policy = str(extra.get("hidden_text_layer") or "strip").lower()
    if policy == "keep" or options.ocr_engine == "none":
        return origin_pdf
    if options.mode not in (ProcessingMode.default, ProcessingMode.redo):
        return origin_pdf  # skip: user wants existing text pages left alone; force: rasterizes

    requested = options.pages
    if isinstance(requested, str):
        requested = _pages_from_ranges(requested)
    pages = sorted(n + 1 for n in requested) if requested else None

    stripped = work_folder / "hidden_text_stripped_input.pdf"
    try:
        report = strip_hidden_text_layer(origin_pdf, stripped, pages=pages)
    except Exception as exc:  # never block OCR on this optimisation
        log.warning("Could not check for a hidden text layer (%s); continuing", exc)
        return origin_pdf

    if report.pages_skipped_no_image:
        log.warning(
            "Kept hidden text on page(s) %s: there is no image behind it, "
            "so removing it would blank the page",
            report.pages_skipped_no_image[:20],
        )
    if report.pages_skipped_shift_risk:
        log.warning(
            "Kept hidden text on page(s) %s: removing it would shift visible text",
            report.pages_skipped_shift_risk[:20],
        )
    if not report.pages_changed:
        return origin_pdf
    log.info(
        "Removed the hidden text layer on %d page(s); they will be OCRed again",
        len(report.pages_changed),
    )
    if report.visible_text_preserved is False:
        log.warning("Visible text changed while removing the hidden layer; using original")
        return origin_pdf
    return stripped


def _run_pipeline(
    options: OcrOptions,
    plugin_manager: OcrmypdfPluginManager,
) -> ExitCode:
    with (
        manage_work_folder(
            work_folder=Path(mkdtemp(prefix="ocrmypdf.io.")),
            retain=options.keep_temporary_files,
            print_location=options.keep_temporary_files,
        ) as work_folder,
        manage_debug_log_handler(options=options, work_folder=work_folder),
    ):
        executor = setup_pipeline(options, plugin_manager)
        check_requested_output_file(options)
        start_input_file, original_filename = create_input_file(options, work_folder)

        # Triage image or pdf
        origin_pdf = triage(
            original_filename, start_input_file, work_folder / 'origin.pdf', options
        )

        extra_attrs = getattr(options, "extra_attrs", {})
        chrome_cleanup = str(
            extra_attrs.get("chrome_screenai_text_cleanup", "sanitize")
            or "sanitize"
        )
        if options.ocr_engine == "chrome_screenai" and chrome_cleanup in {"sanitize", "mode_default"}:
            requested_pages = options.pages
            if isinstance(requested_pages, str):
                requested_pages = _pages_from_ranges(requested_pages)
            cleanup_pages = sorted(n + 1 for n in requested_pages) if requested_pages else None
            prepared_pdf = work_folder / "chrome_screenai_sanitized_input.pdf"
            cleanup_report = sanitize_pdf_for_screenai(origin_pdf, prepared_pdf, pages=cleanup_pages)
            log.info("ScreenAI 净化：移除隐藏文字操作 %d 项、注释 %d 项；文件 %d → %d 字节",
                     cleanup_report["hidden_operations_removed"], cleanup_report["annotations_removed"],
                     cleanup_report["input_bytes"], cleanup_report["output_bytes"])
            origin_pdf = prepared_pdf
        else:
            origin_pdf = _strip_hidden_text_before_ocr(options, work_folder, origin_pdf)
        if options.ocr_engine == "chrome_screenai" and chrome_cleanup in {
            "clear_all",
            "rasterize",
        }:
            requested_pages = options.pages
            if isinstance(requested_pages, str):
                requested_pages = _pages_from_ranges(requested_pages)
            cleanup_pages = (
                sorted(page_number + 1 for page_number in requested_pages)
                if requested_pages
                else None
            )
            prepared_pdf = work_folder / f"chrome_screenai_{chrome_cleanup}_input.pdf"
            if chrome_cleanup == "clear_all":
                cleanup_report = strip_hidden_text_layer(
                    origin_pdf,
                    prepared_pdf,
                    pages=cleanup_pages,
                    scope="all",
                )
                log.info(
                    "Chrome explicit text cleanup removed %d text operation(s), including %d visible operation(s)",
                    cleanup_report.text_show_operations_removed,
                    cleanup_report.visible_show_operations_removed,
                )
            else:
                cleanup_report = rasterize_pdf_pages(
                    origin_pdf,
                    prepared_pdf,
                    pages=cleanup_pages,
                    dpi=int(extra_attrs.get("chrome_screenai_rasterize_dpi", 300) or 300),
                    jpeg_quality=int(
                        extra_attrs.get("chrome_screenai_rasterize_quality", 88) or 88
                    ),
                )
                log.info(
                    "Chrome explicit rasterization processed %d page(s) at %d DPI",
                    cleanup_report["pages_processed"],
                    cleanup_report["dpi"],
                )
            origin_pdf = prepared_pdf

        chrome_text_layer_mode = str(
            extra_attrs.get("chrome_screenai_text_layer_mode", "native_direct")
            or "native_direct"
        )
        chrome_native_default = chrome_text_layer_mode == "native_direct"
        if (
            options.ocr_engine == "chrome_screenai"
            and extra_attrs.get("_chrome_screenai_native_pdf", chrome_native_default)
        ):
            return _run_chrome_screenai_native_pdf(
                options=options,
                work_folder=work_folder,
                origin_pdf=origin_pdf,
                start_input_file=start_input_file,
                plugin_manager=plugin_manager, executor=executor,
            )

        # Gather pdfinfo and create context
        pdfinfo = do_get_pdfinfo(origin_pdf, executor, options)
        context = PdfContext(options, work_folder, origin_pdf, pdfinfo, plugin_manager)

        # Validate options are okay for this pdf
        validate_pdfinfo_options(context)

        # Execute the pipeline
        optimize_messages = exec_concurrent(context, executor)

        exitcode = report_output_pdf(options, start_input_file, optimize_messages)
        return exitcode


def run_pipeline_cli(
    options: OcrOptions,
    *,
    plugin_manager: OcrmypdfPluginManager,
) -> ExitCode:
    """Run the OCR pipeline with command line exception handling.

    Args:
        options: The parsed OCR options.
        plugin_manager: The plugin manager to use. If not provided, one will be
            created.
    """
    return cli_exception_handler(_run_pipeline, options, plugin_manager)


def run_pipeline(
    options: OcrOptions,
    *,
    plugin_manager: OcrmypdfPluginManager,
) -> ExitCode:
    """Run the OCR pipeline without command line exception handling.

    Args:
        options: The parsed OCR options.
        plugin_manager: The plugin manager to use. If not provided, one will be
            created.
    """
    return _run_pipeline(options, plugin_manager)
