# OCRmyPDF-AIH 0.5.0

OCRmyPDF-AIH 0.5.0 improves batch PDF OCR workflows while keeping the
standard OCRmyPDF command-line interface.

## What’s new

- Added the Streamlit Web UI for multi-file PDF OCR and page-level progress.
- Added optional Calamari HTTP backend support alongside Tesseract.
- Added Chrome ScreenAI integration for local OCR workflows.
- Added configurable OCR preprocessing and output modes for ordinary PDF and
  PDF/A generation.
- Added Windows, macOS, and Linux installation and startup scripts.
- Added a packaged Web UI launcher: `ocrmypdf-aih-webui`.

## What’s improved

- Batch uploads and temporary results are handled through task-level disk
  storage, reducing memory use for multi-file jobs.
- Web UI port discovery is more reliable when the default port is unavailable.
- Tesseract and Ghostscript detection and installation guidance are clearer.
- The Calamari backend is clearly separated as an optional external service;
  Tesseract remains the default local backend.

## Installation

Run the platform-specific `install` script, then the matching `start` script.
System OCR dependencies and optional Calamari/Chrome ScreenAI components are
configured separately on the user’s computer.
