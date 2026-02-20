"""Speed test: server-side batch predict vs client-side batch endpoint."""
import os
import subprocess
import tempfile
import time
import io
import re
from pathlib import Path

import requests
import cv2
import numpy as np

CALAMARI_URL = "http://localhost:11800"
tmpdir = tempfile.gettempdir()
conda_bin = r'D:\Runtime\OCRmyPDF_py312\Library\bin'
os.environ['PATH'] = conda_bin + os.pathsep + os.environ.get('PATH', '')

inp = r'C:\Users\vellichor\Desktop\J. G. Herder_(Einige Gespräche Über Spinozas System - Nebst Shaftesburis Naturhymnus)_Karl Wilhelm Ettinger_1800.pdf'

# Render page 30 (body text) at 300 DPI
img_path = os.path.join(tmpdir, 'herder_page30.png')
print("Rendering page 30 at 300 DPI...")
subprocess.run(
    ['gswin64c', '-dNOPAUSE', '-dBATCH', '-dSAFER', '-sDEVICE=png16m',
     '-r300', '-dFirstPage=30', '-dLastPage=30',
     f'-sOutputFile={img_path}', inp],
    capture_output=True, timeout=60,
)
sz = os.path.getsize(img_path)
print(f"Image: {sz//1024} KB\n")


# === Mode 1: Server-side segmentation (/ocr endpoint, batch predict inside) ===
print("=" * 60)
print("Mode 1: Server-side segmentation (/ocr, batch predict)")
print("=" * 60)
t0 = time.time()
with open(img_path, 'rb') as f:
    resp = requests.post(
        f"{CALAMARI_URL}/ocr",
        files={"file": ("page30.png", f, "image/png")},
        data={"model": "gt4histocr"},
        timeout=300,
    )
elapsed_server = time.time() - t0
data = resp.json()
text_server = data.get('text', '')
print(f"[{elapsed_server:.2f}s] {len(text_server)} chars")
print(f"Text preview:\n{text_server[:500]}\n")


# === Mode 2: Client-side segmentation → /ocr/batch ===
print("=" * 60)
print("Mode 2: Client-side segmentation → /ocr/batch")
print("=" * 60)

# Client-side line segmentation
t0_seg = time.time()
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
h_proj = np.sum(binary, axis=1) / 255.0
threshold = w * 0.02
in_line = h_proj > threshold

lines = []
start = None
for i in range(len(in_line)):
    if in_line[i] and start is None:
        start = i
    elif not in_line[i] and start is not None:
        if i - start >= 15:
            lines.append((start, i))
        start = None
if start is not None and len(in_line) - start >= 15:
    lines.append((start, len(in_line)))

line_images = []
for y0, y1 in lines:
    y0p = max(0, y0 - 5)
    y1p = min(h, y1 + 5)
    crop = img[y0p:y1p, 0:w]
    _, buf = cv2.imencode('.png', crop)
    line_images.append(buf.tobytes())

elapsed_seg = time.time() - t0_seg
print(f"Segmentation: {len(line_images)} lines in {elapsed_seg:.3f}s")

# Send batch to /ocr/batch with indexed filenames
t0_batch = time.time()
files = []
for i, img_bytes in enumerate(line_images):
    filename = f"image_{i:06d}.png"
    files.append(("files", (filename, img_bytes, "image/png")))

resp = requests.post(
    f"{CALAMARI_URL}/ocr/batch",
    files=files,
    data={"model": "gt4histocr", "mode": "indexed"},
    timeout=300,
)
elapsed_batch = time.time() - t0_batch
payload = resp.json()

# Reorder by index
ordered = [""] * len(line_images)
for item in payload.get("results", []):
    idx = item.get("index")
    if idx is None:
        fn = item.get("filename", "")
        m = re.search(r"image_(\d+)", fn)
        if m:
            idx = int(m.group(1))
    if isinstance(idx, int) and 0 <= idx < len(line_images):
        ordered[idx] = (item.get("text") or "").strip()

text_client = "\n".join(t for t in ordered if t)
elapsed_total = elapsed_seg + elapsed_batch

print(f"Batch API: {elapsed_batch:.2f}s (binding={payload.get('binding_method')}, server_elapsed={payload.get('elapsed_seconds', 0):.2f}s)")
print(f"Total (seg+batch): {elapsed_total:.2f}s")
print(f"[{elapsed_total:.2f}s] {len(text_client)} chars")
print(f"Text preview:\n{text_client[:500]}\n")


# === Summary ===
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Server-side (/ocr):     {elapsed_server:.2f}s, {len(text_server)} chars")
print(f"Client-side (/ocr/batch): {elapsed_total:.2f}s, {len(text_client)} chars")
speedup = elapsed_server / max(elapsed_total, 0.01)
print(f"Speedup: {speedup:.1f}x")
