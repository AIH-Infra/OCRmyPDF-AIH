#!/usr/bin/env python3
"""Patch: inject line segmentation into Calamari container's app.py."""
import subprocess
import sys

# The line segmentation + updated do_ocr_single code to inject
PATCH_CODE = r'''
import cv2
import numpy as np

def segment_lines(image_path, min_line_height=15, padding=5):
    """Split full-page image into text line images using projection profile."""
    img = cv2.imread(image_path)
    if img is None:
        return [image_path]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_proj = np.sum(binary, axis=1)
    threshold = w * 0.5
    in_line = h_proj > threshold
    lines = []
    start = None
    for i in range(len(in_line)):
        if in_line[i] and start is None:
            start = i
        elif not in_line[i] and start is not None:
            if i - start >= min_line_height:
                lines.append((start, i))
            start = None
    if start is not None and len(in_line) - start >= min_line_height:
        lines.append((start, len(in_line)))
    if not lines:
        return [image_path]
    import tempfile as _tf
    tmpdir = _tf.mkdtemp(prefix="cal_lines_")
    line_paths = []
    for idx, (y0, y1) in enumerate(lines):
        y0p = max(0, y0 - padding)
        y1p = min(h, y1 + padding)
        lp = os.path.join(tmpdir, f"line_{idx:04d}.png")
        cv2.imwrite(lp, img[y0p:y1p, 0:w])
        line_paths.append(lp)
    return line_paths


def do_ocr_single_with_segmentation(temp_img_path, model):
    """OCR with automatic line segmentation for full-page images."""
    line_paths = segment_lines(temp_img_path)
    print(f"[Calamari] Segmented into {len(line_paths)} lines")

    predictor = get_predictor(model)
    if predictor is None:
        raise ValueError(f"No checkpoints for '{model}'")

    from calamari_ocr.ocr.dataset.datareader.file import FileDataParams

    all_texts = []
    for lp in line_paths:
        data_params = FileDataParams(images=[lp])
        for sample in predictor.predict(data_params):
            if sample.outputs and len(sample.outputs) >= 2:
                voted = sample.outputs[1]
                if hasattr(voted, 'sentence') and voted.sentence.strip():
                    all_texts.append(voted.sentence)

    # Cleanup line images
    if len(line_paths) > 1:
        import shutil
        parent = os.path.dirname(line_paths[0])
        shutil.rmtree(parent, ignore_errors=True)

    return "\n".join(all_texts)
'''

# Read current app.py from container
print("Reading current app.py from container...")
result = subprocess.run(
    ['docker', 'exec', 'calamari-ocr', 'cat', '/app/app.py'],
    capture_output=True, text=True
)
app_code = result.stdout

# Check if already patched
if 'segment_lines' in app_code:
    print("Already patched! Skipping.")
    sys.exit(0)

# Insert segmentation code after imports (after "import threading" line)
insert_after = "import threading"
if insert_after not in app_code:
    print(f"ERROR: Cannot find '{insert_after}' in app.py")
    sys.exit(1)

patched = app_code.replace(
    insert_after,
    insert_after + "\n" + PATCH_CODE,
    1
)

# Replace do_ocr_single call in /ocr endpoint to use segmentation version
patched = patched.replace(
    "text = await loop.run_in_executor(_executor, do_ocr_single, temp_img_path, model)",
    "text = await loop.run_in_executor(_executor, do_ocr_single_with_segmentation, temp_img_path, model)"
)

# Write patched app.py back
print("Writing patched app.py to container...")
proc = subprocess.run(
    ['docker', 'exec', '-i', 'calamari-ocr', 'bash', '-c', 'cat > /app/app.py'],
    input=patched, text=True, capture_output=True
)
if proc.returncode != 0:
    print(f"ERROR writing: {proc.stderr}")
    sys.exit(1)

print("Patched successfully! Restarting container...")
subprocess.run(['docker', 'restart', 'calamari-ocr'], check=True)
print("Done. Wait ~30s for model warmup.")
