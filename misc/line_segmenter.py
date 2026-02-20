"""Line segmentation module for Calamari API - uses projection profile to split page into lines."""
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os


def segment_lines(image_path: str, min_line_height: int = 15, padding: int = 5) -> list[str]:
    """Split a full-page image into individual text line images.

    Uses horizontal projection profile to detect text lines.

    Args:
        image_path: Path to the full page image.
        min_line_height: Minimum pixel height to consider a line.
        padding: Extra pixels above/below each line crop.

    Returns:
        List of paths to cropped line images (temp files).
    """
    img = cv2.imread(image_path)
    if img is None:
        return [image_path]  # fallback: return original

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Binarize (Otsu)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection profile
    h_proj = np.sum(binary, axis=1)

    # Find line boundaries: regions where projection > threshold
    threshold = w * 0.5  # at least 0.5% of width has ink
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
        return [image_path]  # no lines detected, return original

    # Crop and save each line
    line_paths = []
    tmpdir = tempfile.mkdtemp(prefix="calamari_lines_")
    for idx, (y0, y1) in enumerate(lines):
        y0_pad = max(0, y0 - padding)
        y1_pad = min(h, y1 + padding)
        line_img = img[y0_pad:y1_pad, 0:w]
        line_path = os.path.join(tmpdir, f"line_{idx:04d}.png")
        cv2.imwrite(line_path, line_img)
        line_paths.append(line_path)

    return line_paths
