# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from ocrmypdf.aih_chrome_screenai import ChromeScreenAIBackend


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(json.dumps({"ok": False, "error": "missing job token"}))
        return 2

    try:
        job = json.loads(base64.b64decode(argv[0]).decode("utf-8"))
        backend = ChromeScreenAIBackend(
            light_mode=bool(job.get("light_mode", False)),
            force_horizontal=bool(job.get("force_horizontal", True)),
        )
        line_boxes = backend.detect_lines(
            Path(job["image_path"]),
            page_number=int(job.get("page_number", 0) or 0),
            filter_printed_text=bool(job.get("filter_printed_text", False)),
            min_confidence=float(job.get("min_confidence", 0.0) or 0.0),
            force_horizontal=bool(job.get("force_horizontal", True)),
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "lines": [
                        {
                            "bbox": list(map(int, line_box.bbox)),
                            "text": line_box.text,
                            "confidence": line_box.confidence,
                            "content_type": int(line_box.content_type),
                            "direction": int(line_box.direction),
                            "block_id": line_box.block_id,
                            "paragraph_id": line_box.paragraph_id,
                            "language": line_box.language,
                            "words": [
                                {
                                    "text": word_text,
                                    "bbox": list(map(int, word_bbox)) if word_bbox else None,
                                }
                                for word_text, word_bbox in line_box.words
                            ],
                        }
                        for line_box in line_boxes
                    ],
                    "bboxes": [list(map(int, line_box.bbox)) for line_box in line_boxes],
                },
            )
        )
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
