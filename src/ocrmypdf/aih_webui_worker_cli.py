# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from ocrmypdf.aih_webui_workers import ocr_one_file_path_worker


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(json.dumps({
            "error": ("", "Usage: python -m ocrmypdf.aih_webui_worker_cli <job-json-b64>")
        }, ensure_ascii=False))
        return 2

    try:
        job = json.loads(base64.b64decode(argv[0]).decode("utf-8"))
        payload = ocr_one_file_path_worker(
            input_path=job["input_path"],
            output_path=job["output_path"],
            fname=job["fname"],
            out_name=job["out_name"],
            ocr_kwargs=job["ocr_kwargs"],
            backend=job["backend"],
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({
            "error": ("", str(exc)),
            "log": ("error", "", f"Chrome worker failed: {exc}"),
        }, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
