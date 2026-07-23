# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from ocrmypdf.aih_chrome_screenai import ChromeScreenAIBackend


class _FileProgressState(dict):
    def __init__(self, progress_path: Path):
        super().__init__(page_current=0, page_total=0, page_desc="")
        self._progress_path = progress_path
        self._flush()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key in {"page_current", "page_total", "page_desc"}:
            self._flush()

    def _flush(self):
        self._progress_path.write_text(
            json.dumps(
                {
                    "page_current": int(self.get("page_current", 0) or 0),
                    "page_total": int(self.get("page_total", 0) or 0),
                    "page_desc": self.get("page_desc", "") or "",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(json.dumps({"ok": False, "error": "missing job token"}))
        return 2

    try:
        job = json.loads(base64.b64decode(argv[0]).decode("utf-8"))
        import builtins

        builtins._ocrmypdf_page_progress_state = _FileProgressState(
            Path(job["progress_json"])
        )
        backend = ChromeScreenAIBackend(light_mode=bool(job.get("light_mode", False)))
        result = backend.ocr_to_searchable_pdf(
            Path(job["input_pdf"]),
            Path(job["output_pdf"]),
            workers=1,
        )
        result_json = {"ok": True}
        if bool(job.get("collect_result", True)):
            result_json["result"] = result.to_dict() if hasattr(result, "to_dict") else {}
        Path(job["result_json"]).write_text(
            json.dumps(result_json, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps({"ok": True}))
        return 0
    except Exception as exc:
        try:
            job = json.loads(base64.b64decode(argv[0]).decode("utf-8"))
            Path(job["result_json"]).write_text(
                json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
