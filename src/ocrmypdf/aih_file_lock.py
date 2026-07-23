# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


LOCK_TTL_SECONDS = 24 * 60 * 60


def _lock_dir() -> Path:
    path = Path(tempfile.gettempdir()) / "ocrmypdf-aih-locks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalise_key(parts: tuple[str, ...]) -> str:
    normalised = []
    for part in parts:
        try:
            normalised.append(str(Path(part).expanduser().resolve()).casefold())
        except OSError:
            normalised.append(str(Path(part).expanduser().absolute()).casefold())
    return "\n".join(normalised)


def lock_path_for(*parts: str) -> Path:
    digest = hashlib.sha256(_normalise_key(tuple(parts)).encode("utf-8")).hexdigest()[:24]
    return _lock_dir() / f"{digest}.lock"


def _is_stale(lock_path: Path) -> bool:
    try:
        return time.time() - lock_path.stat().st_mtime > LOCK_TTL_SECONDS
    except OSError:
        return True


@contextmanager
def file_job_lock(*parts: str, label: str = "file") -> Iterator[None]:
    lock_path = lock_path_for(*parts)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as exc:
            if _is_stale(lock_path):
                try:
                    lock_path.unlink()
                    continue
                except OSError:
                    pass
            raise RuntimeError(f"{label} 正在被另一个任务处理，请等待当前任务结束后再试。") from exc

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pid": os.getpid(),
                    "label": label,
                    "paths": list(parts),
                    "created": time.time(),
                },
                f,
                ensure_ascii=False,
            )
        yield
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass
