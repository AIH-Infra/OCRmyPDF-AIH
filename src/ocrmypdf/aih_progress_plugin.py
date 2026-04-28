# SPDX-FileCopyrightText: 2025 James R. Barlow
# SPDX-License-Identifier: AGPL-3.0-or-later

"""OCRmyPDF 进度插件 — 将页面级进度写入共享 dict.

通过 builtins 模块传递状态，确保跨 importlib 加载的模块实例共享。
"""

from __future__ import annotations

import builtins

from ocrmypdf import hookimpl

_STATE_KEY = '_ocrmypdf_page_progress_state'


class _StateProgressBar:
    def __init__(self, *, total=None, desc=None, unit=None, disable=False, **kw):
        self.total = total or 0
        self.desc = desc or ""
        self.current = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, n=1, *, completed=None):
        if completed is not None:
            self.current = completed
        else:
            self.current += n
        state = getattr(builtins, _STATE_KEY, None)
        if state is not None and self.total > 0:
            state["page_current"] = int(self.current)
            state["page_total"] = self.total
            state["page_desc"] = self.desc


@hookimpl
def get_progressbar_class():
    return _StateProgressBar
