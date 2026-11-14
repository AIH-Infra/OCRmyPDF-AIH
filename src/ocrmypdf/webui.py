# SPDX-FileCopyrightText: 2025 James R. Barlow
# SPDX-License-Identifier: AGPL-3.0-or-later

"""OCRmyPDF Pro - 增强版批量处理界面."""

from __future__ import annotations

import io
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 禁用所有警告（包括子进程）
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import pikepdf
import streamlit as st
import streamlit.components.v1 as components

from ocrmypdf import aih_webui_upload_server as _upload_server
from ocrmypdf.aih_file_lock import file_job_lock

# 禁用 Streamlit 日志（主进程和子进程）
for logger_name in ['streamlit', 'streamlit.runtime', 'streamlit.runtime.scriptrunner_utils',
                    'streamlit.runtime.state', 'streamlit.runtime.caching']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
logging.getLogger('streamlit.runtime').setLevel(logging.ERROR)

_CONFIG_PATH = Path(os.environ.get("AIH_OCRMYPDF_CONFIG") or Path.home() / ".aih-ocrmypdf-webui.json")
_LEGACY_CONFIG_PATH = Path.home() / ".ocrmypdf-aih-webui.json"  # pre-rename settings

def _load_config() -> dict:
    try:
        path = _CONFIG_PATH if _CONFIG_PATH.exists() else _LEGACY_CONFIG_PATH
        return json.loads(path.read_text("utf-8")) if path.exists() else {}
    except Exception:
        return {}

def _save_config(cfg: dict):
    try:
        _CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")
    except Exception:
        pass

# 确保当前包目录在 sys.path 中，以便加载同包内插件模块
_pkg_dir = str(Path(__file__).parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# 预初始化 OCRmyPDF 插件基础设施（防止 Streamlit 模块重载丢失注册状态）
from ocrmypdf.api import setup_plugin_infrastructure as _setup_plugins
_setup_plugins()

try:
    from ocrmypdf.aih_line_detection import discover_paddle_env
except Exception:
    discover_paddle_env = None

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
_favicon = Path(__file__).parent / "logo.png"
st.set_page_config(
    page_title="AIH-OCRmyPDF",
    page_icon=str(_favicon) if _favicon.exists() else "\U0001F4D6",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 橘白主题 CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* --- 主色调：橘色 #E8751A，辅助：深橘 #C45E10，浅橘背景 #FFF5EB --- */

/* 顶部 header 栏 */
header[data-testid="stHeader"] {
    background: linear-gradient(90deg, #E8751A 0%, #F5A623 100%);
}

/* 侧边栏 */
section[data-testid="stSidebar"] {
    background-color: #FFF5EB;
    border-right: 3px solid #E8751A;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #C45E10 !important;
}

/* 主按钮 */
button[kind="primary"], .stButton>button[kind="primary"] {
    background-color: #E8751A !important;
    border-color: #E8751A !important;
    color: white !important;
}
button[kind="primary"]:hover {
    background-color: #C45E10 !important;
    border-color: #C45E10 !important;
}

/* 进度条 */
.stProgress > div > div > div {
    background-color: #E8751A !important;
}

/* 链接色 */
a { color: #E8751A !important; }

/* 分割线 */
hr { border-color: #F5D6B8 !important; }

/* 数据表格表头 */
.stDataFrame thead th {
    background-color: #E8751A !important;
    color: white !important;
}

/* expander 标题 */
details summary span {
    color: #C45E10 !important;
}

/* 页脚致谢样式 */
.credits-footer {
    text-align: center;
    color: #999;
    font-size: 0.8em;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid #F5D6B8;
    margin-top: 3rem;
}
.credits-footer a { color: #E8751A !important; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 会话状态初始化（用普通 dict 做线程共享状态，避免后台线程访问 session_state 报错）
# ---------------------------------------------------------------------------
if "ocr" not in st.session_state:
    st.session_state.ocr = {
        "running": False,
        "logs": [],          # [{type, file, msg}]
        "progress": 0.0,
        "progress_text": "",
        "outputs": [],       # [(name, path)]
        "errors": [],        # [(name, err)]
        "stop": False,
        "done": False,
        "output_dir": "",
        "page_current": 0,
        "page_total": 0,
        "page_desc": "",
        "task_dir": "",
    }
# 取出引用，后台线程直接操作这个 dict
_ocr_state: dict = st.session_state.ocr

# ---------------------------------------------------------------------------
# 常量与工具函数
# ---------------------------------------------------------------------------

FALLBACK_LANGUAGES = [
    "eng", "chi_sim", "chi_tra", "deu", "fra", "jpn",
    "kor", "spa", "por", "rus", "ara", "hin", "ita",
    "nld", "pol", "tur", "vie", "osd",
]

COMMON_LANGUAGES = [
    "eng", "chi_sim", "chi_tra", "deu", "fra", "jpn", "kor", "spa",
    "rus", "por", "ita", "ara", "hin", "nld",
    "frk", "deu_frak",  # 古德语 Fraktur — 用户核心需求
]

LANG_NAMES = {
    # 常用语言
    "eng": "eng - 英语",
    "chi_sim": "chi_sim - 简体中文",
    "chi_tra": "chi_tra - 繁体中文",
    "deu": "deu - 德语",
    "fra": "fra - 法语",
    "jpn": "jpn - 日语",
    "kor": "kor - 韩语",
    "spa": "spa - 西班牙语",
    "por": "por - 葡萄牙语",
    "rus": "rus - 俄语",
    "ara": "ara - 阿拉伯语",
    "hin": "hin - 印地语",
    "ita": "ita - 意大利语",
    "nld": "nld - 荷兰语",
    "pol": "pol - 波兰语",
    "tur": "tur - 土耳其语",
    "vie": "vie - 越南语",
    # 历史/古语言
    "frk": "frk - 德语 Fraktur（仅传统引擎）",
    "deu_frak": "deu_frak - 古德语 Fraktur（仅传统引擎）",
    "frm": "frm - 中古法语",
    "enm": "enm - 中古英语",
    "ita_old": "ita_old - 古意大利语",
    "spa_old": "spa_old - 古西班牙语",
    "grc": "grc - 古希腊语",
    "lat": "lat - 拉丁语",
    "san": "san - 梵语",
    "heb": "heb - 希伯来语",
    "chi_sim_vert": "chi_sim_vert - 简体中文（竖排）",
    "chi_tra_vert": "chi_tra_vert - 繁体中文（竖排）",
    "jpn_vert": "jpn_vert - 日语（竖排）",
    # 欧洲语言
    "bul": "bul - 保加利亚语",
    "cat": "cat - 加泰罗尼亚语",
    "ces": "ces - 捷克语",
    "dan": "dan - 丹麦语",
    "ell": "ell - 希腊语",
    "est": "est - 爱沙尼亚语",
    "fin": "fin - 芬兰语",
    "glg": "glg - 加利西亚语",
    "hrv": "hrv - 克罗地亚语",
    "hun": "hun - 匈牙利语",
    "isl": "isl - 冰岛语",
    "lav": "lav - 拉脱维亚语",
    "lit": "lit - 立陶宛语",
    "mkd": "mkd - 马其顿语",
    "nor": "nor - 挪威语",
    "ron": "ron - 罗马尼亚语",
    "slk": "slk - 斯洛伐克语",
    "slv": "slv - 斯洛文尼亚语",
    "sqi": "sqi - 阿尔巴尼亚语",
    "srp": "srp - 塞尔维亚语",
    "srp_latn": "srp_latn - 塞尔维亚语（拉丁）",
    "swe": "swe - 瑞典语",
    "ukr": "ukr - 乌克兰语",
    # 亚洲语言
    "ben": "ben - 孟加拉语",
    "guj": "guj - 古吉拉特语",
    "kan": "kan - 卡纳达语",
    "mal": "mal - 马拉雅拉姆语",
    "mar": "mar - 马拉地语",
    "nep": "nep - 尼泊尔语",
    "pan": "pan - 旁遮普语",
    "tam": "tam - 泰米尔语",
    "tel": "tel - 泰卢固语",
    "tha": "tha - 泰语",
    "mya": "mya - 缅甸语",
    "khm": "khm - 高棉语",
    "lao": "lao - 老挝语",
    "bod": "bod - 藏语",
    "urd": "urd - 乌尔都语",
    # 其他
    "afr": "afr - 南非荷兰语",
    "amh": "amh - 阿姆哈拉语",
    "aze": "aze - 阿塞拜疆语",
    "bel": "bel - 白俄罗斯语",
    "eus": "eus - 巴斯克语",
    "fas": "fas - 波斯语",
    "fil": "fil - 菲律宾语",
    "gle": "gle - 爱尔兰语",
    "hat": "hat - 海地克里奥尔语",
    "ind": "ind - 印尼语",
    "jav": "jav - 爪哇语",
    "kat": "kat - 格鲁吉亚语",
    "kaz": "kaz - 哈萨克语",
    "kir": "kir - 吉尔吉斯语",
    "msa": "msa - 马来语",
    "mon": "mon - 蒙古语",
    "sin": "sin - 僧伽罗语",
    "swa": "swa - 斯瓦希里语",
    "tgl": "tgl - 他加禄语",
    "uzb": "uzb - 乌兹别克语",
    "cym": "cym - 威尔士语",
    "yid": "yid - 意第绪语",
    # 脚本检测
    "osd": "osd - 方向/脚本检测",
    # 脚本变体
    "script/Latin": "Latin - 拉丁文脚本",
    "script/Fraktur": "Fraktur - Fraktur 脚本",
}

# 仅支持传统引擎（OEM 0）的语言包，LSTM 模式下不可用
LEGACY_ONLY_LANGS = {"frk", "deu_frak", "equ"}

# Tesseract 官方提供的全部语言包（用于显示"未安装"列表）
ALL_OFFICIAL_LANGS = {
    "afr", "amh", "ara", "asm", "aze", "aze_cyrl", "bel", "ben", "bod",
    "bos", "bre", "bul", "cat", "ceb", "ces", "chi_sim", "chi_sim_vert",
    "chi_tra", "chi_tra_vert", "chr", "cos", "cym", "dan", "deu",
    "deu_frak", "deu_latf", "div", "dzo", "ell", "eng", "enm", "epo",
    "equ", "est", "eus", "fao", "fas", "fil", "fin", "fra", "frk", "frm",
    "fry", "gla", "gle", "glg", "grc", "guj", "hat", "heb", "hin", "hrv",
    "hun", "hye", "iku", "ind", "isl", "ita", "ita_old", "jav", "jpn",
    "jpn_vert", "kan", "kat", "kat_old", "kaz", "khm", "kir", "kmr",
    "kor", "lao", "lat", "lav", "lit", "ltz", "mal", "mar", "mkd", "mlt",
    "mon", "mri", "msa", "mya", "nep", "nld", "nor", "oci", "ori", "osd",
    "pan", "pol", "por", "pus", "que", "ron", "rus", "san", "sin", "slk",
    "slv", "snd", "spa", "spa_old", "sqi", "srp", "srp_latn", "sun",
    "swa", "swe", "syr", "tam", "tat", "tel", "tgk", "tgl", "tha", "tir",
    "ton", "tur", "uig", "ukr", "urd", "uzb", "uzb_cyrl", "vie", "yid",
    "yor",
}

OEM_OPTIONS = {
    0: "0 - 传统引擎（支持 Fraktur 等古语言）",
    1: "1 - LSTM 神经网络（现代语言，精度最高）",
    2: "2 - 传统 + LSTM 混合",
    3: "3 - 自动选择（按语言包内容决定）",
}

OEM_DESCRIPTIONS = {
    0: "适合古德语 Fraktur (frk/deu_frak) 等历史文献，使用传统模式匹配引擎",
    1: "适合现代印刷体，基于深度学习，精度最高（默认推荐）",
    2: "同时运行两种引擎取最优结果，速度较慢",
    3: "根据语言包中包含的数据自动选择引擎",
}

PSM_OPTIONS = {
    0: "0 - 仅方向和脚本检测",
    1: "1 - 自动 OSD + OCR",
    3: "3 - 全自动分割（默认）",
    4: "4 - 单列文本",
    6: "6 - 单个文本块",
    7: "7 - 单行文本",
    8: "8 - 单个词",
    11: "11 - 稀疏文本",
    12: "12 - 稀疏文本 + OSD",
    13: "13 - 原始行",
}


def _lang_display(code: str) -> str:
    return LANG_NAMES.get(code, code)


@st.cache_data(ttl=600)
def detect_tesseract_languages() -> tuple[list[str], list[str]]:
    """获取已安装语言，返回 (普通语言列表, script列表)."""
    try:
        from ocrmypdf._exec.tesseract import get_languages
        all_langs = get_languages()
        scripts = sorted(l for l in all_langs if l.startswith("script"))
        langs = sorted(l for l in all_langs if not l.startswith("script") and l != "osd")
        return langs, scripts
    except Exception:
        return FALLBACK_LANGUAGES, []


@st.cache_data(ttl=60)
def fetch_calamari_models(url: str) -> list[str] | None:
    """从 Calamari API 获取模型列表. 返回 None 表示连接失败."""
    try:
        import requests
        resp = requests.get(f"{url}/models", timeout=5)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except Exception:
        return None


def get_pdf_page_count(data: bytes) -> int:
    try:
        with pikepdf.open(io.BytesIO(data)) as pdf:
            return len(pdf.pages)
    except Exception:
        return -1


def _log(state: dict, level: str, fname: str, msg: str):
    """向共享状态追加一条日志."""
    state["logs"].append({"type": level, "file": fname, "msg": msg})


# ---------------------------------------------------------------------------
# 磁盘暂存缓存：实现集中在 aih_webui_upload_server（Web UI 线程与旁路上传
# 线程共享同一把 manifest 锁、原子落盘），这里只留薄封装供界面与后台线程调用。
# ---------------------------------------------------------------------------
def _cleanup_task_dir(state: dict):
    """删除当前任务拥有的整个磁盘暂存批次（inputs/outputs/manifest）。"""
    _upload_server.cleanup_batch(state)


def _sweep_orphan_batches() -> None:
    _upload_server.sweep_orphans()


def _load_batch_manifest(batch_dir: Path) -> dict:
    return _upload_server.load_manifest(Path(batch_dir))


def _save_batch_manifest(batch_dir: Path, manifest: dict) -> None:
    _upload_server.save_manifest(Path(batch_dir), manifest)


def _get_batch_dir(state: dict, create: bool = False) -> Path | None:
    return _upload_server.get_batch_dir(state, create)


def _staged_files(state: dict) -> list[dict]:
    return _upload_server.list_staged(state)


def _remove_staged_files(state: dict, paths: set[str]) -> None:
    _upload_server.remove_staged(state, paths)


def _reset_batch_for_run(batch_dir: Path, staged: list[dict]) -> None:
    _upload_server.mark_running(Path(batch_dir), staged)


def _apply_ocr_worker_result(state: dict, payload: dict):
    if payload.get("output"):
        state["outputs"].append(payload["output"])
    if payload.get("output_path") and payload.get("out_name"):
        output_path = Path(payload["output_path"])
        if output_path.exists():
            state["outputs"].append((payload["out_name"], str(output_path)))
    if payload.get("error"):
        state["errors"].append(payload["error"])
    if payload.get("log"):
        level, fname, msg = payload["log"]
        _log(state, level, fname, msg)


def _run_chrome_file_parallel_subprocess(
    state: dict,
    files_data: list[tuple[str, str]],
    out_names: list[str],
    ocr_kwargs: dict,
    backend: str,
    save_to_disk: bool,
    output_dir: str,
    file_concurrency: int,
) -> int:
    """Run Chrome file-level parallelism with explicit subprocess workers."""
    total = len(files_data)
    completed = 0
    next_index = 0
    running: list[dict] = []
    worker_tmpdirs: list[tempfile.TemporaryDirectory] = []

    def _launch(index: int):
        fname, source_path = files_data[index]
        out_name = out_names[index]
        tmpdir = tempfile.TemporaryDirectory(prefix="ocrmypdf-aih-chrome-worker-")
        worker_tmpdirs.append(tmpdir)
        tmp_path = Path(tmpdir.name)
        stem = Path(fname).stem[:40]
        output_path = tmp_path / "output.pdf"
        job = {
            "input_path": source_path,
            "output_path": str(output_path),
            "fname": fname,
            "out_name": out_name,
            "ocr_kwargs": ocr_kwargs,
            "backend": backend,
        }
        token = base64.b64encode(json.dumps(job, ensure_ascii=False).encode("utf-8")).decode("ascii")
        env = os.environ.copy()
        src_dir = str(Path(__file__).parents[1])
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            src_dir if not existing_pythonpath else f"{src_dir}{os.pathsep}{existing_pythonpath}"
        )
        # Files, not pipes: a pipe nobody drains blocks the worker once it fills up,
        # and with stderr discarded a crashed worker left no error message at all.
        out_file = tmp_path / "worker.stdout"
        err_file = tmp_path / "worker.stderr"
        out_fh = open(out_file, "wb")
        err_fh = open(err_file, "wb")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "ocrmypdf.aih_webui_worker_cli", token],
                stdin=subprocess.DEVNULL,
                stdout=out_fh,
                stderr=err_fh,
                env=env,
                cwd=str(Path(__file__).parents[1]),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        finally:
            out_fh.close()
            err_fh.close()
        running.append({
            "index": index,
            "fname": fname,
            "out_name": out_name,
            "output_path": output_path,
            "proc": proc,
            "out_file": out_file,
            "err_file": err_file,
        })
        state["progress_text"] = f"Chrome ScreenAI 已启动 {index + 1}/{total}：{fname}"

    try:
        while next_index < total and len(running) < file_concurrency:
            _launch(next_index)
            next_index += 1

        while running:
            if state["stop"]:
                for item in running:
                    item["proc"].terminate()
                break

            for item in list(running):
                proc = item["proc"]
                if proc.poll() is None:
                    continue

                stdout = item["out_file"].read_bytes().decode("utf-8", "replace")
                stderr = item["err_file"].read_bytes().decode("utf-8", "replace")
                stderr = "\n".join(stderr.strip().splitlines()[-8:])
                payload = None
                for line in reversed(stdout.splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

                if payload is None:
                    fname = item["fname"]
                    err = stderr.strip() or stdout.strip() or f"worker exited with {proc.returncode}"
                    payload = {
                        "fname": fname,
                        "out_name": item["out_name"],
                        "output_path": None,
                        "error": (fname, err),
                        "log": ("error", fname, f"{fname}：Chrome worker failed: {err}"),
                    }

                if payload.get("output_path") and save_to_disk and output_dir:
                    src_output = Path(payload["output_path"])
                    if src_output.exists():
                        dst_output = Path(output_dir) / item["out_name"]
                        with file_job_lock(str(dst_output), label=item["out_name"]):
                            shutil.copy2(src_output, dst_output)
                        payload["output_path"] = str(dst_output)
                elif payload.get("output_path"):
                    src_output = Path(payload["output_path"])
                    retained_output = Path(state["task_dir"]) / "outputs" / item["out_name"]
                    shutil.copy2(src_output, retained_output)
                    payload["output_path"] = str(retained_output)

                _apply_ocr_worker_result(state, payload)
                running.remove(item)
                completed += 1
                state["progress"] = completed / total if total > 0 else 0
                state["progress_text"] = f"Chrome ScreenAI 已完成 {completed}/{total}"

                while next_index < total and len(running) < file_concurrency:
                    _launch(next_index)
                    next_index += 1
                break
            else:
                time.sleep(0.2)
    finally:
        for tmpdir in worker_tmpdirs:
            try:
                tmpdir.cleanup()
            except Exception:
                pass

    return completed


def _ocr_one_file(
    state: dict, input_path: str, fname: str, out_name: str,
    ocr_kwargs: dict, backend: str,
    save_to_disk: bool, output_dir: str,
):
    """处理单个文件（可在子线程中调用）."""
    import builtins
    import ocrmypdf
    import time

    start_time = time.time()

    # 通过 builtins 传递状态，确保插件（被 importlib 加载的独立模块实例）能读取
    builtins._ocrmypdf_page_progress_state = state
    state["page_current"] = 0
    state["page_total"] = 0
    state["page_desc"] = ""

    tmp_out_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_out:
            tmp_out_path = tmp_out.name

        plugins = [str(Path(__file__).parent / "aih_progress_plugin.py")]
        if backend in ("Calamari（HTTP服务）", "PaddleOCR（本地全链路）", "Chrome ScreenAI（本地）"):
            plugin_path = Path(__file__).parent / "aih_calamari_plugin.py"
            print(f"[DEBUG] Plugin path: {plugin_path}")
            print(f"[DEBUG] Plugin exists: {plugin_path.exists()}")
            if plugin_path.exists():
                plugins.append(str(plugin_path))
                print(f"[DEBUG] Plugins to load: {plugins}")
            else:
                raise FileNotFoundError(f"AIH OCR 插件未找到：{plugin_path}")

        # 覆盖 progress_bar 为 True，让进度插件生效
        kw = {**ocr_kwargs, "progress_bar": True}
        result = ocrmypdf.ocr(
            input_path, tmp_out_path,
            plugins=plugins,
            **kw,
        )

        elapsed = time.time() - start_time
        elapsed_str = f"{elapsed:.1f}秒"

        if result == ocrmypdf.ExitCode.ok:
            saved_path = None
            if save_to_disk and output_dir:
                saved_path = Path(output_dir) / out_name
                with file_job_lock(str(saved_path), label=out_name):
                    shutil.copy2(tmp_out_path, saved_path)
            else:
                saved_path = Path(state["task_dir"]) / "outputs" / out_name
                shutil.move(tmp_out_path, saved_path)
                tmp_out_path = ""
            state["outputs"].append((out_name, str(saved_path)))
            if saved_path:
                _log(state, "success", fname, f"{fname} -> {saved_path} (耗时: {elapsed_str})")
            else:
                _log(state, "success", fname, f"{fname} -> {out_name} (耗时: {elapsed_str})")
        elif result == ocrmypdf.ExitCode.pdfa_conversion_failed:
            saved_path = None
            if Path(tmp_out_path).stat().st_size > 0:
                if save_to_disk and output_dir:
                    saved_path = Path(output_dir) / out_name
                    with file_job_lock(str(saved_path), label=out_name):
                        shutil.copy2(tmp_out_path, saved_path)
                else:
                    saved_path = Path(state["task_dir"]) / "outputs" / out_name
                    shutil.move(tmp_out_path, saved_path)
                    tmp_out_path = ""
                state["outputs"].append((out_name, str(saved_path)))
            selected_output_type = kw.get("output_type")
            if selected_output_type == "auto":
                suffix = f"，输出：{saved_path}" if saved_path else ""
                _log(state, "warning", fname, f"{fname}：未能生成 PDF/A，已自动回退为普通 PDF{suffix} (耗时: {elapsed_str})")
            else:
                suffix = f"，输出：{saved_path}" if saved_path else ""
                _log(state, "warning", fname, f"{fname}：PDF/A 转换失败，输出可能不符合标准{suffix} (耗时: {elapsed_str})")
        elif result == ocrmypdf.ExitCode.already_done_ocr:
            hint = (
                "请勾选“OCR 前先移除旧的隐形文字层”，或改用“跳过 / 强制”模式"
                if kw.get("hidden_text_layer") == "keep"
                else "页面含可见文字（或隐形文字背后没有图片），请改用“跳过 / 强制”模式"
            )
            state["errors"].append((fname, "页面已有文字"))
            _log(state, "warning", fname, f"{fname}：已包含文字层，未处理；{hint} (耗时: {elapsed_str})")
        elif result == ocrmypdf.ExitCode.encrypted_pdf:
            state["errors"].append((fname, "PDF 已加密"))
            _log(state, "error", fname, f"{fname}：PDF 已加密，请先解密 (耗时: {elapsed_str})")
        else:
            state["errors"].append((fname, f"退出码：{result.name}"))
            _log(state, "error", fname, f"{fname}：处理失败（{result.name}）(耗时: {elapsed_str})")
    except Exception as e:
        elapsed = time.time() - start_time
        elapsed_str = f"{elapsed:.1f}秒"
        state["errors"].append((fname, str(e)))
        _log(state, "error", fname, f"{fname}：{e} (耗时: {elapsed_str})")
    finally:
        for p in (tmp_out_path,):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


def _run_ocr_batch(
    state: dict,
    files_data: list[tuple[str, str]],
    ocr_kwargs: dict, backend: str,
    output_suffix: str, save_to_disk: bool, output_dir: str,
    file_concurrency: int,
):
    """后台线程入口：批量处理所有文件."""
    total = len(files_data)
    seen_names: dict[str, int] = {}
    _name_lock = threading.Lock()

    def _make_out_name(fname):
        stem = Path(fname).stem
        out = f"{stem}{output_suffix}.pdf"
        with _name_lock:
            if out in seen_names:
                seen_names[out] += 1
                out = f"{stem}{output_suffix}_{seen_names[out]}.pdf"
            else:
                seen_names[out] = 0
        return out

    def _process_one(item):
        idx, (fname, input_path) = item
        if state["stop"]:
            return
        out_name = _make_out_name(fname)
        state["progress_text"] = f"正在处理 {idx + 1}/{total}：{fname}"
        _ocr_one_file(state, input_path, fname, out_name, ocr_kwargs, backend, save_to_disk, output_dir)
        state["progress"] = (idx + 1) / total if total > 0 else 0

    try:
        chrome_file_parallel = (
            backend == "Chrome ScreenAI（本地）"
            and file_concurrency > 1
            and int(ocr_kwargs.get("jobs", 1) or 1) <= 1
        )
        if chrome_file_parallel:
            state["progress_text"] = (
                f"Chrome ScreenAI 多本书外部进程并行：{file_concurrency} worker"
            )
            out_names = [_make_out_name(fname) for fname, _data in files_data]
            _run_chrome_file_parallel_subprocess(
                state,
                files_data,
                out_names,
                ocr_kwargs,
                backend,
                save_to_disk,
                output_dir,
                file_concurrency,
            )
        elif file_concurrency > 1:
            with ThreadPoolExecutor(max_workers=file_concurrency) as pool:
                futs = [pool.submit(_process_one, (i, fd)) for i, fd in enumerate(files_data)]
                for f in as_completed(futs):
                    if state["stop"]:
                        break
                    f.result()
        else:
            for i, fd in enumerate(files_data):
                if state["stop"]:
                    break
                _process_one((i, fd))
    except Exception as e:
        _log(state, "error", "", f"批量处理异常：{e}")
    finally:
        task_dir = state.get("task_dir")
        if task_dir:
            # 任务彻底完成：清理磁盘上的输入缓存，输出保留到用户清除。
            shutil.rmtree(Path(task_dir) / "inputs", ignore_errors=True)
            try:
                _upload_server.mark_done(Path(task_dir))
            except Exception:
                pass
        state["running"] = False
        state["done"] = True
        state["progress"] = 1.0
        state["progress_text"] = "全部完成！" if not state["stop"] else "已停止"


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<h2 style="color:#E8751A">\U0001F4D6 AIH-OCRmyPDF</h2>',
    unsafe_allow_html=True,
)

# -- 后端选择 --
st.sidebar.header("OCR 引擎")
backend = st.sidebar.radio(
    "选择后端",
    ["Tesseract", "PaddleOCR（本地全链路）", "Calamari（HTTP服务）", "Chrome ScreenAI（本地）", "无（仅 PDF/A 转换）"],
    index=3,  # 默认选择 Chrome ScreenAI（本地）
)

calamari_url = ""
calamari_model = ""
calamari_batch_mode = "indexed"
calamari_timeout = 300
calamari_retry = 2
perf_strategy = "standard"
preflight_chunk = 10
preflight_dpi = 300
preflight_workers = 1
calamari_line_detector = "projection_debug"
binarize_lines = True
chrome_screenai_light = False
chrome_screenai_filter_types = "printed"
chrome_screenai_min_confidence = 0.0
chrome_screenai_text_layer_mode = "native_direct"
chrome_screenai_force_horizontal = True
chrome_screenai_text_cleanup = "mode_default"
chrome_screenai_rasterize_dpi = 300
chrome_screenai_rasterize_quality = 88
paddle_det_model = "PP-OCRv5_server_det"
paddle_det_device = "gpu:0"
paddle_det_min_confidence = 0.0
_discovered_paddle_env = discover_paddle_env() if discover_paddle_env else None
paddle_det_python = str(_discovered_paddle_env.python) if _discovered_paddle_env else ""
paddle_det_source_dir = str(_discovered_paddle_env.source_dir) if _discovered_paddle_env else ""
paddle_ocr_version = "PP-OCRv6"
paddle_ocr_lang = "en"
paddle_ocr_device = "gpu:0"
paddle_ocr_rec_batch_size = 16
paddle_ocr_min_confidence = 0.0
paddle_ocr_python = paddle_det_python
paddle_ocr_source_dir = paddle_det_source_dir
paddle_ocr_enable_mkldnn = False
paddle_ocr_cpu_threads = 0
if backend == "Calamari（HTTP服务）":
    calamari_url = st.sidebar.text_input(
        "Calamari API 地址", value="http://localhost:11800"
    )
    models = fetch_calamari_models(calamari_url)
    if models is None:
        st.sidebar.warning("无法连接 Calamari 服务")
    elif models:
        st.sidebar.success(f"已连接 — 共 {len(models)} 个模型")
    else:
        st.sidebar.info("已连接，模型按需加载")
    _model_options = (models or []) + ["（自定义）"]
    # 默认选择 gt4histocr 或 fraktur_19th_century
    default_idx = 0
    for preferred in ["gt4histocr", "fraktur_19th_century"]:
        if preferred in _model_options:
            default_idx = _model_options.index(preferred)
            break
    _sel = st.sidebar.selectbox(
        "选择模型", options=_model_options, index=default_idx, key="calamari_model_sel",
    )
    if _sel == "（自定义）":
        calamari_model = st.sidebar.text_input(
            "自定义模型名称", value="gt4histocr", key="calamari_model_custom",
        )
    else:
        calamari_model = _sel

    # -- Calamari 性能策略 --
    PERF_STRATEGIES = {
        "fast": "快速 — 客户端分割 + 批量发送",
        "preflight": "预飞 — 跨页合并批量推理（最大 GPU 利用率）",
    }
    perf_strategy = st.sidebar.selectbox(
        "性能策略",
        list(PERF_STRATEGIES.keys()),
        format_func=lambda x: PERF_STRATEGIES[x],
        index=1,  # 默认选择 preflight，优先整体批量吞吐
        help="快速：按页进入识别流程；预飞：先分批预渲染并缓存，整体吞吐通常更高",
    )
    if perf_strategy == "preflight":
        st.sidebar.warning("预飞会先分批渲染/切行并写缓存，大文件会更久才看到 OCR 结果。")

    # -- Calamari 高级选项 --
    with st.sidebar.expander("Calamari 高级选项"):
        LINE_DETECTOR_LABELS = {
            "paddle_det": "PaddleOCR — 稳定文本检测",
            "chrome_screenai": "Chrome ScreenAI — 本地版面/行检测",
            "tesseract": "Tesseract — 传统行分割",
            "projection_debug": "投影法 — 兼容旧流程",
        }
        calamari_line_detector = st.selectbox(
            "行检测器",
            ["paddle_det", "chrome_screenai", "tesseract", "projection_debug"],
            format_func=lambda x: LINE_DETECTOR_LABELS[x],
            index=0,
            help="选择 Calamari 识别前的切行来源",
        )
        if calamari_line_detector == "paddle_det":
            if _discovered_paddle_env:
                st.caption(f"Paddle 环境：{_discovered_paddle_env.source} · {_discovered_paddle_env.python}")
            else:
                st.caption("Paddle 环境：未自动发现，可手动填写 Python 与源码目录")
            paddle_det_model = st.text_input(
                "Paddle 检测模型",
                value="PP-OCRv5_server_det",
                help="默认使用 PaddleOCR 3.x 文本检测模型",
            )
            paddle_det_device = st.selectbox(
                "Paddle 推理设备",
                ["gpu:0", "cpu"],
                index=0,
            )
            paddle_det_min_confidence = st.number_input(
                "Paddle 最低置信度",
                value=0.0,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
            )
            paddle_det_python = st.text_input(
                "Paddle Python",
                value=paddle_det_python,
            )
            paddle_det_source_dir = st.text_input(
                "Paddle 源码目录",
                value=paddle_det_source_dir,
            )
        if calamari_line_detector == "chrome_screenai":
            chrome_screenai_light = st.checkbox(
                "Chrome ScreenAI 轻量模式",
                value=False,
                help="启用轻量模型路径；可用性取决于本地 locro/ScreenAI 环境",
            )
            chrome_screenai_filter_types = st.selectbox(
                "Chrome 内容过滤",
                ["printed", "all"],
                format_func=lambda x: {
                    "printed": "仅印刷文本",
                    "all": "全部文本类型",
                }[x],
            )
            chrome_screenai_min_confidence = st.number_input(
                "Chrome 最低置信度",
                value=0.0,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
            )
            chrome_screenai_force_horizontal = st.checkbox(
                "强制横排坐标匹配",
                value=True,
                help="忽略 Chrome 返回的竖排方向，按从上到下、同一行从左到右匹配坐标。",
            )
        binarize_lines = st.checkbox(
            "对行图像进行二值化",
            value=True,  # 默认启用
            help="使用 Otsu 自动阈值对切割的行图像进行二值化，改善泛黄背景文档的识别质量"
        )
        calamari_batch_mode = st.selectbox(
            "批量模式",
            ["indexed", "sequential"],
            format_func=lambda x: {
                "indexed": "indexed — 索引编码（快速）",
                "sequential": "sequential — 逐张串行（稳定）",
            }[x],
        )
        calamari_timeout = st.number_input(
            "请求超时（秒）", value=300, min_value=10, max_value=3600,
        )
        calamari_retry = st.number_input(
            "重试次数", value=2, min_value=0, max_value=10,
        )
        if perf_strategy == "preflight":
            import os
            cpu_count = os.cpu_count() or 4
            default_workers = 10  # 默认10个线程

            preflight_chunk = st.slider(
                "预飞每批页数", 5, 200, 100,
                help="每批合并多少页的行图片发送到识别服务（默认 100，建议根据内存调整）",
            )
            preflight_dpi = st.selectbox(
                "预飞渲染 DPI", [150, 200, 300, 400], index=2,
                help="越高越清晰但越慢，300 适合大多数场景",
            )
            preflight_workers = st.slider(
                "渲染并行线程数", 1, 10, default_workers,
                help=f"并行渲染页面的线程数（默认 {default_workers}）",
            )
        else:
            preflight_chunk = 10
            preflight_dpi = 300
            preflight_workers = 1

    # -- Calamari 健康诊断 --
    if calamari_url:
        with st.sidebar.expander("服务诊断", expanded=False):
            if st.button("检测连接", key="calamari_health"):
                try:
                    import requests as _req
                    resp = _req.get(f"{calamari_url}/health", timeout=5)
                    info = resp.json()
                    st.success(f"状态：{info.get('status', '?')}")
                    st.text(f"版本：{info.get('version', '?')}")
                    st.text(f"模式：{info.get('mode', '?')}")
                    cached = info.get("cached_models", [])
                    st.text(f"已缓存模型：{', '.join(cached) if cached else '无'}")
                except Exception as e:
                    st.error(f"连接失败：{e}")
elif backend == "PaddleOCR（本地全链路）":
    with st.sidebar.expander("PaddleOCR 选项", expanded=True):
        if _discovered_paddle_env:
            st.caption(f"Paddle 环境：{_discovered_paddle_env.source} · {_discovered_paddle_env.python}")
        else:
            st.caption("Paddle 环境：未自动发现，可手动填写 Python 与源码目录")
        paddle_ocr_version = st.selectbox(
            "PaddleOCR 基线",
            ["PP-OCRv6", "PP-OCRv5", "auto"],
            index=0,
            help="PP-OCRv6 是 PaddleOCR 3.7 的新基线；auto 由当前 PaddleOCR 环境决定。",
        )
        paddle_ocr_lang = st.text_input(
            "PaddleOCR 语言",
            value="en",
            help="例如 en、ch、japan；PP-OCRv6 覆盖多种拉丁语系。",
        )
        paddle_ocr_device = st.selectbox(
            "PaddleOCR 推理设备",
            ["gpu:0", "cpu", "auto"],
            index=0,
        )
        paddle_ocr_rec_batch_size = st.slider(
            "识别 batch size",
            1,
            64,
            16,
            help="单页内文本行识别批大小；GPU 可适当调高，CPU 建议保守。",
        )
        paddle_ocr_min_confidence = st.number_input(
            "最低识别置信度",
            value=0.0,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
        )
        paddle_ocr_python = st.text_input(
            "Paddle Python",
            value=paddle_ocr_python,
        )
        paddle_ocr_source_dir = st.text_input(
            "Paddle 源码目录",
            value=paddle_ocr_source_dir,
        )
        st.caption("高级性能选项")
        paddle_ocr_enable_mkldnn = st.checkbox(
            "CPU 启用 MKL-DNN",
            value=False,
            help="仅 CPU 模式下有意义；实际可用性取决于 Paddle 安装。",
        )
        paddle_ocr_cpu_threads = st.number_input(
            "CPU 线程数（0=默认）",
            value=0,
            min_value=0,
            max_value=max(1, os.cpu_count() or 1),
        )
elif backend == "Chrome ScreenAI（本地）":
    with st.sidebar.expander("Chrome ScreenAI 选项", expanded=True):
        chrome_screenai_text_layer_mode = st.selectbox(
            "文本层模式",
            ["native_direct", "proxy_match"],
            format_func=lambda x: {
                "native_direct": "完全直通 — Chrome 原生生成文本层",
                "proxy_match": "代理匹配 — AIH 按坐标生成文本层",
            }[x],
            index=0,
            help="完全直通最快、最接近 Chrome 原生；代理匹配可应用强制横排坐标逻辑。",
        )
        chrome_screenai_text_cleanup = st.selectbox(
            "OCR 前文字层处理",
            ["mode_default", "clear_all", "rasterize"],
            format_func=lambda x: {
                "mode_default": "按处理模式（默认，不自动栅格化）",
                "clear_all": "轻量清除全部 PDF 文字对象",
                "rasterize": "高性能栅格化清除（显式启用）",
            }[x],
            index=0,
            help="轻量清除直接重写 PDF 内容流；栅格化仅在这里明确选择时启用。",
        )
        if chrome_screenai_text_cleanup == "rasterize":
            chrome_screenai_rasterize_dpi = st.slider(
                "栅格化 DPI",
                150,
                600,
                300,
                step=25,
            )
            chrome_screenai_rasterize_quality = st.slider(
                "JPEG 质量",
                60,
                100,
                88,
            )
        chrome_screenai_light = st.checkbox(
            "轻量模式",
            value=False,
            help="启用轻量模型路径；可用性取决于本地 locro/ScreenAI 环境",
        )
        chrome_screenai_filter_types = st.selectbox(
            "内容过滤",
            ["printed", "all"],
            format_func=lambda x: {
                "printed": "仅印刷文本",
                "all": "全部文本类型",
            }[x],
        )
        chrome_screenai_min_confidence = st.number_input(
            "最低置信度",
            value=0.0,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
        )
        chrome_screenai_force_horizontal = st.checkbox(
            "强制横排坐标匹配",
            value=True,
            disabled=chrome_screenai_text_layer_mode != "proxy_match",
            help="忽略 Chrome 返回的竖排方向，按从上到下、同一行从左到右匹配坐标。",
        )
        if chrome_screenai_text_layer_mode == "native_direct":
            st.caption("当前为完全直通模式：文本层由 Chrome ScreenAI 原生生成，不经过 AIH 坐标重排。")

# -- Tesseract 引擎选择（在语言之前，因为引擎决定可用语言）--
tess_oem = 1  # 默认值，非 Tesseract 后端时使用
if backend == "Tesseract":
    st.sidebar.header("识别引擎")
    tess_oem = st.sidebar.selectbox(
        "OEM（引擎模式）",
        options=list(OEM_OPTIONS.keys()),
        format_func=lambda x: OEM_OPTIONS[x],
        index=1,
    )
    st.sidebar.caption(OEM_DESCRIPTIONS[tess_oem])

# -- 语言设置 --
st.sidebar.header("语言设置")
if backend == "Tesseract":
    installed_langs, installed_scripts = detect_tesseract_languages()

    # 根据 OEM 过滤：OEM 1 (LSTM) 不支持仅传统引擎的语言
    if tess_oem == 1:
        available_langs = [l for l in installed_langs if l not in LEGACY_ONLY_LANGS]
        hidden = [l for l in installed_langs if l in LEGACY_ONLY_LANGS]
    else:
        available_langs = installed_langs
        hidden = []

    # 未安装的官方语言包
    not_installed = sorted(ALL_OFFICIAL_LANGS - set(installed_langs) - {"osd"})
    if tess_oem == 1:
        not_installed = [l for l in not_installed if l not in LEGACY_ONLY_LANGS]

    st.sidebar.caption(f"已安装 {len(installed_langs)} 个语言包，当前 OEM {tess_oem} 可用 {len(available_langs)} 个")

    common = [l for l in COMMON_LANGUAGES if l in available_langs]
    other = [l for l in available_langs if l not in COMMON_LANGUAGES]

    selected_langs = []
    cols = st.sidebar.columns(2)
    for i, lang in enumerate(common):
        default_on = lang == "eng"
        if cols[i % 2].checkbox(_lang_display(lang), value=default_on, key=f"oem{tess_oem}_lang_{lang}"):
            selected_langs.append(lang)

    if other:
        with st.sidebar.expander(f"其他已安装语言（{len(other)} 个）"):
            cols2 = st.columns(2)
            for i, lang in enumerate(other):
                if cols2[i % 2].checkbox(_lang_display(lang), value=False, key=f"oem{tess_oem}_lang_{lang}"):
                    selected_langs.append(lang)

    if hidden:
        st.sidebar.info(f"OEM {tess_oem} 下隐藏了 {len(hidden)} 个仅传统引擎语言：{', '.join(hidden)}")

    if not_installed:
        with st.sidebar.expander(f"📦 未安装的官方语言包（{len(not_installed)} 个）"):
            st.caption("可通过 `apt install tesseract-ocr-<code>` 或下载 .traineddata 安装")
            for lang in not_installed:
                st.text(f"  {_lang_display(lang)}")

    custom_lang = st.sidebar.text_input(
        "手动输入语言组合", value="",
        help="覆盖上方选择，格式如 chi_sim+eng",
    )
elif backend == "Calamari（HTTP服务）":
    selected_langs = [calamari_model] if calamari_model else ["gt4histocr"]
    custom_lang = ""
elif backend == "PaddleOCR（本地全链路）":
    selected_langs = []
    custom_lang = ""
    st.sidebar.caption("PaddleOCR 全链路使用 PaddleOCR 自身语言配置，不依赖 Tesseract 语言包。")
elif backend == "Chrome ScreenAI（本地）":
    selected_langs = []
    custom_lang = ""
    st.sidebar.caption("Chrome ScreenAI 使用本地版面分析与识别模型，不依赖 Tesseract 语言包。")
else:
    selected_langs = []
    custom_lang = ""

# -- OCR 模式 --
st.sidebar.header("OCR 模式")
MODE_LABELS = {
    "default": "默认（遇到可见文字报错）",
    "skip": "跳过（跳过已有文字的页面）",
    "force": "强制（光栅化全部内容重新 OCR）",
    "redo": "重做（删除旧 OCR 层重新识别）",
}
mode = st.sidebar.selectbox(
    "处理模式",
    list(MODE_LABELS.keys()),
    format_func=lambda x: MODE_LABELS[x],
    index=3,  # 默认选择 redo
)
_strip_applies = mode in ("default", "redo")
strip_hidden_layer = st.sidebar.checkbox(
    "OCR 前先移除旧的隐形文字层",
    value=True,
    disabled=not _strip_applies,
    help="扫描图 + 上次 OCR 留下的隐形文字层：先去掉再重新识别，不会因“已有文字”而中止。"
    "只删不可见文字，且仅限背后有图片的页面；可见文字不动，原文件不改。"
    "跳过/强制模式下不适用（跳过保留原文字，强制本来就会栅格化）。",
)
if not _strip_applies:
    st.sidebar.caption("当前处理模式不使用此选项。")

# -- 预处理 --
with st.sidebar.expander("预处理选项"):
    rotate_pages = st.checkbox("自动旋转页面", value=False)
    deskew = st.checkbox("纠偏", value=False)
    clean = st.checkbox("OCR 前清理（unpaper）", value=False)
    clean_final = st.checkbox("最终输出也清理", value=False)
    remove_background = st.checkbox("去除背景", value=False)
    remove_vectors = st.checkbox("去除矢量图形", value=False)
    oversample = st.slider("过采样 DPI", 0, 5000, 0, step=50,
                           help="0 = 不过采样")

# -- Tesseract 专属选项 --
if backend == "Tesseract":
    with st.sidebar.expander("Tesseract 高级选项"):
        tess_psm = st.selectbox(
            "PSM（页面分割模式）",
            options=list(PSM_OPTIONS.keys()),
            format_func=lambda x: PSM_OPTIONS[x],
            index=2,
        )
        tess_threshold = st.selectbox(
            "二值化方法",
            ["auto", "otsu", "sauvola"],
            format_func=lambda x: {"auto": "自动", "otsu": "Otsu", "sauvola": "Sauvola"}[x],
            help="Sauvola 需要 Tesseract 5.0+",
        )
        tess_timeout = st.number_input("超时时间（秒）", value=180.0, min_value=10.0)
        tess_downsample = st.checkbox("自动缩小超大图片", value=True)

# -- 输出设置 --
OUTPUT_MODE_OPTIONS = {
    "普通 PDF": "pdf",
    "尽量 PDF/A（失败则回退普通 PDF）": "auto",
    "严格 PDF/A": "pdfa",
}

with st.sidebar.expander("输出设置"):
    output_mode = st.selectbox(
        "输出模式",
        list(OUTPUT_MODE_OPTIONS.keys()),
        index=0,
        help="普通 PDF：成功率最高；尽量 PDF/A：能归档则归档，否则自动回退普通 PDF；严格 PDF/A：明确要求 PDF/A 输出。",
    )
    output_type = OUTPUT_MODE_OPTIONS[output_mode]
    pdf_renderer = st.selectbox(
        "PDF 渲染器",
        ["auto", "hocr", "sandwich"],
        help="auto=自动选择, hocr=hOCR文字层, sandwich=Tesseract原生PDF",
    )
    optimize_level = st.selectbox(
        "优化级别",
        [1, 0, 2, 3],
        format_func=lambda x: {0: "0 - 关闭", 1: "1 - 无损（默认）",
                                2: "2 - 有损", 3: "3 - 激进"}[x],
    )

# -- 性能设置 --
with st.sidebar.expander("性能设置"):
    cpu_count = os.cpu_count() or 4
    if backend == "Chrome ScreenAI（本地）":
        current_mode = st.session_state.get("chrome_concurrency_mode", "page_parallel")
        if current_mode not in ("native_direct", "page_parallel", "file_parallel"):
            current_mode = "page_parallel"
        chrome_concurrency_mode = st.radio(
            "Chrome 并发模式",
            ["native_direct", "page_parallel", "file_parallel"],
            index=["native_direct", "page_parallel", "file_parallel"].index(current_mode),
            format_func=lambda x: {
                "native_direct": "原生直通（不拆分）",
                "page_parallel": "单本书页内并行",
                "file_parallel": "多本书整本并行",
            }[x],
            help="三种模式互斥：原生直通最稳且后处理最少；页段并行适合单本大书；文件并行适合多本书批处理。",
            key="chrome_concurrency_mode",
        )
        use_threads = True
        if chrome_concurrency_mode == "native_direct":
            jobs = 1
            file_concurrency = 1
            st.caption(
                "当前策略：Chrome/locro 原生单文档直通。不拆页段、不做最终 chunk 合并，"
                "优先使用最少中间层和最稳定路径。"
            )
        elif chrome_concurrency_mode == "page_parallel":
            st.caption(
                "将单本书拆成多个页段，由外部独立子进程并行处理，再按页序合并。"
                "适合单本大书，不再使用同进程线程直接调用 Chrome DLL。"
            )
            jobs = st.slider(
                "单书并行页段数",
                1,
                24,
                8,
                help="把一本书切成多个页段，交给多个独立子进程并行 OCR。默认 8，最高 24。",
                key="chrome_page_workers",
            )
            file_concurrency = 1
            st.caption(f"当前策略：单本书页段并行。文件并发=1，页段 worker={jobs}。")
        else:
            file_concurrency = st.slider(
                "并行书本数",
                1,
                24,
                min(3, max(1, cpu_count // 2)),
                help="同时处理多本书；Chrome 会使用进程隔离，每本书内部固定为单线程顺序处理页面。"
                     "上限 24；超过 CPU 核心数时属于超配，吞吐未必线性提升。",
                key="chrome_file_concurrency",
            )
            jobs = 1
            st.caption(f"当前策略：多本书整本并行。进程并发={file_concurrency}，每本书内部顺序处理。")
    else:
        if backend == "Calamari（HTTP服务）":
            default_jobs = 1
            jobs_help = "Calamari GPU 后端建议 jobs=1，避免多进程争抢 GPU 资源"
        elif backend == "PaddleOCR（本地全链路）":
            default_jobs = 1 if paddle_ocr_device.startswith("gpu") else max(1, cpu_count // 2)
            jobs_help = "PaddleOCR GPU 后端建议先用 jobs=1；CPU 可用多进程按页并行"
        else:
            default_jobs = max(1, cpu_count // 2)
            jobs_help = f"默认为 CPU 线程数的一半（{default_jobs}/{cpu_count}），留余量给系统"
        jobs = st.slider("页面并行数（每文件）", 1, cpu_count, default_jobs, help=jobs_help)
        use_threads = st.checkbox(
            "使用线程（而非进程）",
            value=False,
            help="线程模式适合轻量或共享内存场景；大多数 CPU OCR 后端默认仍建议进程模式。",
        )
        # 文件级并发（仅 Tesseract）
        if backend == "Tesseract":
            file_concurrency = st.slider(
                "文件并发数", 1, max(1, cpu_count // 2), 1,
                help="同时处理多个文件。注意：文件并发数 × 页面并行数 = 总进程数，避免超过 CPU 核心数",
            )
        else:
            file_concurrency = 1

# -- 输出存储 --
_user_cfg = _load_config()
_default_output_dir = _user_cfg.get("output_dir", str(Path.home() / "OCRmyPDF_output"))
_default_suffix = _user_cfg.get("output_suffix", "_ocr")
_default_save = bool(_user_cfg.get("save_to_disk", True))
with st.sidebar.expander("输出存储"):
    output_suffix = st.text_input(
        "输出文件名后缀", value=_default_suffix,
        help="添加在 .pdf 扩展名之前，可留空；修改后自动记住，下次启动沿用。"
        "不能含 \\ / : * ? \" < > | 等文件名非法字符。",
    )
    if re.search(r'[\\/:*?"<>|\x00-\x1f]', output_suffix):
        st.error("后缀含文件名非法字符，已按默认 _ocr 处理")
        output_suffix = "_ocr"
    save_to_disk = st.checkbox("自动保存到磁盘", value=_default_save)
    output_dir = ""
    if save_to_disk:
        output_dir = st.text_input(
            "输出目录", value=_default_output_dir,
            help="修改后自动记住，下次启动沿用。",
        )
    # 变更即持久化（无需再点按钮）
    _new_cfg = {
        **_user_cfg,
        "output_suffix": output_suffix,
        "save_to_disk": save_to_disk,
        **({"output_dir": output_dir} if save_to_disk and output_dir else {}),
    }
    if _new_cfg != _user_cfg:
        _save_config(_new_cfg)
        _user_cfg = _new_cfg

# ---------------------------------------------------------------------------
# 主区域 — Logo + 头部
# ---------------------------------------------------------------------------
_logo_path = Path(__file__).parent / "logo.png"
_logo_col, _title_col = st.columns([1, 5])
with _logo_col:
    if _logo_path.exists():
        st.image(str(_logo_path), width=90)
    else:
        st.markdown("#### \U0001F4D6")
with _title_col:
    st.markdown(
        '<h1 style="color:#E8751A;margin-bottom:0">AIH-OCRmyPDF Infra \u7279\u5316\u7248</h1>'
        '<p style="color:#999;margin-top:0">\u7279\u5316 by <b>Baireinhold</b> &mdash; '
        '\u6279\u91cf PDF \u6587\u5b57\u8bc6\u522b\uff0c\u652f\u6301 Tesseract / PaddleOCR / Calamari / Chrome ScreenAI \u540e\u7aef</p>',
        unsafe_allow_html=True,
    )
st.markdown("<hr style='margin:0.5rem 0'>", unsafe_allow_html=True)

# 内嵌上传页：直连旁路上传服务，逐本顺序上传，字节由服务端流式落盘。
# __PORT__/__TOKEN__ 由下方代码替换。注意 components.html 的 iframe 以
# srcdoc 渲染，自身 window.location 是 about:srcdoc（hostname 为空），
# 必须从父页面地址探测主机名，本机和局域网其他设备才能都直连。
_UPLOAD_PAGE_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  html,body{margin:0;font-family:"Segoe UI","Microsoft YaHei",sans-serif;}
  #drop{border:2px dashed #E8751A;border-radius:10px;background:#FFF5EB;color:#C45E10;
        text-align:center;padding:24px 12px;cursor:pointer;font-size:14px;user-select:none;}
  #drop.hover{background:#FCE6D4;}
  #bar-wrap{height:8px;background:#F0E0D0;border-radius:4px;margin:10px 0 6px;overflow:hidden;display:none;}
  #bar{height:100%;width:0%;background:#E8751A;border-radius:4px;transition:width .15s;}
  #status{font-size:12.5px;color:#666;line-height:1.5;max-height:110px;overflow:auto;}
  #status .err{color:#C0392B;}
  #status .ok{color:#2E7D32;}
</style></head>
<body>
<div id="drop">将 PDF 文件拖拽到此处，或点击选择文件（支持多选，可分批追加）</div>
<input id="picker" type="file" multiple accept=".pdf,application/pdf" style="display:none">
<div id="bar-wrap"><div id="bar"></div></div>
<div id="status"></div>
<script>
var PORT = "__PORT__";
var TOKEN = "__TOKEN__";
var queue = [], totalBytes = 0, sentDone = 0, curLoaded = 0;
var doneCount = 0, failCount = 0, busy = false;
var drop = document.getElementById("drop"),
    picker = document.getElementById("picker"),
    barWrap = document.getElementById("bar-wrap"),
    bar = document.getElementById("bar"),
    statusEl = document.getElementById("status");

function detectHost(){
  try { var h = window.parent.location.hostname; if (h) return h; } catch (e) {}
  try {
    if (document.referrer) { var r = new URL(document.referrer).hostname; if (r) return r; }
  } catch (e) {}
  try { var s = window.location.hostname; if (s) return s; } catch (e) {}
  return "127.0.0.1";
}
var API = "http://" + detectHost() + ":" + PORT;

function line(msg, cls){
  var d = document.createElement("div");
  if (cls) d.className = cls;
  d.textContent = msg;
  statusEl.appendChild(d);
  statusEl.scrollTop = statusEl.scrollHeight;
}
function fmt(n){ return (n / 1048576).toFixed(1) + " MB"; }
function progress(){
  if (!totalBytes) return;
  barWrap.style.display = "block";
  var p = (sentDone + curLoaded) / totalBytes * 100;
  bar.style.width = Math.min(100, p).toFixed(1) + "%";
}

function enqueue(files){
  var added = 0, bytes = 0;
  for (var i = 0; i < files.length; i++){
    var f = files[i];
    if (!/\.pdf$/i.test(f.name)){ line("已跳过非 PDF 文件：" + f.name, "err"); continue; }
    queue.push(f); totalBytes += f.size; bytes += f.size; added++;
  }
  if (added) line("已加入队列：" + added + " 个文件，共 " + fmt(bytes) + "；开始上传……");
  pump();
}
function pump(){
  if (busy) return;
  var f = queue.shift();
  if (!f){
    if (totalBytes && sentDone + curLoaded >= totalBytes){
      line("上传完成：成功 " + doneCount + " 个，失败 " + failCount + " 个；暂存列表将自动刷新。",
           failCount ? "err" : "ok");
    }
    return;
  }
  busy = true; curLoaded = 0;
  var xhr = new XMLHttpRequest();
  try {
    xhr.open("POST", API + "/upload?token=" + encodeURIComponent(TOKEN) +
             "&name=" + encodeURIComponent(f.name), true);
  } catch (err) {
    busy = false; failCount++;
    line("无法构造上传地址（" + err.message + "），请刷新页面重试", "err");
    return;
  }
  xhr.upload.onprogress = function(e){
    if (e.lengthComputable){ curLoaded = e.loaded; progress(); }
  };
  xhr.onload = function(){
    busy = false;
    if (xhr.status === 200){
      doneCount++; sentDone += f.size; curLoaded = 0; progress();
      var extra = "";
      try { var r = JSON.parse(xhr.responseText); if (r && r.staged) extra = "（累计 " + r.staged + " 个）"; } catch (err) {}
      line("已暂存至服务器磁盘缓存" + extra + "：" + f.name, "ok");
    } else {
      failCount++;
      var msg = "上传失败：" + f.name;
      try { msg += " — " + (JSON.parse(xhr.responseText).error || xhr.status); }
      catch (err) { msg += " — HTTP " + xhr.status; }
      line(msg, "err");
      if (xhr.status === 0)
        line("无法连接上传服务（" + API + "），请检查服务器防火墙是否放行端口 " + PORT, "err");
    }
    pump();
  };
  xhr.onerror = function(){
    busy = false; failCount++;
    line("网络错误，已跳过：" + f.name + "（可重新选择该文件继续上传）", "err");
    pump();
  };
  xhr.send(f);
}

drop.onclick = function(){ picker.click(); };
picker.onchange = function(){ enqueue(picker.files); picker.value = ""; };
drop.ondragover = function(e){ e.preventDefault(); drop.className = "hover"; };
drop.ondragleave = function(){ drop.className = ""; };
drop.ondrop = function(e){ e.preventDefault(); drop.className = ""; enqueue(e.dataTransfer.files); };
</script>
</body></html>
"""

# -- 文件上传（旁路流式通道：字节边收边落盘，不经过 Streamlit 内存缓冲）--
_upload_server.sweep_orphans()

if not _ocr_state["running"]:
    _upload_port = _upload_server.ensure_server()
    _upload_token = st.session_state.get("aih_upload_token") or ""
    if not _upload_token:
        _upload_token = _upload_server.mint_token()
        st.session_state["aih_upload_token"] = _upload_token
    _upload_server.register_session(_upload_token, _ocr_state)
    components.html(
        _UPLOAD_PAGE_HTML
        .replace("__PORT__", str(_upload_port))
        .replace("__TOKEN__", _upload_token),
        height=260,
    )
else:
    st.info("处理进行中，待本批结束后可继续追加上传。")


@st.fragment(run_every=2)
def _staging_list_fragment():
    """暂存列表（磁盘缓存；处理期间就是排队队列）。

    独立片段每 2 秒刷新：旁路上传线程直接写 manifest，无需整页 rerun
    即可看到新暂存的书；暂存区空/非空状态翻转时整页 rerun 一次，让
    下方的页码过滤和开始按钮出现或隐藏。
    """
    staged = _staged_files(_ocr_state)
    if bool(staged) != st.session_state.get("aih_had_staged", False):
        st.session_state["aih_had_staged"] = bool(staged)
        st.rerun()
    if not staged:
        return
    total_mb = sum(f["size"] for f in staged) / (1024 * 1024)
    st.subheader(f"已暂存：{len(staged)} 个文件（磁盘缓存 {total_mb:.1f} MB）")
    st.dataframe(
        [{"文件名": f["name"], "大小 (MB)": f["size"] / (1024 * 1024)} for f in staged],
        width="stretch", hide_index=True,
    )
    if not _ocr_state["running"]:
        rm_cols = st.columns([3, 1])
        remove_sel = rm_cols[0].multiselect(
            "从暂存区移除所选文件",
            options=[Path(f["path"]).name for f in staged],
        )
        if rm_cols[1].button("移除", disabled=not remove_sel):
            path_by_option = {Path(f["path"]).name: f["path"] for f in staged}
            _remove_staged_files(_ocr_state, {path_by_option[n] for n in remove_sel})
            st.rerun()
    st.caption("提示：页数统计将在处理时显示；上传过程中可随时追加分批文件。")


_staging_list_fragment()

staged_files = _staged_files(_ocr_state)

if staged_files:
    # -- 页码过滤 --
    pages_filter = st.text_input(
        "页码过滤（可选）", value="", key="pages_filter",
        help="例如 1-5,8,10-12 — 对所有文件生效",
    )

    # -- 开始 / 停止 按钮 --
    btn_cols = st.columns([3, 1])
    start_clicked = btn_cols[0].button(
        "开始 OCR", type="primary", disabled=_ocr_state["running"],
    )
    if _ocr_state["running"]:
        if btn_cols[1].button("停止", type="secondary"):
            _ocr_state["stop"] = True

    if start_clicked and not _ocr_state["running"]:
        # 准备输出目录
        if save_to_disk and output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                st.error(f"无法创建输出目录：{e}")
                st.stop()

        # [DEBUG] 显示UI参数状态
        print(f"[UI-DEBUG] deskew checkbox value: {deskew}", flush=True)
        print(f"[UI-DEBUG] backend: {backend}", flush=True)

        # 构建 OCR 参数
        # 处理 mode=redo 的兼容性问题
        if mode == "redo":
            # --mode redo 与这些选项不兼容，自动禁用
            if clean_final or remove_background:
                st.warning("⚠️ 'redo' 模式与最终清理、移除背景不兼容，已自动禁用这些选项")
            clean_final = False
            remove_background = False

        ocr_kwargs: dict = {
            "mode": mode if mode != "default" else None,
            "output_type": output_type,
            "pdf_renderer": pdf_renderer,
            "optimize": optimize_level,
            "jobs": jobs,
            "use_threads": use_threads,
            "rotate_pages": rotate_pages or None,
            "deskew": deskew or None,
            "clean": clean or None,
            "clean_final": clean_final or None,
            "remove_background": remove_background or None,
            "remove_vectors": remove_vectors or None,
            "progress_bar": False,
            "hidden_text_layer": "strip" if strip_hidden_layer else "keep",
        }

        if oversample > 0:
            ocr_kwargs["oversample"] = oversample
        if pages_filter:
            ocr_kwargs["pages"] = pages_filter

        # 语言
        if custom_lang:
            ocr_kwargs["language"] = custom_lang.split("+")
        elif selected_langs:
            ocr_kwargs["language"] = selected_langs

        # 后端参数
        if backend == "Tesseract":
            ocr_kwargs["tesseract_oem"] = tess_oem
            ocr_kwargs["tesseract_pagesegmode"] = tess_psm
            ocr_kwargs["tesseract_timeout"] = tess_timeout
            ocr_kwargs["tesseract_downsample_large_images"] = tess_downsample
            thresh_map = {"auto": 0, "otsu": 1, "sauvola": 2}
            ocr_kwargs["tesseract_thresholding"] = thresh_map.get(tess_threshold, 0)
        elif backend == "Calamari（HTTP服务）":
            ocr_kwargs["ocr_engine"] = "calamari"
            seg_mode = "client"  # 总是使用客户端分割
            is_preflight = perf_strategy == "preflight"
            print(f"[DEBUG] perf_strategy={perf_strategy}, is_preflight={is_preflight}", flush=True)
            print(f"[UI-DEBUG] calamari_line_detector={calamari_line_detector}", flush=True)
            ocr_kwargs["calamari_url"] = calamari_url or "http://localhost:11800"
            ocr_kwargs["calamari_model"] = calamari_model or "gt4histocr"
            ocr_kwargs["calamari_timeout"] = calamari_timeout
            ocr_kwargs["calamari_batch_mode"] = calamari_batch_mode
            ocr_kwargs["calamari_segmentation"] = seg_mode
            ocr_kwargs["calamari_line_detector"] = calamari_line_detector
            if calamari_line_detector == "tesseract":
                ocr_kwargs["calamari_use_tesseract_segmentation"] = True
            if calamari_line_detector == "paddle_det":
                ocr_kwargs["calamari_paddle_model"] = paddle_det_model
                ocr_kwargs["calamari_paddle_device"] = paddle_det_device
                ocr_kwargs["calamari_paddle_min_confidence"] = paddle_det_min_confidence
                ocr_kwargs["calamari_paddle_python"] = paddle_det_python
                ocr_kwargs["calamari_paddle_source_dir"] = paddle_det_source_dir
            if calamari_line_detector == "chrome_screenai":
                ocr_kwargs["chrome_screenai_light"] = chrome_screenai_light
                ocr_kwargs["chrome_screenai_filter_types"] = chrome_screenai_filter_types
                ocr_kwargs["chrome_screenai_min_confidence"] = chrome_screenai_min_confidence
                ocr_kwargs["chrome_screenai_force_horizontal"] = chrome_screenai_force_horizontal
                ocr_kwargs["_ocr_worker_strategy"] = "serial"
            if binarize_lines:
                ocr_kwargs["calamari_binarize_lines"] = True
            if deskew:
                ocr_kwargs["calamari_deskew"] = True
                print(f"[UI-DEBUG] Set calamari_deskew=True", flush=True)
            else:
                print(f"[UI-DEBUG] deskew is False, not setting calamari_deskew", flush=True)
            if is_preflight:
                ocr_kwargs["calamari_preflight"] = True
                ocr_kwargs["calamari_preflight_chunk"] = preflight_chunk
                ocr_kwargs["calamari_preflight_dpi"] = preflight_dpi
                ocr_kwargs["calamari_preflight_workers"] = preflight_workers
            # Calamari使用自己的deskew实现，移除OCRmyPDF内置的deskew参数
            ocr_kwargs.pop("deskew", None)
        elif backend == "PaddleOCR（本地全链路）":
            ocr_kwargs["ocr_engine"] = "paddle_ocr"
            ocr_kwargs["paddle_ocr_version"] = paddle_ocr_version
            ocr_kwargs["paddle_ocr_lang"] = paddle_ocr_lang
            ocr_kwargs["paddle_ocr_device"] = paddle_ocr_device
            ocr_kwargs["paddle_ocr_rec_batch_size"] = paddle_ocr_rec_batch_size
            ocr_kwargs["paddle_ocr_min_confidence"] = paddle_ocr_min_confidence
            ocr_kwargs["paddle_ocr_python"] = paddle_ocr_python
            ocr_kwargs["paddle_ocr_source_dir"] = paddle_ocr_source_dir
            ocr_kwargs["paddle_ocr_enable_mkldnn"] = bool(paddle_ocr_enable_mkldnn)
            if paddle_ocr_cpu_threads:
                ocr_kwargs["paddle_ocr_cpu_threads"] = int(paddle_ocr_cpu_threads)
        elif backend == "Chrome ScreenAI（本地）":
            ocr_kwargs["ocr_engine"] = "chrome_screenai"
            ocr_kwargs["chrome_screenai_light"] = chrome_screenai_light
            ocr_kwargs["chrome_screenai_filter_types"] = chrome_screenai_filter_types
            ocr_kwargs["chrome_screenai_min_confidence"] = chrome_screenai_min_confidence
            ocr_kwargs["chrome_screenai_text_layer_mode"] = chrome_screenai_text_layer_mode
            ocr_kwargs["chrome_screenai_force_horizontal"] = chrome_screenai_force_horizontal
            ocr_kwargs["chrome_screenai_text_cleanup"] = chrome_screenai_text_cleanup
            ocr_kwargs["chrome_screenai_rasterize_dpi"] = chrome_screenai_rasterize_dpi
            ocr_kwargs["chrome_screenai_rasterize_quality"] = chrome_screenai_rasterize_quality
            ocr_kwargs["_chrome_screenai_native_pdf"] = chrome_screenai_text_layer_mode == "native_direct"
            ocr_kwargs["_ocr_worker_strategy"] = "serial"
        elif backend == "无（仅 PDF/A 转换）":
            ocr_kwargs["ocr_engine"] = "none"
            ocr_kwargs["pdf_renderer"] = "auto"

        # 移除 None 值
        ocr_kwargs = {k: v for k, v in ocr_kwargs.items() if v is not None}

        # [DEBUG] 显示最终参数
        print(f"[UI-DEBUG] Final ocr_kwargs keys: {list(ocr_kwargs.keys())}", flush=True)
        if "calamari_deskew" in ocr_kwargs:
            print(f"[UI-DEBUG] calamari_deskew in final kwargs: {ocr_kwargs['calamari_deskew']}", flush=True)
        else:
            print(f"[UI-DEBUG] calamari_deskew NOT in final kwargs", flush=True)

        # 队列只携带磁盘缓存路径；书的内容从上传起就只活在磁盘暂存区。
        batch_dir = _get_batch_dir(_ocr_state)
        if batch_dir is None or not staged_files:
            st.error("暂存区为空，请先上传文件。")
            st.stop()
        _reset_batch_for_run(batch_dir, staged_files)
        files_data = [(f["name"], f["path"]) for f in staged_files]

        # 重置状态
        _ocr_state["running"] = True
        _ocr_state["done"] = False
        _ocr_state["stop"] = False
        _ocr_state["logs"] = []
        _ocr_state["outputs"] = []
        _ocr_state["errors"] = []
        _ocr_state["progress"] = 0.0
        _ocr_state["progress_text"] = "准备中..."
        _ocr_state["output_dir"] = output_dir if save_to_disk else ""

        # 启动后台线程
        t = threading.Thread(
            target=_run_ocr_batch,
            args=(_ocr_state, files_data, ocr_kwargs, backend, output_suffix,
                  save_to_disk, output_dir, file_concurrency),
            daemon=True,
        )
        t.start()
        st.rerun()

else:
    st.info("请通过上方上传区添加 PDF 文件，支持拖拽、多选与分批追加。")

# ---------------------------------------------------------------------------
# 处理状态显示区（从 session_state 读取，不受 rerun 影响）
# ---------------------------------------------------------------------------
if _ocr_state["running"] or _ocr_state["done"]:
    st.markdown("<hr style='margin:0.5rem 0'>", unsafe_allow_html=True)
    st.subheader("处理状态")

    # 进度条（页面级进度放前面，防止长文件名截断）
    pg_cur = _ocr_state.get("page_current", 0)
    pg_tot = _ocr_state.get("page_total", 0)
    pg_desc = _ocr_state.get("page_desc", "")
    if _ocr_state["running"] and pg_tot > 0:
        progress_text = f"[{pg_desc} {pg_cur}/{pg_tot}]  {_ocr_state['progress_text']}"
    else:
        progress_text = _ocr_state["progress_text"]
    st.progress(
        _ocr_state["progress"],
        text=progress_text,
    )

    # 滚动日志容器
    log_container = st.container(height=300)
    with log_container:
        for entry in _ocr_state["logs"]:
            t = entry["type"]
            if t == "success":
                st.success(entry["msg"])
            elif t == "warning":
                st.warning(entry["msg"])
            elif t == "info":
                st.info(entry["msg"])
            else:
                st.error(entry["msg"])

    # 处理中自动刷新
    if _ocr_state["running"]:
        time.sleep(1)
        st.rerun()

    # 处理完成 — 下载区
    if _ocr_state["done"]:
        output_files = _ocr_state["outputs"]
        if output_files:
            st.divider()
            if _ocr_state["output_dir"]:
                st.info(f"文件已同步保存到：{_ocr_state['output_dir']}")
            elif len(output_files) == 1:
                name, output_path = output_files[0]
                with Path(output_path).open("rb") as data:
                    st.download_button(
                        f"下载 {name}", data=data,
                        file_name=name, mime="application/pdf",
                    )
            else:
                zip_path = Path(_ocr_state["task_dir"]) / "ocrmypdf_output.zip"
                if not zip_path.exists():
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                        for name, output_path in output_files:
                            zf.write(output_path, arcname=name)
                with zip_path.open("rb") as data:
                    st.download_button(
                        f"下载全部（{len(output_files)} 个文件，ZIP）",
                        data=data,
                        file_name="ocrmypdf_output.zip",
                        mime="application/zip",
                    )

        # 错误汇总
        if _ocr_state["errors"]:
            with st.expander(f"错误详情（{len(_ocr_state['errors'])} 个）"):
                for fname, err in _ocr_state["errors"]:
                    st.text(f"{fname}：{err}")

        # 清除按钮
        if st.button("清除结果"):
            _cleanup_task_dir(_ocr_state)
            _ocr_state["done"] = False
            _ocr_state["logs"] = []
            _ocr_state["outputs"] = []
            _ocr_state["errors"] = []
            st.rerun()

# ---------------------------------------------------------------------------
# 页脚致谢
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="credits-footer">'
    'Powered by '
    '<a href="https://github.com/ocrmypdf/OCRmyPDF">OCRmyPDF</a> · '
    '<a href="https://github.com/tesseract-ocr/tesseract">Tesseract OCR</a> · '
    '<a href="https://github.com/Calamari-OCR/calamari">Calamari OCR</a> · '
    '<a href="https://github.com/chromium/chromium">Chrome ScreenAI</a>'
    '<br>AIH-Infra 特化版 by Baireinhold'
    '</div>',
    unsafe_allow_html=True,
)




