# SPDX-FileCopyrightText: 2025 James R. Barlow
# SPDX-License-Identifier: AGPL-3.0-or-later

"""AIH-OCR - 增强版批量处理界面."""

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
from html import escape
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
from ocrmypdf.aih_chrome_screenai import scan_screenai_environment
from ocrmypdf.aih_webui_performance import detect_hardware, recommend_performance
from ocrmypdf.exceptions import InputFileError

# 禁用 Streamlit 日志（主进程和子进程）
for logger_name in ['streamlit', 'streamlit.runtime', 'streamlit.runtime.scriptrunner_utils',
                    'streamlit.runtime.state', 'streamlit.runtime.caching']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
logging.getLogger('streamlit.runtime').setLevel(logging.ERROR)

_CONFIG_PATH = Path(os.environ.get("AIH_OCRMYPDF_CONFIG") or Path.home() / ".aih-ocrmypdf-webui.json")
_LEGACY_CONFIG_PATH = Path.home() / ".ocrmypdf-aih-webui.json"  # pre-rename settings

def _load_config() -> dict:
    try:
        path = _CONFIG_PATH
        if not os.environ.get("AIH_OCRMYPDF_CONFIG") and not path.exists():
            path = _LEGACY_CONFIG_PATH
        cfg = json.loads(path.read_text("utf-8")) if path.exists() else {}
        if not isinstance(cfg, dict):
            raise ValueError("配置内容应为 JSON 对象")
        return cfg
    except (OSError, ValueError) as exc:
        st.error(f"配置读取失败：{exc}。请检查配置文件 {_CONFIG_PATH}。")
        st.stop()

def _save_config(cfg: dict):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=_CONFIG_PATH.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            json.dump(cfg, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, _CONFIG_PATH)
        return True
    except OSError as exc:
        st.error(f"配置保存失败：{exc}")
        return False
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _saved_value(key, default, *, kind, minimum=None, maximum=None):
    value = _user_cfg.get("settings", {}).get(key, default)
    valid = {
        "text": isinstance(value, str),
        "bool": isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "list": isinstance(value, list),
    }[kind]
    if not valid:
        st.error(f"保存的配置 {key} 类型无效，请检查配置文件。")
        st.stop()
    if kind == "number":
        if minimum is not None and value < minimum or maximum is not None and value > maximum:
            st.error(f"保存的配置 {key} 超出当前可用范围，请检查配置文件。")
            st.stop()
        if isinstance(default, float):
            value = float(value)
        elif not isinstance(value, int):
            st.error(f"保存的配置 {key} 应为整数，请检查配置文件。")
            st.stop()
    return value


def _saved_index(key, options, default):
    saved = _user_cfg.get("settings", {})
    if key not in saved:
        return default
    if saved[key] not in options:
        st.warning(f"已保存的 {key} 当前不可用，请重新选择并保存。")
        return default
    return list(options).index(saved[key])


def _saved_selection(key, options, default):
    selected = _saved_value(key, default, kind="list")
    available = [value for value in selected if value in options]
    if available != selected:
        st.warning("部分已保存的语言包当前未安装，请重新选择并保存。")
    return available

# 确保当前包目录在 sys.path 中，以便加载同包内插件模块
_pkg_dir = str(Path(__file__).parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# 预初始化 OCRmyPDF 插件基础设施（防止 Streamlit 模块重载丢失注册状态）
from ocrmypdf.api import setup_plugin_infrastructure as _setup_plugins
_setup_plugins()

from ocrmypdf.aih_line_detection import (
    discover_paddle_env, probe_paddle_environment, refresh_paddle_environment_cache,
)

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
_favicon = Path(__file__).parent / "logo.png"
st.set_page_config(
    page_title="AIH-OCR",
    page_icon=str(_favicon) if _favicon.exists() else "\U0001F4D6",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 橘白主题 CSS
# ---------------------------------------------------------------------------
_theme_css = Path(__file__).with_name("webui.css").read_text(encoding="utf-8")
st.markdown(f"<style>{_theme_css}</style>", unsafe_allow_html=True)

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
    1: "1 - LSTM（现代印刷体）",
    2: "2 - 传统 + LSTM 混合",
    3: "3 - 自动选择（按语言包内容决定）",
}

OEM_DESCRIPTIONS = {
    0: "适合古德语 Fraktur (frk/deu_frak) 等历史文献，使用传统模式匹配引擎",
    1: "使用 LSTM 识别现代印刷体；Fraktur 等传统语言包需选择传统引擎。",
    2: "结合传统引擎与 LSTM，需要语言包同时包含两类数据。",
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
    input_path = payload.get("input_path")
    file_state = state.get("file_states", {}).get(input_path)
    if payload.get("output"):
        state["outputs"].append(payload["output"])
        if file_state is not None:
            file_state["output"] = payload["out_name"]
    if payload.get("output_path") and payload.get("out_name"):
        output_path = Path(payload["output_path"])
        if output_path.exists():
            state["outputs"].append((payload["out_name"], str(output_path)))
            if file_state is not None:
                file_state["output"] = str(output_path)
    if payload.get("error"):
        state["errors"].append(payload["error"])
        if file_state is not None:
            error = payload["error"]
            file_state["error"] = error[1] if isinstance(error, (tuple, list)) and len(error) > 1 else str(error)
            file_state["status"] = "failed"
    elif file_state is not None:
        file_state["status"] = "completed"
    if payload.get("log"):
        level, fname, msg = payload["log"]
        _log(state, level, fname, msg)

    progress_lock = state.get("_batch_progress_lock")
    if progress_lock is not None:
        with progress_lock:
            completed = state.get("_batch_completed_count", 0) + 1
            state["_batch_completed_count"] = completed
            state["completed_count"] = completed
            total = state.get("_batch_total", 0)
            progress = completed / total if total else 1.0
            state["progress"] = progress
            state["progress_text"] = f"已处理 {completed}/{total} 个文件"


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
        file_state = state.get("file_states", {}).get(source_path)
        if file_state is not None:
            file_state["status"] = "running"
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
        state["progress_text"] = (
            f"已处理 {state.get('_batch_completed_count', 0)}/{total} 个文件，正在启动：{fname}"
        )

    try:
        while next_index < total and len(running) < file_concurrency:
            _launch(next_index)
            next_index += 1

        while running:
            if state["stop"]:
                state["progress_text"] = "正在停止 Chrome worker"
                for item in running:
                    item["proc"].terminate()
                for item in running:
                    try:
                        item["proc"].wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        item["proc"].kill()
                        item["proc"].wait()
                    file_state = state.get("file_states", {}).get(files_data[item["index"]][1])
                    if file_state is not None and file_state["status"] == "running":
                        file_state["status"] = "stopped"
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
                payload.setdefault("input_path", files_data[item["index"]][1])

                if (
                    payload.get("log")
                    and payload["log"][0] == "warning"
                    and ocr_kwargs.get("output_type") != "auto"
                ):
                    payload["error"] = payload.get("error") or (
                        item["fname"], "PDF/A 转换失败，输出可能不符合标准"
                    )

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

                if payload.get("output_path") and not Path(payload["output_path"]).exists():
                    payload["output_path"] = None
                if not payload.get("error") and not payload.get("output") and not payload.get("output_path"):
                    payload["error"] = (item["fname"], "Chrome worker 未生成输出文件")
                    payload["log"] = ("error", item["fname"], f"{item['fname']}：Chrome worker 未生成输出文件")

                _apply_ocr_worker_result(state, payload)
                running.remove(item)
                completed += 1

                while next_index < total and len(running) < file_concurrency:
                    _launch(next_index)
                    next_index += 1
                break
            else:
                time.sleep(0.2)
    finally:
        for item in running:
            if item["proc"].poll() is None:
                item["proc"].terminate()
                try:
                    item["proc"].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    item["proc"].kill()
                    item["proc"].wait()
            file_state = state.get("file_states", {}).get(files_data[item["index"]][1])
            if file_state is not None and file_state["status"] == "running":
                file_state["status"] = "stopped" if state.get("stop") else "failed"
                if file_state["status"] == "failed":
                    file_state["error"] = file_state["error"] or "Chrome worker 未完成"
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

    file_state = state.setdefault("file_states", {}).setdefault(input_path, {
        "name": fname,
        "status": "pending",
        "output": None,
        "error": None,
        "page_current": 0,
        "page_total": 0,
    })
    file_state.update({
        "name": fname,
        "status": "running",
        "output": None,
        "error": None,
        "page_current": 0,
        "page_total": 0,
    })
    # OCRmyPDF may import this plugin independently; bind the progress bar to
    # this worker's file state while retaining the legacy builtins state.
    builtins._ocrmypdf_page_progress_state = state
    progress_state_local = builtins.__dict__.setdefault(
        "_ocrmypdf_page_progress_state_local", threading.local()
    )
    progress_state_local.state = file_state
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
            file_state["status"] = "completed"
            file_state["output"] = str(saved_path)
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
                file_state["output"] = str(saved_path)
            selected_output_type = kw.get("output_type")
            if selected_output_type == "auto":
                suffix = f"，输出：{saved_path}" if saved_path else ""
                _log(state, "warning", fname, f"{fname}：未能生成 PDF/A，已自动回退为普通 PDF{suffix} (耗时: {elapsed_str})")
                if saved_path:
                    file_state["status"] = "completed"
                    file_state["output"] = str(saved_path)
                else:
                    file_state["status"] = "failed"
                    file_state["error"] = "PDF/A 转换失败，未生成输出文件"
                    state["errors"].append((fname, file_state["error"]))
            else:
                suffix = f"，输出：{saved_path}" if saved_path else ""
                _log(state, "warning", fname, f"{fname}：PDF/A 转换失败，输出可能不符合标准{suffix} (耗时: {elapsed_str})")
                file_state["status"] = "failed"
                file_state["error"] = "PDF/A 转换失败，输出可能不符合标准"
                state["errors"].append((fname, file_state["error"]))
        elif result == ocrmypdf.ExitCode.already_done_ocr:
            hint = (
                "请勾选“OCR 前先移除旧的隐形文字层”，或改用“跳过 / 强制”模式"
                if kw.get("hidden_text_layer") == "keep"
                else "页面含可见文字（或隐形文字背后没有图片），请改用“跳过 / 强制”模式"
            )
            state["errors"].append((fname, "页面已有文字"))
            file_state["status"] = "failed"
            file_state["error"] = "页面已有文字"
            _log(state, "warning", fname, f"{fname}：已包含文字层，未处理；{hint} (耗时: {elapsed_str})")
        elif result == ocrmypdf.ExitCode.encrypted_pdf:
            state["errors"].append((fname, "PDF 已加密"))
            file_state["status"] = "failed"
            file_state["error"] = "PDF 已加密"
            _log(state, "error", fname, f"{fname}：PDF 已加密，请先解密 (耗时: {elapsed_str})")
        else:
            state["errors"].append((fname, f"退出码：{result.name}"))
            file_state["status"] = "failed"
            file_state["error"] = f"退出码：{result.name}"
            _log(state, "error", fname, f"{fname}：处理失败（{result.name}）(耗时: {elapsed_str})")
    except Exception as e:
        elapsed = time.time() - start_time
        elapsed_str = f"{elapsed:.1f}秒"
        error_message = str(e) or type(e).__name__
        state["errors"].append((fname, error_message))
        file_state["status"] = "failed"
        if isinstance(e, InputFileError) and not str(e):
            file_state["error"] = "PDF 输入读取失败，请检查文件完整性与读取权限。"
        else:
            file_state["error"] = error_message
        _log(state, "error", fname, f"{fname}：{error_message} (耗时: {elapsed_str})")
    finally:
        if file_state["status"] == "running":
            file_state["status"] = "failed"
            file_state["error"] = file_state["error"] or "未生成处理结果"
        try:
            del progress_state_local.state
        except AttributeError:
            pass
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
    state["file_states"] = {
        input_path: {
            "name": fname,
            "status": "pending",
            "output": None,
            "error": None,
            "page_current": 0,
            "page_total": 0,
        }
        for fname, input_path in files_data
    }
    state["completed_count"] = 0
    state["progress"] = 0.0
    state["status"] = "running"
    state["_batch_total"] = total
    state["_batch_completed_count"] = 0
    progress_lock = threading.Lock()
    state["_batch_progress_lock"] = progress_lock
    state["_batch_exception"] = False
    state["_stop_behavior"] = "停止新文件调度，已启动的 OCR 处理结束后停止"
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
        _idx, (fname, input_path) = item
        if state["stop"]:
            return
        file_state = state["file_states"][input_path]
        file_state["status"] = "running"
        out_name = _make_out_name(fname)
        state["progress_text"] = f"正在处理：{fname}"
        try:
            _ocr_one_file(state, input_path, fname, out_name, ocr_kwargs, backend, save_to_disk, output_dir)
        except Exception as e:
            state["_batch_exception"] = True
            file_state["status"] = "failed"
            file_state["error"] = str(e)
            state["errors"].append((fname, str(e)))
            _log(state, "error", fname, f"{fname}：{e}")
            raise
        else:
            if file_state["status"] in ("completed", "failed"):
                with progress_lock:
                    completed_count = state["_batch_completed_count"] + 1
                    state["_batch_completed_count"] = completed_count
                    state["completed_count"] = completed_count
                    progress = completed_count / total if total else 1.0
                    state["progress"] = progress
                    state["progress_text"] = f"已处理 {completed_count}/{total} 个文件"

    try:
        chrome_file_parallel = (
            backend == "Chrome ScreenAI（本地）"
            and file_concurrency > 1
            and int(ocr_kwargs.get("jobs", 1) or 1) <= 1
        )
        if chrome_file_parallel:
            state["_stop_behavior"] = "终止已启动的 Chrome worker，并停止后续文件"
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
                    f.result()
        else:
            for i, fd in enumerate(files_data):
                if state["stop"]:
                    break
                _process_one((i, fd))
    except Exception as e:
        state["_batch_exception"] = True
        _log(state, "error", "", f"批量处理异常：{e}")
        state["errors"].append(("", f"批量处理异常：{e}"))
    finally:
        task_dir = state.get("task_dir")
        if task_dir:
            try:
                for input_path, file_state in state["file_states"].items():
                    if file_state["status"] == "pending":
                        if state["stop"]:
                            file_state["status"] = "stopped"
                    elif file_state["status"] == "running":
                        file_state["status"] = "failed"
                        file_state["error"] = file_state["error"] or "批次中断，文件未完成"
                    if file_state["status"] == "completed":
                        try:
                            Path(input_path).unlink(missing_ok=True)
                        except OSError as e:
                            state["_batch_exception"] = True
                            state["errors"].append((file_state["name"], f"清理已处理输入失败：{e}"))
                retained = {
                    input_path
                    for input_path, file_state in state["file_states"].items()
                    if file_state["status"] != "completed"
                }
                _upload_server.mark_done(Path(task_dir), retained=retained)
            except Exception as e:
                state["_batch_exception"] = True
                state["errors"].append(("", f"批次收尾异常：{e}"))
                _log(state, "error", "", f"批次收尾异常：{e}")
        state["running"] = False
        state["done"] = True
        statuses = [file_state["status"] for file_state in state["file_states"].values()]
        success_count = statuses.count("completed")
        failed_count = statuses.count("failed")
        incomplete_count = sum(status in ("pending", "stopped", "running") for status in statuses)
        if state["stop"] and incomplete_count:
            state["status"] = "stopped"
            state["progress"] = state.get("completed_count", 0) / total if total else 0.0
            state["progress_text"] = (
                f"已停止：{state['_stop_behavior']}；已处理 {state.get('completed_count', 0)}/{total} 个文件；"
                "未完成输入已保留，可重新加入队列"
            )
        elif state["_batch_exception"]:
            state["status"] = "partial_failed" if success_count else "failed"
            state["progress"] = state.get("completed_count", 0) / total if total else 0.0
            state["progress_text"] = f"批次异常中止：已处理 {state.get('completed_count', 0)}/{total} 个文件"
        elif failed_count or incomplete_count:
            state["status"] = "partial_failed" if success_count else "failed"
            state["progress"] = state.get("completed_count", 0) / total if total else 0.0
            state["progress_text"] = f"文件处理结束：成功 {success_count}，失败 {failed_count}，未处理 {incomplete_count}"
        else:
            state["status"] = "completed"
            state["progress"] = 1.0
            state["progress_text"] = "全部文件处理完成"
        state.pop("_batch_progress_lock", None)
        state.pop("_batch_total", None)
        state.pop("_batch_completed_count", None)
        state.pop("_batch_exception", None)
        state.pop("_stop_behavior", None)


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
_user_cfg = _load_config()
if not isinstance(_user_cfg.get("settings", {}), dict):
    st.error("配置 settings 应为 JSON 对象，请检查配置文件。")
    st.stop()
_PERSISTED_WIDGET_KEYS = ['backend', 'binarize_lines', 'calamari_batch_mode', 'calamari_line_detector',
 'calamari_model_custom', 'calamari_model_sel', 'calamari_retry', 'calamari_timeout',
 'calamari_url', 'chrome_screenai_filter_types', 'chrome_screenai_force_horizontal',
 'chrome_screenai_light', 'chrome_screenai_min_confidence', 'chrome_screenai_rasterize_dpi',
 'chrome_screenai_rasterize_quality', 'chrome_screenai_text_cleanup',
 'chrome_screenai_text_layer_mode', 'clean', 'clean_final', 'custom_lang', 'deskew', 'mode',
 'oem0_languages', 'oem1_languages', 'oem2_languages', 'oem3_languages', 'optimize_level',
 'output_dir', 'output_mode', 'output_suffix', 'oversample', 'paddle_det_device',
 'paddle_det_min_confidence', 'paddle_det_model', 'paddle_det_python', 'paddle_det_source_dir',
 'paddle_ocr_device', 'paddle_ocr_enable_mkldnn', 'paddle_ocr_lang',
 'paddle_ocr_min_confidence', 'paddle_ocr_python', 'paddle_ocr_source_dir',
 'paddle_ocr_version', 'pdf_renderer', 'perf_strategy', 'preflight_chunk', 'preflight_dpi',
 'preflight_workers', 'remove_background', 'remove_vectors', 'rotate_pages', 'save_to_disk',
 'strip_hidden_layer', 'tess_downsample', 'tess_oem', 'tess_psm', 'tess_threshold',
 'tess_timeout']

_brand_logo = Path(__file__).with_name("logo.png")
if not _brand_logo.exists():
    _brand_logo = Path(__file__).parents[2] / "misc" / "logo.png"
_logo_data = base64.b64encode(_brand_logo.read_bytes()).decode("ascii")
st.markdown(
    f'<div class="aih-header"><img class="aih-brand-mark" src="data:image/png;base64,{_logo_data}" alt="AIH-OCR">'
    '<div><div class="aih-eyebrow">AIH · DOCUMENT WORKSPACE</div><h1>AIH-OCR</h1>'
    '<p class="aih-subtitle">批量识别 PDF · 保留文献版面 · 生成可检索文字层</p></div></div>',
    unsafe_allow_html=True,
)
_work_column, _config_column = st.columns([1.65, 1], gap="large")
_work_area = _work_column.container()
_config_column.subheader("下一次处理配置" if _ocr_state["running"] else "处理配置", anchor="processing-configuration")
_engine_settings, _document_settings, _output_settings, _performance_settings = _config_column.tabs(
    ["识别", "文档", "输出", "性能"]
)
_settings = _engine_settings

if "screenai_environment_scan" not in st.session_state:
    st.session_state["screenai_environment_scan"] = scan_screenai_environment()
_initial_screenai = st.session_state["screenai_environment_scan"]
_default_backend_index = 3 if not _initial_screenai["error"] and not _initial_screenai["missing"] else 0

_BACKEND_LABELS = {'PaddleOCR（本地全链路）': 'PaddleOCR',
 'Calamari（HTTP服务）': 'Calamari',
 'Chrome ScreenAI（本地）': 'Chrome ScreenAI',
 '无（仅 PDF/A 转换）': '仅转换 PDF'}

# -- 后端选择 --
backend = _settings.selectbox(
    "OCR 后端",
    ["Tesseract", "PaddleOCR（本地全链路）", "Calamari（HTTP服务）", "Chrome ScreenAI（本地）", "无（仅 PDF/A 转换）"],
    index=_saved_index('backend', ["Tesseract", "PaddleOCR（本地全链路）", "Calamari（HTTP服务）", "Chrome ScreenAI（本地）", "无（仅 PDF/A 转换）"], _default_backend_index),
    format_func=lambda value: _BACKEND_LABELS.get(value, value),
    key='backend',
)

_settings.caption({
    "Tesseract": "在本机运行 Tesseract，使用已安装的语言包识别文档。",
    "PaddleOCR（本地全链路）": "在本机 Paddle 环境中完成文本检测与识别，可选择 CPU 或 GPU。",
    "Calamari（HTTP服务）": "需先在本机自行部署并启动 Calamari 服务，再通过下方服务地址调用。",
    "Chrome ScreenAI（本地）": "在本机使用 Chrome ScreenAI 模型与原生组件识别文档。",
    "无（仅 PDF/A 转换）": "按输出选项转换 PDF，不运行文字识别。",
}[backend])

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
chrome_screenai_text_cleanup = "sanitize"
chrome_screenai_rasterize_dpi = 300
chrome_screenai_rasterize_quality = 88
paddle_det_model = "PP-OCRv5_server_det"
paddle_det_device = "gpu:0"
paddle_det_min_confidence = 0.0
_discovered_paddle_env = None
_paddle_probe = None
def _detect_paddle_settings(container, key):
    if container.button("重新检测 Paddle 环境", key=key):
        refresh_paddle_environment_cache()
    try:
        return discover_paddle_env()
    except (ValueError, RuntimeError, OSError) as exc:
        container.error(f"Paddle 环境配置错误：{exc}")
        return None

def _screenai_environment_settings(container, key):
    refresh = container.button("重新扫描 ScreenAI", key=key)
    if refresh or "screenai_environment_scan" not in st.session_state:
        st.session_state["screenai_environment_scan"] = scan_screenai_environment()
    status = st.session_state["screenai_environment_scan"]
    if status["error"]:
        container.error(status["error"])
    elif status["missing"]:
        container.error(f"ScreenAI 组件缺失或为空的文件：{len(status['missing'])} 项，请修复模型文件后重新扫描。")
    else:
        container.success(f"已发现 ScreenAI {status['version'] or '（版本未标注）'} · OCR 文件齐全（{status['model_file_count']} 项）")
    if status["directory"]:
        container.caption(f"当前组件目录：{status['directory']}")
    with container.expander("扫描详情"):
        st.caption("搜索范围：Chrome 组件目录、本机 locro 模型目录；优先级与实际识别一致。")
        for directory in status["search_roots"]:
            st.text(directory)
        st.caption(f"原生库：{status['library']}")
        if status["missing"]:
            st.code("\n".join(status["missing"]), language=None)


paddle_det_python = str(_discovered_paddle_env.python) if _discovered_paddle_env else ""
paddle_det_source_dir = str(_discovered_paddle_env.source_dir) if _discovered_paddle_env else ""
paddle_ocr_version = "PP-OCRv6"
paddle_ocr_lang = "en"
paddle_ocr_device = "auto"
paddle_ocr_rec_batch_size = 16
paddle_ocr_min_confidence = 0.0
paddle_ocr_python = paddle_det_python
paddle_ocr_source_dir = paddle_det_source_dir
paddle_ocr_enable_mkldnn = False
paddle_ocr_cpu_threads = 0
if backend == "Calamari（HTTP服务）":
    calamari_url = _settings.text_input(
        "Calamari 服务地址",
        value=_saved_value('calamari_url', "http://localhost:11800", kind='text'),
        key='calamari_url',
    )
    models = fetch_calamari_models(calamari_url)
    if models is None:
        _settings.warning("无法连接 Calamari 服务")
    elif models:
        _settings.success(f"已连接 — 共 {len(models)} 个模型")
    else:
        _settings.info("已连接，模型按需加载")
    _model_options = list(models or [])
    _saved_model = _user_cfg.get("settings", {}).get("calamari_model_sel")
    if isinstance(_saved_model, str) and _saved_model != "（自定义）" and _saved_model not in _model_options:
        _model_options.append(_saved_model)
    _model_options.append("（自定义）")
    # 默认选择 gt4histocr 或 fraktur_19th_century
    default_idx = 0
    for preferred in ["gt4histocr", "fraktur_19th_century"]:
        if preferred in _model_options:
            default_idx = _model_options.index(preferred)
            break
    _sel = _settings.selectbox(
        "选择模型",
        options=_model_options,
        index=_saved_index("calamari_model_sel", _model_options, default_idx),
        key="calamari_model_sel",
    )
    if _sel == "（自定义）":
        calamari_model = _settings.text_input(
            "自定义模型名称",
            value=_saved_value("calamari_model_custom", "gt4histocr", kind='text'),
            key="calamari_model_custom",
        )
    else:
        calamari_model = _sel

    # -- Calamari 性能策略 --
    PERF_STRATEGIES = {
        "fast": "按页分割与识别",
        "preflight": "预渲染后批量识别",
    }
    perf_strategy = _performance_settings.selectbox(
        "性能策略",
        list(PERF_STRATEGIES.keys()),
        format_func=lambda x: PERF_STRATEGIES[x],
        index=_saved_index('perf_strategy', list(PERF_STRATEGIES.keys()), 1),
        help="按页：分割后逐页识别；预渲染：先分批渲染和切行，再跨页批量识别。",
        key='perf_strategy',
    )
    if perf_strategy == "preflight":
        _performance_settings.caption("先渲染、切行并写入缓存，随后开始批量识别。")

    # -- Calamari 高级选项 --
    with _settings.expander("Calamari 高级选项"):
        LINE_DETECTOR_LABELS = {
            "paddle_det": "PaddleOCR — 文字区域检测",
            "chrome_screenai": "Chrome ScreenAI — 本地版面/行检测",
            "tesseract": "Tesseract — 传统行分割",
            "projection_debug": "投影法 — 兼容旧流程",
        }
        calamari_line_detector = st.selectbox(
            "行检测器",
            ["paddle_det", "chrome_screenai", "tesseract", "projection_debug"],
            format_func=lambda x: LINE_DETECTOR_LABELS[x],
            index=_saved_index('calamari_line_detector', ["paddle_det", "chrome_screenai", "tesseract", "projection_debug"], 0),
            help="选择 Calamari 识别前的切行来源",
            key='calamari_line_detector',
        )
        if calamari_line_detector == "paddle_det":
            st.caption("Paddle 仅负责前置文字区域检测，文字识别由 Calamari 服务完成。")
            _discovered_paddle_env = _detect_paddle_settings(st, "refresh_calamari_paddle")
            paddle_det_python = str(_discovered_paddle_env.python) if _discovered_paddle_env else ""
            paddle_det_source_dir = str(_discovered_paddle_env.source_dir) if _discovered_paddle_env else ""
            if _discovered_paddle_env:
                st.caption(f"Paddle 环境：{_discovered_paddle_env.source} · {_discovered_paddle_env.python}")
            else:
                st.caption("Paddle 环境：未自动发现，可手动填写 Python 与源码目录")
            paddle_det_model = st.text_input(
                "Paddle 检测模型",
                value=_saved_value('paddle_det_model', "PP-OCRv5_server_det", kind='text'),
                help="默认使用 PaddleOCR 3.x 文本检测模型",
                key='paddle_det_model',
            )
            paddle_det_device = st.selectbox(
                "Paddle 推理设备",
                ["gpu:0", "cpu"],
                index=_saved_index('paddle_det_device', ["gpu:0", "cpu"], 0),
                key='paddle_det_device',
            )
            paddle_det_min_confidence = st.number_input(
                "Paddle 最低置信度",
                value=_saved_value('paddle_det_min_confidence', 0.0, kind='number', minimum=0.0, maximum=1.0),
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key='paddle_det_min_confidence',
            )
            paddle_det_python = st.text_input(
                "Paddle Python",
                value=_saved_value('paddle_det_python', paddle_det_python, kind='text'),
                key='paddle_det_python',
            )
            paddle_det_source_dir = st.text_input(
                "Paddle 源码目录",
                value=_saved_value('paddle_det_source_dir', paddle_det_source_dir, kind='text'),
                key='paddle_det_source_dir',
            )
        if calamari_line_detector == "chrome_screenai":
            _screenai_environment_settings(st, "scan_calamari_screenai")
            chrome_screenai_light = st.checkbox(
                "Chrome ScreenAI 轻量模式",
                value=_saved_value('chrome_screenai_light', False, kind='bool'),
                help="启用轻量模型路径；可用性取决于本地 locro/ScreenAI 环境",
                key='chrome_screenai_light',
            )
            chrome_screenai_filter_types = st.selectbox(
                "Chrome 内容过滤",
                ["printed", "all"],
                format_func=lambda x: {
                    "printed": "仅印刷文本",
                    "all": "全部文本类型",
                }[x],
                key='chrome_screenai_filter_types',
                index=_saved_index('chrome_screenai_filter_types', ["printed", "all"], 0),
            )
            chrome_screenai_min_confidence = st.number_input(
                "Chrome 最低置信度",
                value=_saved_value('chrome_screenai_min_confidence', 0.0, kind='number', minimum=0.0, maximum=1.0),
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key='chrome_screenai_min_confidence',
            )
            chrome_screenai_force_horizontal = st.checkbox(
                "强制横排坐标匹配",
                value=_saved_value('chrome_screenai_force_horizontal', True, kind='bool'),
                help="忽略 Chrome 返回的竖排方向，按从上到下、同一行从左到右匹配坐标。",
                key='chrome_screenai_force_horizontal',
            )
        binarize_lines = st.checkbox(
            "对行图像进行二值化",
            value=_saved_value('binarize_lines', True, kind='bool'),
            help="使用 Otsu 自动阈值对切割的行图像进行二值化，改善泛黄背景文档的识别质量",
            key='binarize_lines',
        )
        calamari_batch_mode = st.selectbox(
            "批量模式",
            ["indexed", "sequential"],
            format_func=lambda x: {
                "indexed": "索引编码（indexed）",
                "sequential": "逐张串行（sequential）",
            }[x],
            key='calamari_batch_mode',
            index=_saved_index('calamari_batch_mode', ["indexed", "sequential"], 0),
        )
        calamari_timeout = st.number_input(
            "请求超时（秒）",
            value=_saved_value('calamari_timeout', 300, kind='number', minimum=10, maximum=3600),
            min_value=10,
            max_value=3600,
            key='calamari_timeout',
        )
        calamari_retry = st.number_input(
            "重试次数",
            value=_saved_value('calamari_retry', 2, kind='number', minimum=0, maximum=10),
            min_value=0,
            max_value=10,
            key='calamari_retry',
        )
        if perf_strategy == "preflight":
            preflight_dpi = st.selectbox(
                "预飞渲染 DPI",
                [150, 200, 300, 400],
                index=_saved_index('preflight_dpi', [150, 200, 300, 400], 2),
                help="越高越清晰但越慢，300 适合大多数场景",
                key='preflight_dpi',
            )
        else:
            preflight_chunk = 10
            preflight_dpi = 300
            preflight_workers = 1

    # -- Calamari 健康诊断 --
    if calamari_url:
        with _settings.expander("服务诊断", expanded=False):
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
    _discovered_paddle_env = _detect_paddle_settings(_settings, "refresh_paddle_ocr")
    paddle_ocr_python = str(_discovered_paddle_env.python) if _discovered_paddle_env else ""
    paddle_ocr_source_dir = str(_discovered_paddle_env.source_dir) if _discovered_paddle_env else ""
    with _settings.container():
        paddle_ocr_version = st.selectbox(
            "PaddleOCR 基线",
            ["PP-OCRv6", "PP-OCRv5", "auto"],
            index=_saved_index('paddle_ocr_version', ["PP-OCRv6", "PP-OCRv5", "auto"], 0),
            help="PP-OCRv6 是 PaddleOCR 3.7 的新基线；auto 由当前 PaddleOCR 环境决定。",
            key='paddle_ocr_version',
        )
        paddle_ocr_lang = st.text_input(
            "PaddleOCR 语言",
            value=_saved_value('paddle_ocr_lang', "en", kind='text'),
            help="例如 en、ch、japan；PP-OCRv6 覆盖多种拉丁语系。",
            key='paddle_ocr_lang',
        )
        paddle_ocr_device = st.selectbox(
            "PaddleOCR 推理设备",
            ["gpu:0", "cpu", "auto"],
            index=_saved_index('paddle_ocr_device', ["gpu:0", "cpu", "auto"], 2),
            key='paddle_ocr_device',
        )
        with st.expander("PaddleOCR 高级选项"):
            if _discovered_paddle_env:
                st.caption(f"Paddle 环境：{_discovered_paddle_env.source} · {_discovered_paddle_env.python}")
            else:
                st.caption("Paddle 环境：未自动发现，可手动填写 Python 与源码目录")
            paddle_ocr_min_confidence = st.number_input(
                "最低识别置信度",
                value=_saved_value('paddle_ocr_min_confidence', 0.0, kind='number', minimum=0.0, maximum=1.0),
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key='paddle_ocr_min_confidence',
            )
            paddle_ocr_python = st.text_input(
                "Paddle Python",
                value=_saved_value('paddle_ocr_python', paddle_ocr_python, kind='text'),
                key='paddle_ocr_python',
            )
            paddle_ocr_source_dir = st.text_input(
                "Paddle 源码目录",
                value=_saved_value('paddle_ocr_source_dir', paddle_ocr_source_dir, kind='text'),
                key='paddle_ocr_source_dir',
            )
        if paddle_ocr_python:
            _paddle_probe = probe_paddle_environment(paddle_ocr_python, paddle_ocr_source_dir or None)
            if _paddle_probe["error"]:
                st.error("Paddle 环境不可用：" + _paddle_probe["error"])
            else:
                st.caption(f"已检测：PaddleOCR {_paddle_probe['paddleocr_version']} · Paddle {_paddle_probe['paddle_version']}")
                st.caption("可用设备：CPU" + ("、GPU" if _paddle_probe["device_available"]["gpu"] else ""))
                if paddle_ocr_device.startswith("gpu") and not _paddle_probe["device_available"]["gpu"]:
                    st.error("所选 GPU 不可用，请安装 GPU 环境或选择 CPU。" + (_paddle_probe["gpu_error"] or ""))
        else:
            st.info("未发现 Paddle 环境。请在项目目录运行下方安装命令，完成后重新检测。")
        st.code("python scripts/install_paddle.py --mode auto", language="powershell")
        st.caption("自动选择安装方案；可用 --mode cpu 或 --mode gpu 指定设备，--upgrade 更新已有环境。")
elif backend == "Chrome ScreenAI（本地）":
    _screenai_environment_settings(_settings, "scan_screenai")
    with _settings.container():
        chrome_screenai_text_layer_mode = st.selectbox(
            "文本层模式",
            ["native_direct", "proxy_match"],
            format_func=lambda x: {
                "native_direct": "Chrome 原生文本层",
                "proxy_match": "AIH 坐标匹配文本层",
            }[x],
            index=_saved_index('chrome_screenai_text_layer_mode', ["native_direct", "proxy_match"], 0),
            help="原生模式由 Chrome 生成文本层；坐标匹配模式可应用强制横排选项。",
            key='chrome_screenai_text_layer_mode',
        )
        if _user_cfg.get("settings", {}).get("chrome_screenai_text_cleanup") in ("mode_default", "clear_all"):
            _user_cfg["settings"]["chrome_screenai_text_cleanup"] = "sanitize"
        chrome_screenai_text_cleanup = st.selectbox(
            "OCR 前 PDF 净化",
            ["sanitize", "rasterize"],
            format_func=lambda x: {
                "sanitize": "保留正文与原图，清除隐藏文字和注释",
                "rasterize": "栅格化后重新识别",
            }[x],
            index=_saved_index('chrome_screenai_text_cleanup', ["sanitize", "rasterize"], 0),
            help="默认不栅格化、不重新编码图片，清除旧隐藏 OCR、批注、高亮、链接及表单注释；已有正文保留一份，OCR 补入未覆盖区域。净化副本超过原文件大小时停止。",
            key='chrome_screenai_text_cleanup',
        )
        if chrome_screenai_text_cleanup == "rasterize":
            chrome_screenai_rasterize_dpi = st.slider(
                "栅格化 DPI",
                150,
                600,
                _saved_value('chrome_screenai_rasterize_dpi', 300, kind='number', minimum=150, maximum=600),
                step=25,
                key='chrome_screenai_rasterize_dpi',
            )
            chrome_screenai_rasterize_quality = st.slider(
                "JPEG 质量",
                60,
                100,
                _saved_value('chrome_screenai_rasterize_quality', 88, kind='number', minimum=60, maximum=100),
                key='chrome_screenai_rasterize_quality',
            )
        with st.expander("ScreenAI 高级选项"):
            chrome_screenai_light = st.checkbox(
                "轻量模式",
                value=_saved_value('chrome_screenai_light', False, kind='bool'),
                help="启用轻量模型路径；可用性取决于本地 locro/ScreenAI 环境",
                key='chrome_screenai_light',
            )
            chrome_screenai_filter_types = st.selectbox(
                "内容过滤",
                ["printed", "all"],
                format_func=lambda x: {
                    "printed": "仅印刷文本",
                    "all": "全部文本类型",
                }[x],
                key='chrome_screenai_filter_types',
                index=_saved_index('chrome_screenai_filter_types', ["printed", "all"], 0),
            )
            chrome_screenai_min_confidence = st.number_input(
                "最低置信度",
                value=_saved_value('chrome_screenai_min_confidence', 0.0, kind='number', minimum=0.0, maximum=1.0),
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key='chrome_screenai_min_confidence',
            )
            chrome_screenai_force_horizontal = st.checkbox(
                "强制横排坐标匹配",
                value=_saved_value('chrome_screenai_force_horizontal', True, kind='bool'),
                disabled=chrome_screenai_text_layer_mode != "proxy_match",
                help="忽略 Chrome 返回的竖排方向，按从上到下、同一行从左到右匹配坐标。",
                key='chrome_screenai_force_horizontal',
            )
            if chrome_screenai_text_layer_mode == "native_direct":
                st.caption("当前为完全直通模式：文本层由 Chrome ScreenAI 原生生成，不经过 AIH 坐标重排。")

# -- Tesseract 引擎选择（在语言之前，因为引擎决定可用语言）--
tess_oem = 1  # 默认值，非 Tesseract 后端时使用
if backend == "Tesseract":
    tess_oem = _settings.selectbox(
        "Tesseract 引擎模式（OEM）",
        options=list(OEM_OPTIONS.keys()),
        format_func=lambda x: OEM_OPTIONS[x],
        index=_saved_index('tess_oem', list(OEM_OPTIONS.keys()), 1),
        key='tess_oem',
    )
    _settings.caption(OEM_DESCRIPTIONS[tess_oem])

# -- 语言设置 --
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

    _settings.caption(f"已安装 {len(installed_langs)} 个语言包，当前 OEM {tess_oem} 可用 {len(available_langs)} 个")

    ordered_langs = [l for l in COMMON_LANGUAGES if l in available_langs]
    ordered_langs += [l for l in available_langs if l not in ordered_langs]
    selected_langs = _settings.multiselect(
        "识别语言",
        options=ordered_langs,
        default=_saved_selection(f"oem{tess_oem}_languages", ordered_langs, ["eng"] if "eng" in available_langs else []),
        format_func=_lang_display,
        key=f"oem{tess_oem}_languages",
        help="可搜索并选择多个语言包；历史语言包根据 OEM 模式显示。",
    )

    if hidden:
        _settings.info(f"OEM {tess_oem} 下隐藏了 {len(hidden)} 个仅传统引擎语言：{', '.join(hidden)}")

    if not_installed:
        with _settings.expander(f"📦 未安装的官方语言包（{len(not_installed)} 个）"):
            st.caption("可通过 `apt install tesseract-ocr-<code>` 或下载 .traineddata 安装")
            for lang in not_installed:
                st.text(f"  {_lang_display(lang)}")

    custom_lang = _settings.text_input(
        "手动输入语言组合",
        value=_saved_value('custom_lang', "", kind='text'),
        help="覆盖上方选择，格式如 chi_sim+eng",
        key='custom_lang',
    )
elif backend == "Calamari（HTTP服务）":
    selected_langs = [calamari_model] if calamari_model else ["gt4histocr"]
    custom_lang = ""
elif backend == "PaddleOCR（本地全链路）":
    selected_langs = []
    custom_lang = ""
    _settings.caption("PaddleOCR 全链路使用 PaddleOCR 自身语言配置，不依赖 Tesseract 语言包。")
elif backend == "Chrome ScreenAI（本地）":
    selected_langs = []
    custom_lang = ""
    _settings.caption("Chrome ScreenAI 使用本地版面分析与识别模型，不依赖 Tesseract 语言包。")
else:
    selected_langs = []
    custom_lang = ""

# -- OCR 模式 --
_settings = _document_settings
_settings.subheader("已有文字处理")
MODE_LABELS = {
    "default": "默认（遇到可见文字报错）",
    "skip": "跳过（跳过已有文字的页面）",
    "force": "强制（光栅化全部内容重新 OCR）",
    "redo": "重做（删除旧 OCR 层重新识别）",
}
mode = _settings.selectbox(
    "处理模式",
    list(MODE_LABELS.keys()),
    format_func=lambda x: MODE_LABELS[x],
    index=_saved_index('mode', list(MODE_LABELS.keys()), 3),
    key='mode',
)
_screenai_sanitize = backend == "Chrome ScreenAI（本地）" and chrome_screenai_text_cleanup == "sanitize"
_strip_applies = _screenai_sanitize or mode in ("default", "redo")
strip_hidden_layer = _settings.checkbox(
    "OCR 前先移除旧的隐形文字层",
    value=True if _screenai_sanitize else _saved_value('strip_hidden_layer', True, kind='bool'),
    disabled=_screenai_sanitize or not _strip_applies,
    help="扫描图 + 上次 OCR 留下的隐形文字层：先去掉再重新识别，不会因“已有文字”而中止。"
    "只删不可见文字，且仅限背后有图片的页面；可见文字不动，原文件不改。"
    "跳过/强制模式下不适用（跳过保留原文字，强制本来就会栅格化）。",
    key='screenai_sanitize_hidden' if _screenai_sanitize else 'strip_hidden_layer',
)
if _screenai_sanitize:
    _settings.caption("已由 ScreenAI PDF 净化统一清除隐藏文字及注释，保留可见正文和原图。")
if not _strip_applies:
    _settings.caption("当前处理模式不使用此选项。")

# -- 预处理 --
with _settings.expander("预处理选项"):
    rotate_pages = st.checkbox(
        "自动旋转页面",
        value=_saved_value('rotate_pages', False, kind='bool'),
        key='rotate_pages',
    )
    deskew = st.checkbox(
        "纠偏",
        value=_saved_value('deskew', False, kind='bool'),
        key='deskew',
    )
    clean = st.checkbox(
        "OCR 前清理（unpaper）",
        value=_saved_value('clean', False, kind='bool'),
        key='clean',
    )
    clean_final = st.checkbox(
        "最终输出也清理",
        value=_saved_value('clean_final', False, kind='bool'),
        disabled=mode == "redo",
        key='clean_final',
    )
    remove_background = st.checkbox(
        "去除背景",
        value=_saved_value('remove_background', False, kind='bool'),
        disabled=mode == "redo",
        key='remove_background',
    )
    remove_vectors = st.checkbox(
        "去除矢量图形",
        value=_saved_value('remove_vectors', False, kind='bool'),
        key='remove_vectors',
    )
    if mode == "redo":
        st.caption("重做模式下，最终清理和去除背景不参与处理。")
    oversample = st.slider(
        "过采样 DPI",
        0,
        5000,
        _saved_value('oversample', 0, kind='number', minimum=0, maximum=5000),
        step=50,
        help="0 = 不过采样",
        key='oversample',
    )

# -- Tesseract 专属选项 --
if backend == "Tesseract":
    _settings = _engine_settings
    with _settings.expander("Tesseract 高级选项"):
        tess_psm = st.selectbox(
            "PSM（页面分割模式）",
            options=list(PSM_OPTIONS.keys()),
            format_func=lambda x: PSM_OPTIONS[x],
            index=_saved_index('tess_psm', list(PSM_OPTIONS.keys()), 2),
            key='tess_psm',
        )
        tess_threshold = st.selectbox(
            "二值化方法",
            ["auto", "otsu", "sauvola"],
            format_func=lambda x: {"auto": "自动", "otsu": "Otsu", "sauvola": "Sauvola"}[x],
            help="Sauvola 需要 Tesseract 5.0+",
            key='tess_threshold',
            index=_saved_index('tess_threshold', ["auto", "otsu", "sauvola"], 0),
        )
        tess_timeout = st.number_input(
            "超时时间（秒）",
            value=_saved_value('tess_timeout', 180.0, kind='number', minimum=10.0, maximum=None),
            min_value=10.0,
            key='tess_timeout',
        )
        tess_downsample = st.checkbox(
            "自动缩小超大图片",
            value=_saved_value('tess_downsample', True, kind='bool'),
            key='tess_downsample',
        )

# -- 输出设置 --
OUTPUT_MODE_OPTIONS = {
    "普通 PDF": "pdf",
    "尽量 PDF/A（失败则回退普通 PDF）": "auto",
    "严格 PDF/A": "pdfa",
}

_settings = _output_settings
with _settings.container():
    st.subheader("输出 PDF")
    output_mode = st.selectbox(
        "输出模式",
        list(OUTPUT_MODE_OPTIONS.keys()),
        index=_saved_index('output_mode', list(OUTPUT_MODE_OPTIONS.keys()), 0),
        help="普通 PDF：添加文字层；尽量 PDF/A：归档转换失败时保留普通 PDF；严格 PDF/A：要求归档格式。",
        key='output_mode',
    )
    output_type = OUTPUT_MODE_OPTIONS[output_mode]
    pdf_renderer = st.selectbox(
        "PDF 渲染器",
        ["auto", "hocr", "sandwich"],
        help="auto=自动选择, hocr=hOCR文字层, sandwich=Tesseract原生PDF",
        key='pdf_renderer',
        index=_saved_index('pdf_renderer', ["auto", "hocr", "sandwich"], 0),
    )
    optimize_level = st.selectbox(
        "优化级别",
        [1, 0, 2, 3],
        format_func=lambda x: {0: "0 - 关闭", 1: "1 - 无损（默认）",
                                2: "2 - 有损", 3: "3 - 激进"}[x],
        key='optimize_level',
        index=_saved_index('optimize_level', [1, 0, 2, 3], 0),
    )

# -- 性能设置 --
_settings = _performance_settings
_hardware = st.cache_data(ttl=30)(detect_hardware)()
_cpu_count = max(1, _hardware.get("cpu_logical_cores") or 1)
_effective_paddle_device = paddle_ocr_device
if paddle_ocr_device == "auto" or (paddle_ocr_device.startswith("gpu") and (not _paddle_probe or not _paddle_probe["device_available"]["gpu"])):
    _effective_paddle_device = "gpu:0" if _paddle_probe and not _paddle_probe["error"] and _paddle_probe["device_available"]["gpu"] else "cpu"
_paddle_configuration_invalid = backend == "PaddleOCR（本地全链路）" and (
    not _paddle_probe or bool(_paddle_probe["error"]) or
    (paddle_ocr_device.startswith("gpu") and not _paddle_probe["device_available"]["gpu"])
)
_perf_prefix = "performance_" + backend.split("（")[0] + "_"
_perf_keys = []
def _perf_key(name):
    key = _perf_prefix + name
    _perf_keys.append(key)
    return key

with _settings.container():
    st.subheader("性能配置")
    _memory = _hardware.get("memory_available_bytes")
    st.caption(f"CPU：{_cpu_count} 个逻辑核 · 可用内存：" +
               (f"{_memory / 1024**3:.1f} GiB" if _memory else "未读取到"))
    _mode_key = _perf_key("level")
    _levels = ["低", "中", "高", "高级自定义"]
    _saved_level = _user_cfg.get("settings", {}).get(_mode_key,
        _user_cfg.get("settings", {}).get(_perf_prefix + "mode", "中"))
    _saved_level = {"自动推荐": "中", "自定义": "高级自定义"}.get(_saved_level, _saved_level)
    _performance_mode = st.radio("性能档位", _levels,
        index=_levels.index(_saved_level) if _saved_level in _levels else 1, key=_mode_key,
        horizontal=True, help="低档减少资源占用，中档平衡系统余量，高档提高资源预算；高级自定义可逐项调整。")
    _level = {"低": "low", "中": "medium", "高": "high", "高级自定义": "medium"}[_performance_mode]
    _screenai_strategy = "page_parallel"
    if backend == "Chrome ScreenAI（本地）":
        _strategy_key = _perf_key("chrome_concurrency_mode")
        if _user_cfg.get("settings", {}).get(_strategy_key) == "native_direct":
            _user_cfg["settings"][_strategy_key] = "page_parallel"
            _user_cfg["settings"].setdefault(_perf_prefix + "chrome_page_workers", 1)
        _screenai_strategy = st.radio("并发策略", ["page_parallel", "file_parallel"],
            index=_saved_index(_strategy_key, ["page_parallel", "file_parallel"], 0), key=_strategy_key,
            format_func=lambda mode: {"page_parallel": "单文件内并发优先", "file_parallel": "多文件进程池并行"}[mode],
            help="单文件内并发：逐个文件处理，将资源分配给页段；多文件进程池：同时处理多个文件，每个文件内部顺序处理。")
    _language_count = len(set(filter(None, custom_lang.split("+")))) if custom_lang else max(1, len(selected_langs))
    _plan = recommend_performance(backend, _hardware, device=_effective_paddle_device,
        file_count=len(_staged_files(_ocr_state)), level=_level, tesseract_oem=tess_oem,
        language_count=_language_count, screenai_light=chrome_screenai_light,
        screenai_concurrency=_screenai_strategy)
    st.caption("档位按资源预算估算，并行度同时受 CPU 和可用内存限制。")
    st.caption("推荐依据：" + _plan["reason"])
    jobs, file_concurrency, use_threads = _plan["jobs"], _plan["file_concurrency"], _plan["use_threads"]
    chrome_concurrency_mode = _plan["chrome_concurrency_mode"] or "page_parallel"
    paddle_ocr_rec_batch_size = _plan["paddle_ocr_rec_batch_size"] or 16
    paddle_ocr_cpu_threads = _plan["paddle_ocr_cpu_threads"] or 0

    def _performance_slider(label, name, maximum, default, minimum=1):
        key = _perf_key(name)
        saved = _saved_value(key, default, kind="number", minimum=minimum)
        if saved > maximum:
            st.caption(f"{label}已按当前设备上限调整为 {maximum}。")
        if maximum == minimum:
            st.caption(f"{label}：{minimum}（当前设备上限）")
            return minimum
        return st.slider(label, minimum, maximum, min(saved, maximum), key=key)

    if backend == "Calamari（HTTP服务）":
        preflight_workers = _plan["preflight_workers"]
        preflight_chunk = _plan["preflight_chunk"]
    if _performance_mode == "高级自定义":
        if backend == "Chrome ScreenAI（本地）":
            jobs = file_concurrency = 1
            if chrome_concurrency_mode == "page_parallel":
                jobs = _performance_slider("单文件并行页段数", "chrome_page_workers", 24, _plan["chrome_page_workers"] or 1)
            elif chrome_concurrency_mode == "file_parallel":
                file_concurrency = _performance_slider("并行文件数", "chrome_file_concurrency", 24, _plan["chrome_file_concurrency"] or 1)
        else:
            jobs = _performance_slider("页面并行数（每文件）", "jobs", _cpu_count, jobs)
            key = _perf_key("use_threads")
            use_threads = st.checkbox("使用线程（而非进程）", value=_saved_value(key, use_threads, kind="bool"), key=key)
            if backend == "Tesseract":
                file_concurrency = _performance_slider("文件并发数", "file_concurrency", _cpu_count, file_concurrency)
        if backend == "PaddleOCR（本地全链路）":
            paddle_ocr_rec_batch_size = _performance_slider("识别批量大小", "paddle_ocr_rec_batch_size", 64, paddle_ocr_rec_batch_size)
            paddle_ocr_cpu_threads = _performance_slider("CPU 线程数（0=默认）", "paddle_ocr_cpu_threads", _cpu_count, paddle_ocr_cpu_threads, minimum=0)
        if backend == "PaddleOCR（本地全链路）" and _effective_paddle_device == "cpu":
            paddle_ocr_enable_mkldnn = st.checkbox("CPU 启用 MKL-DNN",
                value=_saved_value("paddle_ocr_enable_mkldnn", False, kind="bool"), key="paddle_ocr_enable_mkldnn")
        if backend == "Calamari（HTTP服务）" and perf_strategy == "preflight":
            preflight_chunk = _performance_slider("预渲染每批页数", "preflight_chunk", 200,
                _saved_value("preflight_chunk", preflight_chunk, kind="number", minimum=5, maximum=200), minimum=5)
            preflight_workers = _performance_slider("渲染并行线程数", "preflight_workers", min(10, _cpu_count),
                _saved_value("preflight_workers", preflight_workers, kind="number", minimum=1, maximum=10))
        if jobs * file_concurrency > _cpu_count:
            st.caption("当前总并行数超过 CPU 逻辑核数，资源竞争可能增加处理时间。")
    st.write(f"当前生效：文件并发 {file_concurrency} · 每文件页面/页段并行 {jobs}")
    if backend == "PaddleOCR（本地全链路）":
        st.caption(f"{paddle_ocr_version} · {_effective_paddle_device} · 识别批量 {paddle_ocr_rec_batch_size} · 每页 CPU 线程 {paddle_ocr_cpu_threads}")
    elif backend == "Tesseract":
        st.caption(f"OEM {tess_oem} · {_language_count} 种语言；内部线程由页面数和页面并行预算共同分配。")
    elif backend == "Calamari（HTTP服务）" and perf_strategy == "preflight":
        st.caption(f"本机预渲染：每批 {preflight_chunk} 页 · 渲染线程 {preflight_workers}；Calamari 识别模型：{calamari_model}")

# -- 输出存储 --
_default_output_dir = _user_cfg.get("output_dir", str(Path.home() / "OCRmyPDF_output"))
_default_suffix = _user_cfg.get("output_suffix", "_ocr")
_default_save = bool(_user_cfg.get("save_to_disk", True))
_settings = _output_settings
with _settings.expander("保存位置与命名", expanded=True):
    st.caption("修改后点击下方“保存配置”，下次启动恢复。路径属于运行 AIH-OCR 的电脑。")
    output_suffix = st.text_input(
        "输出文件名后缀",
        value=_saved_value('output_suffix', _default_suffix, kind='text'),
        help="添加在 .pdf 扩展名之前，可留空；点击“保存配置”后，下次启动沿用。"
        "不能含 \\ / : * ? \" < > | 等文件名非法字符。",
        key='output_suffix',
    )
    _suffix_invalid = bool(re.search(r'[\\/:*?"<>|\x00-\x1f]', output_suffix))
    if _suffix_invalid:
        st.error("后缀含文件名非法字符，请修改后保存。")
    save_to_disk = st.checkbox(
        "自动保存到磁盘",
        value=_saved_value('save_to_disk', _default_save, kind='bool'),
        key='save_to_disk',
    )
    output_dir = ""
    if save_to_disk:
        output_dir = st.text_input(
            "输出目录",
            value=_saved_value('output_dir', _default_output_dir, kind='text'),
            help="点击“保存配置”后，下次启动沿用。",
            key='output_dir',
        )

if _config_column.button("保存配置", type="primary", use_container_width=True, disabled=_suffix_invalid):
    _settings_to_save = dict(_user_cfg.get("settings", {}))
    for _key in _PERSISTED_WIDGET_KEYS + _perf_keys:
        if _key in st.session_state:
            _settings_to_save[_key] = st.session_state[_key]
    _new_cfg = {**_user_cfg, "settings": _settings_to_save,
                "output_suffix": output_suffix, "save_to_disk": save_to_disk}
    if save_to_disk and output_dir:
        _new_cfg["output_dir"] = output_dir
    if _save_config(_new_cfg):
        _config_column.success("配置已保存，下次启动自动恢复。")

with _work_area:
    # ---------------------------------------------------------------------------
    # 主区域 — Logo + 头部
    # ---------------------------------------------------------------------------
    st.subheader("01 · 添加文件")
    st.caption("支持 PDF 多选、拖放和分批追加。文件完整上传后进入待处理列表。")

    # 内嵌上传页：直连旁路上传服务，逐本顺序上传，字节由服务端流式落盘。
    # __PORT__/__TOKEN__ 由下方代码替换。注意 components.html 的 iframe 以
    # srcdoc 渲染，自身 window.location 是 about:srcdoc（hostname 为空），
    # 必须从父页面地址探测主机名，本机和局域网其他设备才能都直连。
    _UPLOAD_PAGE_HTML = r"""<!doctype html>
    <html><head><meta charset="utf-8">
    <style>
      html,body{margin:0;font-family:"Segoe UI","Microsoft YaHei",sans-serif;color:#29231D;}
      #drop{display:block;width:100%;border:1px dashed #B45309;border-radius:10px;background:#FFFDF9;color:#8A3D07;
            text-align:center;padding:20px 12px;cursor:pointer;font-size:14px;user-select:none;}
      #drop.hover{background:#FFF2DF;} #drop:focus-visible{outline:3px solid #B45309;outline-offset:2px;}
      #bar-wrap{height:8px;background:#F0E0D0;border-radius:4px;margin:10px 0 6px;overflow:hidden;display:none;}
      #bar{height:100%;width:0%;background:#B45309;border-radius:4px;transition:width .15s;}
      #status{font-size:12.5px;color:#62584F;line-height:1.5;max-height:56px;overflow:auto;}
      #status .err{color:#C0392B;}
      #status .ok{color:#2E7D32;}
    </style></head>
    <body>
    <button id="drop" type="button">选择 PDF 文件，或拖放到这里</button>
    <input id="picker" type="file" multiple accept=".pdf,application/pdf" style="display:none">
    <div id="bar-wrap" role="progressbar" aria-label="上传进度" aria-valuemin="0" aria-valuemax="100"><div id="bar"></div></div>
    <div id="status" role="status" aria-live="polite"></div>
    <script>
    var PORT = "__PORT__";
    var TOKEN = "__TOKEN__";
    var queue = [], totalBytes = 0, curLoaded = 0;
    var doneCount = 0, failCount = 0, finishedBytes = 0, busy = false;
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
      var p = (finishedBytes + curLoaded) / totalBytes * 100;
      barWrap.setAttribute("aria-valuenow", Math.min(100, p).toFixed(0));
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
        if (doneCount + failCount > 0){
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
          doneCount++;
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
        finishedBytes += f.size; curLoaded = 0; progress();
        pump();
      };
      xhr.onerror = function(){
        busy = false; failCount++;
        finishedBytes += f.size; curLoaded = 0; progress();
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
            height=140,
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
        uploads = _upload_server.uploading_count(_ocr_state)
        snapshot = (tuple(f["path"] for f in staged), uploads)
        if snapshot != st.session_state.get("aih_queue_snapshot"):
            st.session_state["aih_queue_snapshot"] = snapshot
            st.rerun()
        if uploads:
            st.info(f"正在上传 {uploads} 个文件，上传结束后可开始处理。")
        if _ocr_state["done"]:
            return
        if not staged:
            return
        total_mb = sum(f["size"] for f in staged) / (1024 * 1024)
        st.subheader(f"待处理文件 · {len(staged)} 个")
        st.caption(f"已上传 {total_mb:.1f} MB")
        st.dataframe(
            [{"文件名": f["name"], "大小 (MB)": f["size"] / (1024 * 1024)} for f in staged],
            width="stretch", hide_index=True,
        )
        if not _ocr_state["running"]:
            rm_cols = st.columns([3, 1])
            remove_sel = rm_cols[0].multiselect(
                "从暂存区移除所选文件",
                options=[f["path"] for f in staged],
                format_func=lambda path: next(f["name"] for f in staged if f["path"] == path),
            )
            if rm_cols[1].button("移除", disabled=not remove_sel):
                _remove_staged_files(_ocr_state, set(remove_sel))
                st.rerun()
        st.caption("页数在处理时统计，页码范围对本批全部文件生效。")


    _staging_list_fragment()

    staged_files = _staged_files(_ocr_state)
    _uploading = _upload_server.uploading_count(_ocr_state)
    if backend == "Tesseract":
        _language_summary = custom_lang or ", ".join(selected_langs) or "引擎默认语言"
    elif backend == "Calamari（HTTP服务）":
        _language_summary = calamari_model
    elif backend == "PaddleOCR（本地全链路）":
        _language_summary = paddle_ocr_lang
    elif backend == "Chrome ScreenAI（本地）":
        _language_summary = "本地模型自动识别"
    else:
        _language_summary = "不执行识别"
    _summary_current = {
        "OCR 后端": _BACKEND_LABELS.get(backend, backend),
        "语言 / 模型": _language_summary,
        "已有文字": MODE_LABELS[mode],
        "输出格式": output_mode,
        "保存位置": output_dir if save_to_disk and output_dir else "浏览器下载",
        "文件名后缀": output_suffix or "无",
        "页码范围": st.session_state.get("pages_filter", "") or "全部页面",
        "性能档位": _performance_mode,
        "并行配置": f"文件 {file_concurrency} · 页面 / 页段 {jobs}",
    }
    if backend == "Chrome ScreenAI（本地）":
        _summary_current["并发策略"] = {"page_parallel": "单文件内并发优先", "file_parallel": "多文件进程池并行"}[_screenai_strategy]
        _summary_current["文字层生成"] = "Chrome 原生文本层" if chrome_screenai_text_layer_mode == "native_direct" else "AIH 坐标匹配"
        _summary_current["处理前文字层"] = {"sanitize": "保留正文与原图，清除隐藏文字和注释", "rasterize": "栅格化后重新识别"}[chrome_screenai_text_cleanup]
    _summary_current["隐形文字层"] = "先移除" if strip_hidden_layer and _strip_applies else "按处理模式执行"
    _preprocessing = []
    for enabled, label in (
        (rotate_pages, "自动旋转"), (deskew, "纠偏"), (clean, "识别前清理"),
        (clean_final and mode != "redo", "最终清理"),
        (remove_background and mode != "redo", "去除背景"),
        (remove_vectors, "去除矢量图形"),
    ):
        if enabled:
            _preprocessing.append(label)
    _summary_current["预处理"] = "、".join(_preprocessing) or "不启用"
    _summary_current["PDF 渲染器"] = pdf_renderer
    _summary_current["PDF 优化级别"] = optimize_level
    if backend == "Tesseract":
        _summary_current["Tesseract 引擎模式"] = OEM_OPTIONS[tess_oem]
        _summary_current["页面分割"] = PSM_OPTIONS[tess_psm]
    elif backend == "PaddleOCR（本地全链路）":
        _summary_current["PaddleOCR 基线"] = paddle_ocr_version
        _summary_current["推理设备"] = paddle_ocr_device
    elif backend == "Calamari（HTTP服务）":
        _summary_current["识别模型"] = calamari_model
        _summary_current["行检测"] = LINE_DETECTOR_LABELS[calamari_line_detector]
        _summary_current["批处理策略"] = PERF_STRATEGIES[perf_strategy]
    st.subheader("02 · 本次处理方案", anchor="processing-plan")
    _summary = _summary_current
    if _ocr_state["running"] or _ocr_state["done"]:
        _summary = _ocr_state.get("run_summary", _summary_current)
        st.caption("任务启动时记录的配置")
    _core_summary_keys = {"OCR 后端", "语言 / 模型", "已有文字", "输出格式", "保存位置", "页码范围"}
    _summary_html = "".join(
        f'<div><span>{escape(key)}</span><strong>{escape(str(value))}</strong></div>'
        for key, value in _summary.items() if key in _core_summary_keys
    )
    st.markdown(f'<div class="aih-summary">{_summary_html}</div>', unsafe_allow_html=True)
    with st.expander("处理细节"):
        st.table([{ "配置": key, "值": str(value)} for key, value in _summary.items() if key not in _core_summary_keys])

    if (staged_files or _uploading) and not _ocr_state["done"]:
        # -- 页码过滤 --
        pages_filter = st.text_input(
            "页码过滤（可选）", value="", key="pages_filter",
            help="例如 1-5,8,10-12 — 对所有文件生效",
        )

        # -- 开始 / 停止 按钮 --
        btn_cols = st.columns([3, 1])
        start_clicked = btn_cols[0].button(
            "开始处理", type="primary", disabled=_ocr_state["running"] or _uploading > 0 or not staged_files or _suffix_invalid or _paddle_configuration_invalid,
            use_container_width=True,
        )
        if _ocr_state["running"]:
            if btn_cols[1].button("停止", type="secondary"):
                _ocr_state["stop"] = True
                st.rerun()

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
                ocr_kwargs["paddle_ocr_device"] = _effective_paddle_device if paddle_ocr_device == "auto" else paddle_ocr_device
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
            try:
                _reset_batch_for_run(batch_dir, staged_files)
            except _upload_server.UploadStateConflict as exc:
                st.error(str(exc))
                st.stop()
            _ocr_state["run_summary"] = {**_summary_current, "页码范围": pages_filter or "全部页面"}
            files_data = [(f["name"], f["path"]) for f in staged_files]

            # 重置状态
            _ocr_state["running"] = True
            _ocr_state["status"] = "running"
            _ocr_state["file_states"] = {}
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

    elif not _ocr_state["done"]:
        st.caption("添加文件后，开始处理按钮会在这里显示。")

    # ---------------------------------------------------------------------------
    # 处理状态显示区（从 session_state 读取，不受 rerun 影响）
    # ---------------------------------------------------------------------------
    if _ocr_state["running"] or _ocr_state["done"]:
        st.subheader("03 · 处理与结果")
        status = _ocr_state.get("status", "running" if _ocr_state["running"] else "completed")
        status_labels = {"running": "正在处理", "completed": "全部完成", "partial_failed": "部分文件失败", "failed": "处理失败", "stopped": "已停止"}
        if _ocr_state["running"]:
            if _ocr_state["stop"]:
                st.warning("正在停止。已运行的文件会结束当前处理后退出，外部子进程将接收停止请求。")
            else:
                st.info("正在按本次方案处理文件。")
        else:
            message = status_labels.get(status, status)
            if status == "completed":
                st.success(message)
            elif status in ("partial_failed", "stopped"):
                st.warning(message)
            else:
                st.error(message)
        st.progress(min(1.0, max(0.0, _ocr_state["progress"])), text=_ocr_state["progress_text"])
        file_states = list(_ocr_state.get("file_states", {}).values())
        if file_states:
            file_labels = {"pending": "等待处理", "running": "处理中", "completed": "已完成", "failed": "失败", "stopped": "已停止"}
            rows = []
            phase_labels = {
                "OCR": "文字识别", "Image processing": "图像处理",
                "Linearizing": "PDF 线性化", "PDF/A conversion": "PDF/A 转换",
                "Recompressing JPEGs": "JPEG 压缩", "Deflating JPEGs": "JPEG 优化",
                "PNGs": "PNG 优化", "JBIG2": "JBIG2 优化",
            }
            for file in file_states:
                page_progress = "—"
                if file.get("page_total"):
                    current = file.get("page_current", 0)
                    unit = file.get("progress_unit", "")
                    if unit == "%":
                        page_progress = f'{current / file["page_total"]:.0%}'
                    else:
                        unit_label = {"page": "页", "image": "图像"}.get(unit, unit)
                        page_progress = f'{current}/{file["page_total"]} {unit_label}'.strip()
                phase = file.get("page_desc", "") or "—"
                rows.append({
                    "文件": file["name"],
                    "状态": file_labels.get(file["status"], file["status"]),
                    "处理阶段": phase_labels.get(phase, phase),
                    "阶段进度": page_progress,
                    "说明": file.get("error", "") or "",
                })
            st.dataframe(rows, hide_index=True, use_container_width=True)
        with st.expander("处理日志", expanded=False):
            with st.container(height=260):
                for entry in list(_ocr_state["logs"]):
                    level = entry["type"]
                    if level == "success":
                        st.success(entry["msg"])
                    elif level == "warning":
                        st.warning(entry["msg"])
                    elif level == "info":
                        st.info(entry["msg"])
                    else:
                        st.error(entry["msg"])
        if _ocr_state["running"]:
            time.sleep(1)
            st.rerun()
        if _ocr_state["done"]:
            output_files = _ocr_state["outputs"]
            if output_files:
                st.subheader(f"处理结果 · {len(output_files)} 个文件")
                if _ocr_state["output_dir"]:
                    st.caption(f"已保存到运行 AIH-OCR 的电脑：{_ocr_state['output_dir']}")
                try:
                    if len(output_files) == 1:
                        name, output_path = output_files[0]
                        with Path(output_path).open("rb") as data:
                            st.download_button(f"下载 {name}", data=data, file_name=name, mime="application/pdf")
                    else:
                        zip_path = Path(_ocr_state["task_dir"]) / "ocrmypdf_output.zip"
                        if not zip_path.exists():
                            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                                for name, output_path in output_files:
                                    zf.write(output_path, arcname=name)
                        with zip_path.open("rb") as data:
                            st.download_button(f"下载全部（{len(output_files)} 个文件，ZIP）", data=data, file_name="ocrmypdf_output.zip", mime="application/zip")
                except OSError as exc:
                    st.error(f"结果文件读取失败：{exc}")
            if _ocr_state["errors"]:
                with st.expander(f"错误详情（{len(_ocr_state['errors'])} 项）"):
                    for fname, err in list(_ocr_state["errors"]):
                        st.text(f"{fname}：{err}")
            retry_files = _staged_files(_ocr_state)
            if retry_files:
                st.info(f"保留了 {len(retry_files)} 个未完成文件，可调整配置后重新处理。")
                if output_files:
                    st.caption("再次开始处理会替换本批下载缓存，请先下载已有结果。")
                if st.button("重新处理未完成文件"):
                    _ocr_state["done"] = False
                    st.rerun()
            if st.button("清理本批缓存"):
                _cleanup_task_dir(_ocr_state)
                _ocr_state.update(done=False, logs=[], outputs=[], errors=[], file_states={})
                st.rerun()

    st.markdown(
        '<div class="aih-footer">AIH-Infra · Baireinhold<br>'
        'OCRmyPDF · Tesseract · PaddleOCR · Calamari · Chrome ScreenAI</div>',
        unsafe_allow_html=True,
    )
