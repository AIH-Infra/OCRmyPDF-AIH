# SPDX-FileCopyrightText: 2025 James R. Barlow
# SPDX-License-Identifier: AGPL-3.0-or-later

"""旁路流式上传服务 + 磁盘暂存缓存（Web UI 专用）.

``st.file_uploader`` 的上传字节全部缓存在 Streamlit 进程内存
（MemoryUploadedFileManager）里，脚本代码运行时文件早已完整驻留
RAM——批量上传几十上百本书会直接把内存撑爆，且释放时机不受控。

本模块在 Streamlit 进程内另起一个只服务上传的 HTTP 线程：请求体按
固定块大小边收边写进批次 ``inputs/``，服务器内存占用与书的数量、
大小无关。上传页通过 ``components.html`` 内嵌 iframe 直连该端口。

同时集中管理磁盘暂存缓存（batch 目录 + manifest.json）：Web UI 线程
与上传线程共享同一把 manifest 锁，写入原子落盘（tmp + os.replace）。
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

log = logging.getLogger(__name__)

CACHE_ROOT = Path(tempfile.gettempdir()) / "ocrmypdf-webui-cache"
MANIFEST_NAME = "manifest.json"
ORPHAN_MAX_AGE_DAYS = 3        # 崩溃/刷新遗留批次的最大保留天数
MIN_FREE_DISK_GB = 5.0         # 每次上传前要求的磁盘剩余空间
MAX_UPLOAD_BYTES = 4 * 1024 ** 3   # 单文件上限 4 GiB
STREAM_CHUNK = 256 * 1024      # 边收边写的块大小
_DRAIN_LIMIT = 8 * 1024 * 1024  # 错误响应前排干未读请求体的上限
DEFAULT_PORT = 8642            # 固定端口便于防火墙一次性放行；占用则回退随机

_manifest_lock = threading.RLock()
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()
_swept = False

_server: ThreadingHTTPServer | None = None
_server_lock = threading.Lock()
_server_port = 0


# ---------------------------------------------------------------------------
# manifest / 批次目录
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def load_manifest(batch_dir: Path) -> dict:
    try:
        with _manifest_lock:
            return json.loads((Path(batch_dir) / MANIFEST_NAME).read_text("utf-8"))
    except Exception:
        return {"created": "", "status": "staging", "files": []}


def save_manifest(batch_dir: Path, manifest: dict) -> None:
    batch_dir = Path(batch_dir)
    with _manifest_lock:
        tmp = batch_dir / (MANIFEST_NAME + ".tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
        os.replace(tmp, batch_dir / MANIFEST_NAME)


def sweep_orphans() -> None:
    """删除超龄或残缺的批次目录，防止异常退出遗留的缓存永久占用磁盘。"""
    global _swept
    with _manifest_lock:
        if _swept:
            return
        _swept = True
    try:
        if not CACHE_ROOT.exists():
            return
        now = time.time()
        for batch_dir in CACHE_ROOT.iterdir():
            if not batch_dir.is_dir():
                continue
            try:
                fresh = now - batch_dir.stat().st_mtime < ORPHAN_MAX_AGE_DAYS * 86400
                if not fresh or not (batch_dir / MANIFEST_NAME).exists():
                    shutil.rmtree(batch_dir, ignore_errors=True)
            except OSError:
                continue
    except OSError:
        pass


def get_batch_dir(state: dict, create: bool = False) -> Path | None:
    """当前会话的批次暂存目录；首次上传时创建，跨 rerun 存活直到清除。

    线程安全：上传线程与 UI 线程都可能触发创建，锁内二次检查防止
    并发时各自建出多个批次目录。
    """
    task_dir = state.get("task_dir")
    if task_dir and (Path(task_dir) / MANIFEST_NAME).exists():
        batch = Path(task_dir)
        if create:
            (batch / "inputs").mkdir(parents=True, exist_ok=True)
            (batch / "outputs").mkdir(parents=True, exist_ok=True)
        return batch
    if not create:
        return None
    with _manifest_lock:
        task_dir = state.get("task_dir")
        if task_dir and (Path(task_dir) / MANIFEST_NAME).exists():
            batch = Path(task_dir)
            if create:
                (batch / "inputs").mkdir(parents=True, exist_ok=True)
                (batch / "outputs").mkdir(parents=True, exist_ok=True)
            return batch
        batch = CACHE_ROOT / f"batch_{time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"
        (batch / "inputs").mkdir(parents=True, exist_ok=True)
        (batch / "outputs").mkdir(parents=True, exist_ok=True)
        save_manifest(batch, {
            "created": _now(),
            "status": "staging",
            "files": [],
        })
        state["task_dir"] = str(batch)
        return batch


def list_staged(state: dict) -> list[dict]:
    """manifest 中仍存在于磁盘上的暂存文件——排队列表的唯一事实来源。"""
    batch_dir = get_batch_dir(state)
    if batch_dir is None:
        return []
    files = load_manifest(batch_dir).get("files", [])
    return [f for f in files if Path(f["path"]).exists()]


def reserve_input(batch_dir: Path, name: str, size: int) -> Path:
    """在 manifest 中预留下一个输入槽位（序号前缀保证唯一），返回磁盘路径。

    调用者随后把字节流写入该路径；成功后 ``commit_input``，失败则
    ``abort_input``。先预留再写，保证并发上传不会撞序号。
    """
    safe_name = Path(name).name.strip() or "upload.pdf"
    inputs_dir = Path(batch_dir) / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    with _manifest_lock:
        manifest = load_manifest(batch_dir)
        files = manifest.setdefault("files", [])
        dest = inputs_dir / f"{len(files):06d}_{safe_name}"
        files.append({
            "name": safe_name,
            "path": str(dest),
            "size": size,
            "staged_at": _now(),
        })
        manifest["status"] = "staging"
        save_manifest(batch_dir, manifest)
    return dest


def commit_input(batch_dir: Path, dest: Path, actual_size: int) -> None:
    with _manifest_lock:
        manifest = load_manifest(batch_dir)
        for entry in manifest.get("files", []):
            if entry["path"] == str(dest):
                entry["size"] = actual_size
                break
        save_manifest(batch_dir, manifest)


def abort_input(batch_dir: Path, dest: Path) -> None:
    with _manifest_lock:
        try:
            Path(dest).unlink()
        except OSError:
            pass
        manifest = load_manifest(batch_dir)
        manifest["files"] = [f for f in manifest.get("files", []) if f["path"] != str(dest)]
        save_manifest(batch_dir, manifest)


def remove_staged(state: dict, paths: set[str]) -> None:
    """从暂存区删除指定文件（按磁盘路径匹配，避免同名歧义）。"""
    batch_dir = get_batch_dir(state)
    if batch_dir is None:
        return
    with _manifest_lock:
        manifest = load_manifest(batch_dir)
        kept = []
        for f in manifest.get("files", []):
            if f["path"] in paths:
                try:
                    Path(f["path"]).unlink()
                except OSError:
                    pass
            else:
                kept.append(f)
        manifest["files"] = kept
        save_manifest(batch_dir, manifest)


def mark_running(batch_dir: Path, staged: list[dict]) -> None:
    """开始新一轮处理：清掉上一轮残留输出，manifest 转入 running。"""
    batch_dir = Path(batch_dir)
    outputs_dir = batch_dir / "outputs"
    if outputs_dir.exists():
        for old in outputs_dir.iterdir():
            try:
                old.unlink()
            except OSError:
                pass
    try:
        (batch_dir / "ocrmypdf_output.zip").unlink()
    except OSError:
        pass
    with _manifest_lock:
        manifest = load_manifest(batch_dir)
        manifest["status"] = "running"
        manifest["files"] = staged
        save_manifest(batch_dir, manifest)


def mark_done(batch_dir: Path) -> None:
    """整批结束（含停止/出错）：manifest 置 done、文件清单清空。

    inputs 目录由调用方删除。
    """
    with _manifest_lock:
        manifest = load_manifest(batch_dir)
        manifest["status"] = "done"
        manifest["files"] = []
        save_manifest(batch_dir, manifest)


def cleanup_batch(state: dict) -> None:
    """删除当前任务拥有的整个磁盘暂存批次（inputs/outputs/manifest）。"""
    task_dir = state.get("task_dir")
    if task_dir:
        shutil.rmtree(task_dir, ignore_errors=True)
    state["task_dir"] = ""


# ---------------------------------------------------------------------------
# 会话令牌（iframe JS ↔ 上传线程 ↔ 会话 state dict）
# ---------------------------------------------------------------------------

def mint_token() -> str:
    return secrets.token_urlsafe(24)


def register_session(token: str, state: dict) -> None:
    """把上传令牌绑定到会话 state dict；Web UI 每次 rerun 重新登记。"""
    with _sessions_lock:
        now = time.time()
        for tok in [t for t, s in _sessions.items() if now - s["seen"] > 86400]:
            _sessions.pop(tok, None)
        _sessions[token] = {"state": state, "seen": now}


def _touch_session(token: str) -> dict | None:
    with _sessions_lock:
        session = _sessions.get(token)
        if session is not None:
            session["seen"] = time.time()
            return session["state"]
    return None


# ---------------------------------------------------------------------------
# 旁路 HTTP 上传服务
# ---------------------------------------------------------------------------

class _UploadHandler(BaseHTTPRequestHandler):
    server_version = "AIHUploadServer/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # 静默访问日志
        pass

    def _drain_request_body(self) -> None:
        """错误响应前排干未读的请求体。

        带 keep-alive 的 HTTP/1.1 下，带着未读数据关闭套接字会触发 RST，
        把已经发出的错误响应也一并摧毁（浏览器表现为网络错误而非错误
        提示）。这里最多排干 _DRAIN_LIMIT，超出则照常关闭。
        """
        try:
            remaining = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return
        drained = 0
        while remaining > 0 and drained < _DRAIN_LIMIT:
            chunk = self.rfile.read(min(STREAM_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            drained += len(chunk)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if code >= 400:
            self._drain_request_body()
            self.close_connection = True
        self.send_response(code)
        if code >= 400:
            self.send_header("Connection", "close")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):  # CORS 预检
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            self._do_upload()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # 上传线程绝不能带崩整个服务
            log.warning("upload handler error: %s", exc, exc_info=True)
            try:
                self._send(500, {"error": str(exc)})
            except Exception:
                pass

    def _do_upload(self):
        url = urlparse(self.path)
        if url.path != "/upload":
            return self._send(404, {"error": "not found"})
        query = parse_qs(url.query)
        token = (query.get("token") or [""])[0]
        name = (query.get("name") or [""])[0]
        state = _touch_session(token)
        if state is None:
            return self._send(403, {"error": "上传令牌无效，请刷新页面重试"})
        if state.get("running"):
            return self._send(409, {"error": "处理进行中，暂不能上传"})

        length_header = self.headers.get("Content-Length")
        if not length_header:
            return self._send(411, {"error": "缺少 Content-Length"})
        length = int(length_header)
        if length <= 0:
            return self._send(400, {"error": "空文件"})
        if length > MAX_UPLOAD_BYTES:
            return self._send(413, {"error": f"文件超过上限 {MAX_UPLOAD_BYTES // (1024**3)} GiB"})

        batch_dir = get_batch_dir(state, create=True)
        free_gb = shutil.disk_usage(str(batch_dir)).free / (1024 ** 3)
        if free_gb < MIN_FREE_DISK_GB:
            return self._send(507, {
                "error": f"磁盘剩余空间不足（{free_gb:.1f} GB < {MIN_FREE_DISK_GB:.0f} GB），请清理后重试",
            })

        dest = reserve_input(batch_dir, name, length)
        committed = False
        try:
            with dest.open("wb") as target:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(STREAM_CHUNK, remaining))
                    if not chunk:
                        raise ConnectionError("客户端中断上传")
                    target.write(chunk)
                    remaining -= len(chunk)
            commit_input(batch_dir, dest, length)
            committed = True
        finally:
            if not committed:
                abort_input(batch_dir, dest)

        staged_count = len(load_manifest(batch_dir).get("files", []))
        self._send(200, {"ok": True, "staged": staged_count})


def ensure_server() -> int:
    """惰性启动旁路上传服务（进程内单例），返回实际监听端口。"""
    global _server, _server_port
    if _server is not None:
        return _server_port
    with _server_lock:
        if _server is not None:
            return _server_port
        try:
            port = int(os.environ.get("OCRMYPDF_WEBUI_UPLOAD_PORT", str(DEFAULT_PORT)))
        except ValueError:
            port = DEFAULT_PORT
        try:
            httpd = ThreadingHTTPServer(("0.0.0.0", port), _UploadHandler)
        except OSError:
            log.warning("upload port %d busy, falling back to ephemeral", port)
            httpd = ThreadingHTTPServer(("0.0.0.0", 0), _UploadHandler)
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True,
                         name="aih-upload-server").start()
        _server = httpd
        _server_port = httpd.server_address[1]
        log.info("upload server listening on port %d", _server_port)
        return _server_port
