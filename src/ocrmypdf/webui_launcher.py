from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_STREAMLIT_PORT = 8501
MAX_PORT_ATTEMPTS = 1000
DEFAULT_STREAMLIT_HOST = "0.0.0.0"
DEFAULT_BROWSER_HOST = "localhost"
STARTING_LOCK_TTL_SECONDS = 120


def _iter_probe_hosts(host: str) -> tuple[str, ...]:
    if host in {"127.0.0.1", "localhost"}:
        return ("127.0.0.1", "localhost", "::1")
    if host == "0.0.0.0":
        return ("127.0.0.1", "localhost", "0.0.0.0")
    return (host,)


def _is_port_listening(host: str, port: int) -> bool:
    for probe_host in _iter_probe_hosts(host):
        try:
            addrinfos = socket.getaddrinfo(
                probe_host,
                port,
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
            )
        except socket.gaierror:
            continue

        for family, socktype, proto, _, sockaddr in addrinfos:
            with socket.socket(family, socktype, proto) as sock:
                sock.settimeout(0.2)
                if sock.connect_ex(sockaddr) == 0:
                    return True
    return False


def _can_bind_to_port(host: str, port: int) -> bool:
    try:
        addrinfos = socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror:
        return False

    for family, socktype, proto, _, sockaddr in addrinfos:
        with socket.socket(family, socktype, proto) as sock:
            if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                sock.bind(sockaddr)
            except OSError:
                return False
    return True


def _is_port_available(host: str, port: int) -> bool:
    if _is_port_listening(host, port):
        return False
    return _can_bind_to_port(host, port)


def _get_windows_excluded_port_ranges() -> list[tuple[int, int]]:
    if os.name != "nt":
        return []

    excluded_ranges: list[tuple[int, int]] = []
    pattern = re.compile(r"^\s*(\d+)\s+(\d+)\s*(?:\*.*)?$")
    for family in ("ipv4", "ipv6"):
        try:
            result = subprocess.run(
                [
                    "netsh",
                    "interface",
                    family,
                    "show",
                    "excludedportrange",
                    "protocol=tcp",
                ],
                capture_output=True,
                check=False,
            )
        except OSError:
            continue

        output = result.stdout.decode(errors="ignore") if result.stdout else ""
        for line in output.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            start, end = map(int, match.groups())
            excluded_ranges.append((start, end))
    return excluded_ranges


def _is_excluded_port(port: int, excluded_ranges: list[tuple[int, int]]) -> bool:
    return any(start <= port <= end for start, end in excluded_ranges)


def _get_windows_excluded_ports() -> set[int]:
    excluded_ports: set[int] = set()
    for start, end in _get_windows_excluded_port_ranges():
        excluded_ports.update(range(start, end + 1))
    return excluded_ports


def _get_start_port() -> int:
    raw_port = os.environ.get("OCRMYPDF_AIH_PORT", str(DEFAULT_STREAMLIT_PORT))
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise RuntimeError(f"Invalid OCRMYPDF_AIH_PORT: {raw_port!r}") from exc

    if not 1 <= port <= 65535:
        raise RuntimeError(f"OCRMYPDF_AIH_PORT out of range: {port}")
    return port


def _find_available_port(
    host: str = DEFAULT_STREAMLIT_HOST,
    start_port: int = DEFAULT_STREAMLIT_PORT,
    max_attempts: int = MAX_PORT_ATTEMPTS,
) -> int:
    excluded_ports = _get_windows_excluded_ports()
    for port in range(start_port, start_port + max_attempts):
        if port > 65535:
            break
        if port in excluded_ports:
            continue
        if _is_port_available(host, port):
            return port
    raise RuntimeError(
        f"No available Streamlit port found in range {start_port}-{min(start_port + max_attempts - 1, 65535)}."
    )


def _build_streamlit_cmd(webui_path: Path, host: str, port: int, browser_host: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(webui_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--browser.serverAddress",
        browser_host,
        "--browser.serverPort",
        str(port),
        "--server.headless",
        "false",
        "--server.fileWatcherType",
        "none",
    ]


def _project_lock_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha1(str(project_root).encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"aih-ocrmypdf-webui-{digest}.lock"


def _read_lock(lock_path: Path) -> dict:
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_lock_active(lock_path: Path, data: dict) -> bool:
    port = data.get("port")
    host = str(data.get("host") or DEFAULT_STREAMLIT_HOST)
    if isinstance(port, int) and _is_port_listening(host, port):
        return True
    try:
        age = time.time() - lock_path.stat().st_mtime
    except OSError:
        return False
    return port is None and age < STARTING_LOCK_TTL_SECONDS


def _write_lock(lock_path: Path, *, host: str, browser_host: str, port: int | None) -> None:
    payload = {
        "pid": os.getpid(),
        "host": host,
        "browser_host": browser_host,
        "port": port,
        "project_root": str(Path(__file__).resolve().parents[2]),
    }
    lock_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _acquire_single_instance_lock(host: str, browser_host: str) -> Path | None:
    lock_path = _project_lock_path()
    if lock_path.exists():
        data = _read_lock(lock_path)
        if _is_lock_active(lock_path, data):
            port = data.get("port")
            if isinstance(port, int):
                print(f"[AIH-OCRmyPDF] Web UI is already running at http://{browser_host}:{port}")
            else:
                print("[AIH-OCRmyPDF] Web UI is already starting. Please use the existing window.")
            return None
        try:
            lock_path.unlink()
        except OSError:
            pass

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lock_path), flags)
    except FileExistsError:
        print("[AIH-OCRmyPDF] Web UI is already starting. Please use the existing window.")
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        payload = {
            "pid": os.getpid(),
            "host": host,
            "browser_host": browser_host,
            "port": None,
            "project_root": str(Path(__file__).resolve().parents[2]),
        }
        json.dump(payload, f, ensure_ascii=False)
    return lock_path


def main() -> int:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    webui_path = Path(__file__).with_name("webui.py")
    host = os.environ.get("OCRMYPDF_AIH_HOST", DEFAULT_STREAMLIT_HOST)
    browser_host = os.environ.get("OCRMYPDF_AIH_BROWSER_HOST", DEFAULT_BROWSER_HOST)
    start_port = _get_start_port()
    last_error: RuntimeError | None = None

    for retry in range(2):
        port = _find_available_port(host=host, start_port=start_port + retry)
        if host == "0.0.0.0":
            print(f"[AIH-OCRmyPDF] Web UI listening on all interfaces at 0.0.0.0:{port}")
            print(f"[AIH-OCRmyPDF] Open locally at http://{browser_host}:{port}")
            print(f"[AIH-OCRmyPDF] Also available at http://127.0.0.1:{port}")
        else:
            print(f"[AIH-OCRmyPDF] Web UI starting at http://{host}:{port}")
            if host == "127.0.0.1":
                print(f"[AIH-OCRmyPDF] Also available at http://localhost:{port}")

        cmd = _build_streamlit_cmd(webui_path, host, port, browser_host)
        rc = subprocess.call(cmd)
        if rc == 0:
            return rc

        port_still_unavailable = not _is_port_available(host, port)
        if retry == 0 and port_still_unavailable:
            print(
                f"[AIH-OCRmyPDF] Streamlit exited with code {rc}, and port {port} is no longer available. "
                "Retrying with the next port."
            )
            last_error = RuntimeError(
                f"Streamlit failed to bind selected port {port}, retrying."
            )
            continue

        print(
            f"[AIH-OCRmyPDF] Streamlit exited with code {rc}. "
            "Not retrying automatically because this does not look like a port conflict."
        )
        return rc

    raise last_error if last_error else RuntimeError("Failed to launch Web UI.")


if __name__ == "__main__":
    raise SystemExit(main())
