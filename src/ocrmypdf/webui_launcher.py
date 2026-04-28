from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
from pathlib import Path

DEFAULT_STREAMLIT_PORT = 8501
MAX_PORT_ATTEMPTS = 1000
DEFAULT_STREAMLIT_HOST = "127.0.0.1"


def _iter_probe_hosts(host: str) -> tuple[str, ...]:
    if host in {"127.0.0.1", "localhost"}:
        return ("127.0.0.1", "localhost", "::1")
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


def _build_streamlit_cmd(webui_path: Path, host: str, port: int) -> list[str]:
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
        "--server.headless",
        "false",
    ]


def main() -> int:
    webui_path = Path(__file__).with_name("webui.py")
    host = os.environ.get("OCRMYPDF_AIH_HOST", DEFAULT_STREAMLIT_HOST)
    start_port = _get_start_port()
    last_error: RuntimeError | None = None

    for retry in range(2):
        port = _find_available_port(host=host, start_port=start_port + retry)
        print(f"[OCRmyPDF-AIH] Web UI starting at http://{host}:{port}")
        if host == "127.0.0.1":
            print(f"[OCRmyPDF-AIH] Also available at http://localhost:{port}")

        cmd = _build_streamlit_cmd(webui_path, host, port)
        rc = subprocess.call(cmd)
        if rc == 0 or retry == 1:
            return rc
        last_error = RuntimeError(f"Streamlit failed to bind selected port {port}, retrying.")

    raise last_error if last_error else RuntimeError("Failed to launch Web UI.")


if __name__ == "__main__":
    raise SystemExit(main())
