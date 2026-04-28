# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import sys

import pytest

from ocrmypdf.webui_launcher import (
    DEFAULT_STREAMLIT_HOST,
    DEFAULT_STREAMLIT_PORT,
    _build_streamlit_cmd,
    _find_available_port,
    _get_start_port,
    _iter_probe_hosts,
    _is_port_available,
)


def test_find_available_port_prefers_default_port(monkeypatch):
    monkeypatch.setattr(
        "ocrmypdf.webui_launcher._get_windows_excluded_ports",
        lambda: set(),
    )
    monkeypatch.setattr(
        "ocrmypdf.webui_launcher._is_port_available",
        lambda host, port: port == DEFAULT_STREAMLIT_PORT,
    )

    assert _find_available_port() == DEFAULT_STREAMLIT_PORT


def test_iter_probe_hosts_checks_all_localhost_aliases():
    assert _iter_probe_hosts("127.0.0.1") == ("127.0.0.1", "localhost", "::1")
    assert _iter_probe_hosts("localhost") == ("127.0.0.1", "localhost", "::1")
    assert _iter_probe_hosts("0.0.0.0") == ("0.0.0.0",)



def test_find_available_port_increments_when_default_is_busy(monkeypatch):
    busy_ports = {8501, 8502}
    monkeypatch.setattr(
        "ocrmypdf.webui_launcher._get_windows_excluded_ports",
        lambda: set(),
    )
    monkeypatch.setattr(
        "ocrmypdf.webui_launcher._is_port_available",
        lambda host, port: port not in busy_ports,
    )

    assert _find_available_port() == 8503



def test_find_available_port_skips_windows_excluded_ports(monkeypatch):
    monkeypatch.setattr(
        "ocrmypdf.webui_launcher._get_windows_excluded_ports",
        lambda: {8501, 8502, 8503},
    )
    monkeypatch.setattr(
        "ocrmypdf.webui_launcher._is_port_available",
        lambda host, port: True,
    )

    assert _find_available_port() == 8504



def test_find_available_port_raises_when_no_port_is_available(monkeypatch):
    monkeypatch.setattr(
        "ocrmypdf.webui_launcher._get_windows_excluded_ports",
        lambda: set(),
    )
    monkeypatch.setattr(
        "ocrmypdf.webui_launcher._is_port_available",
        lambda host, port: False,
    )

    with pytest.raises(RuntimeError, match="No available Streamlit port found"):
        _find_available_port(max_attempts=3)


def test_is_port_available_rejects_port_with_existing_listener(monkeypatch):
    monkeypatch.setattr(
        "ocrmypdf.webui_launcher._is_port_listening",
        lambda host, port: True,
    )
    monkeypatch.setattr(
        "ocrmypdf.webui_launcher._can_bind_to_port",
        lambda host, port: True,
    )

    assert _is_port_available(DEFAULT_STREAMLIT_HOST, DEFAULT_STREAMLIT_PORT) is False


def test_is_port_available_requires_bindable_port(monkeypatch):
    monkeypatch.setattr(
        "ocrmypdf.webui_launcher._is_port_listening",
        lambda host, port: False,
    )
    monkeypatch.setattr(
        "ocrmypdf.webui_launcher._can_bind_to_port",
        lambda host, port: False,
    )

    assert _is_port_available(DEFAULT_STREAMLIT_HOST, DEFAULT_STREAMLIT_PORT) is False



def test_get_start_port_from_environment(monkeypatch):
    monkeypatch.setenv("OCRMYPDF_AIH_PORT", "8600")
    assert _get_start_port() == 8600



def test_get_start_port_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("OCRMYPDF_AIH_PORT", "bad-port")
    with pytest.raises(RuntimeError, match="Invalid OCRMYPDF_AIH_PORT"):
        _get_start_port()



def test_build_streamlit_cmd_includes_selected_port_and_browser_flag(tmp_path):
    webui_path = tmp_path / "webui.py"
    cmd = _build_streamlit_cmd(webui_path, DEFAULT_STREAMLIT_HOST, 8504)

    assert cmd[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert "--server.address" in cmd
    assert cmd[cmd.index("--server.address") + 1] == DEFAULT_STREAMLIT_HOST
    assert "--server.port" in cmd
    assert cmd[cmd.index("--server.port") + 1] == "8504"
    assert "--server.headless" in cmd
    assert cmd[cmd.index("--server.headless") + 1] == "false"
