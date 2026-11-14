# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import builtins
import importlib
import threading
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("AIH_OCRMYPDF_CONFIG", str(tmp_path / "webui-config.json"))


@pytest.fixture
def webui(_isolated_config):
    pytest.importorskip("streamlit")
    return importlib.import_module("ocrmypdf.webui")


def _state(tmp_path, files):
    task_dir = tmp_path / "batch"
    inputs_dir = task_dir / "inputs"
    (task_dir / "outputs").mkdir(parents=True)
    inputs_dir.mkdir(parents=True)
    files_data = []
    for name in files:
        path = inputs_dir / name
        path.write_bytes(b"input")
        files_data.append((name, str(path)))
    state = {
        "task_dir": str(task_dir),
        "stop": False,
        "outputs": [],
        "errors": [],
        "logs": [],
        "progress": 0.0,
    }
    return state, files_data


def test_parallel_progress_counts_completed_files_not_input_index(webui, tmp_path, monkeypatch):
    state, files_data = _state(tmp_path, ["slow.pdf", "quick.pdf"])
    first_done = threading.Event()
    observed_progress = []

    class RecordingState(dict):
        def __setitem__(self, key, value):
            if key == "progress":
                observed_progress.append(value)
                if value >= 0.5:
                    first_done.set()
            super().__setitem__(key, value)

    state = RecordingState(state)

    def fake_ocr_one_file(shared_state, input_path, fname, out_name, *_args):
        file_state = shared_state["file_states"][input_path]
        if fname == "slow.pdf":
            assert first_done.wait(5)
            assert shared_state["progress"] == 0.5
        file_state["status"] = "completed"

    retained_calls = []
    monkeypatch.setattr(webui, "_ocr_one_file", fake_ocr_one_file)
    monkeypatch.setattr(webui._upload_server, "mark_done", lambda _path, retained=None: retained_calls.append(retained))

    webui._run_ocr_batch(state, files_data, {}, "Tesseract", "_ocr", False, "", 2)

    assert observed_progress[1] == 0.5
    assert state["completed_count"] == 2
    assert state["progress"] == 1.0
    assert state["status"] == "completed"
    assert all(item["status"] == "completed" for item in state["file_states"].values())
    assert retained_calls == [set()]
    assert all(not Path(path).exists() for _name, path in files_data)


def test_stop_keeps_unfinished_file_and_never_reports_100_percent(webui, tmp_path, monkeypatch):
    state, files_data = _state(tmp_path, ["done.pdf", "waiting.pdf"])
    retained_calls = []

    def fake_ocr_one_file(shared_state, input_path, *_args):
        shared_state["file_states"][input_path]["status"] = "completed"
        shared_state["stop"] = True

    monkeypatch.setattr(webui, "_ocr_one_file", fake_ocr_one_file)
    monkeypatch.setattr(webui._upload_server, "mark_done", lambda _path, retained=None: retained_calls.append(retained))

    webui._run_ocr_batch(state, files_data, {}, "Tesseract", "_ocr", False, "", 1)

    assert state["status"] == "stopped"
    assert state["progress"] == 0.5
    assert state["progress_text"].startswith("已停止：")
    assert state["file_states"][files_data[0][1]]["status"] == "completed"
    assert state["file_states"][files_data[1][1]]["status"] == "stopped"
    assert retained_calls == [{files_data[1][1]}]


def test_batch_exception_is_recorded_and_input_is_retryable(webui, tmp_path, monkeypatch):
    state, files_data = _state(tmp_path, ["broken.pdf"])
    retained_calls = []

    def fail_ocr(*_args):
        raise RuntimeError("worker setup failed")

    monkeypatch.setattr(webui, "_ocr_one_file", fail_ocr)
    monkeypatch.setattr(webui._upload_server, "mark_done", lambda _path, retained=None: retained_calls.append(retained))

    webui._run_ocr_batch(state, files_data, {}, "Tesseract", "_ocr", False, "", 1)

    assert state["status"] == "failed"
    assert state["progress"] < 1.0
    assert any("批量处理异常" in error[1] for error in state["errors"])
    assert state["file_states"][files_data[0][1]]["status"] == "failed"
    assert retained_calls == [{files_data[0][1]}]


def test_file_failure_keeps_processed_count_and_failed_status(webui, tmp_path, monkeypatch):
    state, files_data = _state(tmp_path, ["failed.pdf"])

    def fail_one_file(shared_state, input_path, fname, *_args):
        file_state = shared_state["file_states"][input_path]
        file_state["status"] = "failed"
        file_state["error"] = "OCR failed"
        shared_state["errors"].append((fname, "OCR failed"))

    monkeypatch.setattr(webui, "_ocr_one_file", fail_one_file)
    monkeypatch.setattr(webui._upload_server, "mark_done", lambda *_args, **_kwargs: None)

    webui._run_ocr_batch(state, files_data, {}, "Tesseract", "_ocr", False, "", 1)

    assert state["status"] == "failed"
    assert state["progress"] == 1.0
    assert state["completed_count"] == 1
    assert not state["progress_text"].startswith("全部文件处理完成")


def test_progress_bar_captures_the_current_file_state(webui):
    from ocrmypdf import aih_progress_plugin

    states = [
        {"page_current": 0, "page_total": 0, "page_desc": ""},
        {"page_current": 0, "page_total": 0, "page_desc": ""},
    ]
    barrier = threading.Barrier(2)

    def update_state(state, current):
        local = builtins.__dict__.setdefault(
            "_ocrmypdf_page_progress_state_local", threading.local()
        )
        local.state = state
        progress = aih_progress_plugin._StateProgressBar(total=10, desc="pages", unit="page")
        barrier.wait(timeout=5)
        progress.update(completed=current)
        del local.state

    threads = [
        threading.Thread(target=update_state, args=(states[0], 3)),
        threading.Thread(target=update_state, args=(states[1], 7)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert states[0]["page_current"] == 3
    assert states[1]["page_current"] == 7
    assert all(state["progress_unit"] == "page" for state in states)
    assert all(not thread.is_alive() for thread in threads)


def test_explicit_pdfa_failure_is_not_marked_completed(webui, tmp_path, monkeypatch):
    import ocrmypdf

    state, files_data = _state(tmp_path, ["pdfa.pdf"])
    input_path = files_data[0][1]
    state["file_states"] = {
        input_path: {
            "name": "pdfa.pdf",
            "status": "pending",
            "output": None,
            "error": None,
            "page_current": 0,
            "page_total": 0,
        }
    }

    def fail_pdfa(_input, output_path, **_kwargs):
        Path(output_path).write_bytes(b"ordinary pdf")
        return ocrmypdf.ExitCode.pdfa_conversion_failed

    monkeypatch.setattr(ocrmypdf, "ocr", fail_pdfa)
    webui._ocr_one_file(
        state,
        input_path,
        "pdfa.pdf",
        "pdfa_ocr.pdf",
        {"output_type": "pdfa"},
        "Tesseract",
        False,
        "",
    )

    file_state = state["file_states"][input_path]
    assert file_state["status"] == "failed"
    assert "PDF/A 转换失败" in file_state["error"]
    assert Path(file_state["output"]).exists()


def test_empty_engine_exception_keeps_diagnostic(webui, tmp_path, monkeypatch):
    import ocrmypdf

    state, files_data = _state(tmp_path, ["broken.pdf"])
    input_path = files_data[0][1]

    def fail_without_message(*args, **kwargs):
        raise ValueError()

    monkeypatch.setattr(ocrmypdf, "ocr", fail_without_message)
    webui._ocr_one_file(state, input_path, "broken.pdf", "out.pdf", {}, "Tesseract", False, "")
    assert state["file_states"][input_path]["error"] == "ValueError"
    assert state["errors"] == [("broken.pdf", "ValueError")]


def test_input_error_explains_recovery_and_keeps_diagnostic(webui, tmp_path, monkeypatch):
    import ocrmypdf
    from ocrmypdf.exceptions import InputFileError

    state, files_data = _state(tmp_path, ["broken.pdf"])
    input_path = files_data[0][1]

    def fail_input(*args, **kwargs):
        raise InputFileError()

    monkeypatch.setattr(ocrmypdf, "ocr", fail_input)
    webui._ocr_one_file(state, input_path, "broken.pdf", "out.pdf", {}, "Tesseract", False, "")
    assert "文件完整性" in state["file_states"][input_path]["error"]
    assert state["errors"] == [("broken.pdf", "InputFileError")]
