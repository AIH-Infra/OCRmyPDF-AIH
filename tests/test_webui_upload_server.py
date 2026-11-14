# SPDX-License-Identifier: MPL-2.0
"""Regression checks for the disk-backed Web UI upload manifest."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import pytest

from ocrmypdf import aih_webui_upload_server as upload_server


@pytest.fixture
def batch(tmp_path):
    batch_dir = tmp_path / "batch"
    (batch_dir / "inputs").mkdir(parents=True)
    (batch_dir / "outputs").mkdir()
    upload_server.save_manifest(batch_dir, {
        "created": "test",
        "status": "staging",
        "files": [],
    })
    return batch_dir


def _state(batch_dir):
    return {"task_dir": str(batch_dir)}


def _ready_file(batch_dir, name="doc.pdf", data=b"pdf"):
    path = upload_server.reserve_input(batch_dir, name, len(data))
    path.write_bytes(data)
    upload_server.commit_input(batch_dir, path, len(data))
    return path


def test_reserved_upload_is_counted_but_not_staged_until_commit(batch):
    state = _state(batch)

    path = upload_server.reserve_input(batch, "half.pdf", 8)

    entry = upload_server.load_manifest(batch)["files"][0]
    assert entry["path"] == str(path)
    assert entry["status"] == "uploading"
    assert not path.exists()
    assert upload_server.uploading_count(state) == 1
    assert upload_server.list_staged(state) == []
    with pytest.raises(upload_server.UploadInProgress):
        upload_server.mark_running(batch, [])

    path.write_bytes(b"complete")
    upload_server.commit_input(batch, path, path.stat().st_size)

    assert upload_server.uploading_count(state) == 0
    assert upload_server.list_staged(state) == [{
        **entry,
        "size": len(b"complete"),
        "status": "ready",
    }]


def test_removing_and_adding_same_name_uses_a_new_path(batch):
    state = _state(batch)
    first = _ready_file(batch, "same.pdf", b"old")

    upload_server.remove_staged(state, {str(first)})
    second = upload_server.reserve_input(batch, "same.pdf", 3)

    assert second != first
    second.write_bytes(b"new")
    upload_server.commit_input(batch, second, 3)

    staged = upload_server.list_staged(state)
    assert len(staged) == 1
    assert staged[0]["path"] == str(second)
    assert Path(staged[0]["path"]).read_bytes() == b"new"


def test_start_and_upload_are_serialized_by_manifest_state(batch):
    path = _ready_file(batch)
    staged = upload_server.list_staged(_state(batch))

    upload_server.mark_running(batch, staged)

    assert upload_server.load_manifest(batch)["status"] == "running"
    with pytest.raises(upload_server.BatchAlreadyRunning):
        upload_server.reserve_input(batch, "later.pdf", 4)
    assert upload_server.load_manifest(batch)["files"][0]["path"] == str(path)


def test_start_and_upload_race_is_serialized_by_manifest_lock(batch, monkeypatch):
    # Seed one complete input before pausing mark_running inside its locked save.
    _ready_file(batch)
    staged = upload_server.list_staged(_state(batch))
    running_save_entered = Event()
    allow_running_save = Event()
    upload_finished = Event()
    errors = []
    original_save = upload_server.save_manifest

    def pause_running_save(batch_dir, manifest):
        if manifest.get("status") == "running":
            running_save_entered.set()
            assert allow_running_save.wait(5)
        original_save(batch_dir, manifest)

    monkeypatch.setattr(upload_server, "save_manifest", pause_running_save)

    def start_processing():
        try:
            upload_server.mark_running(batch, staged)
        except Exception as exc:  # surfaced in the main test thread below
            errors.append(exc)

    def reserve_upload():
        try:
            upload_server.reserve_input(batch, "racing.pdf", 4)
        except Exception as exc:
            errors.append(exc)
        finally:
            upload_finished.set()

    start_thread = Thread(target=start_processing)
    start_thread.start()
    assert running_save_entered.wait(5)

    upload_thread = Thread(target=reserve_upload)
    upload_thread.start()
    assert not upload_finished.wait(0.05)
    allow_running_save.set()
    start_thread.join(5)
    upload_thread.join(5)

    assert not start_thread.is_alive() and not upload_thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], upload_server.BatchAlreadyRunning)
    assert upload_server.load_manifest(batch)["status"] == "running"


def test_mark_running_rejects_a_stale_queue(batch):
    state = _state(batch)
    path = _ready_file(batch)
    stale_queue = upload_server.list_staged(state)
    upload_server.remove_staged(state, {str(path)})

    with pytest.raises(upload_server.StaleStagedQueue):
        upload_server.mark_running(batch, stale_queue)
    assert upload_server.load_manifest(batch)["status"] == "staging"


def test_legacy_complete_manifest_entries_are_ready(batch):
    path = batch / "inputs" / "legacy.pdf"
    path.write_bytes(b"legacy")
    upload_server.save_manifest(batch, {
        "created": "old",
        "status": "staging",
        "files": [{
            "name": "legacy.pdf",
            "path": str(path),
            "size": len(b"legacy"),
            "staged_at": "old",
        }],
    })

    state = _state(batch)
    staged = upload_server.list_staged(state)
    assert len(staged) == 1 and staged[0]["status"] == "ready"
    upload_server.mark_running(batch, staged)
    assert upload_server.load_manifest(batch)["status"] == "running"


def test_legacy_manifest_size_mismatch_is_not_ready(batch):
    path = batch / "inputs" / "partial.pdf"
    path.write_bytes(b"part")
    upload_server.save_manifest(batch, {
        "created": "old",
        "status": "staging",
        "files": [{
            "name": "partial.pdf",
            "path": str(path),
            "size": 10,
            "staged_at": "old",
        }],
    })

    assert upload_server.list_staged(_state(batch)) == []
    with pytest.raises(upload_server.StaleStagedQueue):
        upload_server.mark_running(batch, [])


def test_mark_done_can_retain_retry_inputs_and_legacy_default_clears(batch):
    retained = _ready_file(batch, "retry.pdf")
    discarded = _ready_file(batch, "done.pdf")
    upload_server.mark_done(batch, retained={str(retained)})

    manifest = upload_server.load_manifest(batch)
    assert manifest["status"] == "done"
    assert manifest["files"] == [{
        "name": "retry.pdf",
        "path": str(retained),
        "size": 3,
        "staged_at": manifest["files"][0]["staged_at"],
        "status": "ready",
    }]
    assert discarded.exists()

    upload_server.mark_done(batch)
    assert upload_server.load_manifest(batch)["files"] == []

