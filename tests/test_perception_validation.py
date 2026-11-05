import shutil
from importlib import import_module
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from probelab.main import app
from probelab.perception_validation.models import SessionCreateRequest, SessionState, UploadResponse
from probelab.perception_validation.router import _persist_upload, get_session
from probelab.perception_validation.storage import ValidationSessionStore

import_module("probelab.perception_validation.router")


def test_existing_and_validation_routes_are_available_through_openapi() -> None:
    route_paths = set(app.openapi()["paths"])

    assert "/" in route_paths
    assert "/api/investigations" in route_paths
    assert "/api/investigations/{item_id}" in route_paths
    assert "/api/perception-validation/sessions" in route_paths


def test_investigation_storage_root_is_unchanged() -> None:
    from probelab import store

    assert store.STORE == Path("data")


def test_validation_session_contract_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SessionCreateRequest.model_validate({"label": "temporary", "unexpected": True})


def test_validation_session_is_isolated_and_rejects_traversal() -> None:
    root = Path(".qa-temp") / f"validation-{uuid4().hex}"
    store = ValidationSessionStore(root)

    try:
        session = store.create(SessionCreateRequest(label="temporary"))

        assert UUID(session.session_id).version == 4
        assert session.artifact_root == f"data/perception-validation/{session.session_id}"
        assert (root / session.session_id / "session.json").exists()
        assert not (root.parent / session.session_id).exists()

        with pytest.raises(ValueError, match="session id"):
            store.get("../escape")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_interrupted_session_creation_does_not_leave_stale_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(".qa-temp") / f"validation-{uuid4().hex}"
    store = ValidationSessionStore(root)

    def fail_write(state: SessionState, directory: Path) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(store, "_write", fail_write)

    try:
        with pytest.raises(OSError, match="simulated interruption"):
            store.create(SessionCreateRequest(label="temporary"))

        assert list(root.glob("*/session.json")) == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_malformed_persisted_session_is_a_controlled_validation_error() -> None:
    root = Path(".qa-temp") / f"validation-{uuid4().hex}"
    store = ValidationSessionStore(root)

    try:
        session = store.create(SessionCreateRequest(label="temporary"))
        (root / session.session_id / "session.json").write_text("{not-json", encoding="utf-8")

        with pytest.raises(ValueError, match="invalid validation session state"):
            store.get(session.session_id)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_upload_save_failure_removes_temporary_and_final_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(".qa-temp") / f"validation-{uuid4().hex}"
    store = ValidationSessionStore(root)

    def fail_write(state: SessionState, directory: Path) -> None:
        raise OSError("simulated persistence failure")

    try:
        session = store.create(SessionCreateRequest(label="temporary"))
        monkeypatch.setattr(store, "_write", fail_write)
        upload_id = uuid4().hex
        response = UploadResponse(
            upload_id=upload_id,
            session_id=session.session_id,
            metadata={
                "kind": "video",
                "original_filename": "clip.mp4",
                "content_type": "video/mp4",
                "byte_size": 1,
            },
            artifact_path=(
                f"data/perception-validation/{session.session_id}/uploads/{upload_id}.mp4"
            ),
        )
        state = store.get(session.session_id).model_copy(
            update={"uploads": [response]}
        )
        target = root / session.session_id / "uploads" / f"{upload_id}.mp4"
        temporary_target = target.with_name(f".{upload_id}.mp4.tmp")
        target.write_bytes(b"stale final artifact")

        def fail_save(_state: SessionState) -> SessionState:
            raise OSError("simulated persistence failure")

        monkeypatch.setattr(store, "save", fail_save)

        with pytest.raises(OSError, match="simulated persistence failure"):
            _persist_upload(store, state, b"new upload", target, temporary_target)

        assert store.get(session.session_id).uploads == []
        assert not temporary_target.exists()
        assert not target.exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_upload_contract_rejects_artifact_path_escape() -> None:
    from probelab.perception_validation.models import UploadResponse

    with pytest.raises(ValidationError):
        UploadResponse.model_validate(
            {
                "upload_id": "a" * 32,
                "session_id": "b" * 32,
                "metadata": {
                    "kind": "video",
                    "original_filename": "clip.mp4",
                    "content_type": "video/mp4",
                    "byte_size": 1,
                },
                "artifact_path": "data/escape.mp4",
            }
        )


def test_storage_marks_existing_session_as_uploaded() -> None:
    root = Path(".qa-temp") / f"validation-{uuid4().hex}"
    store = ValidationSessionStore(root)

    try:
        session = store.create(SessionCreateRequest(label="temporary"))
        upload_id = uuid4().hex
        response = UploadResponse(
            upload_id=upload_id,
            session_id=session.session_id,
            metadata={
                "kind": "video",
                "original_filename": "clip.mp4",
                "content_type": "video/mp4",
                "byte_size": 1,
            },
            artifact_path=(
                f"data/perception-validation/{session.session_id}/uploads/{upload_id}.mp4"
            ),
        )
        state = store.get(session.session_id).model_copy(
            update={
                "uploads": [response],
                "session": session.model_copy(update={"status": "uploaded"}),
            }
        )
        store.save(state)
        artifact = root / session.session_id / "uploads" / f"{upload_id}.mp4"
        artifact.write_bytes(b"x")

        assert store.get(session.session_id).session.status == "uploaded"
        assert artifact.read_bytes() == b"x"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_validation_http_surface_rejects_missing_and_malformed_requests() -> None:
    with pytest.raises(HTTPException) as invalid:
        get_session("missing")

    assert invalid.value.status_code == 400

    with pytest.raises(HTTPException) as missing:
        get_session("0" * 32)

    assert missing.value.status_code == 404

    with pytest.raises(ValidationError):
        SessionCreateRequest.model_validate({"label": "temporary", "unexpected": True})
