import csv
import json
from pathlib import Path

import pytest

from probelab.engine_test.storage import (
    EngineTestStorage,
    InvalidRunIdError,
    ResultNotFoundError,
)


def test_completed_result_round_trips_and_isolated_from_investigations(tmp_path: Path) -> None:
    storage = EngineTestStorage(tmp_path)
    payload = {
        "run_id": "run-20261102-abc123",
        "status": "completed",
        "metadata": {"engine_version": "test-engine-1"},
        "result": {"selected_angle_deg": 25.0, "metrics": {"rmse_deg": 0.2}},
    }

    stored = storage.save_result(payload)

    assert stored == payload
    assert storage.get_result("run-20261102-abc123") == payload
    assert (tmp_path / "run-20261102-abc123.json").exists()
    assert not (tmp_path.parent / "run-20261102-abc123.json").exists()


def test_history_is_sorted_newest_first_and_retrieves_metadata(tmp_path: Path) -> None:
    storage = EngineTestStorage(tmp_path)
    storage.save_result({"run_id": "run-old", "status": "failed", "created_at": "2026-11-01T00:00:00Z"})
    storage.save_result({"run_id": "run-new", "status": "completed", "created_at": "2026-11-02T00:00:00Z"})

    history = storage.list_history()

    assert [item["run_id"] for item in history] == ["run-new", "run-old"]
    assert history[0]["status"] == "completed"


def test_invalid_or_traversal_run_ids_are_rejected_without_writing(tmp_path: Path) -> None:
    storage = EngineTestStorage(tmp_path)

    for run_id in ["../escape", "nested/run", "", ".", "run with spaces"]:
        with pytest.raises(InvalidRunIdError):
            storage.save_result({"run_id": run_id, "status": "completed"})

    assert list(tmp_path.iterdir()) == []


def test_missing_result_has_typed_error(tmp_path: Path) -> None:
    with pytest.raises(ResultNotFoundError):
        EngineTestStorage(tmp_path).get_result("run-missing")


def test_json_and_csv_exports_include_status_and_metadata(tmp_path: Path) -> None:
    storage = EngineTestStorage(tmp_path)
    payload = {
        "run_id": "run-export",
        "status": "completed",
        "created_at": "2026-11-02T00:00:00Z",
        "metadata": {"engine_version": "test-engine-1", "candidates": [5, 10]},
        "result": {"selected_angle_deg": 10.0},
    }
    storage.save_result(payload)

    exported_json = json.loads(storage.export_json("run-export"))
    rows = list(csv.DictReader(storage.export_csv("run-export").splitlines()))

    assert exported_json == payload
    assert rows == [
        {
            "run_id": "run-export",
            "status": "completed",
            "created_at": "2026-11-02T00:00:00Z",
            "metadata_json": '{"candidates":[5,10],"engine_version":"test-engine-1"}',
            "result_json": '{"selected_angle_deg":10.0}',
        }
    ]


def test_partial_write_does_not_replace_existing_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage = EngineTestStorage(tmp_path)
    original = {"run_id": "run-safe", "status": "completed", "result": {"value": 1}}
    storage.save_result(original)

    def fail_replace(source: str, destination: str) -> str:
        raise OSError("simulated interruption")

    monkeypatch.setattr("probelab.engine_test.storage.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated interruption"):
        storage.save_result({"run_id": "run-safe", "status": "failed", "result": {"value": 2}})

    assert storage.get_result("run-safe") == original
    assert not list(tmp_path.glob("*.tmp"))


def test_artifact_size_limit_rejects_oversized_result(tmp_path: Path) -> None:
    storage = EngineTestStorage(tmp_path, max_bytes=100)

    with pytest.raises(ValueError, match="size"):
        storage.save_result({"run_id": "run-large", "status": "completed", "result": {"text": "x" * 200}})

    assert list(tmp_path.iterdir()) == []
