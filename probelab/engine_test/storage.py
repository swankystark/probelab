"""Atomic, isolated JSON storage for engine-test results."""

import csv
import io
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = Mapping[str, JsonValue]

_DEFAULT_ROOT: Final = Path("data") / "engine-test"
_RUN_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CSV_FIELDS: Final = ("run_id", "status", "created_at", "metadata_json", "result_json")


class InvalidRunIdError(ValueError):
    """Raised when a run ID cannot safely address one result file."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(run_id)

    def __str__(self) -> str:
        return f"invalid engine-test run ID: {self.run_id!r}"


class ResultNotFoundError(FileNotFoundError):
    """Raised when an engine-test result does not exist."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(run_id)

    def __str__(self) -> str:
        return f"engine-test result not found: {self.run_id!r}"


class StorageCorruptionError(ValueError):
    """Raised when a stored artifact is not a valid result record."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(str(path))

    def __str__(self) -> str:
        return f"invalid engine-test result artifact: {self.path}"


class EngineTestStorage:
    """Store engine-test records below one dedicated filesystem root."""

    def __init__(self, root: Path | str = _DEFAULT_ROOT, max_bytes: int = 5_000_000) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._root = Path(root)
        self._max_bytes = max_bytes

    def save_result(self, payload: JsonObject) -> dict[str, JsonValue]:
        """Atomically publish one complete result record and return its snapshot."""
        record = dict(payload)
        run_id = self._run_id(record)
        encoded = self._encode(record)
        if len(encoded) > self._max_bytes:
            raise ValueError(f"engine-test result exceeds size limit of {self._max_bytes} bytes")

        self._root.mkdir(parents=True, exist_ok=True)
        destination = self._path_for(run_id)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self._root, prefix=f".{run_id}.", suffix=".tmp", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(encoded)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(str(temporary_path), str(destination))
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return record

    def get_result(self, run_id: str) -> dict[str, JsonValue]:
        """Read and validate one complete result record."""
        path = self._path_for(run_id)
        if not path.exists():
            raise ResultNotFoundError(run_id)
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StorageCorruptionError(path) from error
        if not isinstance(decoded, dict) or decoded.get("run_id") != run_id:
            raise StorageCorruptionError(path)
        return decoded

    def list_history(self) -> list[dict[str, JsonValue]]:
        """Return valid result records, newest created timestamp first."""
        if not self._root.exists():
            return []
        records = [self.get_result(path.stem) for path in self._root.glob("*.json")]
        return sorted(records, key=self._history_key, reverse=True)

    def export_json(self, run_id: str) -> str:
        """Export one result as stable, human-readable JSON."""
        return json.dumps(self.get_result(run_id), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def export_csv(self, run_id: str) -> str:
        """Export one result with metadata and nested result data preserved as JSON."""
        record = self.get_result(run_id)
        row = self._csv_row(record)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
        return output.getvalue()

    def export_history_json(self) -> str:
        """Export all stored history as a JSON array."""
        return json.dumps(self.list_history(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def export_history_csv(self) -> str:
        """Export all stored history with one result per CSV row."""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in self.list_history():
            writer.writerow(self._csv_row(record))
        return output.getvalue()

    def _path_for(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        return self._root / f"{run_id}.json"

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise InvalidRunIdError(run_id)

    @staticmethod
    def _run_id(record: Mapping[str, JsonValue]) -> str:
        run_id = record.get("run_id")
        if not isinstance(run_id, str):
            raise InvalidRunIdError(str(run_id))
        EngineTestStorage._validate_run_id(run_id)
        return run_id

    @staticmethod
    def _encode(record: Mapping[str, JsonValue]) -> bytes:
        try:
            return (json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
                "utf-8"
            )
        except (TypeError, ValueError) as error:
            raise ValueError("engine-test result must contain JSON-compatible values") from error

    @staticmethod
    def _history_key(record: Mapping[str, JsonValue]) -> str:
        created_at = record.get("created_at")
        return created_at if isinstance(created_at, str) else ""

    @staticmethod
    def _csv_row(record: Mapping[str, JsonValue]) -> dict[str, str]:
        metadata = record.get("metadata", {})
        result = record.get("result", {})
        return {
            "run_id": str(record.get("run_id", "")),
            "status": str(record.get("status", "")),
            "created_at": str(record.get("created_at", "")),
            "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            "result_json": json.dumps(result, sort_keys=True, separators=(",", ":")),
        }
