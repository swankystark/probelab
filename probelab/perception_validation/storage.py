"""Filesystem storage restricted to temporary validation artifacts."""

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from .models import SessionCreateRequest, SessionResponse, SessionState

SESSION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class InvalidSessionStateError(ValueError):
    """Persisted session state is unreadable or violates its contract."""


class InvalidSessionTransitionError(ValueError):
    """A validation session attempted to move backwards in its lifecycle."""


class ValidationSessionStore:
    """Persist validation state below one caller-provided validation root."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def create(self, request: SessionCreateRequest) -> SessionResponse:
        """Create a UUID-scoped session directory and state file."""
        session_id = uuid4().hex
        now = datetime.now(UTC)
        session = SessionResponse(
            session_id=session_id,
            status="created",
            label=request.label,
            artifact_root=f"data/perception-validation/{session_id}",
            created_at=now,
            updated_at=now,
        )
        directory = self._directory(session_id)
        try:
            (directory / "uploads").mkdir(parents=True, exist_ok=False)
            self._write(SessionState(session=session), directory)
        except OSError:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return session

    def get(self, session_id: str) -> SessionState:
        """Load a session after validating its path-safe identifier."""
        directory = self._directory(session_id)
        path = directory / "session.json"
        if not path.exists():
            raise FileNotFoundError(session_id)
        try:
            return SessionState.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValidationError, json.JSONDecodeError) as error:
            raise InvalidSessionStateError("invalid validation session state") from error

    def save(self, state: SessionState) -> SessionState:
        """Replace a session state inside its existing isolated directory."""
        directory = self._directory(state.session.session_id)
        previous = self.get(state.session.session_id)
        if _status_rank(state.session.status) < _status_rank(previous.session.status):
            raise InvalidSessionTransitionError(
                f"cannot move validation session from {previous.session.status} "
                f"back to {state.session.status}"
            )
        self._write(state, directory)
        return state

    def _directory(self, session_id: str) -> Path:
        if SESSION_ID_PATTERN.fullmatch(session_id) is None:
            raise ValueError("invalid validation session id")
        directory = (self.root / session_id).resolve()
        root = self.root.resolve()
        if directory.parent != root:
            raise ValueError("invalid validation session id")
        return directory

    @staticmethod
    def _write(state: SessionState, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "session.json"
        temporary = directory / f".session-{uuid4().hex}.json.tmp"
        try:
            temporary.write_text(
                json.dumps(state.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def _status_rank(status: str) -> int:
    statuses = {
        "created": 0,
        "uploaded": 1,
        "annotated": 2,
        "diagnosed": 3,
        "quality_checked": 4,
        "handed_off": 5,
    }
    return statuses[status]
