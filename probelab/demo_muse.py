"""One-time, non-production Muse interpretation for a saved demo run."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from threading import Lock
from typing import Any

from .meta_client import MetaMuseClient, NarrativeResult

_SESSION_ID = re.compile(r"^[0-9a-f]{32}$")
_INVOCATION_LOCK = Lock()
_MODEL = "muse-spark-1.3"


class DemoMuseError(ValueError):
    """The requested cached run cannot be safely interpreted."""


def interpret_saved_demo(session_id: str, client: MetaMuseClient) -> dict[str, Any]:
    """Make at most one demo-only Muse request and persist its provenance."""
    if not _SESSION_ID.fullmatch(session_id):
        raise DemoMuseError("Invalid saved demo session identifier.")

    with _INVOCATION_LOCK:
        root = Path("data/perception-validation") / session_id
        evaluation_path = root / "demo-evaluation.json"
        state_path = root / "session.json"
        try:
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DemoMuseError("Saved demo evidence is unavailable or invalid.") from error

        if evaluation.get("session_id") != session_id or state.get("session", {}).get("session_id") != session_id:
            raise DemoMuseError("Saved evidence does not belong to the requested session.")

        existing = evaluation.get("muse") or {}
        if existing.get("invocation_metadata", {}).get("attempted"):
            return existing

        if not client.is_configured:
            result = _unavailable_result(session_id, "META_API is not configured; no request was sent.")
            _save_result(evaluation_path, root, evaluation, result)
            return result

        tracking = state.get("sanitized_sam_tracking") or {}
        geometry = state.get("geometry_quality") or {}
        rows = geometry.get("pixel_geometry") or []
        tracks = tracking.get("tracks") or []
        fit = evaluation.get("fit") or {}
        selector = evaluation.get("selector") or {}
        prediction = evaluation.get("frozen_prediction") or {}
        validation = evaluation.get("validation") or {}
        expected_count = int(evaluation.get("measurement_count", 0))
        tracks_by_frame = {
            int(track["frame_index"]): track
            for track in tracks
            if "frame_index" in track
        }

        if (
            tracking.get("method") != "sam_guided_mask"
            or expected_count != len(rows)
            or len(tracks_by_frame) != len(rows)
            or fit.get("adequacy") != "good"
            or not selector.get("angle_deg")
            or prediction.get("frozen_before_validation") is not True
        ):
            raise DemoMuseError("Saved SAM, fit, selector, or frozen-prediction evidence is incomplete.")
        if evaluation.get("csv_used_for_estimation") is not False:
            raise DemoMuseError("The saved run does not certify comparison-CSV exclusion.")

        angles = [float(row["angle_deg"]) for row in rows]
        times = [float(row["timestamp_s"]) for row in rows]
        confidences = []
        frame_indices = []
        for row in rows:
            frame_index = int(row["frame_index"])
            track = tracks_by_frame.get(frame_index)
            if (
                track is None
                or abs(float(track["timestamp_s"]) - float(row["timestamp_s"])) > 1e-6
                or str(track.get("object_id")) != str(tracking.get("object_id"))
            ):
                raise DemoMuseError("SAM tracks and mask-derived geometry do not identify the same frames/object.")
            confidences.append(float(track["confidence"]))
            frame_indices.append(frame_index)
        scores = selector.get("scores") or []
        if not scores or max(scores, key=lambda score: float(score["eig_nats"]))["angle_deg"] != selector["angle_deg"]:
            raise DemoMuseError("The saved EIG action does not match the highest stored candidate score.")
        sam_observation = {
            "method": tracking["method"],
            "object_id": tracking.get("object_id"),
            "frame_indices": frame_indices,
            "timestamps_s": times,
            "angles_deg": angles,
            "confidences": confidences,
        }
        evidence = {
            "session_id": session_id,
            "classification": "phone-recorded simulator; synthetic validation only",
            "sam_observation": sam_observation,
            "fit": fit,
            "selector": selector,
            "frozen_prediction": {
                key: prediction.get(key)
                for key in (
                    "status",
                    "source",
                    "angle_deg",
                    "duration_s",
                    "length_m",
                    "damping_s_minus_1",
                    "initial_velocity_rad_s",
                    "times_s",
                    "angles_deg",
                    "frozen_before_validation",
                )
            },
            "validation_status": validation.get("status", "awaiting_validation"),
        }
        fingerprint = _fingerprint(evidence)
        prompt = _build_prompt(
            rows=rows,
            tracking=tracking,
            confidences=confidences,
            fit=fit,
            selector=selector,
            prediction=prediction,
            validation=validation,
        )
        now = datetime.now(UTC).isoformat()
        invocation = {
            "attempted": True,
            "started_at_utc": now,
            "endpoint": "isolated_demo_interpretation",
            "request_attempts": 1,
            "evidence_fingerprint": fingerprint,
            "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
        }
        pending = {
            "session_id": session_id,
            "evidence_fingerprint": fingerprint,
            "status": "Running",
            "request_status": "running",
            "completion_status": "not_started",
            "model": _MODEL,
            "finish_reason": "",
            "response_text_length": 0,
            "text": "",
            "error_detail": "",
            "invocation_metadata": invocation,
        }
        _save_result(evaluation_path, root, evaluation, pending)

        narrative = client.reason(prompt, single_attempt=True)
        completed_at = datetime.now(UTC).isoformat()
        result = _narrative_result(session_id, fingerprint, narrative, invocation, completed_at)
        _save_result(evaluation_path, root, evaluation, result)
        return result


def _build_prompt(
    *,
    rows: list[dict[str, Any]],
    tracking: dict[str, Any],
    confidences: list[float],
    fit: dict[str, Any],
    selector: dict[str, Any],
    prediction: dict[str, Any],
    validation: dict[str, Any],
) -> str:
    angles = [float(row["angle_deg"]) for row in rows]
    times = [float(row["timestamp_s"]) for row in rows]
    return (
        "You are Muse Spark, interpreting one saved ProbeLab demo run. This is a "
        "demo-only interpretation, not a production handoff. The evidence is from a "
        "phone-recorded simulator and is synthetic validation, not physical experimental proof.\n\n"
        f"SAM 3.1 mask-derived tracking: object {tracking.get('object_id')}, "
        f"{len(rows)} retained detections; observed angular range "
        f"{min(angles):.2f} to {max(angles):.2f} degrees over source timestamps "
        f"{min(times):.3f} to {max(times):.3f} seconds; mean mask confidence "
        f"{sum(confidences) / len(confidences):.3f}.\n"
        f"Production Bayesian fit: effective length {fit['length_m']:.4f} m, 68% interval "
        f"{fit['length_interval_m'][0]:.4f}–{fit['length_interval_m'][1]:.4f} m; "
        f"damping {fit['damping_s_minus_1']:.5f} s^-1, 68% interval "
        f"{fit['damping_interval_s_minus_1'][0]:.5f}–{fit['damping_interval_s_minus_1'][1]:.5f} s^-1; "
        f"angular RMS residual {fit['residual_rms_deg']:.3f} degrees against the existing "
        f"{fit['adequacy_threshold_deg']:.3f}-degree adequacy threshold. The selected window "
        "starts mid-motion; initial angular velocity was estimated as a nuisance state.\n"
        f"The unchanged production EIG selector chose {selector['angle_deg']:.0f} degrees "
        f"for {selector['duration_s']:.1f} seconds, {selector['eig_nats']:.3f} nats. "
        f"Its rationale: {selector['rationale']} Candidate angles evaluated: "
        f"{', '.join(f'{float(value):.0f}' for value in selector['candidate_angles_deg'])} degrees.\n"
        f"Frozen prediction: {len(prediction.get('angles_deg', []))} model-generated samples "
        f"for a {prediction['angle_deg']:.0f}-degree release over {prediction['duration_s']:.1f} seconds, "
        f"using length {prediction['length_m']:.4f} m and damping "
        f"{prediction['damping_s_minus_1']:.5f} s^-1. It is a forecast, not an observation.\n"
        f"Validation status: {validation.get('status', 'awaiting_validation')}; no independent "
        "follow-up observation has been performed.\n\n"
        f"In at most 90 words, explain (1) what the observation and fit indicate, (2) why the "
        f"selected {selector['angle_deg']:.0f}-degree experiment is informative based only on its highest EIG score among "
        "the candidates evaluated, (3) what the frozen prediction represents, and (4) what is "
        "needed for validation. Do not invent measurements, change estimates, select another "
        "action, claim validation, or present this as physical experimental proof. Do not use "
        "reference CSV values or simulator ground-truth parameters."
    )


def _narrative_result(
    session_id: str,
    fingerprint: str,
    narrative: NarrativeResult,
    invocation: dict[str, Any],
    completed_at: str,
) -> dict[str, Any]:
    status = narrative.status
    result = {
        "session_id": session_id,
        "evidence_fingerprint": fingerprint,
        "status": status,
        "request_status": narrative.request_status,
        "completion_status": narrative.completion_status,
        "model": narrative.model,
        "finish_reason": narrative.finish_reason,
        "response_text_length": narrative.response_text_length,
        "text": narrative.text if status in {"Completed", "Incomplete response"} else "",
        "error_detail": narrative.error_detail,
        "invocation_metadata": {
            **invocation,
            "completed_at_utc": completed_at,
            "request_attempts": 1 if narrative.request_status != "not_invoked" else 0,
        },
    }
    return result


def _unavailable_result(session_id: str, detail: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "evidence_fingerprint": "",
        "status": "Failed",
        "request_status": "not_invoked",
        "completion_status": "not_started",
        "model": _MODEL,
        "finish_reason": "not_called",
        "response_text_length": 0,
        "text": "",
        "error_detail": detail,
        "invocation_metadata": {"attempted": False, "endpoint": "isolated_demo_interpretation", "request_attempts": 0},
    }


def _fingerprint(value: dict[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _save_result(
    evaluation_path: Path,
    root: Path,
    evaluation: dict[str, Any],
    result: dict[str, Any],
) -> None:
    evaluation["muse"] = result
    _atomic_json_write(evaluation_path, evaluation)
    _atomic_json_write(root / "muse-interpretation.json", result)


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
