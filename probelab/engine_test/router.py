"""Isolated engine-validation HTTP API and dashboard entry point."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from ..engine_test_evaluation import ComparisonMetrics
from ..science import FitResult
from .schemas import (
    CsvComparisonRunRequest,
    ExperimentSelectionRequest,
    ParameterRecoveryRunRequest,
    PredictionRequest,
    SolverValidationRequest,
    utc_now,
)
from .schemas import (
    ParameterRecoveryDataset as GenerationDataset,
)
from .schemas import (
    ParameterRecoveryGenerationRequest as GenerationRequest,
)
from .service import (
    ENGINE_VERSION,
    REFERENCE_VERSION,
    generate_parameter_recovery_dataset,
    run_csv_comparison,
    run_prediction,
    run_recovery,
    run_selection,
    run_solver,
)
from .storage import (
    EngineTestStorage,
    InvalidRunIdError,
    JsonValue,
    ResultNotFoundError,
    StorageCorruptionError,
)

router = APIRouter(tags=["engine-test"])
_STORAGE: Final[EngineTestStorage] = EngineTestStorage()


class Exporter(Protocol):
    """Callable storage export contract."""

    def __call__(self, run_id: str) -> str: ...


@router.get("/engine_test")
def engine_test_page() -> FileResponse:
    """Serve the isolated engine-test dashboard."""
    return FileResponse(Path(__file__).resolve().parents[2] / "web" / "engine_test.html")


@router.get("/api/engine_test")
def engine_test_info() -> JSONResponse:
    """Return capabilities and storage isolation metadata."""
    return JSONResponse(
        {
            "status": "ready",
            "dashboard": "/engine_test",
            "api": "/api/engine_test",
            "engine_version": ENGINE_VERSION,
            "reference_version": REFERENCE_VERSION,
        }
    )


@router.post("/api/engine_test", response_class=JSONResponse)
def validate_solver(request: SolverValidationRequest) -> JSONResponse:
    """Validate the production solver against the independent reference solver."""
    try:
        run = run_solver(request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    result = {
        "status": "completed",
        "production": {
            "times_s": run.production_times.tolist(),
            "angles_deg": run.production_angles.tolist(),
        },
        "reference": {
            "times_s": run.reference_times.tolist(),
            "angles_deg": run.reference_angles.tolist(),
        },
        "comparison": _metrics(run.metrics),
        "engine": _engine_metadata(),
    }
    return _persist("solver-validation", request, result)


@router.post("/api/engine_test/solver", response_class=JSONResponse)
@router.post("/api/engine_test/solver-validation", response_class=JSONResponse)
def validate_solver_alias(request: SolverValidationRequest) -> JSONResponse:
    """Alias for the dashboard's explicit solver-validation action."""
    return validate_solver(request)


@router.post("/api/engine_test/parameter-recovery", response_class=JSONResponse)
def parameter_recovery(request: ParameterRecoveryRunRequest) -> JSONResponse:
    """Recover parameters using the production posterior fitter."""
    try:
        run = run_recovery(request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    fit = run.fit
    result = {
        "status": "completed",
        "fit": {
            "length_m": fit.length,
            "damping_s_inv": fit.damping,
            "length_interval_m": list(fit.length_interval),
            "damping_interval_s_inv": list(fit.damping_interval),
            "residual_rms_deg": fit.residual_rms,
            "adequacy": fit.adequacy,
            "posterior": {
                "lengths_m": fit.posterior.lengths.tolist(),
                "damping_s_inv": fit.posterior.damping.tolist(),
                "weights": fit.posterior.weights.tolist(),
            },
        },
        "observations": {
            "times_s": request.observations.times_s,
            "angles_deg": request.observations.angles_deg,
            "noise_sigma_deg": request.observations.noise_sigma_deg,
            "gravity_m_s2": request.observations.gravity_m_s2,
            "time_unit": request.observations.time_unit,
            "angle_unit": request.observations.angle_unit,
        },
        "truth": _truth_error(request, fit),
        "engine": _engine_metadata(),
    }
    return _persist("parameter-recovery", request, result)


@router.post("/api/engine_test/parameter-recovery/generate", response_model=GenerationDataset)
def parameter_recovery_generation(request: GenerationRequest) -> GenerationDataset:
    """Generate standard measured observations for parameter-recovery estimation."""
    return generate_parameter_recovery_dataset(request.noise_sigma_deg, request.seed)


@router.post("/api/engine_test/prediction", response_class=JSONResponse)
def frozen_prediction(request: PredictionRequest) -> JSONResponse:
    """Generate an unseen-condition prediction from explicitly frozen parameters."""
    times, angles = run_prediction(request)
    result = {
        "status": "completed",
        "model_version": request.model_version,
        "parameter_source": request.parameter_source,
        "frozen_parameters": {"length_m": request.length_m, "damping_s_inv": request.damping_s_inv},
        "prediction": {"times_s": times.tolist(), "angles_deg": angles.tolist()},
        "engine": _engine_metadata(),
    }
    return _persist("prediction", request, result)


@router.post("/api/engine_test/experiment-selection", response_class=JSONResponse)
def experiment_selection(request: ExperimentSelectionRequest) -> JSONResponse:
    """Select an experiment from dashboard-supplied candidates with production EIG."""
    try:
        selection = run_selection(request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    angle, duration, eig, rationale = selection.selected
    result = {
        "status": "completed",
        "selected_angle_deg": angle,
        "duration_s": duration,
        "eig_nats": eig,
        "rationale": rationale,
        "candidate_angles_deg": request.candidate_angles_deg,
        "completed_angles_deg": request.completed_angles_deg,
        "scores": [
            {"angle_deg": score.angle_deg, "eig_nats": score.eig_nats}
            for score in selection.scores
        ],
        "engine": _engine_metadata(),
    }
    return _persist("experiment-selection", request, result)


@router.post("/api/engine_test/csv-comparison", response_class=JSONResponse)
def csv_comparison(request: CsvComparisonRunRequest) -> JSONResponse:
    """Compare exact ``time_s,angle_deg`` CSV samples with the reference trajectory."""
    try:
        metrics = run_csv_comparison(request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    result = {"status": "completed", "comparison": _metrics(metrics), "engine": _engine_metadata()}
    return _persist("csv-comparison", request, result)


@router.get("/api/engine_test/history", response_class=JSONResponse)
def history() -> JSONResponse:
    """List persisted engine-test runs, newest first."""
    return JSONResponse(_STORAGE.list_history())


@router.get("/api/engine_test/results/{run_id}", response_class=JSONResponse)
def result(run_id: str) -> JSONResponse:
    """Return one persisted engine-test run."""
    try:
        return JSONResponse(_STORAGE.get_result(run_id))
    except (InvalidRunIdError, ResultNotFoundError, StorageCorruptionError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/api/engine_test/results/{run_id}/export.json")
def export_json(run_id: str) -> Response:
    """Download one result as JSON."""
    return _download(run_id, _STORAGE.export_json, "application/json", ".json")


@router.get("/api/engine_test/results/{run_id}/export.csv")
def export_csv(run_id: str) -> Response:
    """Download one result as CSV."""
    return _download(run_id, _STORAGE.export_csv, "text/csv", ".csv")


@router.get("/api/engine_test/export.json")
def export_history_json() -> Response:
    """Download all result history as JSON."""
    return Response(
        _STORAGE.export_history_json(),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=engine-test-history.json"},
    )


@router.get("/api/engine_test/export.csv")
def export_history_csv() -> Response:
    """Download all result history as CSV."""
    return Response(
        _STORAGE.export_history_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=engine-test-history.csv"},
    )


def _persist(kind: str, request: BaseModel, result: dict[str, JsonValue]) -> JSONResponse:
    run_id = f"run-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    record = {
        "run_id": run_id,
        "status": "completed",
        "created_at": utc_now().isoformat(),
        "metadata": {"kind": kind, **_engine_metadata(), "request": _json(request)},
        "result": result,
    }
    return JSONResponse(_STORAGE.save_result(record), status_code=201)


def _download(run_id: str, exporter: Exporter, media_type: str, suffix: str) -> Response:
    try:
        content = exporter(run_id)
    except (InvalidRunIdError, ResultNotFoundError, StorageCorruptionError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return Response(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={run_id}{suffix}"},
    )


def _engine_metadata() -> dict[str, str]:
    return {"engine_version": ENGINE_VERSION, "reference_version": REFERENCE_VERSION}


def _metrics(metrics: ComparisonMetrics) -> dict[str, float]:
    return {
        name: float(getattr(metrics, name))
        for name in (
            "angle_rmse_deg",
            "angle_max_error_deg",
            "velocity_rmse_deg_s",
            "velocity_max_error_deg_s",
            "energy_rmse_per_mass",
            "energy_max_error_per_mass",
            "tolerance_deg",
        )
    }


def _truth_error(request: ParameterRecoveryRunRequest, fit: FitResult) -> dict[str, float] | None:
    if request.truth is None:
        return None
    return {
        "length_error_m": abs(fit.length - request.truth.length_m),
        "damping_error_s_inv": abs(fit.damping - request.truth.damping_s_inv),
    }


def _json(value: BaseModel) -> JsonValue:
    return json.loads(value.model_dump_json())
