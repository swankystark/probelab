from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from probelab.engine_test.schemas import (
    CsvComparisonRequest,
    ExperimentSelectionRequest,
    ParameterRecoveryRequest,
    ParameterRecoveryTruth,
    PredictionRequest,
    PosteriorGrid,
    SolverValidationRequest,
    ValidationErrorPayload,
    parse_csv_request,
)


def valid_posterior() -> dict[str, object]:
    return {
        "lengths_m": [0.5, 1.0],
        "damping_s_inv": [0.05, 0.1, 0.2],
        "weights": [[0.1, 0.2, 0.1], [0.2, 0.25, 0.15]],
    }


def test_solver_request_accepts_explicit_units_and_finite_ranges() -> None:
    request = SolverValidationRequest(
        angle_deg=35.0,
        duration_s=8.0,
        length_m=1.0,
        damping_s_inv=0.1,
        initial_velocity_rad_s=0.0,
        gravity_m_s2=9.80665,
        samples=120,
    )

    assert request.angle_unit == "deg"
    assert request.time_unit == "s"
    assert request.velocity_unit == "rad/s"


@pytest.mark.parametrize(
    ("field", "value"),
    [("angle_deg", float("nan")), ("duration_s", float("inf")), ("length_m", 0.0), ("samples", 1)],
)
def test_solver_request_rejects_nonfinite_or_invalid_values(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        SolverValidationRequest(**{field: value})


def test_posterior_preserves_full_grid_shape_and_marginals() -> None:
    posterior = PosteriorGrid(**valid_posterior())

    assert posterior.lengths_m == [0.5, 1.0]
    assert posterior.damping_s_inv == [0.05, 0.1, 0.2]
    assert len(posterior.weights) == 2
    assert [len(row) for row in posterior.weights] == [3, 3]


def test_posterior_rejects_shape_mismatch_and_nonfinite_weights() -> None:
    with pytest.raises(ValidationError):
        PosteriorGrid(lengths_m=[0.5], damping_s_inv=[0.1], weights=[[float("nan")]])
    with pytest.raises(ValidationError):
        PosteriorGrid(lengths_m=[0.5, 1.0], damping_s_inv=[0.1], weights=[[1.0]])


def test_recovery_truth_is_separate_from_observed_request() -> None:
    request = ParameterRecoveryRequest(
        times_s=[float(index) for index in range(12)],
        angles_deg=[20.0] * 12,
        noise_sigma_deg=0.5,
    )
    truth = ParameterRecoveryTruth(length_m=1.0, damping_s_inv=0.1)

    assert "truth" not in request.model_fields
    assert truth.length_m == 1.0


def test_prediction_request_requires_frozen_parameters_and_explicit_units() -> None:
    request = PredictionRequest(
        angle_deg=48.0,
        duration_s=8.0,
        length_m=1.0,
        damping_s_inv=0.1,
        model_version="posterior-2026-11-02T00:00:00Z",
    )

    assert request.model_version.startswith("posterior-")
    assert request.parameter_source == "frozen_posterior"


def test_selection_rejects_empty_or_ambiguous_candidates() -> None:
    with pytest.raises(ValidationError):
        ExperimentSelectionRequest(posterior=valid_posterior(), candidate_angles_deg=[])
    with pytest.raises(ValidationError):
        ExperimentSelectionRequest(
            posterior=valid_posterior(),
            candidate_angles_deg=[12.0, 12.0],
        )


def test_csv_requires_exact_columns_uniform_time_and_minimum_samples() -> None:
    csv = "time_s,angle_deg\n" + "".join(f"{index * 0.5},{20 - index * 0.1}\n" for index in range(12))

    request = parse_csv_request(csv)

    assert isinstance(request, CsvComparisonRequest)
    assert request.times_s[0] == 0.0
    assert request.time_step_s == 0.5


@pytest.mark.parametrize(
    "csv",
    [
        "time,angle_deg\n0,20\n1,19\n",
        "time_s,angle_deg\n0,20\n0,19\n",
        "time_s,angle_deg\n0,20\n1,nan\n",
        "time_s,angle_deg\n" + "".join(f"{index},{20 - index}\n" for index in range(11)),
        "time_s,angle_deg\n" + "".join(f"{index * (0.5 if index < 6 else 0.7)},{20 - index}\n" for index in range(12)),
    ],
)
def test_csv_rejects_malformed_or_irregular_payloads(csv: str) -> None:
    with pytest.raises(ValidationError):
        parse_csv_request(csv)


def test_result_error_payload_is_stable_and_utc_metadata_is_required() -> None:
    error = ValidationErrorPayload(code="invalid_request", message="candidate list is empty", field="candidate_angles_deg")
    metadata = {
        "run_id": "run-123",
        "created_at": datetime(2026, 11, 2, tzinfo=UTC),
        "engine_version": "probelab-science-0.1",
        "reference_version": "reference-0.1",
        "candidate_angles_deg": [12.0],
        "noise_sigma_deg": 1.0,
        "tolerances": {"rmse_deg": 0.1},
    }

    assert error.model_dump() == {
        "code": "invalid_request",
        "message": "candidate list is empty",
        "field": "candidate_angles_deg",
    }
    assert metadata["created_at"].tzinfo is UTC
