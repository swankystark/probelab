"""Typed, isolated request and result contracts for engine evaluations."""

from __future__ import annotations

import csv
import io
import math
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

Angle = Annotated[FiniteFloat, Field(ge=0.0, le=180.0)]
PositiveFloat = Annotated[FiniteFloat, Field(gt=0.0)]
SampleCount = Annotated[int, Field(ge=2, le=10_000)]
Status = Literal["completed", "failed", "rejected"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValidationErrorPayload(StrictModel):
    """Stable, serializable error details for a 4xx-compatible response."""

    code: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=500)
    field: str | None = Field(default=None, min_length=1, max_length=120)


class RunMetadata(StrictModel):
    """Reproducibility metadata shared by every engine-test result."""

    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
    created_at: datetime
    engine_version: str = Field(min_length=1, max_length=120)
    reference_version: str = Field(min_length=1, max_length=120)
    candidate_angles_deg: list[Angle] = Field(min_length=1, max_length=180)
    noise_sigma_deg: FiniteFloat = Field(ge=0.0, le=180.0)
    tolerances: dict[str, FiniteFloat] = Field(min_length=1)

    @field_validator("created_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
            or value.utcoffset() != UTC.utcoffset(value)
        ):
            raise ValueError("created_at must include a UTC timezone")
        return value

    @field_validator("candidate_angles_deg")
    @classmethod
    def require_unique_candidates(cls, value: list[float]) -> list[float]:
        if len(set(value)) != len(value):
            raise ValueError("candidate_angles_deg must contain unique angles")
        return value


class SolverValidationRequest(StrictModel):
    """Inputs for production/reference trajectory comparison."""

    angle_deg: Angle = 20.0
    duration_s: PositiveFloat = Field(default=8.0, le=300.0)
    length_m: PositiveFloat = 1.0
    damping_s_inv: FiniteFloat = Field(default=0.1, ge=0.0, le=10.0)
    initial_velocity_rad_s: FiniteFloat = Field(default=0.0, ge=-100.0, le=100.0)
    gravity_m_s2: PositiveFloat = Field(default=9.80665, le=100.0)
    samples: SampleCount = 240
    angle_unit: Literal["deg"] = "deg"
    time_unit: Literal["s"] = "s"
    length_unit: Literal["m"] = "m"
    velocity_unit: Literal["rad/s"] = "rad/s"


class PosteriorGrid(StrictModel):
    """Complete posterior grid, including marginals and the selector weight matrix."""

    lengths_m: list[PositiveFloat] = Field(min_length=1, max_length=10_000)
    damping_s_inv: list[FiniteFloat] = Field(min_length=1, max_length=10_000)
    weights: list[list[FiniteFloat]] = Field(min_length=1, max_length=10_000)

    @field_validator("damping_s_inv")
    @classmethod
    def validate_damping(cls, value: list[float]) -> list[float]:
        if any(item < 0.0 for item in value):
            raise ValueError("damping_s_inv values must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_shape_and_mass(self) -> PosteriorGrid:
        if len(self.weights) != len(self.lengths_m) or any(
            len(row) != len(self.damping_s_inv) for row in self.weights
        ):
            raise ValueError("weights shape must be lengths_m by damping_s_inv")
        if (
            any(weight < 0.0 for row in self.weights for weight in row)
            or math.fsum(weight for row in self.weights for weight in row) <= 0.0
        ):
            raise ValueError("weights must be non-negative with positive total mass")
        return self


class ParameterRecoveryRequest(StrictModel):
    """Observed data only; hidden truth is represented by ParameterRecoveryTruth."""

    times_s: list[FiniteFloat] = Field(min_length=12, max_length=10_000)
    angles_deg: list[FiniteFloat] = Field(min_length=12, max_length=10_000)
    noise_sigma_deg: FiniteFloat = Field(ge=0.0, le=180.0)
    gravity_m_s2: PositiveFloat = Field(default=9.81, le=100.0)
    time_unit: Literal["s"] = "s"
    angle_unit: Literal["deg"] = "deg"

    @model_validator(mode="after")
    def validate_series(self) -> ParameterRecoveryRequest:
        if len(self.times_s) != len(self.angles_deg):
            raise ValueError("times_s and angles_deg must have equal length")
        if any(right <= left for left, right in zip(self.times_s, self.times_s[1:], strict=False)):
            raise ValueError("times_s must be strictly increasing")
        return self


class ParameterRecoveryTruth(StrictModel):
    """Evaluation-only truth, kept separate from estimator observations."""

    length_m: PositiveFloat
    damping_s_inv: FiniteFloat = Field(ge=0.0, le=10.0)
    initial_angle_deg: Angle = 20.0
    initial_velocity_rad_s: FiniteFloat = Field(default=0.0, ge=-100.0, le=100.0)


class ParameterRecoveryGenerationTruth(StrictModel):
    """Full synthetic-generation truth, never required by the estimator."""

    length_m: PositiveFloat
    damping_s_inv: FiniteFloat = Field(ge=0.0, le=10.0)
    gravity_m_s2: PositiveFloat = Field(le=100.0)
    initial_angle_deg: Angle
    initial_velocity_rad_s: FiniteFloat = Field(default=0.0, ge=-100.0, le=100.0)
    duration_s: PositiveFloat = Field(le=300.0)
    sample_rate_hz: PositiveFloat
    sample_count: SampleCount


class ParameterRecoveryGenerationRequest(StrictModel):
    """Optional measurement controls for the standard recovery dataset."""

    noise_sigma_deg: FiniteFloat = Field(default=0.0, ge=0.0, le=180.0)
    seed: int | None = Field(default=None, ge=0)


class ParameterRecoveryDataset(StrictModel):
    """Synthetic observations paired with separately reported evaluation truth."""

    observations: ParameterRecoveryRequest
    truth: ParameterRecoveryGenerationTruth


class ParameterRecoveryRunRequest(StrictModel):
    """Observed samples with optional evaluation-only truth metadata."""

    observations: ParameterRecoveryRequest
    truth: ParameterRecoveryTruth | None = None


class PredictionRequest(StrictModel):
    """Frozen-parameter prediction inputs; no test observations are accepted."""

    angle_deg: Angle
    duration_s: PositiveFloat = Field(le=300.0)
    length_m: PositiveFloat
    damping_s_inv: FiniteFloat = Field(ge=0.0, le=10.0)
    model_version: str = Field(min_length=1, max_length=160)
    parameter_source: Literal["frozen_posterior"] = "frozen_posterior"
    angle_unit: Literal["deg"] = "deg"
    time_unit: Literal["s"] = "s"


class ExperimentSelectionRequest(StrictModel):
    """Production selector inputs with an explicit, non-empty candidate set."""

    posterior: PosteriorGrid
    candidate_angles_deg: list[Angle] = Field(min_length=1, max_length=180)
    completed_angles_deg: list[Angle] = Field(default_factory=list, max_length=180)
    noise_sigma_deg: FiniteFloat = Field(default=1.0, gt=0.0, le=180.0)
    angle_unit: Literal["deg"] = "deg"

    @field_validator("candidate_angles_deg")
    @classmethod
    def validate_candidates(cls, value: list[float]) -> list[float]:
        if len(set(value)) != len(value):
            raise ValueError("candidate_angles_deg must contain unique angles")
        return value


class CsvComparisonRequest(StrictModel):
    """Normalized solver-validation CSV with exact, unit-bearing columns."""

    times_s: list[FiniteFloat] = Field(min_length=12, max_length=10_000)
    angles_deg: list[FiniteFloat] = Field(min_length=12, max_length=10_000)
    time_step_s: PositiveFloat
    columns: tuple[Literal["time_s"], Literal["angle_deg"]] = ("time_s", "angle_deg")

    @model_validator(mode="before")
    @classmethod
    def parse_raw_csv(cls, value: dict[str, object]) -> dict[str, object]:
        if "csv_text" not in value:
            return value
        raw = value["csv_text"]
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 2_000_000:
            raise ValueError("csv_text must be a UTF-8 payload no larger than 2 MB")
        rows = list(csv.DictReader(io.StringIO(raw)))
        if not rows or list(rows[0].keys()) != ["time_s", "angle_deg"]:
            raise ValueError("CSV must have exactly the columns time_s,angle_deg")
        try:
            times = [float(row["time_s"]) for row in rows]
            angles = [float(row["angle_deg"]) for row in rows]
        except (TypeError, ValueError) as error:
            raise ValueError("CSV values must be finite numbers") from error
        return {"times_s": times, "angles_deg": angles, "time_step_s": _uniform_step(times)}

    @model_validator(mode="after")
    def validate_series(self) -> CsvComparisonRequest:
        if len(self.times_s) != len(self.angles_deg):
            raise ValueError("CSV columns must have equal length")
        if any(right <= left for left, right in zip(self.times_s, self.times_s[1:], strict=False)):
            raise ValueError("CSV time_s values must be strictly increasing")
        if any(
            abs((right - left) - self.time_step_s) > 1e-9
            for left, right in zip(self.times_s, self.times_s[1:], strict=False)
        ):
            raise ValueError("CSV timestamps must be uniformly sampled")
        return self


class CsvComparisonRunRequest(StrictModel):
    """Production solver configuration paired with normalized CSV samples."""

    solver: SolverValidationRequest
    csv: CsvComparisonRequest


def _uniform_step(times: list[float]) -> float:
    if len(times) < 12:
        raise ValueError("CSV must contain at least 12 samples")
    if any(not math.isfinite(value) for value in times):
        raise ValueError("CSV values must be finite numbers")
    steps = [right - left for left, right in zip(times, times[1:], strict=False)]
    if any(step <= 0.0 for step in steps) or max(steps) - min(steps) > 1e-9:
        raise ValueError("CSV time_s values must be increasing and uniformly sampled")
    return steps[0]


def parse_csv_request(csv_text: str) -> CsvComparisonRequest:
    """Parse an exact solver-validation CSV at the trust boundary."""
    return CsvComparisonRequest.model_validate({"csv_text": csv_text})


def utc_now() -> datetime:
    """Return an aware UTC timestamp for reproducibility metadata."""
    return datetime.now(UTC)
