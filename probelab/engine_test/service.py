"""Typed adapters around ProbeLab's production numerical engine."""

from dataclasses import dataclass
from typing import Final

import numpy as np

from ..engine_test_evaluation import (
    ComparisonMetrics,
    ReferenceConfig,
    compare_trajectories,
    reference_solve,
)
from ..science import (
    ExperimentScore,
    FitResult,
    Posterior,
    choose_experiment,
    fit_posterior,
    score_experiments,
    simulate,
)
from .schemas import (
    CsvComparisonRunRequest,
    ExperimentSelectionRequest,
    ParameterRecoveryDataset,
    ParameterRecoveryGenerationRequest,
    ParameterRecoveryGenerationTruth,
    ParameterRecoveryRequest,
    ParameterRecoveryRunRequest,
    PredictionRequest,
    SolverValidationRequest,
)

ENGINE_VERSION: Final[str] = "probelab.science.production"
REFERENCE_VERSION: Final[str] = "probelab.engine_test_evaluation.dop853-1"
RECOVERY_GENERATION_GRAVITY: Final[float] = 9.81
RECOVERY_GENERATION_ANGLE: Final[float] = 35.0
RECOVERY_GENERATION_DURATION: Final[float] = 15.0
RECOVERY_GENERATION_LENGTH: Final[float] = 0.8
RECOVERY_GENERATION_DAMPING: Final[float] = 0.08
RECOVERY_GENERATION_SAMPLE_RATE: Final[float] = 30.0
RECOVERY_GENERATION_COUNT: Final[int] = 450
@dataclass(frozen=True, slots=True)
class SolverRun:
    """Production trajectory and independent comparison for one solver run."""

    request: SolverValidationRequest
    production_times: np.ndarray
    production_angles: np.ndarray
    reference_times: np.ndarray
    reference_angles: np.ndarray
    metrics: ComparisonMetrics


@dataclass(frozen=True, slots=True)
class RecoveryRun:
    """Production posterior fit and optional truth error."""

    request: ParameterRecoveryRunRequest
    fit: FitResult


@dataclass(frozen=True, slots=True)
class SelectionRun:
    """Production selection and the scores for the supplied candidates."""

    selected: tuple[float, float, float, str]
    scores: tuple[ExperimentScore, ...]


def run_solver(request: SolverValidationRequest) -> SolverRun:
    """Run production ``simulate`` and compare it with the independent reference."""
    times, angles = simulate(
        request.angle_deg,
        request.duration_s,
        request.length_m,
        request.damping_s_inv,
        request.samples,
        gravity=request.gravity_m_s2,
        initial_velocity=request.initial_velocity_rad_s,
    )
    reference = reference_solve(
        ReferenceConfig(
            request.angle_deg,
            request.duration_s,
            request.length_m,
            request.damping_s_inv,
            request.samples,
            np.degrees(request.initial_velocity_rad_s),
            request.gravity_m_s2,
        )
    )
    metrics = compare_trajectories(reference, times, angles)
    return SolverRun(
        request, times, angles, reference.times, np.degrees(reference.angle_rad), metrics
    )


def run_recovery(request: ParameterRecoveryRunRequest) -> RecoveryRun:
    """Fit observed samples with production ``fit_posterior``."""
    fit = fit_posterior(
        np.asarray(request.observations.times_s), np.asarray(request.observations.angles_deg)
    )
    return RecoveryRun(request, fit)


def generate_parameter_recovery_dataset(
    noise_sigma_deg: float = 0.0,
    seed: int | None = None,
) -> ParameterRecoveryDataset:
    """Generate estimator-compatible observations with isolated evaluation truth."""
    request = ParameterRecoveryGenerationRequest(noise_sigma_deg=noise_sigma_deg, seed=seed)
    times, angles = simulate(
        RECOVERY_GENERATION_ANGLE,
        RECOVERY_GENERATION_DURATION,
        RECOVERY_GENERATION_LENGTH,
        RECOVERY_GENERATION_DAMPING,
        RECOVERY_GENERATION_COUNT,
        gravity=RECOVERY_GENERATION_GRAVITY,
        initial_velocity=0.0,
    )
    rng = np.random.default_rng(request.seed)
    measured_angles = angles + rng.normal(
        0.0, request.noise_sigma_deg, RECOVERY_GENERATION_COUNT
    )
    observations = ParameterRecoveryRequest(
        times_s=times.tolist(),
        angles_deg=measured_angles.tolist(),
        noise_sigma_deg=request.noise_sigma_deg,
        gravity_m_s2=RECOVERY_GENERATION_GRAVITY,
    )
    truth = ParameterRecoveryGenerationTruth(
        length_m=RECOVERY_GENERATION_LENGTH,
        damping_s_inv=RECOVERY_GENERATION_DAMPING,
        gravity_m_s2=RECOVERY_GENERATION_GRAVITY,
        initial_angle_deg=RECOVERY_GENERATION_ANGLE,
        initial_velocity_rad_s=0.0,
        duration_s=RECOVERY_GENERATION_DURATION,
        sample_rate_hz=RECOVERY_GENERATION_SAMPLE_RATE,
        sample_count=RECOVERY_GENERATION_COUNT,
    )
    return ParameterRecoveryDataset(observations=observations, truth=truth)


def run_prediction(request: PredictionRequest) -> tuple[np.ndarray, np.ndarray]:
    """Generate a frozen-condition trajectory with production ``simulate``."""
    return simulate(request.angle_deg, request.duration_s, request.length_m, request.damping_s_inv)


def run_selection(request: ExperimentSelectionRequest) -> SelectionRun:
    """Select from dashboard candidates using production selector mathematics."""
    posterior = Posterior(
        np.asarray(request.posterior.lengths_m),
        np.asarray(request.posterior.damping_s_inv),
        np.asarray(request.posterior.weights),
    )
    completed = request.completed_angles_deg
    candidates = request.candidate_angles_deg
    selected = choose_experiment(
        posterior,
        completed,
        candidate_angles=candidates,
        noise_sigma_deg=request.noise_sigma_deg,
    )
    scores = score_experiments(
        posterior,
        completed,
        candidate_angles=candidates,
        noise_sigma_deg=request.noise_sigma_deg,
    )
    return SelectionRun(selected, scores)


def run_csv_comparison(request: CsvComparisonRunRequest) -> ComparisonMetrics:
    """Compare normalized CSV samples against the production trajectory."""
    reference = reference_solve(
        ReferenceConfig(
            request.solver.angle_deg,
            request.solver.duration_s,
            request.solver.length_m,
            request.solver.damping_s_inv,
            request.solver.samples,
            np.degrees(request.solver.initial_velocity_rad_s),
            request.solver.gravity_m_s2,
        )
    )
    return compare_trajectories(
        reference, np.asarray(request.csv.times_s), np.asarray(request.csv.angles_deg)
    )

