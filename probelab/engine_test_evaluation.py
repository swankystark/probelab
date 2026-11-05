"""Evaluation-only numerical reference tools for the engine-test dashboard.

This module deliberately does not import production routes or production physics.
It owns truth generation, an independent DOP853 integration, and comparison summaries.
"""

from dataclasses import dataclass, replace
import math
from typing import Final, Mapping

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

DEFAULT_GRAVITY: Final[float] = 9.80665
REFERENCE_RTOL: Final[float] = 1e-11
REFERENCE_ATOL: Final[float] = 1e-13


@dataclass(frozen=True, slots=True)
class EvaluationInputError(ValueError):
    """Structured input rejection for evaluation-only calls."""

    field: str
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.field}: {self.message}"


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    """Physical and sampling inputs for one deterministic evaluation run."""

    angle_deg: float
    duration_s: float
    length_m: float
    damping: float
    count: int = 240
    initial_velocity_deg_s: float = 0.0
    gravity: float = DEFAULT_GRAVITY


@dataclass(frozen=True, slots=True)
class ReferenceTrajectory:
    """Reference state and specific mechanical energy sampled on one time grid."""

    times: np.ndarray
    angle_rad: np.ndarray
    angular_velocity_rad_s: np.ndarray
    energy_per_mass: np.ndarray
    config: ReferenceConfig


@dataclass(frozen=True, slots=True)
class SyntheticData:
    """Observed samples plus separate hidden truth for recovery experiments."""

    truth: ReferenceConfig
    times: np.ndarray
    truth_angles_deg: np.ndarray
    observed_angles_deg: np.ndarray


@dataclass(frozen=True, slots=True)
class ComparisonMetrics:
    """Finite comparison metrics and the declared angle tolerance."""

    angle_rmse_deg: float
    angle_max_error_deg: float
    velocity_rmse_deg_s: float
    velocity_max_error_deg_s: float
    energy_rmse_per_mass: float
    energy_max_error_per_mass: float
    tolerance_deg: float = 1e-4


@dataclass(frozen=True, slots=True)
class PeriodSummary:
    """Estimated oscillation period and the spread of detected cycles."""

    period_s: float
    uncertainty_s: float
    cycles: int


@dataclass(frozen=True, slots=True)
class ConvergenceSummary:
    """Errors against the finest reference grid and observed convergence order."""

    sample_counts: tuple[int, ...]
    errors_deg: tuple[float, ...]
    observed_order: float


@dataclass(frozen=True, slots=True)
class ParameterInterval:
    """Point estimate and central interval for one recovered parameter."""

    name: str
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class UncertaintySummary:
    """Normalized uncertainty and a deterministic identifiability flag."""

    name: str
    estimate: float
    interval: tuple[float, float]
    relative_width: float
    identifiable: bool


def reference_solve(config: ReferenceConfig) -> ReferenceTrajectory:
    """Integrate the nonlinear pendulum with high-accuracy independent settings."""
    _validate_config(config)
    times = np.linspace(0.0, config.duration_s, config.count)
    initial = [math.radians(config.angle_deg), math.radians(config.initial_velocity_deg_s)]

    def dynamics(_: float, state: np.ndarray) -> list[float]:
        return [state[1], -2.0 * config.damping * state[1] - config.gravity / config.length_m * math.sin(state[0])]

    solution = solve_ivp(dynamics, (0.0, config.duration_s), initial, t_eval=times, method="DOP853", rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
    if not solution.success or solution.y.shape != (2, config.count):
        raise EvaluationInputError("integration", "reference_failed", "SciPy did not return the requested state grid.")
    angle = solution.y[0]
    velocity = solution.y[1]
    energy = 0.5 * (config.length_m * velocity) ** 2 + config.gravity * config.length_m * (1.0 - np.cos(angle))
    return ReferenceTrajectory(times, angle, velocity, energy, config)


def generate_synthetic_data(config: ReferenceConfig, sample_count: int = 240, noise_std_deg: float = 0.0, seed: int = 0) -> SyntheticData:
    """Generate reproducible observations while retaining truth in a separate field."""
    if sample_count < 12:
        raise EvaluationInputError("sample_count", "too_short", "At least 12 samples are required.")
    if not math.isfinite(noise_std_deg) or noise_std_deg < 0.0:
        raise EvaluationInputError("noise_std_deg", "invalid_value", "Noise must be finite and non-negative.")
    truth = reference_solve(replace(config, count=sample_count))
    rng = np.random.default_rng(seed)
    observed = truth.angle_rad * 180.0 / math.pi + rng.normal(0.0, noise_std_deg, sample_count)
    return SyntheticData(config, truth.times.copy(), np.degrees(truth.angle_rad), observed)


def compare_trajectories(reference: ReferenceTrajectory, times: np.ndarray, angles_deg: np.ndarray) -> ComparisonMetrics:
    """Compare sampled angles with the reference, deriving velocity and energy consistently."""
    sample_times = np.asarray(times, dtype=float)
    observed = np.asarray(angles_deg, dtype=float)
    _validate_samples(sample_times, observed)
    reference_angle = np.interp(sample_times, reference.times, reference.angle_rad)
    reference_velocity = np.interp(sample_times, reference.times, reference.angular_velocity_rad_s)
    observed_rad = np.radians(observed)
    observed_velocity = np.gradient(observed_rad, sample_times)
    observed_energy = 0.5 * (reference.config.length_m * observed_velocity) ** 2 + reference.config.gravity * reference.config.length_m * (1.0 - np.cos(observed_rad))
    reference_energy = np.interp(sample_times, reference.times, reference.energy_per_mass)
    angle_error = np.degrees(observed_rad - reference_angle)
    velocity_error = np.degrees(observed_velocity - reference_velocity)
    energy_error = observed_energy - reference_energy
    return ComparisonMetrics(_rmse(angle_error), float(np.max(np.abs(angle_error))), _rmse(velocity_error), float(np.max(np.abs(velocity_error))), _rmse(energy_error), float(np.max(np.abs(energy_error))))


def estimate_period(times: np.ndarray, angle_rad: np.ndarray) -> PeriodSummary:
    """Estimate period from positive-angle peaks and report cycle spread."""
    sample_times = np.asarray(times, dtype=float)
    angles = np.asarray(angle_rad, dtype=float)
    _validate_samples(sample_times, angles)
    peaks, _ = find_peaks(angles)
    periods = np.diff(sample_times[peaks])
    if periods.size == 0:
        raise EvaluationInputError("angle_rad", "period_unavailable", "At least two positive-angle peaks are required.")
    return PeriodSummary(float(np.mean(periods)), float(np.std(periods)), int(periods.size))


def convergence_summary(config: ReferenceConfig, sample_counts: tuple[int, ...] = (80, 160, 320)) -> ConvergenceSummary:
    """Measure self-convergence against the highest-resolution reference trajectory."""
    if len(sample_counts) < 2 or any(count < 12 for count in sample_counts):
        raise EvaluationInputError("sample_counts", "invalid_value", "At least two counts of 12 or more are required.")
    ordered = tuple(sorted(set(sample_counts)))
    finest = reference_solve(replace(config, count=ordered[-1]))
    errors = tuple(compare_trajectories(finest, reference_solve(replace(config, count=count)).times, np.degrees(reference_solve(replace(config, count=count)).angle_rad)).angle_rmse_deg for count in ordered[:-1]) + (0.0,)
    ratio = errors[-3] / max(errors[-2], np.finfo(float).tiny) if len(errors) >= 3 else 0.0
    order = math.log2(ratio) if ratio > 0.0 else 0.0
    return ConvergenceSummary(ordered, errors, order)


def summarize_uncertainty(intervals: tuple[ParameterInterval, ...]) -> tuple[UncertaintySummary, ...]:
    """Convert parameter intervals into finite relative widths and flags."""
    summaries: list[UncertaintySummary] = []
    for item in intervals:
        values = (item.estimate, item.lower, item.upper)
        if not all(math.isfinite(value) for value in values) or item.lower > item.upper:
            raise EvaluationInputError(item.name, "invalid_interval", "Interval values must be finite and ordered.")
        scale = max(abs(item.estimate), np.finfo(float).eps)
        width = (item.upper - item.lower) / scale
        summaries.append(UncertaintySummary(item.name, item.estimate, (item.lower, item.upper), width, width <= 0.25))
    return tuple(summaries)


def _validate_config(config: ReferenceConfig) -> None:
    values = (config.angle_deg, config.duration_s, config.length_m, config.damping, config.initial_velocity_deg_s, config.gravity)
    if not all(math.isfinite(value) for value in values):
        raise EvaluationInputError("config", "nonfinite", "All physical inputs must be finite.")
    if config.duration_s <= 0.0 or config.length_m <= 0.0 or config.gravity <= 0.0 or config.damping < 0.0:
        raise EvaluationInputError("config", "invalid_physics", "Duration, length, and gravity must be positive; damping cannot be negative.")
    if config.count < 12:
        raise EvaluationInputError("count", "too_short", "At least 12 samples are required.")


def _validate_samples(times: np.ndarray, values: np.ndarray) -> None:
    if times.ndim != 1 or values.ndim != 1 or times.size != values.size or times.size < 3:
        raise EvaluationInputError("samples", "invalid_shape", "Times and values must be aligned one-dimensional arrays with at least 3 rows.")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)) or not np.all(np.diff(times) > 0.0):
        raise EvaluationInputError("samples", "invalid_samples", "Samples must be finite and strictly increasing.")


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))
