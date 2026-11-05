"""Deterministic pendulum simulation, fitting, and information-gain selection."""

from dataclasses import dataclass
import math
from typing import Final
import numpy as np
from scipy.integrate import solve_ivp

G: Final[float] = 9.80665
DEFAULT_CANDIDATE_ANGLES: Final[tuple[float, ...]] = (12.0, 24.0, 38.0, 52.0, 66.0)


@dataclass(frozen=True, slots=True)
class Posterior:
    lengths: np.ndarray
    damping: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True, slots=True)
class FitResult:
    length: float
    damping: float
    length_interval: tuple[float, float]
    damping_interval: tuple[float, float]
    residual_rms: float
    adequacy: str
    posterior: Posterior


def simulate(
    angle_deg: float,
    duration_s: float,
    length: float,
    damping: float,
    count: int = 240,
    *,
    gravity: float = G,
    initial_velocity: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the nonlinear damped pendulum with configurable physics defaults."""
    times = np.linspace(0.0, duration_s, count)
    initial = [math.radians(angle_deg), initial_velocity]

    def dynamics(_: float, state: np.ndarray) -> list[float]:
        return [state[1], -2 * damping * state[1] - (gravity / length) * math.sin(state[0])]

    solution = solve_ivp(dynamics, (0.0, duration_s), initial, t_eval=times, rtol=1e-7, atol=1e-9)
    return times, np.degrees(solution.y[0])


def fit_posterior(times: np.ndarray, angles_deg: np.ndarray) -> FitResult:
    """Evaluate a bounded grid posterior against measured angular data."""
    observed = np.asarray(angles_deg, dtype=float)
    sample_times = np.asarray(times, dtype=float)
    if observed.size < 12 or sample_times.size != observed.size or not np.all(np.isfinite(observed)) or not np.all(np.isfinite(sample_times)) or not np.all(np.diff(sample_times) > 0):
        raise ValueError("Observation timestamps and angles must be finite, increasing, and aligned.")
    lengths = np.linspace(0.25, 1.8, 32)
    damping = np.linspace(0.005, 0.35, 28)
    length_grid, damping_grid = np.meshgrid(lengths, damping, indexing="ij")
    scores = np.empty((len(lengths), len(damping)))
    scale = max(float(np.std(observed) * 0.08), 0.35)
    predicted = _grid_trajectory(float(observed[0]), float(sample_times[-1]), length_grid, damping_grid, len(observed))
    scores[:, :] = -0.5 * np.mean((observed[None, None, :] - predicted) ** 2, axis=2) / (scale**2)
    scores -= float(np.max(scores))
    weights = np.exp(scores)
    weights /= float(np.sum(weights))
    l_mesh, b_mesh = length_grid, damping_grid
    best_length = _weighted_quantile(l_mesh, weights, 0.5)
    best_damping = _weighted_quantile(b_mesh, weights, 0.5)
    _, predicted = simulate(float(observed[0]), float(sample_times[-1]), best_length, best_damping, len(observed))
    residual = float(np.sqrt(np.mean((observed - predicted) ** 2)))
    adequacy = "good" if residual < 3.0 else "needs more evidence"
    return FitResult(best_length, best_damping, _credible(l_mesh, weights), _credible(b_mesh, weights), residual, adequacy, Posterior(lengths, damping, weights))


def _grid_trajectory(angle_deg: float, duration_s: float, lengths: np.ndarray, damping: np.ndarray, count: int) -> np.ndarray:
    """Integrate every grid member together with a fixed-step RK4 scheme."""
    dt = duration_s / max(count - 1, 1)
    theta = np.full_like(lengths, math.radians(angle_deg), dtype=float)
    velocity = np.zeros_like(theta)
    result = np.empty((*lengths.shape, count), dtype=float)
    result[..., 0] = np.degrees(theta)
    for index in range(1, count):
        def acceleration(current_theta: np.ndarray, current_velocity: np.ndarray) -> np.ndarray:
            return -2 * damping * current_velocity - (G / lengths) * np.sin(current_theta)
        k1_theta, k1_velocity = velocity, acceleration(theta, velocity)
        k2_theta, k2_velocity = velocity + 0.5 * dt * k1_velocity, acceleration(theta + 0.5 * dt * k1_theta, velocity + 0.5 * dt * k1_velocity)
        k3_theta, k3_velocity = velocity + 0.5 * dt * k2_velocity, acceleration(theta + 0.5 * dt * k2_theta, velocity + 0.5 * dt * k2_velocity)
        k4_theta, k4_velocity = velocity + dt * k3_velocity, acceleration(theta + dt * k3_theta, velocity + dt * k3_velocity)
        theta += dt * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6
        velocity += dt * (k1_velocity + 2 * k2_velocity + 2 * k3_velocity + k4_velocity) / 6
        result[..., index] = np.degrees(theta)
    return result


def _credible(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    return _weighted_quantile(values, weights, 0.16), _weighted_quantile(values, weights, 0.84)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    flat = values.ravel()
    order = np.argsort(flat)
    sorted_weights = weights.ravel()[order]
    cumulative = np.cumsum(sorted_weights / np.sum(sorted_weights))
    return float(flat[order[np.searchsorted(cumulative, quantile)]])


@dataclass(frozen=True, slots=True)
class ExperimentScore:
    """Expected information gain for one candidate experiment angle."""

    angle_deg: float
    eig_nats: float


def score_experiments(
    posterior: Posterior,
    completed_angles: list[float],
    candidate_angles: list[float] | None = None,
    *,
    noise_sigma_deg: float = 1.0,
) -> tuple[ExperimentScore, ...]:
    """Score each usable candidate with the production EIG implementation."""
    if not math.isfinite(noise_sigma_deg) or noise_sigma_deg <= 0.0:
        raise ValueError("noise_sigma_deg must be finite and positive")
    candidates = _validated_candidates(candidate_angles)
    usable = [angle for angle in candidates if all(abs(angle - done) > 6 for done in completed_angles)]
    if not usable:
        usable = candidates
    prior_entropy = _entropy(posterior.weights)
    gains = []
    for angle in usable:
        lengths, damping = np.meshgrid(posterior.lengths, posterior.damping, indexing="ij")
        endpoint = _grid_trajectory(angle, 8.0, lengths, damping, 60)[..., -1]
        values = endpoint.ravel()
        weights = posterior.weights.ravel()
        noise = noise_sigma_deg
        predictions = np.linspace(float(np.min(values)), float(np.max(values)), 9)
        predictive = np.array([float(np.sum(weights * np.exp(-0.5 * ((value - values) / noise) ** 2))) for value in predictions])
        predictive /= float(np.sum(predictive))
        expected_after = 0.0
        for probability, value in zip(predictive, predictions):
            likelihood = np.exp(-0.5 * ((value - values) / noise) ** 2)
            posterior_weights = weights * likelihood
            posterior_weights /= float(np.sum(posterior_weights))
            expected_after += float(probability) * _entropy(posterior_weights)
        gains.append(prior_entropy - expected_after)
    return tuple(ExperimentScore(float(angle), float(eig)) for angle, eig in zip(usable, gains))


def choose_experiment(
    posterior: Posterior,
    completed_angles: list[float],
    candidate_angles: list[float] | None = None,
    *,
    noise_sigma_deg: float = 1.0,
) -> tuple[float, float, float, str]:
    """Select the highest-EIG experiment while preserving the legacy tuple API."""
    scores = score_experiments(
        posterior,
        completed_angles,
        candidate_angles,
        noise_sigma_deg=noise_sigma_deg,
    )
    index = int(np.argmax([score.eig_nats for score in scores]))
    selected = scores[index]
    eig = selected.eig_nats
    return selected.angle_deg, 8.0, eig, f"This angle has the highest expected posterior entropy reduction ({eig:.3f} nats), integrating over noisy endpoint observations with {noise_sigma_deg:g}° measurement noise."


def _validated_candidates(candidate_angles: list[float] | None) -> tuple[float, ...]:
    """Validate caller-supplied candidates without embedding dashboard configuration."""
    if candidate_angles is None:
        return DEFAULT_CANDIDATE_ANGLES
    if not candidate_angles:
        raise ValueError("candidate_angles must not be empty")
    if any(not math.isfinite(angle) for angle in candidate_angles):
        raise ValueError("candidate_angles must contain only finite angles")
    if any(angle <= 0.0 for angle in candidate_angles):
        raise ValueError("candidate_angles must contain only positive angles")
    if any(angle > 180.0 for angle in candidate_angles):
        raise ValueError("candidate_angles must not exceed 180 degrees")
    if len(set(candidate_angles)) != len(candidate_angles):
        raise ValueError("candidate_angles must not contain duplicate angles")
    return tuple(candidate_angles)


def _entropy(weights: np.ndarray) -> float:
    return -float(np.sum(weights * np.log(weights + 1e-12)))
