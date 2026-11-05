import math

import numpy as np
import pytest

from probelab.engine_test_evaluation import (
    EvaluationInputError,
    ParameterInterval,
    ReferenceConfig,
    compare_trajectories,
    generate_synthetic_data,
    reference_solve,
    summarize_uncertainty,
    estimate_period,
    convergence_summary,
)


def test_reference_matches_production_default_solver() -> None:
    """Given default settings, the independent reference agrees with ProbeLab."""
    from probelab.science import simulate

    config = ReferenceConfig(angle_deg=35.0, duration_s=6.0, length_m=0.8, damping=0.06, count=240)

    reference = reference_solve(config)
    times, production_angles = simulate(35.0, 6.0, 0.8, 0.06, 240)
    metrics = compare_trajectories(reference, times, production_angles)

    assert metrics.angle_rmse_deg < 1e-5
    assert metrics.angle_max_error_deg < 5e-5
    assert metrics.tolerance_deg == 1e-4


def test_reference_supports_damping_velocity_gravity_and_nonlinear_angle() -> None:
    """Given a nonlinear moving start, the reference produces finite physical state."""
    result = reference_solve(
        ReferenceConfig(
            angle_deg=78.0,
            duration_s=8.0,
            length_m=1.2,
            damping=0.18,
            initial_velocity_deg_s=24.0,
            gravity=3.7,
            count=400,
        )
    )

    assert np.all(np.isfinite(result.angle_rad))
    assert np.all(np.isfinite(result.angular_velocity_rad_s))
    assert np.all(result.energy_per_mass >= 0.0)
    assert result.angle_rad[0] == pytest.approx(math.radians(78.0))
    assert result.angular_velocity_rad_s[0] == pytest.approx(math.radians(24.0))
    assert result.energy_per_mass[-1] < result.energy_per_mass[0]


def test_synthetic_data_keeps_truth_separate_from_observations() -> None:
    """Given a truth record and noise, observed input is distinct and reproducible."""
    truth = ReferenceConfig(angle_deg=30.0, duration_s=4.0, length_m=0.7, damping=0.04)

    first = generate_synthetic_data(truth, sample_count=120, noise_std_deg=0.25, seed=7)
    second = generate_synthetic_data(truth, sample_count=120, noise_std_deg=0.25, seed=7)

    assert first.truth == truth
    assert first.observed_angles_deg is not first.truth_angles_deg
    assert np.array_equal(first.observed_angles_deg, second.observed_angles_deg)
    assert not np.array_equal(first.observed_angles_deg, first.truth_angles_deg)


def test_convergence_and_period_summary_are_finite() -> None:
    """Given a low-damping oscillator, convergence and period are measurable."""
    config = ReferenceConfig(angle_deg=12.0, duration_s=12.0, length_m=1.0, damping=0.0, count=600)
    result = reference_solve(config)

    period = estimate_period(result.times, result.angle_rad)
    convergence = convergence_summary(config, sample_counts=(80, 160, 320))

    assert period.period_s == pytest.approx(2.006, abs=0.03)
    assert period.uncertainty_s >= 0.0
    assert convergence.errors_deg[-1] < convergence.errors_deg[0]
    assert convergence.observed_order > 3.0


def test_uncertainty_summary_marks_wide_intervals_unidentifiable() -> None:
    """Given estimate intervals, summary reports finite width and identifiability."""
    summary = summarize_uncertainty(
        (
            ParameterInterval("length_m", 0.8, 0.79, 0.81),
            ParameterInterval("damping", 0.12, 0.02, 0.30),
        )
    )

    assert summary[0].identifiable is True
    assert summary[1].identifiable is False
    assert all(math.isfinite(item.relative_width) for item in summary)


@pytest.mark.parametrize(
    "config",
    [
        ReferenceConfig(angle_deg=float("nan"), duration_s=4.0, length_m=1.0, damping=0.0),
        ReferenceConfig(angle_deg=10.0, duration_s=0.0, length_m=1.0, damping=0.0),
        ReferenceConfig(angle_deg=10.0, duration_s=4.0, length_m=-1.0, damping=0.0),
        ReferenceConfig(angle_deg=10.0, duration_s=4.0, length_m=1.0, damping=-0.1),
    ],
)
def test_reference_rejects_malformed_or_nonfinite_configuration(config: ReferenceConfig) -> None:
    """Given invalid physics inputs, the reference raises a structured error."""
    with pytest.raises(EvaluationInputError):
        reference_solve(config)


def test_comparison_rejects_stale_or_nonmonotonic_samples() -> None:
    """Given stale or malformed samples, comparison does not report success."""
    reference = reference_solve(ReferenceConfig(20.0, 3.0, 1.0, 0.05, count=60))

    with pytest.raises(EvaluationInputError):
        compare_trajectories(reference, np.array([0.0, 0.5, 0.4]), np.array([20.0, 10.0, 5.0]))
