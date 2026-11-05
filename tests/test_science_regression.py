"""Regression coverage for the production science engine seams."""

import numpy as np
import pytest

from probelab.science import Posterior, choose_experiment, simulate


def test_default_simulate_and_choose_experiment_characterization() -> None:
    """Given legacy inputs, defaults retain the established solver and selector behavior."""
    times, angles = simulate(24.0, 2.0, 1.2, 0.08, count=32)
    posterior = Posterior(
        lengths=np.array([0.8, 1.2, 1.6]),
        damping=np.array([0.04, 0.08, 0.12]),
        weights=np.full((3, 3), 1 / 9),
    )

    selected_angle, duration, eig, rationale = choose_experiment(posterior, [])

    assert times.shape == (32,)
    assert angles.shape == (32,)
    assert np.isclose(times[0], 0.0)
    assert np.isclose(times[-1], 2.0)
    assert selected_angle in (12.0, 24.0, 38.0, 52.0, 66.0)
    assert duration == 8.0
    assert np.isfinite(eig)
    assert "highest expected posterior entropy reduction" in rationale


def test_simulate_accepts_custom_gravity_and_initial_velocity() -> None:
    """Given valid custom physics inputs, simulation starts with the requested velocity."""
    _, default_angles = simulate(24.0, 2.0, 1.2, 0.08, count=32)
    _, custom_angles = simulate(
        24.0,
        2.0,
        1.2,
        0.08,
        count=32,
        gravity=3.0,
        initial_velocity=0.4,
    )

    assert not np.allclose(default_angles, custom_angles)


def test_choose_experiment_uses_only_supplied_candidates() -> None:
    """Given custom candidates, selection returns one of those candidates only."""
    posterior = Posterior(
        lengths=np.array([0.8, 1.2, 1.6]),
        damping=np.array([0.04, 0.08, 0.12]),
        weights=np.full((3, 3), 1 / 9),
    )

    selected_angle, _, _, _ = choose_experiment(posterior, [], candidate_angles=[180.0])

    assert selected_angle == 180.0


@pytest.mark.parametrize(
    ("candidate_angles", "message"),
    [
        ([], "candidate_angles"),
        ([12.0, 12.0], "duplicate"),
        ([float("nan")], "finite"),
        ([0.0], "positive"),
        ([-5.0], "positive"),
        ([181.0], "180"),
        ([float("inf")], "finite"),
    ],
)
def test_choose_experiment_rejects_invalid_candidates(
    candidate_angles: list[float], message: str,
) -> None:
    """Given malformed candidate angles, the selector names the rejected input."""
    posterior = Posterior(
        lengths=np.array([0.8, 1.2]),
        damping=np.array([0.04, 0.08]),
        weights=np.full((2, 2), 0.25),
    )

    with pytest.raises(ValueError, match=message):
        choose_experiment(posterior, [], candidate_angles=candidate_angles)
