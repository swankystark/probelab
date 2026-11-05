from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from probelab.engine_test.service import generate_parameter_recovery_dataset
from probelab.main import app


def test_generation_has_expected_grid_and_initial_condition() -> None:
    dataset = generate_parameter_recovery_dataset()

    assert len(dataset.observations.times_s) == 450
    assert dataset.observations.times_s[0] == pytest.approx(0.0)
    assert dataset.observations.times_s[-1] == pytest.approx(15.0)
    assert np.diff(dataset.observations.times_s).min() == pytest.approx(15.0 / 449.0)
    assert dataset.observations.angles_deg[0] == pytest.approx(35.0)


def test_generation_noise_and_seed_are_reproducible() -> None:
    first = generate_parameter_recovery_dataset(noise_sigma_deg=0.5, seed=42)
    second = generate_parameter_recovery_dataset(noise_sigma_deg=0.5, seed=42)
    different = generate_parameter_recovery_dataset(noise_sigma_deg=0.5, seed=43)

    assert first.observations.angles_deg == second.observations.angles_deg
    assert first.observations.angles_deg != different.observations.angles_deg


def test_generation_response_keeps_truth_out_of_observations() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/engine_test/parameter-recovery/generate",
            json={"noise_sigma_deg": 0.0, "seed": 7},
        )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload["observations"]) == {
        "times_s",
        "angles_deg",
        "noise_sigma_deg",
        "time_unit",
        "angle_unit",
    }
    assert payload["truth"] == {
        "length_m": 0.8,
        "damping_s_inv": 0.08,
        "gravity_m_s2": 9.81,
        "initial_angle_deg": 35.0,
        "initial_velocity_rad_s": 0.0,
        "duration_s": 15.0,
        "sample_rate_hz": 30.0,
        "sample_count": 450,
    }


def test_generation_does_not_call_posterior_fitter(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fit(*args: np.ndarray, **kwargs: float) -> None:
        raise AssertionError("fit_posterior must not be used to generate observations")

    monkeypatch.setattr("probelab.engine_test.service.fit_posterior", fail_fit)

    dataset = generate_parameter_recovery_dataset()

    assert dataset.observations.angles_deg[0] == pytest.approx(35.0)
