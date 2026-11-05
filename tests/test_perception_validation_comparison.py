from __future__ import annotations

import numpy as np
import pytest

from probelab.perception_validation.comparison import (
    _find_offset,
    compare_reference_csv,
)
from probelab.perception_validation.models import VideoTrajectorySample


@pytest.mark.parametrize("known_offset_s", [0.37, -0.42])
def test_find_offset_returns_shift_added_to_reference_clock(known_offset_s: float) -> None:
    reference_times = np.linspace(2.0, 12.0, 101)
    reference_angles = 3.0 * reference_times**2 - 4.0 * reference_times
    video_times = np.linspace(2.5, 11.5, 73) + known_offset_s
    video_angles = np.interp(video_times - known_offset_s, reference_times, reference_angles)

    estimated_offset_s, _uncertainty_s = _find_offset(
        reference_times,
        reference_angles,
        video_times,
        video_angles,
        0.8,
    )

    assert estimated_offset_s == pytest.approx(known_offset_s, abs=0.002)


def test_comparison_converts_radians_and_preserves_signed_video_angles(tmp_path) -> None:
    timestamps_s = [2.0, 2.7, 3.5, 4.4, 5.0]
    signed_angles_deg = [-25.0, -11.0, 2.0, 17.0, 31.0]
    csv_path = tmp_path / "reference.csv"
    csv_path.write_text(
        "time_s,angle_rad\n"
        + "".join(
            f"{timestamp},{np.deg2rad(angle)}\n"
            for timestamp, angle in zip(timestamps_s, signed_angles_deg, strict=True)
        ),
        encoding="utf-8",
    )
    video = [
        VideoTrajectorySample(
            timestamp_s=timestamp,
            angle_deg=angle,
            frame_index=index,
        )
        for index, (timestamp, angle) in enumerate(zip(timestamps_s, signed_angles_deg, strict=True))
    ]

    report = compare_reference_csv(csv_path, "a" * 32, video, 0.05)

    assert report.estimated_time_offset_s == pytest.approx(0.0, abs=0.001)
    assert report.angular_unit == "rad"
    assert report.angular_rmse_deg == pytest.approx(0.0, abs=1e-10)
