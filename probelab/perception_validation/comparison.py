"""Independent ScienceQuest CSV parsing, alignment, and comparison."""

from __future__ import annotations

from csv import DictReader
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np

from .models import CsvComparisonReport, VideoTrajectorySample


class CsvComparisonError(ValueError):
    """A reference CSV cannot be used as an evaluation source."""


@dataclass(frozen=True, slots=True)
class _ReferenceTrajectory:
    timestamps_s: np.ndarray
    angles_deg: np.ndarray
    timestamp_column: str
    angular_column: str
    angular_unit: str


def compare_reference_csv(
    path: Path,
    csv_upload_id: str,
    video_trajectory: list[VideoTrajectorySample],
    alignment_search_s: float,
) -> CsvComparisonReport:
    """Compare signed video angles in degrees against a CSV in evaluation only.

    Video angles use the geometry convention: degrees from the image's downward
    vertical, with rightward displacement positive. Radian CSV values are
    converted to degrees without changing their sign.
    """
    reference = _read_reference_csv(path)
    video_times = np.asarray([sample.timestamp_s for sample in video_trajectory], dtype=float)
    video_angles = np.asarray([sample.angle_deg for sample in video_trajectory], dtype=float)
    offset, uncertainty = _find_offset(
        reference.timestamps_s,
        reference.angles_deg,
        video_times,
        video_angles,
        alignment_search_s,
    )
    aligned_times = video_times
    shifted_reference_times = reference.timestamps_s + offset
    overlap = (aligned_times >= shifted_reference_times[0]) & (
        aligned_times <= shifted_reference_times[-1]
    )
    if int(np.count_nonzero(overlap)) < 2:
        raise CsvComparisonError("CSV and video trajectory do not overlap at two sampled frames")
    interpolated = np.interp(aligned_times[overlap], shifted_reference_times, reference.angles_deg)
    errors = video_angles[overlap] - interpolated
    return CsvComparisonReport(
        csv_upload_id=csv_upload_id,
        timestamp_column=reference.timestamp_column,
        angular_column=reference.angular_column,
        angular_unit=reference.angular_unit,
        row_count=int(reference.timestamps_s.size),
        alignment_method="timestamp_offset_plus_linear_interpolation",
        alignment_search_s=alignment_search_s,
        estimated_time_offset_s=float(offset),
        alignment_uncertainty_s=float(uncertainty),
        overlap_duration_s=float(aligned_times[overlap][-1] - aligned_times[overlap][0]),
        overlap_frame_count=int(np.count_nonzero(overlap)),
        angular_rmse_deg=float(np.sqrt(np.mean(errors**2))),
        angular_max_error_deg=float(np.max(np.abs(errors))),
        mismatch_diagnostics=[
            *_mismatch_diagnostics(reference, video_times, overlap),
            (
                "offset candidates required at least 50% of the shorter trajectory "
                f"({0.5 * min(np.ptp(reference.timestamps_s), np.ptp(video_times)):.3f} s) "
                "to overlap"
            ),
        ],
    )


def _read_reference_csv(path: Path) -> _ReferenceTrajectory:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            data_lines = (
                line for line in handle if line.strip() and not line.lstrip().startswith("#")
            )
            reader = DictReader(data_lines)
            headers = reader.fieldnames or []
            timestamp_column = _select_column(
                headers, {"timestamp_s", "time_s", "Time (s)"}, "timestamp"
            )
            angular_column = _select_column(
                headers, {"angle_deg", "angle_rad", "Angle (deg)"}, "angular"
            )
            rows = list(reader)
    except (OSError, UnicodeDecodeError) as error:
        raise CsvComparisonError("reference CSV could not be read as UTF-8 text") from error
    if not rows:
        raise CsvComparisonError("reference CSV must contain at least two data rows")
    timestamps: list[float] = []
    angles: list[float] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            timestamp = float(row[timestamp_column] or "")
            angle = float(row[angular_column] or "")
        except (KeyError, TypeError, ValueError) as error:
            raise CsvComparisonError(f"CSV row {row_number} has non-numeric timestamp or angle") from error
        if not isfinite(timestamp) or not isfinite(angle):
            raise CsvComparisonError(f"CSV row {row_number} contains a non-finite value")
        timestamps.append(timestamp)
        angles.append(angle)
    timestamp_array = np.asarray(timestamps, dtype=float)
    angle_array = np.asarray(angles, dtype=float)
    if np.any(timestamp_array < 0):
        raise CsvComparisonError("CSV timestamps must be non-negative seconds")
    if np.any(np.diff(timestamp_array) <= 0):
        raise CsvComparisonError("CSV timestamps must be strictly increasing with no duplicates")
    unit = "rad" if angular_column == "angle_rad" else "deg"
    if unit == "rad":
        angle_array = np.rad2deg(angle_array)
    return _ReferenceTrajectory(timestamp_array, angle_array, timestamp_column, angular_column, unit)


def _select_column(headers: list[str], candidates: set[str], kind: str) -> str:
    matches = [header for header in headers if header in candidates]
    if len(matches) != 1:
        raise CsvComparisonError(
            f"CSV must contain exactly one supported {kind} column: {', '.join(sorted(candidates))}"
        )
    return matches[0]


def _find_offset(
    reference_times: np.ndarray,
    reference_angles: np.ndarray,
    video_times: np.ndarray,
    video_angles: np.ndarray,
    search_s: float,
) -> tuple[float, float]:
    """Return the shift added to reference timestamps to align them to video.

    A positive offset advances the reference clock (``reference_time + offset``);
    a negative offset delays it. Both timestamp arrays are measured in seconds.
    """
    minimum_overlap_s = 0.5 * min(
        float(np.ptp(reference_times)),
        float(np.ptp(video_times)),
    )
    sample_interval_s = float(np.median(np.diff(video_times)))
    maximum_grid_interval_s = sample_interval_s / 2
    grid_intervals = max(1, int(np.ceil(2 * search_s / maximum_grid_interval_s)))
    grid = np.linspace(-search_s, search_s, grid_intervals + 1)
    scores = np.asarray(
        [
            _score_offset(
                offset,
                reference_times,
                reference_angles,
                video_times,
                video_angles,
                minimum_overlap_s,
            )
            for offset in grid
        ]
    )
    valid = np.isfinite(scores)
    if not np.any(valid):
        raise CsvComparisonError(
            "no alignment candidate has at least 50% overlap of the shorter trajectory "
            "within the configured search window"
        )
    best_index = int(np.nanargmin(scores))
    left = grid[max(0, best_index - 1)]
    right = grid[min(len(grid) - 1, best_index + 1)]
    refined = np.linspace(left, right, 101)
    refined_scores = np.asarray(
        [
            _score_offset(
                offset,
                reference_times,
                reference_angles,
                video_times,
                video_angles,
                minimum_overlap_s,
            )
            for offset in refined
        ]
    )
    refined_index = int(np.nanargmin(refined_scores))
    return float(refined[refined_index]), float((refined[1] - refined[0]) / 2)


def _score_offset(
    offset: float,
    reference_times: np.ndarray,
    reference_angles: np.ndarray,
    video_times: np.ndarray,
    video_angles: np.ndarray,
    minimum_overlap_s: float,
) -> float:
    """Score signed angles in degrees after shifting reference times in seconds."""
    shifted = reference_times + offset
    overlap = (video_times >= shifted[0]) & (video_times <= shifted[-1])
    if (
        int(np.count_nonzero(overlap)) < 2
        or float(video_times[overlap][-1] - video_times[overlap][0]) < minimum_overlap_s
    ):
        return float("nan")
    predicted = np.interp(video_times[overlap], shifted, reference_angles)
    return float(np.sqrt(np.mean((video_angles[overlap] - predicted) ** 2)))


def _mismatch_diagnostics(
    reference: _ReferenceTrajectory,
    video_times: np.ndarray,
    overlap: np.ndarray,
) -> list[str]:
    diagnostics: list[str] = []
    if int(np.count_nonzero(overlap)) < video_times.size:
        diagnostics.append("comparison uses only the timestamp intersection; non-overlap frames were excluded")
    if reference.angular_unit == "rad":
        diagnostics.append("reference angles converted from radians to degrees before error calculation")
    return diagnostics
