from __future__ import annotations

from datetime import UTC, datetime
from importlib import import_module

import pytest

from probelab.perception_validation.geometry import evaluate_geometry
from probelab.perception_validation.models import (
    AnnotationRequest,
    GeometryQualityRequest,
    LeakageControlMetadata,
    MaskCentroidTrack,
    Point,
    SamplingMetadata,
    SamplingSelection,
    SamTrackingResult,
    SessionResponse,
    SessionState,
    VideoSource,
)

router_module = import_module("probelab.perception_validation.router")


def _request(frame_indices: list[int], *, calibrated: bool = False) -> GeometryQualityRequest:
    calibration = {
        "reference_start": Point(x=0, y=0),
        "reference_end": Point(x=3, y=4),
        "physical_distance_m": 1.0,
    } if calibrated else {}
    annotation = AnnotationRequest(
        source_width=100,
        source_height=100,
        frame_index=0,
        bob_box=[1, 1, 10, 10],
        positive_points=[Point(x=5, y=5)],
        pivot=Point(x=0, y=0),
        **calibration,
    )
    return GeometryQualityRequest(
        annotation=annotation,
        tracks=[
            MaskCentroidTrack(
                frame_index=frame,
                timestamp_s=frame / 30,
                centroid=Point(x=3, y=4),
                mask_area_px=100,
                confidence=0.9,
                object_id="bob",
            )
            for frame in frame_indices
        ],
        sampling=SamplingMetadata(frame_start=0, frame_end=240, frame_stride=3),
        leakage_controls=LeakageControlMetadata(),
    )


def test_stride_gaps_are_not_reported_as_missing_sampled_slots() -> None:
    result = evaluate_geometry(_request(list(range(0, 241, 3))))

    assert result.retained_frame_count == 81
    assert result.discontinuities.missing_frame_indices == []
    assert result.discontinuities.gap_count == 0
    assert result.discontinuities.largest_gap_frames == 0
    assert result.pixel_tracking_approved is True


def test_missing_sampled_detection_remains_visible_as_one_gap() -> None:
    result = evaluate_geometry(_request([frame for frame in range(0, 241, 3) if frame != 9]))

    assert result.discontinuities.missing_frame_indices == [9]
    assert result.discontinuities.gap_count == 1
    assert result.discontinuities.largest_gap_frames == 1


def test_pixel_gate_is_independent_from_calibration_and_leakage() -> None:
    result = evaluate_geometry(_request(list(range(0, 241, 3))))

    assert result.pixel_tracking_approved is True
    assert result.pixel_geometry_approved is True
    assert result.metric_calibration_approved is False
    assert result.leakage_compliance_approved is False
    assert result.production_handoff_approved is False
    assert result.sendable is False


def test_radius_summary_reports_median_minimum_maximum_and_cv() -> None:
    request = _request([0, 3, 6])
    request = request.model_copy(
        update={
            "tracks": [
                request.tracks[0].model_copy(update={"centroid": Point(x=3, y=4)}),
                request.tracks[1].model_copy(update={"centroid": Point(x=6, y=8)}),
                request.tracks[2].model_copy(update={"centroid": Point(x=9, y=12)}),
            ]
        }
    )

    radius = evaluate_geometry(request).radius

    assert radius.median_px == 10.0
    assert radius.min_px == 5.0
    assert radius.max_px == 15.0
    assert radius.cv > 0


def test_pixel_angle_convention_uses_downward_pivot_axis() -> None:
    request = _request([0, 3, 6]).model_copy(
        update={
            "annotation": _request([0]).annotation.model_copy(
                update={"pivot": Point(x=50, y=0)}
            ),
            "tracks": [
                _request([0]).tracks[0].model_copy(
                    update={"frame_index": 0, "centroid": Point(x=50, y=10)}
                ),
                _request([0]).tracks[0].model_copy(
                    update={"frame_index": 3, "centroid": Point(x=60, y=10)}
                ),
                _request([0]).tracks[0].model_copy(
                    update={"frame_index": 6, "centroid": Point(x=40, y=10)}
                ),
            ]
        }
    )

    angles = [item.angle_deg for item in evaluate_geometry(request).pixel_geometry]

    assert angles == pytest.approx([0.0, 45.0, -45.0])


def test_geometry_endpoint_uses_stored_sampled_sam_tracks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stored SAM observations supply all geometry records and sampled slots."""
    sampled_indices = list(range(0, 241, 3))
    tracks = [
        MaskCentroidTrack(
            frame_index=frame,
            timestamp_s=frame / 30.08,
            centroid=Point(x=3, y=4),
            mask_area_px=100,
            confidence=0.9,
            object_id="bob",
        )
        for frame in sampled_indices
    ]
    sampling = SamplingSelection(
        upload_id="a" * 32,
        source=VideoSource(width=100, height=100, fps=30.08, frame_count=241, duration_s=8),
        source_start_s=0,
        source_end_s=8,
        frame_stride=3,
        frame_index=0,
        source_timestamp_s=0,
    )
    tracking = SamTrackingResult(
        status="completed",
        method="sam_guided_mask",
        propagation_direction="forward",
        sampling=sampling,
        frames_tracked=len(tracks),
        selected_frame_indices=sampled_indices,
        selected_timestamps_s=[frame / 30.08 for frame in sampled_indices],
        tracks=tracks,
    )
    now = datetime.now(UTC)
    state = SessionState(
        session=SessionResponse(
            session_id="b" * 32,
            status="annotated",
            label="tracking regression",
            artifact_root=f"data/perception-validation/{'b' * 32}",
            created_at=now,
            updated_at=now,
        ),
        sam_tracking=tracking,
    )
    monkeypatch.setattr(router_module, "_session_or_404", lambda _session_id: state)
    monkeypatch.setattr(router_module.STORE, "save", lambda updated: updated)

    annotation = _request([]).annotation
    result = router_module.calculate_geometry_quality(
        state.session.session_id,
        GeometryQualityRequest(
            annotation=annotation,
            leakage_controls=LeakageControlMetadata(),
        ),
    )

    assert result.retained_frame_count == 81
    assert [item.frame_index for item in result.pixel_geometry] == sampled_indices
    assert [item.timestamp_s for item in result.pixel_geometry] == [
        frame / 30.08 for frame in sampled_indices
    ]
    assert result.pixel_geometry[-1].timestamp_s == pytest.approx(7.9787, abs=0.0001)
    assert result.discontinuities.missing_frame_indices == []
    assert result.coverage_fraction == 1.0
