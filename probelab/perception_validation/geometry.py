"""Temporary, non-production geometry and observation-quality calculations."""

from __future__ import annotations

from collections.abc import Iterable
import math
from statistics import fmean, median

from .models import (
    AnnotationRequest,
    ConfidenceSummary,
    DiscontinuitySummary,
    ExclusionRegion,
    FrameGeometry,
    GeometryQualityRequest,
    GeometryQualityResult,
    IndependentAccuracyReport,
    MaskCentroidTrack,
    Point,
    QualityGateSummary,
    RadiusSummary,
)


def evaluate_geometry(request: GeometryQualityRequest) -> GeometryQualityResult:
    """Compute blind geometry and quality from annotations and mask centroids."""
    annotation = request.annotation
    separation_px = annotation.endpoint_pixel_separation
    metres_per_pixel = (
        annotation.physical_distance_m / separation_px
        if separation_px is not None and annotation.physical_distance_m is not None
        else None
    )
    expected_frames = _expected_frames(request)
    expected_frame_set = set(expected_frames)
    excluded_frames = _excluded_frames(request.tracks, request.leakage_controls.regions)
    retained = [
        track
        for track in request.tracks
        if track.frame_index in expected_frame_set and track.frame_index not in excluded_frames
    ]
    retained.sort(key=lambda track: track.frame_index)
    pixel_geometry = [_frame_geometry(track, annotation, None) for track in retained]
    calibrated_geometry = (
        [_frame_geometry(track, annotation, metres_per_pixel) for track in retained]
        if metres_per_pixel is not None
        else []
    )
    confidence = _confidence_summary(retained, request.thresholds.minimum_confidence)
    radius = _radius_summary(pixel_geometry, metres_per_pixel, request.thresholds.maximum_radius_cv)
    discontinuities = _discontinuities(
        expected_frames,
        retained,
        request.thresholds.maximum_jump_px,
    )
    coverage = len(retained) / len(expected_frames) if expected_frames else 0.0
    pixel_reasons = _pixel_geometry_reasons(
        coverage,
        radius,
        discontinuities,
        confidence,
        request,
    )
    sam_reasons = _sam_tracking_quality_reasons(request, retained)
    calibration_reasons = _metric_calibration_reasons(metres_per_pixel)
    leakage_reasons = _leakage_compliance_reasons(request)
    production_reasons = [
        *sam_reasons,
        *pixel_reasons,
        *calibration_reasons,
        *leakage_reasons,
    ]
    if len(pixel_geometry) < 12:
        production_reasons.append("pixel tracking requires at least 12 retained measurements")
    production_approved = not production_reasons
    result = GeometryQualityResult(
        endpoint_separation_px=separation_px,
        confirmed_physical_distance_m=annotation.physical_distance_m,
        metres_per_pixel=metres_per_pixel,
        calibration_available=metres_per_pixel is not None,
        calibration_status="available" if metres_per_pixel is not None else "unavailable",
        coverage_fraction=coverage,
        retained_frame_count=len(retained),
        excluded_frames=excluded_frames,
        confidence=confidence,
        radius=radius,
        discontinuities=discontinuities,
        sam_tracking_quality=QualityGateSummary(
            approved=not sam_reasons,
            reasons=sam_reasons,
        ),
        pixel_geometry_quality=QualityGateSummary(
            approved=not pixel_reasons,
            reasons=pixel_reasons,
        ),
        metric_calibration_validity=QualityGateSummary(
            approved=not calibration_reasons,
            reasons=calibration_reasons,
        ),
        simulator_leakage_compliance=QualityGateSummary(
            approved=not leakage_reasons,
            reasons=leakage_reasons,
        ),
        production_handoff_eligibility=QualityGateSummary(
            approved=production_approved,
            reasons=production_reasons,
        ),
        thresholds=request.thresholds,
        pixel_geometry=pixel_geometry,
        calibrated_geometry=calibrated_geometry,
        pixel_tracking_approved=not pixel_reasons,
        pixel_geometry_approved=not pixel_reasons,
        pixel_geometry_reasons=pixel_reasons,
        sam_tracking_quality_approved=not sam_reasons,
        sam_tracking_quality_reasons=sam_reasons,
        metric_calibration_approved=not calibration_reasons,
        metric_calibration_reasons=calibration_reasons,
        leakage_compliance_approved=not leakage_reasons,
        leakage_compliance_reasons=leakage_reasons,
        leakage_sanitization_applied=request.leakage_sanitization_applied,
        production_handoff_approved=production_approved,
        production_handoff_reasons=production_reasons,
        sendable=production_approved,
        sendability_reasons=production_reasons,
        leakage_controls=request.leakage_controls,
        independent_accuracy=IndependentAccuracyReport(),
    )
    return result


def _expected_frames(request: GeometryQualityRequest) -> tuple[int, ...]:
    if request.expected_frame_indices is not None:
        return tuple(request.expected_frame_indices)
    sampling = request.sampling
    if sampling is None:
        return ()
    return tuple(range(sampling.frame_start, sampling.frame_end + 1, sampling.frame_stride))


def _distance(first: Point, second: Point) -> float:
    """Return Euclidean distance for the strict Point model's coordinates."""
    return math.hypot(second.x - first.x, second.y - first.y)


def _frame_geometry(
    track: MaskCentroidTrack,
    annotation: AnnotationRequest,
    distance_scale: float | None,
) -> FrameGeometry:
    dx = track.centroid.x - annotation.pivot.x
    dy = track.centroid.y - annotation.pivot.y
    pixel_distance = math.hypot(dx, dy)
    return FrameGeometry(
        frame_index=track.frame_index,
        timestamp_s=track.timestamp_s,
        pixel_distance=pixel_distance,
        metric_distance_m=pixel_distance * distance_scale if distance_scale is not None else None,
        angle_deg=math.degrees(math.atan2(dx, dy)),
        confidence=track.confidence,
    )


def _excluded_frames(
    tracks: Iterable[MaskCentroidTrack],
    regions: list[ExclusionRegion],
) -> dict[int, str]:
    enabled_regions = [region for region in regions if region.enabled]
    excluded: dict[int, str] = {}
    for track in tracks:
        for region in enabled_regions:
            if region.left <= track.centroid.x <= region.right and region.top <= track.centroid.y <= region.bottom:
                excluded[track.frame_index] = f"excluded by {region.kind} {region.mode} region"
                break
    return excluded


def _confidence_summary(
    tracks: list[MaskCentroidTrack], threshold: float
) -> ConfidenceSummary:
    values = [track.confidence for track in tracks]
    return ConfidenceSummary(
        mean=fmean(values) if values else 0.0,
        minimum=min(values) if values else 0.0,
        below_threshold_count=sum(value < threshold for value in values),
    )


def _radius_summary(
    geometry: list[FrameGeometry], metres_per_pixel: float | None, maximum_cv: float
) -> RadiusSummary:
    radii = [item.pixel_distance for item in geometry]
    if not radii:
        return RadiusSummary(
            median_px=0.0,
            min_px=0.0,
            max_px=0.0,
            cv=0.0,
            median_m=None,
            status="not_checked",
        )
    center = median(radii)
    deviation = math.sqrt(fmean((radius - center) ** 2 for radius in radii))
    coefficient = deviation / center if center > 0.0 else math.inf
    return RadiusSummary(
        median_px=center,
        min_px=min(radii),
        max_px=max(radii),
        cv=coefficient,
        median_m=center * metres_per_pixel if metres_per_pixel is not None else None,
        status="stable" if coefficient <= maximum_cv else "unstable",
    )


def _discontinuities(
    expected_frames: tuple[int, ...],
    tracks: list[MaskCentroidTrack],
    maximum_jump_px: float,
) -> DiscontinuitySummary:
    expected_set = set(expected_frames)
    observed = {
        track.frame_index: track for track in tracks if track.frame_index in expected_set
    }
    missing = [frame for frame in expected_frames if frame not in observed]
    missing_positions = [index for index, frame in enumerate(expected_frames) if frame not in observed]
    gap_count = 0
    largest_gap_frames = 0
    current_gap = 0
    previous_position: int | None = None
    for position in missing_positions:
        if previous_position is None or position != previous_position + 1:
            gap_count += 1
            current_gap = 1
        else:
            current_gap += 1
        largest_gap_frames = max(largest_gap_frames, current_gap)
        previous_position = position
    observed_tracks = sorted(observed.values(), key=lambda track: track.frame_index)
    jumps = [
        right.frame_index
        for left, right in zip(observed_tracks, observed_tracks[1:])
        if _distance(left.centroid, right.centroid) > maximum_jump_px
    ]
    return DiscontinuitySummary(
        missing_frame_indices=missing,
        gap_count=gap_count,
        largest_gap_frames=largest_gap_frames,
        jump_frame_indices=jumps,
        jump_count=len(jumps),
    )


def _pixel_geometry_reasons(
    coverage: float,
    radius: RadiusSummary,
    discontinuities: DiscontinuitySummary,
    confidence: ConfidenceSummary,
    request: GeometryQualityRequest,
) -> list[str]:
    thresholds = request.thresholds
    reasons: list[str] = []
    if coverage < thresholds.minimum_coverage_fraction:
        reasons.append("coverage below configured threshold")
    if radius.status == "unstable":
        reasons.append("radius stability below configured threshold")
    if discontinuities.largest_gap_frames > thresholds.maximum_gap_frames:
        reasons.append("tracking gap exceeds configured threshold")
    if discontinuities.jump_count:
        reasons.append("centroid jump exceeds configured threshold")
    if confidence.below_threshold_count:
        reasons.append("one or more retained masks are below confidence threshold")
    return reasons


def _sam_tracking_quality_reasons(
    request: GeometryQualityRequest,
    retained: list[MaskCentroidTrack],
) -> list[str]:
    reasons = list(request.sam_tracking_rejection_reasons)
    if request.sam_tracking_status in {"failed", "partial"}:
        reasons.append(f"SAM tracking status is {request.sam_tracking_status}")
    if not retained:
        reasons.append("SAM tracking produced no sampled mask-derived tracks")
    return list(dict.fromkeys(reasons))


def _metric_calibration_reasons(metres_per_pixel: float | None) -> list[str]:
    if metres_per_pixel is None:
        return ["metric calibration is required for production handoff"]
    return []


def _leakage_compliance_reasons(request: GeometryQualityRequest) -> list[str]:
    required_kinds = {
        "simulator_label",
        "displayed_parameters",
        "live_angle_values",
        "controls",
    }
    enabled_kinds = {
        region.kind
        for region in request.leakage_controls.regions
        if region.enabled and region.mode == "mask"
    }
    if not request.leakage_controls.regions:
        reasons = ["no simulator leakage exclusion metadata was supplied"]
    else:
        reasons = []
    missing_kinds = sorted(required_kinds - enabled_kinds)
    if missing_kinds:
        reasons.append(
            "missing enabled mask regions for: " + ", ".join(missing_kinds)
        )
    if not request.leakage_sanitization_applied:
        reasons.append(
            "tracking was not produced from video frames with the saved exclusion masks applied"
        )
    return reasons


def attach_independent_accuracy(
    result: GeometryQualityResult,
    accuracy: IndependentAccuracyReport,
) -> GeometryQualityResult:
    """Attach evaluation-only accuracy without changing blind geometry or sendability."""
    return result.model_copy(update={"independent_accuracy": accuracy})
