"""Strict contracts for the temporary perception-validation workflow.

# noqa: SIZE_OK — this pre-existing boundary module owns the temporary API
# contracts; Task 4 calculations remain isolated in geometry.py.
"""

from __future__ import annotations

from datetime import datetime
from math import isfinite
from math import hypot
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SessionCreateRequest(StrictModel):
    label: str = Field(default="temporary validation", min_length=1, max_length=120)


class SessionResponse(StrictModel):
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    status: Literal[
        "created", "uploaded", "annotated", "diagnosed", "quality_checked", "handed_off"
    ]
    label: str
    artifact_root: str = Field(pattern=r"^data/perception-validation/[0-9a-f]{32}$")
    created_at: datetime
    updated_at: datetime


class UploadMetadata(StrictModel):
    kind: Literal["video", "reference_csv"]
    original_filename: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[^/\\\x00-\x1f\x7f]+$",
    )
    content_type: Literal["video/mp4", "video/quicktime", "video/x-msvideo", "text/csv"]
    byte_size: int = Field(gt=0, le=100_000_000)


class UploadResponse(StrictModel):
    upload_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    metadata: UploadMetadata
    artifact_path: str = Field(
        pattern=r"^data/perception-validation/[0-9a-f]{32}/uploads/[0-9a-f]{32}\.(mp4|mov|avi|csv)$"
    )


class VideoSource(StrictModel):
    width: int = Field(gt=0, le=100_000)
    height: int = Field(gt=0, le=100_000)
    fps: float = Field(gt=0, le=10_000)
    frame_count: int = Field(gt=0, le=10_000_000)
    duration_s: float = Field(gt=0, le=86_400)


class SamplingRequest(StrictModel):
    upload_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_start_s: float = Field(ge=0)
    source_end_s: float = Field(gt=0)
    frame_stride: int | None = Field(default=None, ge=1, le=10_000)
    target_fps: float | None = Field(default=None, gt=0)
    representative_time_s: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_sampling_choice(self) -> "SamplingRequest":
        if self.source_end_s <= self.source_start_s:
            raise ValueError("source_end_s must be greater than source_start_s")
        if (self.frame_stride is None) == (self.target_fps is None):
            raise ValueError("provide exactly one of frame_stride or target_fps")
        return self


class SamplingSelection(StrictModel):
    upload_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source: VideoSource
    source_start_s: float = Field(ge=0)
    source_end_s: float = Field(gt=0)
    frame_stride: int = Field(ge=1)
    target_fps: float | None = Field(default=None, gt=0)
    frame_index: int = Field(ge=0)
    source_timestamp_s: float = Field(ge=0)


class RepresentativeFrameResponse(StrictModel):
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    upload_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    frame_data_url: str = Field(pattern=r"^data:image/[^;]+;base64,")
    source: VideoSource
    source_start_s: float = Field(ge=0)
    source_end_s: float = Field(gt=0)
    frame_stride: int = Field(ge=1)
    target_fps: float | None = Field(default=None, gt=0)
    frame_index: int = Field(ge=0)
    source_timestamp_s: float = Field(ge=0)
    sampled_frames: list[tuple[int, float]] = Field(min_length=1, max_length=10_000_000)


class AnnotationResponse(SessionResponse):
    frame_index: int = Field(ge=0)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    endpoint_pixel_separation: float | None = Field(default=None, gt=0)
    physical_distance_m: float | None = Field(default=None, gt=0)


class Point(StrictModel):
    x: float = Field(ge=0, le=100_000)
    y: float = Field(ge=0, le=100_000)


class AnnotationRequest(StrictModel):
    source_width: int = Field(gt=0, le=100_000)
    source_height: int = Field(gt=0, le=100_000)
    frame_index: int = Field(ge=0)
    bob_box: list[float] = Field(
        min_length=4,
        max_length=4,
    )
    positive_points: list[Point] = Field(min_length=1, max_length=8)
    negative_points: list[Point] = Field(default_factory=list, max_length=8)
    pivot: Point
    reference_start: Point | None = None
    reference_end: Point | None = None
    physical_distance_m: float | None = Field(default=None, gt=0, le=100_000)

    @model_validator(mode="after")
    def validate_geometry(self) -> "AnnotationRequest":
        points = [*self.positive_points, *self.negative_points, self.pivot]
        calibration_values = (self.reference_start, self.reference_end, self.physical_distance_m)
        if any(value is not None for value in calibration_values) and not all(
            value is not None for value in calibration_values
        ):
            raise ValueError("calibration requires both endpoints and physical distance")
        if self.reference_start is not None and self.reference_end is not None:
            points.extend((self.reference_start, self.reference_end))
        if any(point.x > self.source_width or point.y > self.source_height for point in points):
            raise ValueError("annotation coordinates must lie within source dimensions")
        left, top, right, bottom = self.bob_box
        if not (0 <= left < right <= self.source_width and 0 <= top < bottom <= self.source_height):
            raise ValueError("bob_box must be ordered and lie within source dimensions")
        if self.reference_start is not None and self.reference_end is not None:
            separation = hypot(
                self.reference_end.x - self.reference_start.x,
                self.reference_end.y - self.reference_start.y,
            )
            if separation <= 0:
                raise ValueError("reference endpoints must be distinct")
        return self

    @property
    def endpoint_pixel_separation(self) -> float | None:
        """Return pixel separation when two calibration endpoints are set."""
        if self.reference_start is None or self.reference_end is None:
            return None
        return hypot(
            self.reference_end.x - self.reference_start.x,
            self.reference_end.y - self.reference_start.y,
        )


class ExclusionRegion(StrictModel):
    """A user-configured crop or mask region removed from blind estimation."""

    kind: Literal["simulator_label", "displayed_parameters", "live_angle_values", "controls"]
    mode: Literal["crop", "mask"]
    left: float = Field(ge=0, le=100_000)
    top: float = Field(ge=0, le=100_000)
    right: float = Field(gt=0, le=100_000)
    bottom: float = Field(gt=0, le=100_000)
    enabled: bool = True

    @model_validator(mode="after")
    def validate_bounds(self) -> "ExclusionRegion":
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("exclusion region bounds must be ordered")
        return self


class LeakageControlMetadata(StrictModel):
    """Metadata describing UI exclusions; it is never a measurement source."""

    regions: list[ExclusionRegion] = Field(default_factory=list, max_length=64)
    simulator_labels_excluded: bool = False
    displayed_parameters_excluded: bool = False
    live_angle_values_excluded: bool = False
    controls_excluded: bool = False


class SanitizedPreviewResponse(StrictModel):
    """One representative frame after server-applied simulator-UI masks."""

    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    frame_index: int = Field(ge=0)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    frame_data_url: str = Field(pattern=r"^data:image/[^;]+;base64,")
    regions_applied: int = Field(gt=0)
    leakage_controls: LeakageControlMetadata


class SamplingMetadata(StrictModel):
    """Expected source frames used to calculate coverage without filling gaps."""

    frame_start: int = Field(ge=0)
    frame_end: int = Field(ge=0)
    frame_stride: int = Field(gt=0, le=10_000)

    @model_validator(mode="after")
    def validate_frame_range(self) -> "SamplingMetadata":
        if self.frame_end < self.frame_start:
            raise ValueError("frame_end must be greater than or equal to frame_start")
        return self


class QualityThresholds(StrictModel):
    """Configurable sendability thresholds for temporary validation only."""

    minimum_coverage_fraction: float = Field(ge=0, le=1, default=0.8)
    maximum_radius_cv: float = Field(ge=0, le=10, default=0.15)
    maximum_jump_px: float = Field(gt=0, le=100_000, default=80.0)
    maximum_gap_frames: int = Field(ge=0, le=10_000, default=2)
    minimum_confidence: float = Field(ge=0, le=1, default=0.55)


class MaskCentroidTrack(StrictModel):
    """One mask-derived centroid; bounding-box centers are not accepted here."""

    frame_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0)
    centroid: Point
    mask_area_px: int = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    object_id: str = Field(min_length=1, max_length=80)


class GeometryQualityRequest(StrictModel):
    """Inputs for blind geometry and quality calculations."""

    annotation: AnnotationRequest
    tracks: list[MaskCentroidTrack] = Field(default_factory=list, max_length=100_000)
    sampling: SamplingMetadata | None = None
    leakage_controls: LeakageControlMetadata = Field(default_factory=LeakageControlMetadata)
    leakage_sanitization_applied: bool = Field(default=False, exclude=True)
    thresholds: QualityThresholds = Field(default_factory=QualityThresholds)
    sam_tracking_status: Literal["completed", "partial", "failed"] | None = None
    sam_tracking_rejection_reasons: list[str] = Field(default_factory=list, max_length=1000)
    expected_frame_indices: list[int] | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_track_identity(self) -> "GeometryQualityRequest":
        frame_indices = [track.frame_index for track in self.tracks]
        if len(set(frame_indices)) != len(frame_indices):
            raise ValueError("mask centroid tracks must contain one record per frame")
        object_ids = {track.object_id for track in self.tracks}
        if len(object_ids) > 1:
            raise ValueError("mask centroid tracks must preserve one selected object id")
        return self


class FrameGeometry(StrictModel):
    """Pixel and calibrated geometry for one retained mask centroid."""

    frame_index: int
    timestamp_s: float
    pixel_distance: float = Field(ge=0)
    metric_distance_m: float | None = Field(default=None, ge=0)
    angle_deg: float
    confidence: float = Field(ge=0, le=1)


class ConfidenceSummary(StrictModel):
    mean: float = Field(ge=0, le=1)
    minimum: float = Field(ge=0, le=1)
    below_threshold_count: int = Field(ge=0)


class RadiusSummary(StrictModel):
    median_px: float = Field(ge=0)
    min_px: float = Field(ge=0)
    max_px: float = Field(ge=0)
    cv: float = Field(ge=0)
    median_m: float | None = Field(default=None, ge=0)
    status: Literal["stable", "unstable", "not_checked"]


class DiscontinuitySummary(StrictModel):
    missing_frame_indices: list[int] = Field(default_factory=list)
    gap_count: int = Field(ge=0)
    largest_gap_frames: int = Field(ge=0)
    jump_frame_indices: list[int] = Field(default_factory=list)
    jump_count: int = Field(ge=0)


class QualityGateSummary(StrictModel):
    approved: bool
    reasons: list[str] = Field(default_factory=list)


class IndependentAccuracyReport(StrictModel):
    """Evaluation-only accuracy; never used to produce blind geometry."""

    metric_distance_rmse_m: float | None = Field(default=None, ge=0)
    length_error_m: float | None = Field(default=None, ge=0)
    angular_rmse_deg: float | None = Field(default=None, ge=0)
    angular_max_error_deg: float | None = Field(default=None, ge=0)


class GeometryQualityResult(StrictModel):
    """Auditable geometry, quality, and sendability output."""

    endpoint_separation_px: float | None = Field(default=None, gt=0)
    confirmed_physical_distance_m: float | None = Field(default=None, gt=0)
    metres_per_pixel: float | None = Field(default=None, gt=0)
    calibration_available: bool = False
    calibration_status: Literal["unavailable", "available"] = "unavailable"
    coverage_fraction: float = Field(ge=0, le=1)
    retained_frame_count: int = Field(ge=0)
    excluded_frames: dict[int, str] = Field(default_factory=dict)
    confidence: ConfidenceSummary
    radius: RadiusSummary
    discontinuities: DiscontinuitySummary
    sam_tracking_quality: QualityGateSummary
    pixel_geometry_quality: QualityGateSummary
    metric_calibration_validity: QualityGateSummary
    simulator_leakage_compliance: QualityGateSummary
    production_handoff_eligibility: QualityGateSummary
    thresholds: QualityThresholds
    pixel_geometry: list[FrameGeometry] = Field(default_factory=list)
    calibrated_geometry: list[FrameGeometry] = Field(default_factory=list)
    pixel_tracking_approved: bool = False
    pixel_geometry_approved: bool = False
    pixel_geometry_reasons: list[str] = Field(default_factory=list)
    sam_tracking_quality_approved: bool = False
    sam_tracking_quality_reasons: list[str] = Field(default_factory=list)
    metric_calibration_approved: bool = False
    metric_calibration_reasons: list[str] = Field(default_factory=list)
    leakage_compliance_approved: bool = False
    leakage_compliance_reasons: list[str] = Field(default_factory=list)
    leakage_sanitization_applied: bool = False
    production_handoff_approved: bool = False
    production_handoff_reasons: list[str] = Field(default_factory=list)
    sendable: bool
    sendability_reasons: list[str] = Field(default_factory=list)
    leakage_controls: LeakageControlMetadata
    independent_accuracy: IndependentAccuracyReport = Field(default_factory=IndependentAccuracyReport)


class VideoTrajectorySample(StrictModel):
    """One sampled video-derived angle used only as the comparison target."""

    timestamp_s: float = Field(ge=0)
    angle_deg: float
    frame_index: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_finite_angle(self) -> "VideoTrajectorySample":
        if not isfinite(self.timestamp_s) or not isfinite(self.angle_deg):
            raise ValueError("video trajectory timestamps and angles must be finite")
        return self


class CsvComparisonRequest(StrictModel):
    """Evaluation-only comparison request; never a production science input."""

    csv_upload_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    video_trajectory: list[VideoTrajectorySample] = Field(min_length=2, max_length=100_000)
    alignment_search_s: float = Field(gt=0, le=10)

    @model_validator(mode="after")
    def validate_video_trajectory(self) -> "CsvComparisonRequest":
        timestamps = [sample.timestamp_s for sample in self.video_trajectory]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("video trajectory timestamps must be strictly increasing")
        return self


class CsvComparisonReport(StrictModel):
    """Transparent, independent comparison metrics for a reference CSV."""

    csv_upload_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    timestamp_column: str
    angular_column: str
    angular_unit: Literal["deg", "rad"]
    row_count: int = Field(gt=1)
    alignment_method: Literal["timestamp_offset_plus_linear_interpolation"]
    alignment_search_s: float | None = Field(default=None, gt=0, le=10)
    estimated_time_offset_s: float
    alignment_uncertainty_s: float = Field(ge=0)
    overlap_duration_s: float = Field(ge=0)
    overlap_frame_count: int = Field(ge=0)
    angular_rmse_deg: float = Field(ge=0)
    angular_max_error_deg: float = Field(ge=0)
    mismatch_diagnostics: list[str] = Field(default_factory=list, max_length=32)
    independent_only: Literal[True] = True


class DiagnosticsResult(StrictModel):
    outcome: Literal["not_started", "completed", "failed", "partial"]
    records: int = Field(ge=0)
    frames_with_masks: int = Field(ge=0)
    frames_tracked: int = Field(ge=0)
    missing_frame_indices: list[int] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list, max_length=1000)
    parser_diagnostics: str = Field(default="", max_length=10_000)


class SamTrackingRequest(StrictModel):
    """Inputs for the temporary sampled guided-SAM adapter."""

    video_path: str = Field(min_length=1, max_length=500)
    sampling: SamplingSelection
    annotation: AnnotationRequest
    propagation_direction: Literal["both", "forward", "backward"] = "forward"


class SamTrackingDirectionRequest(StrictModel):
    """Validated direction selected for a temporary sampled tracking run."""

    propagation_direction: Literal["both", "forward", "backward"] = "forward"


class SamTrackingResult(StrictModel):
    """Mask-derived tracking diagnostics; no production handoff is implied."""

    status: Literal["completed", "partial", "failed"]
    method: Literal["sam_guided_mask", "sam_guided_mask_forward_fallback"]
    propagation_direction: Literal["both", "forward", "backward"]
    fallback_label: Literal["none", "forward_after_reverse_failure"] = "none"
    sampling: SamplingSelection
    outcome: str | None = None
    records: int = Field(default=0, ge=0)
    masks_returned: int = Field(default=0, ge=0)
    frames_with_masks: int = Field(default=0, ge=0)
    frames_tracked: int = Field(default=0, ge=0)
    object_id: str | None = None
    input_sanitization: Literal["none", "exclusion_masks_applied"] = "none"
    applied_exclusion_regions: list[ExclusionRegion] = Field(default_factory=list, max_length=64)
    sanitized_frame_count: int = Field(default=0, ge=0)
    confidence: float = Field(default=0, ge=0, le=1)
    selected_frame_indices: list[int] = Field(default_factory=list)
    selected_timestamps_s: list[float] = Field(default_factory=list)
    missing_frame_indices: list[int] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list, max_length=1000)
    raw_output_excerpt: str = Field(default="", max_length=10_000)
    parser_diagnostics: str = Field(default="", max_length=10_000)
    stream_event_excerpt: str = Field(default="", max_length=10_000)
    tracks: list[MaskCentroidTrack] = Field(default_factory=list, max_length=100_000)


class QualityResult(StrictModel):
    status: Literal["not_checked", "approved", "rejected"]
    coverage_fraction: float = Field(ge=0, le=1)
    radius_stability: Literal["not_checked", "stable", "unstable"]
    discontinuity_count: int = Field(ge=0)
    metric_accuracy: float | None = Field(default=None, ge=0)
    angular_accuracy_deg: float | None = Field(default=None, ge=0)
    reasons: list[str] = Field(default_factory=list, max_length=1000)
    sam_tracking_quality_approved: bool = False
    sam_tracking_quality_reasons: list[str] = Field(default_factory=list, max_length=1000)
    pixel_geometry_approved: bool = False
    pixel_geometry_reasons: list[str] = Field(default_factory=list, max_length=1000)
    metric_calibration_approved: bool = False
    metric_calibration_reasons: list[str] = Field(default_factory=list, max_length=1000)
    leakage_compliance_approved: bool = False
    leakage_compliance_reasons: list[str] = Field(default_factory=list, max_length=1000)
    production_handoff_approved: bool = False
    production_handoff_reasons: list[str] = Field(default_factory=list, max_length=1000)


class HandoffMeasurement(StrictModel):
    """One quality-approved calibrated angle permitted into production fitting."""

    frame_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0)
    angle_deg: float

    @model_validator(mode="after")
    def validate_finite_measurement(self) -> "HandoffMeasurement":
        if not all(isfinite(value) for value in (self.timestamp_s, self.angle_deg)):
            raise ValueError("handoff measurements must be finite")
        return self


class HandoffRequest(StrictModel):
    """Explicit send action; comparison CSV data has no field in this contract."""

    measurements: list[HandoffMeasurement] = Field(min_length=12, max_length=100_000)
    completed_angles: list[float] = Field(default_factory=list, max_length=100)
    objective: str = Field(default="Interpret the approved validation measurements.", min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_measurement_order(self) -> "HandoffRequest":
        timestamps = [item.timestamp_s for item in self.measurements]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("handoff measurement timestamps must be strictly increasing")
        if any(not isfinite(angle) or angle <= 0 or angle > 180 for angle in self.completed_angles):
            raise ValueError("completed_angles must contain finite angles from (0, 180]")
        return self


class ExportRequest(StrictModel):
    format: Literal["json", "csv", "report"]


class ExportResponse(StrictModel):
    format: Literal["json", "csv", "report"]
    artifact_path: str
    status: Literal["ready"]


class DeterministicFitSummary(StrictModel):
    """Production fit values copied without reinterpretation."""

    length_m: float
    damping_s_minus_1: float
    length_interval_m: tuple[float, float]
    damping_interval_s_minus_1: tuple[float, float]
    residual_rms_deg: float
    adequacy: str


class SelectorSummary(StrictModel):
    """Production EIG selection; Muse cannot replace these values."""

    angle_deg: float
    duration_s: float
    eig_nats: float
    rationale: str


class MuseSummary(StrictModel):
    """Muse boundary diagnostics, including explicit unavailable states."""

    text: str
    status: str
    model: str
    finish_reason: str
    request_status: str
    completion_status: str
    response_text_length: int = Field(ge=0)
    error_detail: str = ""


class HandoffState(StrictModel):
    status: Literal["not_requested", "blocked", "approved", "completed", "failed"]
    reason: str = Field(default="", max_length=2000)
    production_invoked: bool = False
    fit: "DeterministicFitSummary | None" = None
    selector: "SelectorSummary | None" = None
    muse: "MuseSummary | None" = None


class HandoffResponse(StrictModel):
    """Auditable result of the deliberate temporary-to-production handoff."""

    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    status: Literal["blocked", "completed", "failed"]
    reason: str = Field(default="", max_length=2000)
    production_invoked: bool = False
    quality: QualityResult
    fit: DeterministicFitSummary | None = None
    selector: SelectorSummary | None = None
    muse: MuseSummary
    muse_interpretation_only: Literal[True] = True
    comparison_csv_excluded: Literal[True] = True


class SessionState(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    session: SessionResponse
    uploads: list[UploadResponse] = Field(default_factory=list)
    sampling: SamplingSelection | None = None
    annotation: AnnotationRequest | None = None
    sam_tracking: SamTrackingResult | None = None
    leakage_controls: LeakageControlMetadata = Field(default_factory=LeakageControlMetadata)
    sanitized_sam_tracking: SamTrackingResult | None = None
    sanitized_tracking_annotation: AnnotationRequest | None = None
    diagnostics: DiagnosticsResult | None = None
    quality: QualityResult | None = None
    geometry_quality: GeometryQualityResult | None = None
    csv_comparison: CsvComparisonReport | None = None
    handoff: HandoffState = Field(default_factory=lambda: HandoffState(status="not_requested"))
