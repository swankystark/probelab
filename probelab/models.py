"""Typed API models for ProbeLab."""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class StartInvestigation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system: str = Field(min_length=3, max_length=160)
    objective: str = Field(min_length=10, max_length=1000)
    calibration_px: float = Field(default=400.0, gt=0)
    calibration_m: float = Field(default=1.0, gt=0)


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    angle_deg: float = Field(ge=5, le=75)
    duration_s: float = Field(ge=2, le=30)


class GuidedPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_index: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    box: tuple[float, float, float, float]
    positive_points: list[tuple[float, float]] = Field(min_length=1, max_length=8)
    negative_points: list[tuple[float, float]] = Field(default_factory=list, max_length=8)


class PreparedVideo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_data_url: str
    width: int
    height: int
    fps: float
    frame_count: int
    frame_index: int


class SamMaskOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_index: int = Field(ge=0)
    left: float
    top: float
    right: float
    bottom: float
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    raster_base64: str


class SamImageTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_data_url: str = Field(pattern=r"^data:image/[^;]+;base64,")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    box: tuple[float, float, float, float]
    positive_points: list[tuple[float, float]] = Field(min_length=1, max_length=8)
    negative_points: list[tuple[float, float]] = Field(default_factory=list, max_length=8)


class SamImageTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["completed", "failed"]
    prompt: str
    outcome: str | None = None
    records: int = 0
    boxes_returned: int = 0
    masks_returned: int = 0
    object_id: str | None = None
    confidence: float | None = None
    raw_output_excerpt: str = ""
    parser_diagnostics: str = ""
    error: str | None = None
    mask_overlay: SamMaskOverlay | None = None


class SamVideoPrepared(BaseModel):
    model_config = ConfigDict(extra="forbid")
    test_id: str
    frame_data_url: str
    width: int
    height: int
    fps: float
    frame_count: int
    frame_index: int


class SamVideoTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["completed", "failed"]
    propagation_direction: Literal["both", "forward", "backward"] = "forward"
    prompt: str
    outcome: str | None = None
    records: int = 0
    boxes_returned: int = 0
    masks_returned: int = 0
    frames_with_masks: int = 0
    frames_tracked: int = 0
    object_id: str | None = None
    confidence: float | None = None
    raw_output_excerpt: str = ""
    parser_diagnostics: str = ""
    stream_event_excerpt: str = ""
    error: str | None = None
    mask_overlay: SamMaskOverlay | None = None
    tracks: list[dict[str, float]] = Field(default_factory=list)


class ObservationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["video", "camera"]
    method: str
    frames: int
    duration_s: float
    confidence: float
    quality: Literal["usable", "insufficient", "unreliable"]
    timestamps: list[float]
    angles_deg: list[float]
    message: str
    calibration_px: float = 0.0
    calibration_m: float = 0.0
    tracked_radius_m: float | None = None
    sam_diagnostics: dict[str, float | int | str] = Field(default_factory=dict)
    tracked_points: list[tuple[float, float, float, float]] = Field(default_factory=list)
    sam_mask_overlay: SamMaskOverlay | None = None


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    angle_deg: float
    duration_s: float
    eig: float
    confidence: float
    quality: str
    created_at: datetime


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    angle_deg: float
    duration_s: float
    model_version: str
    times: list[float]
    angles_deg: list[float]
    frozen_parameters: dict[str, float]
    created_at: datetime


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: str
    actual_times: list[float]
    actual_angles_deg: list[float]
    rmse_deg: float
    max_error_deg: float
    status: str
    created_at: datetime


class Investigation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    system: str
    objective: str
    calibration_px: float
    calibration_m: float
    status: str
    model: str
    reasoning: str
    parameters: dict[str, float | list[float] | str]
    recommendation: dict[str, float | str]
    posterior: dict[str, list[float]]
    experiments: list[ExperimentRecord]
    observations: list[ObservationResult]
    prediction: PredictionRecord | None = None
    validation: ValidationResult | None = None
    report: str | None = None
    agent_model: str = "muse-spark-1.3"
    agent_status: str = "Pending"
    agent_request_status: str = "not_invoked"
    agent_completion_status: str = "not_started"
    agent_finish_reason: str = ""
    agent_response_length: int = 0
    agent_error: str = ""
    created_at: datetime
    updated_at: datetime


class DemoReplaySummary(BaseModel):
    """Non-sensitive index row for an existing saved investigation."""

    id: str
    system: str
    status: str
    updated_at: datetime
    observation_count: int = Field(ge=0)
    latest_observation_quality: str | None = None
    latest_observation_frames: int = Field(ge=0)
    has_saved_muse: bool
    has_frozen_prediction: bool
    has_validation: bool


class DemoReplayResponse(BaseModel):
    """Local replay results; never eligible for production handoff."""

    id: str
    system: str
    objective: str
    saved_status: str
    replay_label: Literal["REPLAYED · LEAKAGE UNVERIFIED"] = "REPLAYED · LEAKAGE UNVERIFIED"
    observation_quality: str
    observation_method: str
    observation_frames: int = Field(ge=0)
    observation_confidence: float = Field(ge=0, le=1)
    representative_frame_data_url: str | None = None
    sam_mask_overlay: SamMaskOverlay | None = None
    tracked_points: list[tuple[float, float, float, float]] = Field(default_factory=list)
    observed_times_s: list[float] = Field(default_factory=list)
    observed_angles_deg: list[float] = Field(default_factory=list)
    inference_status: Literal["computed", "unavailable"]
    length_m: float | None = None
    length_interval_m: list[float] = Field(default_factory=list)
    damping_s_minus_1: float | None = None
    damping_interval_s_minus_1: list[float] = Field(default_factory=list)
    residual_rms_deg: float | None = None
    model_adequacy: str
    eig_angle_deg: float | None = None
    eig_duration_s: float | None = None
    eig_nats: float | None = None
    eig_rationale: str
    prediction_source: Literal["saved_frozen_prediction", "local_replay_forecast", "none"]
    predicted_times_s: list[float] = Field(default_factory=list)
    predicted_angles_deg: list[float] = Field(default_factory=list)
    validation_status: str
    validation_rmse_deg: float | None = None
    validation_max_error_deg: float | None = None
    validation_times_s: list[float] = Field(default_factory=list)
    validation_angles_deg: list[float] = Field(default_factory=list)
    muse_status: str
    muse_model: str
    muse_text: str
