"""HTTP router for the temporary perception-validation boundary."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from ..meta_client import MetaMuseClient
from .models import (
    AnnotationResponse,
    AnnotationRequest,
    GeometryQualityRequest,
    GeometryQualityResult,
    HandoffRequest,
    HandoffResponse,
    HandoffState,
    LeakageControlMetadata,
    QualityResult,
    CsvComparisonReport,
    CsvComparisonRequest,
    RepresentativeFrameResponse,
    SanitizedPreviewResponse,
    SessionCreateRequest,
    SessionResponse,
    SessionState,
    SamplingRequest,
    SamTrackingDirectionRequest,
    SamTrackingRequest,
    SamTrackingResult,
    UploadMetadata,
    UploadResponse,
)
from .geometry import evaluate_geometry
from .comparison import CsvComparisonError, compare_reference_csv
from .storage import InvalidSessionStateError, ValidationSessionStore
from .service import execute_handoff
from .sam_tracking import track_sampled_guided_video
from .sanitization import sanitized_frame_data_url
from .workflow import prepare_validation_frame

router = APIRouter(prefix="/api/perception-validation", tags=["perception-validation"])
STORE = ValidationSessionStore(Path("data") / "perception-validation")
HANDOFF_MUSE = MetaMuseClient()
ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".csv"}
UPLOAD_METADATA_FORM = Form(...)
UPLOAD_FILE = File(...)


def _persist_upload(
    store: ValidationSessionStore,
    state: SessionState,
    data: bytes,
    target: Path,
    temporary_target: Path,
) -> None:
    """Persist upload state and atomically publish its artifact."""
    try:
        temporary_target.write_bytes(data)
        temporary_target.replace(target)
        store.save(state)
    except (OSError, ValueError):
        target.unlink(missing_ok=True)
        raise
    finally:
        temporary_target.unlink(missing_ok=True)


def _session_or_404(session_id: str) -> SessionState:
    try:
        return STORE.get(session_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid validation session id.") from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Validation session not found.") from error
    except InvalidSessionStateError as error:
        raise HTTPException(
            status_code=422, detail="Validation session state is invalid."
        ) from error


def _video_upload_path(state: SessionState, upload_id: str) -> Path:
    """Resolve a previously stored video upload inside the session boundary."""
    upload = next((item for item in state.uploads if item.upload_id == upload_id), None)
    if upload is None or upload.metadata.kind != "video":
        raise HTTPException(status_code=404, detail="Validation video upload not found.")
    return STORE.root / state.session.session_id / "uploads" / f"{upload_id}{Path(upload.artifact_path).suffix}"


def _csv_upload_path(state: SessionState, upload_id: str) -> Path:
    """Resolve a comparison-only CSV inside the validation session boundary."""
    upload = next((item for item in state.uploads if item.upload_id == upload_id), None)
    if upload is None or upload.metadata.kind != "reference_csv":
        raise HTTPException(status_code=404, detail="Validation comparison CSV not found.")
    return STORE.root / state.session.session_id / "uploads" / f"{upload_id}{Path(upload.artifact_path).suffix}"


def _validate_leakage_controls(
    state: SessionState,
    controls: LeakageControlMetadata,
) -> None:
    """Reject metadata-only, out-of-frame, unsupported, or bob-covering masks."""
    if state.sampling is None or state.annotation is None:
        raise HTTPException(
            status_code=409,
            detail="Prepare a representative frame and save the bob annotation first.",
        )
    active_kinds = {region.kind for region in controls.regions if region.enabled}
    if not active_kinds:
        raise HTTPException(
            status_code=422,
            detail="Draw and save at least one enabled exclusion mask.",
        )
    flags = {
        "simulator_label": controls.simulator_labels_excluded,
        "displayed_parameters": controls.displayed_parameters_excluded,
        "live_angle_values": controls.live_angle_values_excluded,
        "controls": controls.controls_excluded,
    }
    for kind, excluded in flags.items():
        if excluded != (kind in active_kinds):
            raise HTTPException(
                status_code=422,
                detail=f"The {kind} flag must match a saved enabled exclusion region.",
            )
    width = state.sampling.source.width
    height = state.sampling.source.height
    left, top, right, bottom = state.annotation.bob_box
    for region in controls.regions:
        if region.mode != "mask":
            raise HTTPException(
                status_code=422,
                detail="Choose mask mode; crop mode is not supported for SAM sanitization.",
            )
        if region.right > width or region.bottom > height:
            raise HTTPException(
                status_code=422,
                detail="Exclusion regions must stay within the source-frame dimensions.",
            )
        if region.enabled and (
            region.left < right
            and region.right > left
            and region.top < bottom
            and region.bottom > top
        ):
            raise HTTPException(
                status_code=422,
                detail="An exclusion mask overlaps the annotated bob box; move it off the object.",
            )


def _complete_leakage_regions(controls: LeakageControlMetadata) -> bool:
    """Return whether all four known simulator UI categories have enabled masks."""
    required = {
        "simulator_label",
        "displayed_parameters",
        "live_angle_values",
        "controls",
    }
    return required.issubset(
        {region.kind for region in controls.regions if region.enabled}
    )


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(payload: SessionCreateRequest) -> SessionResponse:
    """Create temporary state without touching Investigation persistence."""
    return STORE.create(payload)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    """Return only the strict session metadata for a valid session."""
    return _session_or_404(session_id).session


@router.post(
    "/sessions/{session_id}/uploads",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_artifact(
    session_id: str,
    metadata: str = UPLOAD_METADATA_FORM,
    file: UploadFile = UPLOAD_FILE,
) -> UploadResponse:
    """Store an upload under the session after strict metadata parsing."""
    state = _session_or_404(session_id)
    try:
        upload_metadata = UploadMetadata.model_validate_json(metadata)
    except (ValidationError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=422, detail="Invalid upload metadata.") from error
    suffix = Path(file.filename or "").suffix.lower()
    if (
        suffix not in ALLOWED_SUFFIXES
        or file.content_type != upload_metadata.content_type
        or Path(file.filename or "").name != file.filename
        or Path(upload_metadata.original_filename).name != upload_metadata.original_filename
        or Path(upload_metadata.original_filename).suffix.lower() != suffix
    ):
        raise HTTPException(
            status_code=422, detail="Upload filename or content type is not allowed."
        )
    data = await file.read()
    if len(data) != upload_metadata.byte_size:
        raise HTTPException(status_code=422, detail="Upload byte size does not match metadata.")
    upload_id = uuid4().hex
    target = STORE.root / session_id / "uploads" / f"{upload_id}{suffix}"
    temporary_target = target.with_name(f".{upload_id}{suffix}.tmp")
    response = UploadResponse(
        upload_id=upload_id,
        session_id=session_id,
        metadata=upload_metadata,
        artifact_path=f"data/perception-validation/{session_id}/uploads/{upload_id}{suffix}",
    )
    updated_state = state.model_copy(
        update={
            "uploads": [*state.uploads, response],
            "session": state.session.model_copy(
                update={
                    "status": (
                        "uploaded" if state.session.status == "created" else state.session.status
                    ),
                    "updated_at": datetime.now(UTC),
                }
            ),
        }
    )
    _persist_upload(STORE, updated_state, data, target, temporary_target)
    return response


@router.post("/sessions/{session_id}/annotations", response_model=AnnotationResponse)
def save_annotation(session_id: str, payload: AnnotationRequest) -> AnnotationResponse:
    """Accept only a fully typed annotation payload for an existing session."""
    state = _session_or_404(session_id)
    if state.sampling is None:
        raise HTTPException(status_code=409, detail="Prepare a representative frame before annotating.")
    if payload.frame_index != state.sampling.frame_index:
        raise HTTPException(status_code=422, detail="Annotation frame_index does not match the prepared frame.")
    if payload.source_width != state.sampling.source.width or payload.source_height != state.sampling.source.height:
        raise HTTPException(status_code=422, detail="Annotation dimensions do not match the video source.")
    updated_state = state.model_copy(
        update={
            "annotation": payload,
            "session": state.session.model_copy(
                update={"status": "annotated", "updated_at": datetime.now(UTC)}
            ),
        }
    )
    STORE.save(updated_state)
    return AnnotationResponse(
        **updated_state.session.model_dump(),
        frame_index=payload.frame_index,
        source_width=payload.source_width,
        source_height=payload.source_height,
        endpoint_pixel_separation=payload.endpoint_pixel_separation,
        physical_distance_m=payload.physical_distance_m,
    )


@router.post(
    "/sessions/{session_id}/leakage-controls",
    response_model=SanitizedPreviewResponse,
)
def save_leakage_controls(
    session_id: str,
    payload: LeakageControlMetadata,
) -> SanitizedPreviewResponse:
    """Persist actual mask regions and return a preview sanitized by the backend."""
    state = _session_or_404(session_id)
    _validate_leakage_controls(state, payload)
    if state.sampling is None:
        raise HTTPException(status_code=409, detail="Prepare a representative frame first.")
    video_path = _video_upload_path(state, state.sampling.upload_id)
    try:
        preview = sanitized_frame_data_url(
            video_path,
            state.sampling.frame_index,
            state.sampling.source.width,
            state.sampling.source.height,
            payload.regions,
        )
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    updated_state = state.model_copy(
        update={
            "leakage_controls": payload,
            "session": state.session.model_copy(
                update={"updated_at": datetime.now(UTC)}
            ),
        }
    )
    STORE.save(updated_state)
    return SanitizedPreviewResponse(
        session_id=session_id,
        frame_index=state.sampling.frame_index,
        source_width=state.sampling.source.width,
        source_height=state.sampling.source.height,
        frame_data_url=preview,
        regions_applied=sum(region.enabled for region in payload.regions),
        leakage_controls=payload,
    )


@router.post(
    "/sessions/{session_id}/sam-tracking",
    response_model=SamTrackingResult,
)
async def track_sampled_session_video(
    session_id: str,
    payload: SamTrackingDirectionRequest,
) -> SamTrackingResult:
    """Run guided SAM only over the session's prepared source-frame selection."""
    state = _session_or_404(session_id)
    if state.sampling is None:
        raise HTTPException(status_code=409, detail="Prepare a representative frame before tracking.")
    if state.annotation is None:
        raise HTTPException(status_code=409, detail="Save an annotation before tracking.")
    video_path = _video_upload_path(state, state.sampling.upload_id)
    request = SamTrackingRequest(
        video_path=str(video_path),
        sampling=state.sampling,
        annotation=state.annotation,
        propagation_direction=payload.propagation_direction,
    )
    try:
        result = await track_sampled_guided_video(request)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    updated_state = state.model_copy(
        update={
            "sam_tracking": result,
            "session": state.session.model_copy(update={"updated_at": datetime.now(UTC)}),
        }
    )
    STORE.save(updated_state)
    return result


@router.post(
    "/sessions/{session_id}/sanitized-sam-tracking",
    response_model=SamTrackingResult,
)
async def track_sanitized_session_video(
    session_id: str,
    payload: SamTrackingDirectionRequest,
) -> SamTrackingResult:
    """Manually run existing SAM over frames after saved exclusions are applied."""
    state = _session_or_404(session_id)
    if state.sampling is None or state.annotation is None:
        raise HTTPException(
            status_code=409,
            detail="Prepare a representative frame and save the bob annotation first.",
        )
    controls = state.leakage_controls
    _validate_leakage_controls(state, controls)
    if not _complete_leakage_regions(controls):
        raise HTTPException(
            status_code=409,
            detail=(
                "Add and save mask regions for simulator labels, displayed parameters, "
                "live angle values, and controls before sanitized SAM tracking."
            ),
        )
    video_path = _video_upload_path(state, state.sampling.upload_id)
    request = SamTrackingRequest(
        video_path=str(video_path),
        sampling=state.sampling,
        annotation=state.annotation,
        propagation_direction=payload.propagation_direction,
    )
    try:
        result = await track_sampled_guided_video(request, controls.regions)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    updated_state = state.model_copy(
        update={
            "sanitized_sam_tracking": result,
            "sanitized_tracking_annotation": state.annotation,
            "session": state.session.model_copy(update={"updated_at": datetime.now(UTC)}),
        }
    )
    STORE.save(updated_state)
    return result


@router.post(
    "/sessions/{session_id}/geometry-quality",
    response_model=GeometryQualityResult,
)
def calculate_geometry_quality(
    session_id: str, payload: GeometryQualityRequest
) -> GeometryQualityResult:
    """Calculate temporary geometry quality without invoking production science."""
    state = _session_or_404(session_id)
    if payload.leakage_controls != state.leakage_controls:
        raise HTTPException(
            status_code=409,
            detail="Save the current exclusion regions before calculating quality.",
        )
    sanitized_tracking = state.sanitized_sam_tracking
    sanitized_matches = bool(
        sanitized_tracking is not None
        and sanitized_tracking.input_sanitization == "exclusion_masks_applied"
        and sanitized_tracking.applied_exclusion_regions == state.leakage_controls.regions
        and sanitized_tracking.sanitized_frame_count > 0
        and sanitized_tracking.sampling == state.sampling
        and state.sanitized_tracking_annotation == state.annotation
    )
    tracking = sanitized_tracking if sanitized_matches else state.sam_tracking
    if tracking is None:
        raise HTTPException(
            status_code=409,
            detail="Run sampled SAM tracking before calculating geometry quality.",
        )
    try:
        result = evaluate_geometry(
            payload.model_copy(
                update={
                    "tracks": tracking.tracks,
                    "expected_frame_indices": sorted(
                        set(tracking.selected_frame_indices)
                        | set(tracking.missing_frame_indices)
                    ),
                    "sam_tracking_status": tracking.status,
                    "sam_tracking_rejection_reasons": tracking.rejection_reasons,
                    "leakage_sanitization_applied": sanitized_matches,
                }
            )
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    updated_state = state.model_copy(
        update={
            "geometry_quality": result,
            "quality": QualityResult(
                status=(
                    "approved"
                    if result.production_handoff_approved
                    else "rejected"
                ),
                coverage_fraction=result.coverage_fraction,
                radius_stability=result.radius.status,
                discontinuity_count=(
                    result.discontinuities.gap_count + result.discontinuities.jump_count
                ),
                metric_accuracy=result.independent_accuracy.metric_distance_rmse_m,
                angular_accuracy_deg=result.independent_accuracy.angular_rmse_deg,
                reasons=result.production_handoff_reasons,
                sam_tracking_quality_approved=result.sam_tracking_quality_approved,
                sam_tracking_quality_reasons=result.sam_tracking_quality_reasons,
                pixel_geometry_approved=result.pixel_geometry_approved,
                pixel_geometry_reasons=result.pixel_geometry_reasons,
                metric_calibration_approved=result.metric_calibration_approved,
                metric_calibration_reasons=result.metric_calibration_reasons,
                leakage_compliance_approved=result.leakage_compliance_approved,
                leakage_compliance_reasons=result.leakage_compliance_reasons,
                production_handoff_approved=result.production_handoff_approved,
                production_handoff_reasons=result.production_handoff_reasons,
            ),
            "session": state.session.model_copy(
                update={"status": "quality_checked", "updated_at": datetime.now(UTC)}
            ),
        }
    )
    STORE.save(updated_state)
    return result


@router.post(
    "/sessions/{session_id}/handoff",
    response_model=HandoffResponse,
)
def handoff_to_production(
    session_id: str,
    payload: HandoffRequest,
) -> HandoffResponse:
    """Deliberately hand off only quality-approved temporary measurements."""
    state = _session_or_404(session_id)
    result = execute_handoff(state, payload, HANDOFF_MUSE)
    updated_state = state.model_copy(
        update={
            "handoff": HandoffState(
                status=result.status,
                reason=result.reason,
                production_invoked=result.production_invoked,
                fit=result.fit,
                selector=result.selector,
                muse=result.muse,
            ),
            "session": state.session.model_copy(
                update={
                    "status": "handed_off"
                    if result.status == "completed"
                    else state.session.status,
                    "updated_at": datetime.now(UTC),
                }
            ),
        }
    )
    STORE.save(updated_state)
    return result


@router.post(
    "/sessions/{session_id}/csv-comparison",
    response_model=CsvComparisonReport,
)
def compare_csv(
    session_id: str,
    payload: CsvComparisonRequest,
) -> CsvComparisonReport:
    """Compare an uploaded reference trajectory without entering production science."""
    state = _session_or_404(session_id)
    path = _csv_upload_path(state, payload.csv_upload_id)
    try:
        report = compare_reference_csv(
            path,
            payload.csv_upload_id,
            payload.video_trajectory,
            payload.alignment_search_s,
        )
    except CsvComparisonError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    updated_state = state.model_copy(
        update={
            "csv_comparison": report,
            "session": state.session.model_copy(
                update={"status": "quality_checked", "updated_at": datetime.now(UTC)}
            ),
        }
    )
    STORE.save(updated_state)
    return report


@router.post(
    "/sessions/{session_id}/representative-frame",
    response_model=RepresentativeFrameResponse,
)
@router.post("/sessions/{session_id}/prepare", response_model=RepresentativeFrameResponse)
def prepare_representative_frame(
    session_id: str,
    payload: SamplingRequest,
) -> RepresentativeFrameResponse:
    """Return one preview and auditable source-frame sampling metadata."""
    state = _session_or_404(session_id)
    if payload.upload_id not in {upload.upload_id for upload in state.uploads}:
        raise HTTPException(status_code=404, detail="Validation upload not found.")
    path = _video_upload_path(state, payload.upload_id)
    try:
        prepared = prepare_validation_frame(session_id, path, payload)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    updated_state = state.model_copy(update={"sampling": prepared.selection})
    STORE.save(updated_state)
    return prepared.response
