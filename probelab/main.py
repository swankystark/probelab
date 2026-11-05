"""ProbeLab FastAPI application."""

from datetime import UTC, datetime
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal
from uuid import uuid4
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from .meta_client import MetaMuseClient, NarrativeResult
from .models import DemoReplayResponse, DemoReplaySummary, ExperimentRecord, ExperimentRequest, GuidedPrompt, Investigation, PredictionRecord, PreparedVideo, SamImageTestRequest, SamImageTestResult, SamMaskOverlay, SamVideoPrepared, SamVideoTestResult, StartInvestigation, ValidationResult
from .perception import extract_from_sam_tracks, extract_video, prepare_video
from .sam_client import SamUnavailableError, segment_image_guided, segment_video, segment_video_guided
from .science import choose_experiment, fit_posterior, simulate
from .store import load, save
from .engine_test.router import router as engine_test_router
from .perception_validation import router as perception_validation_router

app = FastAPI(title="ProbeLab", version="0.1.0")
app.include_router(engine_test_router)
app.include_router(perception_validation_router)
app.mount("/static", StaticFiles(directory="web"), name="static")
muse = MetaMuseClient()


def _agent_updates(narrative: NarrativeResult) -> dict[str, str | int]:
    """Persist non-secret Muse request and response diagnostics."""
    return {
        "agent_model": narrative.model,
        "agent_status": narrative.status,
        "agent_request_status": narrative.request_status,
        "agent_completion_status": narrative.completion_status,
        "agent_finish_reason": narrative.finish_reason,
        "agent_response_length": narrative.response_text_length,
        "agent_error": narrative.error_detail,
    }


@app.get("/")
def home() -> FileResponse:
    return FileResponse("web/index.html")


@app.get("/sam-test")
def sam_test_page() -> FileResponse:
    """Serve the isolated single-frame SAM access test."""
    return FileResponse("web/sam_test.html")


@app.get("/sam-video-test")
def sam_video_test_page() -> FileResponse:
    """Serve the isolated video-processing SAM test."""
    return FileResponse("web/sam_video_test.html")


@app.post("/api/sam-test/image", response_model=SamImageTestResult)
async def sam_test_image(payload: SamImageTestRequest) -> SamImageTestResult:
    """Test SAM 3.1 on one user-selected image without touching an investigation."""
    try:
        result = await segment_image_guided(payload.image_data_url, payload.width, payload.height, payload.box, payload.positive_points, payload.negative_points)
    except SamUnavailableError as error:
        diagnostics = error.diagnostics
        return SamImageTestResult(status="failed", prompt=str(diagnostics.get("prompt", "")), outcome=str(diagnostics.get("outcome")) if diagnostics.get("outcome") is not None else None, records=int(diagnostics.get("records", 0)), boxes_returned=int(diagnostics.get("boxes_returned", 0)), masks_returned=int(diagnostics.get("masks_returned", 0)), raw_output_excerpt=str(diagnostics.get("raw_output_excerpt", "")), parser_diagnostics=str(diagnostics.get("parser_diagnostics", "")), error=str(error))
    diagnostics = result.diagnostics
    return SamImageTestResult(status="completed", prompt=str(diagnostics["prompt"]), outcome=str(diagnostics["outcome"]), records=int(diagnostics["records"]), boxes_returned=int(diagnostics["boxes_returned"]), masks_returned=int(diagnostics["masks_returned"]), object_id="1", confidence=float(diagnostics["confidence"]), raw_output_excerpt=str(diagnostics["raw_output_excerpt"]), parser_diagnostics=str(diagnostics["parser_diagnostics"]), mask_overlay=SamMaskOverlay.model_validate(result.mask_overlay))


@app.post("/api/sam-test/video/prepare", response_model=SamVideoPrepared)
async def sam_test_video_prepare(file: UploadFile = File(...), frame_index: int | None = Form(None)) -> SamVideoPrepared:
    """Store an isolated video-test upload and return its representative frame."""
    suffix = Path(file.filename or "sam-test.mp4").suffix or ".mp4"
    test_id = uuid4().hex
    path = Path("data") / f"sam-video-test-{test_id}{suffix}"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(await file.read())
    try:
        prepared = prepare_video(path, frame_index)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return SamVideoPrepared(test_id=test_id, frame_data_url=prepared.frame_data_url, width=prepared.width, height=prepared.height, fps=prepared.fps, frame_count=prepared.frame_count, frame_index=prepared.frame_index)


@app.post("/api/sam-test/video", response_model=SamVideoTestResult)
async def sam_test_video(test_id: str = Form(...), prompt: str = Form(...), propagation_direction: Literal["both", "forward", "backward"] = Form("forward")) -> SamVideoTestResult:
    """Run isolated SAM video tracking without fallback or scientific updates."""
    if Path(test_id).name != test_id:
        raise HTTPException(status_code=400, detail="Invalid SAM test id.")
    try:
        guided = GuidedPrompt.model_validate(json.loads(prompt))
    except (json.JSONDecodeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Invalid guided prompt JSON.") from error
    candidates = sorted(Path("data").glob(f"sam-video-test-{test_id}.*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise HTTPException(status_code=404, detail="Video test upload not found.")
    try:
        result = await segment_video_guided(candidates[0], guided.frame_index, guided.width, guided.height, guided.box, guided.positive_points, guided.negative_points, propagation_direction)
    except SamUnavailableError as error:
        diagnostics = error.diagnostics
        return SamVideoTestResult(status="failed", propagation_direction=propagation_direction, prompt=str(diagnostics.get("prompt", "")), outcome=str(diagnostics.get("outcome")) if diagnostics.get("outcome") is not None else None, records=int(diagnostics.get("records", 0)), boxes_returned=int(diagnostics.get("boxes_returned", 0)), masks_returned=int(diagnostics.get("masks_returned", 0)), frames_with_masks=int(diagnostics.get("frames_with_masks", 0)), frames_tracked=int(diagnostics.get("frames_with_selected_mask", 0)), raw_output_excerpt=str(diagnostics.get("raw_output_excerpt", "")), parser_diagnostics=str(diagnostics.get("parser_diagnostics", "")), stream_event_excerpt=str(diagnostics.get("stream_event_excerpt", "")), error=str(error))
    diagnostics = result.diagnostics
    confidence = float(np.mean([track.get("confidence", 0.0) for track in result.tracks])) if result.tracks else 0.0
    return SamVideoTestResult(status="completed", propagation_direction=propagation_direction, prompt=str(diagnostics["prompt"]), outcome=str(diagnostics["outcome"]), records=int(diagnostics["records"]), boxes_returned=int(diagnostics["boxes_returned"]), masks_returned=int(diagnostics["masks_returned"]), frames_with_masks=int(diagnostics["frames_with_masks"]), frames_tracked=len(result.tracks), object_id=str(diagnostics.get("selected_object_id", "1")), confidence=confidence, raw_output_excerpt=str(diagnostics["raw_output_excerpt"]), parser_diagnostics=str(diagnostics["parser_diagnostics"]), stream_event_excerpt=str(diagnostics.get("stream_event_excerpt", "")), mask_overlay=SamMaskOverlay.model_validate(result.mask_overlay) if result.mask_overlay is not None else None, tracks=result.tracks)


@app.post("/api/investigations", response_model=Investigation)
def create_investigation(payload: StartInvestigation) -> Investigation:
    now = datetime.now(UTC)
    recommendation = {"angle_deg": 24.0, "duration_s": 8.0, "rationale": "Upload an initial observation first; EIG will be recomputed from its posterior."}
    item = Investigation(id=str(uuid4()), system=payload.system, objective=payload.objective, calibration_px=payload.calibration_px, calibration_m=payload.calibration_m, status="awaiting observation", model="Damped nonlinear pendulum", reasoning="The investigation is ready. Start with a clear, fixed-camera observation so the perception layer can establish a trajectory.", parameters={"length_m": "pending", "damping_s-1": "pending", "adequacy": "pending"}, recommendation=recommendation, posterior={"length": [], "damping": []}, experiments=[], observations=[], prediction=None, validation=None, report=None, agent_status="Pending", agent_request_status="not_invoked", agent_completion_status="not_started", agent_finish_reason="", agent_response_length=0, agent_error="", created_at=now, updated_at=now)
    return save(item)


@app.get("/api/investigations/{item_id}", response_model=Investigation)
def get_investigation(item_id: str) -> Investigation:
    item = load(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Investigation not found")
    return item


@app.get("/api/demo/replays", response_model=list[DemoReplaySummary])
def list_demo_replays() -> list[DemoReplaySummary]:
    """List local saved investigations suitable for a no-network replay."""
    items = [
        item
        for path in Path("data").glob("*.json")
        if (item := load(path.stem)) is not None and item.observations
    ]
    items.sort(key=lambda item: item.updated_at, reverse=True)
    return [
        DemoReplaySummary(
            id=item.id,
            system=item.system,
            status=item.status,
            updated_at=item.updated_at,
            observation_count=len(item.observations),
            latest_observation_quality=item.observations[-1].quality,
            latest_observation_frames=item.observations[-1].frames,
            has_saved_muse=(
                item.agent_status.casefold() == "completed" and item.agent_response_length > 0
            ),
            has_frozen_prediction=item.prediction is not None,
            has_validation=item.validation is not None,
        )
        for item in items
    ]


@app.get("/api/demo/source-video")
def demo_source_video() -> FileResponse:
    """Serve the user-supplied demo video stored locally and excluded from Git."""
    path = Path("data") / "demo-source-video.mp4"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="The supplied demo video is not installed locally.")
    return FileResponse(path, media_type="video/mp4", filename="WhatsApp Video 2026-11-02 at 2.24.27 PM.mp4")


@app.get("/api/demo/synthetic-replay")
def synthetic_validation_replay() -> dict[str, object]:
    """Return the latest cached, no-network replay of the sanitized simulator run."""
    candidates = sorted(
        Path("data/perception-validation").glob("*/demo-evaluation.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for evaluation_path in candidates:
        try:
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            state_path = evaluation_path.parent / "session.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        geometry = state.get("geometry_quality") or {}
        tracking = state.get("sanitized_sam_tracking") or {}
        sampling = state.get("sampling") or {}
        source = sampling.get("source") or {}
        return {
            **evaluation,
            "video_url": "/api/demo/source-video",
            "sampling": {
                "source_width": source.get("width"),
                "source_height": source.get("height"),
                "source_fps": source.get("fps"),
                "source_frame_count": source.get("frame_count"),
                "start_s": sampling.get("source_start_s"),
                "end_s": sampling.get("source_end_s"),
                "frame_stride": sampling.get("frame_stride"),
            },
            "tracking": {
                "method": tracking.get("method"),
                "status": tracking.get("status"),
                "object_id": tracking.get("object_id"),
                "frames_tracked": tracking.get("frames_tracked", 0),
                "frames_with_masks": tracking.get("frames_with_masks", 0),
                "confidence": tracking.get("confidence", 0.0),
                "input_sanitization": tracking.get("input_sanitization"),
                "sanitized_frame_count": tracking.get("sanitized_frame_count", 0),
                "tracks": tracking.get("tracks", []),
            },
            "annotation": state.get("annotation"),
            "geometry_quality": geometry,
            "comparison": state.get("csv_comparison"),
        }
    raise HTTPException(status_code=404, detail="No cached synthetic validation replay is available.")


@app.get("/api/demo/replays/{item_id}", response_model=DemoReplayResponse)
def replay_saved_investigation(item_id: str) -> DemoReplayResponse:
    """Recompute inference locally from one saved observation without API calls."""
    item = load(item_id)
    if item is None or not item.observations:
        raise HTTPException(status_code=404, detail="Saved observation not found.")
    observation = item.observations[-1]
    fit = None
    if len(observation.timestamps) >= 12 and len(observation.timestamps) == len(observation.angles_deg):
        try:
            fit = fit_posterior(
                np.asarray(observation.timestamps),
                np.asarray(observation.angles_deg),
            )
        except ValueError:
            fit = None

    eig_angle = None
    eig_duration = None
    eig_nats = None
    eig_rationale = "No new experiment selected because the replayed fit is not adequate."
    predicted_times: list[float] = []
    predicted_angles: list[float] = []
    prediction_source: Literal[
        "saved_frozen_prediction", "local_replay_forecast", "none"
    ] = "none"
    inference_status: Literal["computed", "unavailable"] = "unavailable"
    if fit is not None:
        inference_status = "computed"
        if fit.adequacy == "good":
            eig_angle, eig_duration, eig_nats, eig_rationale = choose_experiment(
                fit.posterior, [experiment.angle_deg for experiment in item.experiments]
            )
            _, replay_angles = simulate(
                eig_angle, eig_duration, fit.length, fit.damping, count=180
            )
            predicted_times = np.linspace(0.0, eig_duration, len(replay_angles)).tolist()
            predicted_angles = replay_angles.tolist()
            prediction_source = "local_replay_forecast"
    if item.prediction is not None:
        predicted_times = item.prediction.times
        predicted_angles = item.prediction.angles_deg
        prediction_source = "saved_frozen_prediction"

    muse_saved = item.agent_status.casefold() == "completed" and item.agent_response_length > 0
    muse_text = ""
    if muse_saved:
        reasoning_parts = item.reasoning.split("\n\nAgent: ", maxsplit=1)
        if len(reasoning_parts) == 2 and "] " in reasoning_parts[1]:
            muse_text = reasoning_parts[1].split("] ", maxsplit=1)[1]
    validation = item.validation
    representative_frame_data_url = None
    for suffix in (".mp4", ".mov", ".avi"):
        saved_video = Path("data") / f"{item.id}-pending{suffix}"
        if saved_video.is_file():
            try:
                representative_frame_data_url = prepare_video(saved_video).frame_data_url
            except ValueError:
                representative_frame_data_url = None
            break
    return DemoReplayResponse(
        id=item.id,
        system=item.system,
        objective=item.objective,
        saved_status=item.status,
        observation_quality=observation.quality,
        observation_method=observation.method,
        observation_frames=observation.frames,
        observation_confidence=observation.confidence,
        representative_frame_data_url=representative_frame_data_url,
        sam_mask_overlay=observation.sam_mask_overlay,
        tracked_points=observation.tracked_points,
        observed_times_s=observation.timestamps,
        observed_angles_deg=observation.angles_deg,
        inference_status=inference_status,
        length_m=fit.length if fit is not None else None,
        length_interval_m=list(fit.length_interval) if fit is not None else [],
        damping_s_minus_1=fit.damping if fit is not None else None,
        damping_interval_s_minus_1=list(fit.damping_interval) if fit is not None else [],
        residual_rms_deg=fit.residual_rms if fit is not None else None,
        model_adequacy=fit.adequacy if fit is not None else "unavailable",
        eig_angle_deg=eig_angle,
        eig_duration_s=eig_duration,
        eig_nats=eig_nats,
        eig_rationale=eig_rationale,
        prediction_source=prediction_source,
        predicted_times_s=predicted_times,
        predicted_angles_deg=predicted_angles,
        validation_status=validation.status if validation is not None else "Awaiting validation",
        validation_rmse_deg=validation.rmse_deg if validation is not None else None,
        validation_max_error_deg=validation.max_error_deg if validation is not None else None,
        validation_times_s=validation.actual_times if validation is not None else [],
        validation_angles_deg=validation.actual_angles_deg if validation is not None else [],
        muse_status=item.agent_status if muse_saved else "Not saved",
        muse_model=item.agent_model if muse_saved else "",
        muse_text=muse_text,
    )


@app.post("/api/investigations/{item_id}/reason", response_model=Investigation)
def retry_muse_reasoning(item_id: str) -> Investigation:
    """Retry only Muse narration from already saved, adequate numerical results."""
    item = load(item_id)
    if item is None or not item.observations:
        raise HTTPException(status_code=404, detail="Saved investigation not found.")
    if item.parameters.get("adequacy") != "good" or item.recommendation.get("angle_deg", 0) <= 0:
        raise HTTPException(
            status_code=409,
            detail="Muse retry is unavailable because the saved model does not support an experiment.",
        )
    prompt = (
        f"Explain these saved ProbeLab results for the objective: {item.objective}. "
        f"The numerical engine estimates length {item.parameters.get('length_m')} m "
        f"with interval {item.parameters.get('length_interval_m')}, damping "
        f"{item.parameters.get('damping_s-1')} s^-1 with interval "
        f"{item.parameters.get('damping_interval_s-1')}, RMS residual "
        f"{item.parameters.get('residual_rms_deg')} degrees, adequacy good. "
        f"The EIG selector chose {item.recommendation.get('angle_deg')} degrees for "
        f"{item.recommendation.get('duration_s')} seconds. Explain uncertainty and "
        "why the selector chose this action. Do not invent or change any result."
    )
    narrative = muse.reason(prompt)
    perception = item.observations[-1].message
    reasoning = f"Perception: {perception}\n\nAgent: [{narrative.status} · {narrative.model} · finish={narrative.finish_reason}] {narrative.text}"
    return save(
        item.model_copy(
            update={
                "reasoning": reasoning,
                **_agent_updates(narrative),
                "updated_at": datetime.now(UTC),
            }
        )
    )


@app.post("/api/investigations/{item_id}/observe", response_model=Investigation)
async def observe(item_id: str, file: UploadFile = File(...)) -> Investigation:
    item = load(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Investigation not found")
    suffix = Path(file.filename or "observation.mp4").suffix or ".mp4"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        video_path = Path(tmp.name)
    perception_note = ""
    try:
        segmentation = await segment_video(video_path)
        observation = extract_from_sam_tracks(segmentation.tracks, 30.0, item.calibration_px, item.calibration_m, segmentation.diagnostics, segmentation.mask_overlay)
    except SamUnavailableError as error:
        observation = extract_video(video_path, item.calibration_px, item.calibration_m)
        perception_note = f" SAM 3.1 unavailable ({error}); used the transparent OpenCV fallback."
        observation = observation.model_copy(update={"message": f"{observation.message}{perception_note} SAM diagnostics: masks={error.diagnostics.get('masks_returned', 'unknown')}, frames_with_masks={error.diagnostics.get('frames_with_masks', 'unknown')}, selected_frames={error.diagnostics.get('frames_with_selected_mask', 'unknown')}.", "sam_diagnostics": error.diagnostics})
    if observation.quality == "insufficient":
        item = item.model_copy(update={"observations": [*item.observations, observation], "status": "needs better observation", "reasoning": observation.message, "agent_status": "Pending", "agent_request_status": "not_invoked", "agent_completion_status": "not_started", "agent_response_length": 0, "agent_error": "Muse Spark not invoked because observation quality was insufficient.", "updated_at": datetime.now(UTC)})
        return save(item)
    if observation.quality != "usable":
        item = item.model_copy(update={"observations": [*item.observations, observation], "status": "needs better observation", "reasoning": observation.message, "agent_status": "Pending", "agent_request_status": "not_invoked", "agent_completion_status": "not_started", "agent_response_length": 0, "agent_error": "Muse Spark not invoked because observation quality was not usable.", "updated_at": datetime.now(UTC)})
        return save(item)
    fit = fit_posterior(np.array(observation.timestamps), np.array(observation.angles_deg))
    completed = [experiment.angle_deg for experiment in item.experiments]
    next_angle, next_duration, eig, rationale = choose_experiment(fit.posterior, completed)
    prompt = f"You are Muse Spark coordinating a physical discovery loop. Objective: {item.objective}. Deterministic engine reports length {fit.length:.3f} m, damping {fit.damping:.3f} s^-1, intervals {fit.length_interval} and {fit.damping_interval}, RMS residual {fit.residual_rms:.2f} degrees, adequacy {fit.adequacy}. Recommended next experiment: {next_angle:.0f} degrees for {next_duration:.0f} seconds. Explain this honestly, never invent measurements."
    narrative = muse.reason(prompt + perception_note)
    valid = fit.adequacy == "good" and observation.quality == "usable"
    recommendation = {"angle_deg": next_angle if valid else 0.0, "duration_s": next_duration if valid else 0.0, "eig_nats": eig if valid else 0.0, "rationale": rationale if valid else "No experiment recommended: the current observation does not support a scientifically adequate model fit."}
    updated = item.model_copy(update={"observations": [*item.observations, observation], "status": "experiment recommended" if valid else "needs better evidence", "reasoning": f"Perception: {observation.message}\n\nAgent: [{narrative.status} · {narrative.model} · finish={narrative.finish_reason}] {narrative.text}", **_agent_updates(narrative), "parameters": {"length_m": fit.length, "damping_s-1": fit.damping, "length_interval_m": list(fit.length_interval), "damping_interval_s-1": list(fit.damping_interval), "residual_rms_deg": fit.residual_rms, "adequacy": fit.adequacy, "calibration_px": observation.calibration_px, "calibration_m": observation.calibration_m, "tracked_radius_m": observation.tracked_radius_m or 0.0}, "recommendation": recommendation, "posterior": {"length": [float(value) for value in fit.posterior.lengths], "damping": [float(value) for value in fit.posterior.damping]}, "updated_at": datetime.now(UTC)})
    return save(updated)


@app.post("/api/investigations/{item_id}/prepare", response_model=PreparedVideo)
async def prepare(item_id: str, file: UploadFile = File(...)) -> PreparedVideo:
    """Persist an upload and return a representative frame for visual prompting."""
    if load(item_id) is None:
        raise HTTPException(status_code=404, detail="Investigation not found")
    suffix = Path(file.filename or "observation.mp4").suffix or ".mp4"
    pending_path = Path("data") / f"{item_id}-pending{suffix}"
    pending_path.parent.mkdir(exist_ok=True)
    pending_path.write_bytes(await file.read())
    try:
        return prepare_video(pending_path)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/investigations/{item_id}/observe-guided", response_model=Investigation)
async def observe_guided(item_id: str, prompt: GuidedPrompt) -> Investigation:
    """Run SAM with a user-selected box and points from the prepared upload."""
    item = load(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Investigation not found")
    pending_files = sorted(Path("data").glob(f"{item_id}-pending.*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not pending_files:
        raise HTTPException(status_code=409, detail="Prepare a video before submitting a visual prompt.")
    video_path = pending_files[0]
    perception_note = ""
    try:
        segmentation = await segment_video_guided(video_path, prompt.frame_index, prompt.width, prompt.height, prompt.box, prompt.positive_points, prompt.negative_points)
        observation = extract_from_sam_tracks(segmentation.tracks, 30.0, item.calibration_px, item.calibration_m, segmentation.diagnostics, segmentation.mask_overlay)
    except SamUnavailableError as error:
        perception_note = f" SAM 3.1 guided prompt rejected ({error}); used the transparent OpenCV fallback."
        fallback = extract_video(video_path, item.calibration_px, item.calibration_m)
        observation = fallback.model_copy(update={"message": f"{fallback.message}{perception_note} Diagnostics: {error.diagnostics}", "sam_diagnostics": error.diagnostics})
    if observation.quality != "usable":
        return save(item.model_copy(update={"observations": [*item.observations, observation], "status": "needs better observation", "reasoning": observation.message, "agent_status": "Pending", "agent_request_status": "not_invoked", "agent_completion_status": "not_started", "agent_response_length": 0, "agent_error": "Muse Spark not invoked because observation quality was not usable.", "updated_at": datetime.now(UTC)}))
    fit = fit_posterior(np.array(observation.timestamps), np.array(observation.angles_deg))
    completed = [experiment.angle_deg for experiment in item.experiments]
    next_angle, next_duration, eig, rationale = choose_experiment(fit.posterior, completed)
    valid = fit.adequacy == "good"
    recommendation = {"angle_deg": next_angle if valid else 0.0, "duration_s": next_duration if valid else 0.0, "eig_nats": eig if valid else 0.0, "rationale": rationale if valid else "No experiment recommended: the guided observation does not support an adequate fit."}
    prompt_text = f"You are Muse Spark coordinating a physical discovery loop. Objective: {item.objective}. Deterministic ProbeLab results: SAM observation quality {observation.quality}; {observation.frames} tracked frames; estimated length {fit.length:.3f} m with credible interval {fit.length_interval}; damping {fit.damping:.3f} s^-1 with credible interval {fit.damping_interval}; RMS residual {fit.residual_rms:.2f} degrees; model adequacy {fit.adequacy}. The deterministic EIG selector chose angle {next_angle:.0f} degrees for {next_duration:.0f} seconds with rationale: {rationale}. The displayed action is {recommendation['angle_deg']:.0f} degrees for {recommendation['duration_s']:.0f} seconds. Explain whether the numerical fit is adequate, interpret uncertainty, and explain the selector's action. Do not invent measurements, alter Bayesian estimates, or choose a different experiment. If the displayed action is 0 degrees, explain that it is an intentional no-action result because the fit is inadequate."
    narrative = muse.reason(prompt_text + perception_note)
    updated = item.model_copy(update={"observations": [*item.observations, observation], "status": "experiment recommended" if valid else "needs better evidence", "reasoning": f"Perception: {observation.message}\n\nAgent: [{narrative.status} · {narrative.model} · finish={narrative.finish_reason}] {narrative.text}", **_agent_updates(narrative), "parameters": {"length_m": fit.length, "damping_s-1": fit.damping, "length_interval_m": list(fit.length_interval), "damping_interval_s-1": list(fit.damping_interval), "residual_rms_deg": fit.residual_rms, "adequacy": fit.adequacy, "calibration_px": observation.calibration_px, "calibration_m": observation.calibration_m, "tracked_radius_m": observation.tracked_radius_m or 0.0}, "recommendation": recommendation, "posterior": {"length": [float(value) for value in fit.posterior.lengths], "damping": [float(value) for value in fit.posterior.damping]}, "updated_at": datetime.now(UTC)})
    return save(updated)


@app.post("/api/investigations/{item_id}/predict", response_model=Investigation)
def predict(item_id: str, payload: ExperimentRequest) -> Investigation:
    """Freeze the learned model and create an unseen-condition prediction."""
    item = load(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Investigation not found")
    length = item.parameters.get("length_m")
    damping = item.parameters.get("damping_s-1")
    if not isinstance(length, float) or not isinstance(damping, float):
        raise HTTPException(status_code=409, detail="A fitted scientific model is required before validation.")
    times, angles = simulate(payload.angle_deg, payload.duration_s, length, damping)
    prediction = PredictionRecord(id=f"P-{uuid4().hex[:8]}", angle_deg=payload.angle_deg, duration_s=payload.duration_s, model_version=f"posterior-{item.updated_at.isoformat()}", times=[float(value) for value in times], angles_deg=[float(value) for value in angles], frozen_parameters={"length_m": length, "damping_s-1": damping}, created_at=datetime.now(UTC))
    return save(item.model_copy(update={"prediction": prediction, "validation": None, "status": "prediction frozen; awaiting validation", "updated_at": datetime.now(UTC)}))


@app.post("/api/investigations/{item_id}/validate", response_model=Investigation)
async def validate(item_id: str, file: UploadFile = File(...)) -> Investigation:
    """Compare validation evidence against the frozen prediction before updating anything."""
    item = load(item_id)
    if item is None or item.prediction is None:
        raise HTTPException(status_code=409, detail="Freeze a prediction before uploading validation evidence.")
    suffix = Path(file.filename or "validation.mp4").suffix or ".mp4"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        video_path = Path(tmp.name)
    try:
        segmentation = await segment_video(video_path)
        actual = extract_from_sam_tracks(segmentation.tracks, 30.0, item.calibration_px, item.calibration_m, segmentation.diagnostics, segmentation.mask_overlay)
    except SamUnavailableError:
        actual = extract_video(video_path, item.calibration_px, item.calibration_m)
    predicted = np.interp(np.array(actual.timestamps), np.array(item.prediction.times), np.array(item.prediction.angles_deg))
    errors = np.array(actual.angles_deg) - predicted
    validation = ValidationResult(prediction_id=item.prediction.id, actual_times=actual.timestamps, actual_angles_deg=actual.angles_deg, rmse_deg=float(np.sqrt(np.mean(errors**2))), max_error_deg=float(np.max(np.abs(errors))), status="validated against frozen prediction", created_at=datetime.now(UTC))
    return save(item.model_copy(update={"validation": validation, "status": "validation complete", "updated_at": datetime.now(UTC)}))


@app.post("/api/investigations/{item_id}/report", response_model=Investigation)
def report(item_id: str) -> Investigation:
    """Generate an evidence-backed report from the stored scientific state."""
    item = load(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Investigation not found")
    if item.prediction is None or item.validation is None:
        raise HTTPException(status_code=409, detail="Complete prediction and validation before generating the report.")
    report_text = f"Objective: {item.objective}\n\nModel: {item.model}. Assumption: damped nonlinear pendulum with fixed gravity {9.80665} m/s².\n\nExperiments: {len(item.experiments)} recorded.\n\nEstimated parameters: length {item.parameters.get('length_m')} m; damping {item.parameters.get('damping_s-1')} s^-1. Credible intervals: {item.parameters.get('length_interval_m')} m and {item.parameters.get('damping_interval_s-1')} s^-1.\n\nModel fit: RMS residual {item.parameters.get('residual_rms_deg')} degrees; adequacy {item.parameters.get('adequacy')}.\n\nPrediction: frozen model version {item.prediction.model_version}; validation RMSE {item.validation.rmse_deg:.3f} degrees; maximum absolute error {item.validation.max_error_deg:.3f} degrees.\n\nLimitations: visual segmentation confidence, calibration quality, unmodeled pivot motion, and model mismatch can affect these results."
    return save(item.model_copy(update={"report": report_text, "status": "report ready", "updated_at": datetime.now(UTC)}))


@app.post("/api/investigations/{item_id}/experiment", response_model=Investigation)
def record_experiment(item_id: str, payload: ExperimentRequest) -> Investigation:
    item = load(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Investigation not found")
    if item.status != "experiment recommended" or item.recommendation.get("angle_deg", 0.0) <= 0:
        raise HTTPException(status_code=409, detail="The current observation is not scientifically adequate for another experiment.")
    eig = item.recommendation.get("eig_nats", 0.0)
    experiment = ExperimentRecord(id=f"E-{len(item.experiments) + 1:02d}", angle_deg=payload.angle_deg, duration_s=payload.duration_s, eig=float(eig) if isinstance(eig, (int, float)) else 0.0, confidence=0.0, quality="awaiting uploaded result", created_at=datetime.now(UTC))
    return save(item.model_copy(update={"experiments": [*item.experiments, experiment], "status": "awaiting experiment observation", "updated_at": datetime.now(UTC)}))
