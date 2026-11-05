"""Temporary video interval and sampling helpers for validation sessions."""

from dataclasses import dataclass
import math
from pathlib import Path

import cv2

from ..perception import prepare_video
from .models import RepresentativeFrameResponse, SamplingRequest, SamplingSelection, VideoSource


@dataclass(frozen=True, slots=True)
class PreparedValidationFrame:
    """Prepared preview plus the source sampling metadata it represents."""

    response: RepresentativeFrameResponse
    selection: SamplingSelection


def prepare_validation_frame(
    session_id: str,
    path: Path,
    request: SamplingRequest,
) -> PreparedValidationFrame:
    """Validate source bounds, choose source frames, and reuse ``prepare_video``."""
    source = _read_source(path)
    if request.source_end_s > source.duration_s:
        raise ValueError("source_end_s exceeds the video duration")
    if request.target_fps is not None:
        if request.target_fps > source.fps:
            raise ValueError("target_fps cannot exceed the source fps")
        frame_stride = max(1, math.ceil(source.fps / request.target_fps))
    else:
        frame_stride = request.frame_stride
        if frame_stride is None:
            raise ValueError("a frame stride is required when target_fps is absent")

    start_frame = math.ceil(request.source_start_s * source.fps)
    end_frame = min(source.frame_count, math.ceil(request.source_end_s * source.fps))
    sampled_indices = list(range(start_frame, end_frame, frame_stride))
    if not sampled_indices:
        raise ValueError("the requested interval does not contain a sampleable source frame")
    representative_time = request.representative_time_s
    if representative_time is None:
        representative_index = sampled_indices[0]
    else:
        if not request.source_start_s <= representative_time < request.source_end_s:
            raise ValueError("representative_time_s must lie within the source interval")
        representative_index = math.floor(representative_time * source.fps)
        if representative_index not in sampled_indices:
            raise ValueError("representative_time_s must select a frame on the configured stride")

    prepared = prepare_video(path, representative_index)
    selection = SamplingSelection(
        upload_id=request.upload_id,
        source=source,
        source_start_s=request.source_start_s,
        source_end_s=request.source_end_s,
        frame_stride=frame_stride,
        target_fps=request.target_fps,
        frame_index=representative_index,
        source_timestamp_s=representative_index / source.fps,
    )
    response = RepresentativeFrameResponse(
        session_id=session_id,
        upload_id=request.upload_id,
        frame_data_url=prepared.frame_data_url,
        source=source,
        source_start_s=request.source_start_s,
        source_end_s=request.source_end_s,
        frame_stride=frame_stride,
        target_fps=request.target_fps,
        frame_index=representative_index,
        source_timestamp_s=representative_index / source.fps,
        sampled_frames=[(index, index / source.fps) for index in sampled_indices],
    )
    return PreparedValidationFrame(response=response, selection=selection)


def _read_source(path: Path) -> VideoSource:
    capture = cv2.VideoCapture(str(path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise ValueError("the uploaded video has incomplete source metadata")
    duration_s = frame_count / fps
    if not math.isfinite(duration_s):
        raise ValueError("the uploaded video has an invalid duration")
    return VideoSource(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=duration_s,
    )
