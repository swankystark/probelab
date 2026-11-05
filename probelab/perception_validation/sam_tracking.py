"""Temporary sampled guided-SAM tracking adapter for validation sessions."""

from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Literal

import cv2

from ..sam_client import SamSegmentation, SamUnavailableError, segment_video_guided
from .models import (
    ExclusionRegion,
    MaskCentroidTrack,
    Point,
    SamTrackingRequest,
    SamTrackingResult,
)
from .sanitization import apply_exclusion_masks


@contextmanager
def _sampled_video(
    request: SamTrackingRequest,
    exclusion_regions: list[ExclusionRegion] | None = None,
) -> Iterator[tuple[Path, list[int]]]:
    """Create a temporary sampled clip, masking configured UI regions per frame."""
    source = request.sampling.source
    start = max(0, math.ceil(request.sampling.source_start_s * source.fps))
    end = min(source.frame_count, math.ceil(request.sampling.source_end_s * source.fps))
    selected = list(range(start, end, request.sampling.frame_stride))
    if request.annotation.frame_index not in selected:
        raise ValueError("The annotated source frame is outside the selected sampling policy.")
    capture = cv2.VideoCapture(request.video_path)
    if not capture.isOpened():
        capture.release()
        raise ValueError("The validation video could not be opened.")
    with TemporaryDirectory(prefix="sam-validation-") as directory:
        path = Path(directory) / "sampled.mp4"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source.fps,
            (source.width, source.height),
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError("The sampled validation video could not be created.")
        written: list[int] = []
        try:
            for source_index in selected:
                capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
                ok, frame = capture.read()
                if ok:
                    if exclusion_regions:
                        frame = apply_exclusion_masks(frame, exclusion_regions)
                    writer.write(frame)
                    written.append(source_index)
        finally:
            writer.release()
            capture.release()
        yield path, written


def _points(points: list[Point]) -> list[tuple[float, float]]:
    """Convert strict annotation points to the existing SAM client shape."""
    return [(point.x, point.y) for point in points]


def _map_segmentation(
    segmentation: SamSegmentation,
    selected: list[int],
    request: SamTrackingRequest,
    fallback_label: Literal["none", "forward_after_reverse_failure"],
) -> SamTrackingResult:
    diagnostics = segmentation.diagnostics
    object_id = str(diagnostics.get("selected_object_id", "1"))
    tracks: list[MaskCentroidTrack] = []
    rejection_reasons: list[str] = []
    for track in segmentation.tracks:
        clip_index = int(track["time"])
        if clip_index < 0 or clip_index >= len(selected):
            rejection_reasons.append(f"returned clip frame {clip_index} has no source-frame mapping")
            continue
        source_index = selected[clip_index]
        tracks.append(
            MaskCentroidTrack(
                frame_index=source_index,
                timestamp_s=source_index / request.sampling.source.fps,
                centroid=Point(x=track["x"], y=track["y"]),
                mask_area_px=max(1, int(track.get("area", 0.0))),
                confidence=max(0.0, min(1.0, track.get("confidence", 0.0))),
                object_id=object_id,
            )
        )
    tracked_indices = {track.frame_index for track in tracks}
    missing = [index for index in selected if index not in tracked_indices]
    if missing:
        rejection_reasons.append(
            f"{len(missing)} selected source frames had no usable selected-object mask"
        )
    return SamTrackingResult(
        status="partial" if missing else "completed",
        method=(
            "sam_guided_mask_forward_fallback"
            if fallback_label != "none"
            else "sam_guided_mask"
        ),
        propagation_direction=request.propagation_direction,
        fallback_label=fallback_label,
        sampling=request.sampling,
        outcome=str(diagnostics.get("outcome")) if diagnostics.get("outcome") is not None else None,
        records=int(diagnostics.get("records", 0)),
        masks_returned=int(diagnostics.get("masks_returned", 0)),
        frames_with_masks=int(diagnostics.get("frames_with_masks", 0)),
        frames_tracked=len(tracks),
        object_id=object_id,
        confidence=sum(track.confidence for track in tracks) / len(tracks) if tracks else 0.0,
        selected_frame_indices=[track.frame_index for track in tracks],
        selected_timestamps_s=[track.timestamp_s for track in tracks],
        missing_frame_indices=missing,
        rejection_reasons=rejection_reasons,
        raw_output_excerpt=str(diagnostics.get("raw_output_excerpt", "")),
        parser_diagnostics=str(diagnostics.get("parser_diagnostics", "")),
        stream_event_excerpt=str(diagnostics.get("stream_event_excerpt", "")),
        tracks=tracks,
    )


async def track_sampled_guided_video(
    request: SamTrackingRequest,
    exclusion_regions: list[ExclusionRegion] | None = None,
) -> SamTrackingResult:
    """Sample frames, optionally mask UI regions, then run existing guided SAM."""
    with _sampled_video(request, exclusion_regions) as (sampled_path, selected):
        annotation_clip_index = selected.index(request.annotation.frame_index)
        forward: SamTrackingResult | None = None
        forward_error: SamUnavailableError | None = None
        try:
            segmentation = await segment_video_guided(
                sampled_path,
                annotation_clip_index,
                request.sampling.source.width,
                request.sampling.source.height,
                tuple(request.annotation.bob_box),
                _points(request.annotation.positive_points),
                _points(request.annotation.negative_points),
                "forward",
            )
            forward = _map_segmentation(segmentation, selected, request, "none")
        except SamUnavailableError as error:
            forward_error = error
        if request.propagation_direction == "forward" and forward is not None:
            return _with_sanitization(forward, exclusion_regions, len(selected))
        if request.propagation_direction != "forward":
            try:
                segmentation = await segment_video_guided(
                    sampled_path,
                    annotation_clip_index,
                    request.sampling.source.width,
                    request.sampling.source.height,
                    tuple(request.annotation.bob_box),
                    _points(request.annotation.positive_points),
                    _points(request.annotation.negative_points),
                    request.propagation_direction,
                )
                result = _map_segmentation(segmentation, selected, request, "none")
                return _with_sanitization(result, exclusion_regions, len(selected))
            except SamUnavailableError as reverse_error:
                if forward is not None:
                    result = forward.model_copy(
                        update={
                            "method": "sam_guided_mask_forward_fallback",
                            "fallback_label": "forward_after_reverse_failure",
                            "rejection_reasons": [
                                *forward.rejection_reasons,
                                f"{request.propagation_direction} propagation failed: {reverse_error}",
                            ],
                        }
                    )
                    return _with_sanitization(result, exclusion_regions, len(selected))
                if forward_error is not None:
                    result = _failed_result(request, forward_error, reverse_error)
                    return _with_sanitization(result, exclusion_regions, len(selected))
                raise
        if forward_error is not None:
            result = _failed_result(request, forward_error, None)
            return _with_sanitization(result, exclusion_regions, len(selected))
        raise RuntimeError("SAM tracking finished without a forward result or error.")


def _with_sanitization(
    result: SamTrackingResult,
    exclusion_regions: list[ExclusionRegion] | None,
    sanitized_frame_count: int,
) -> SamTrackingResult:
    """Attach server-derived provenance for the exact masked frame set."""
    if not exclusion_regions:
        return result
    return result.model_copy(
        update={
            "input_sanitization": "exclusion_masks_applied",
            "applied_exclusion_regions": exclusion_regions,
            "sanitized_frame_count": sanitized_frame_count,
        }
    )


def _failed_result(
    request: SamTrackingRequest,
    forward_error: SamUnavailableError,
    reverse_error: SamUnavailableError | None,
) -> SamTrackingResult:
    """Keep expected provider failures inspectable without inventing observations."""
    diagnostics = forward_error.diagnostics
    reasons = [str(forward_error)]
    if reverse_error is not None:
        reasons.append(f"{request.propagation_direction} propagation failed: {reverse_error}")
    return SamTrackingResult(
        status="failed",
        method="sam_guided_mask",
        propagation_direction=request.propagation_direction,
        sampling=request.sampling,
        outcome=str(diagnostics.get("outcome")) if diagnostics.get("outcome") is not None else None,
        records=int(diagnostics.get("records", 0)),
        masks_returned=int(diagnostics.get("masks_returned", 0)),
        frames_with_masks=int(diagnostics.get("frames_with_masks", 0)),
        object_id=str(diagnostics.get("selected_object_id"))
        if diagnostics.get("selected_object_id") is not None
        else None,
        rejection_reasons=reasons,
        raw_output_excerpt=str(diagnostics.get("raw_output_excerpt", "")),
        parser_diagnostics=str(diagnostics.get("parser_diagnostics", "")),
        stream_event_excerpt=str(diagnostics.get("stream_event_excerpt", "")),
    )
