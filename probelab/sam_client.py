"""Hosted SAM 3.1 video segmentation through Meta's documented Responses API."""

from __future__ import annotations

import os
import mimetypes
import base64
from dataclasses import dataclass
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import aclosing
from pathlib import Path
from typing import Any, Literal

import numpy as np
from dotenv import load_dotenv
from meta_sam_parser import FramePrompt, PromptBox, PromptObject, PromptPoint, ResponsesStreamError, SegmentationMaskRecord, build_object_prompt, decode_mask_to_raster, image_segmentation_format, parse_responses_stream, video_segmentation_format
from openai import AsyncOpenAI, OpenAIError


class SamUnavailableError(RuntimeError):
    """Raised when SAM cannot produce a usable segmentation result."""

    def __init__(self, message: str, diagnostics: dict[str, float | int | str] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True, slots=True)
class SamSegmentation:
    tracks: list[dict[str, float]]
    diagnostics: dict[str, float | int | str]
    mask_overlay: dict[str, float | int | str] | None = None


@dataclass(frozen=True, slots=True)
class SamImageSegmentation:
    diagnostics: dict[str, float | int | str]
    mask_overlay: dict[str, float | int | str]


def _error_detail(error: BaseException) -> str:
    """Return a short provider detail without credential-like substrings."""
    detail = str(error).replace("Bearer ", "Bearer [redacted] ")
    return detail[:240] or type(error).__name__


def _client() -> AsyncOpenAI:
    load_dotenv(override=False)
    key = os.getenv("META_API")
    if not key:
        raise SamUnavailableError("META_API is not configured for SAM 3.1.")
    return AsyncOpenAI(base_url="https://api.meta.ai/v1", api_key=key, timeout=45.0, max_retries=0)


async def _events(stream: AsyncIterable[Any]) -> AsyncIterator[dict[str, Any]]:
    """Adapt SDK events and close output-text lanes for the official parser."""
    completed: set[tuple[Any, Any, Any]] = set()
    async for event in stream:
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        lane = (data.get("item_id"), data.get("output_index"), data.get("content_index"))
        if data.get("type") == "response.output_text.done":
            completed.add(lane)
            yield data
            continue
        yield data
        part = data.get("part")
        if data.get("type") == "response.content_part.done" and lane not in completed and isinstance(part, dict) and part.get("type") == "output_text":
            completed.add(lane)
            yield {"type": "response.output_text.done", "item_id": data.get("item_id"), "output_index": data.get("output_index"), "content_index": data.get("content_index"), "text": part.get("text", "")}


def build_guided_prompt(frame_index: int, width: int, height: int, box: tuple[float, float, float, float], positive_points: list[tuple[float, float]], negative_points: list[tuple[float, float]]) -> str:
    """Build the documented fixed-object box/point prompt with stable id 1."""
    raw_left, raw_top, raw_right, raw_bottom = (int(value) for value in box)
    left = max(0, min(raw_left, width - 1))
    top = max(0, min(raw_top, height - 1))
    right = max(left + 1, min(raw_right + 1, width))
    bottom = max(top + 1, min(raw_bottom + 1, height))
    points = [PromptPoint(max(0, min(int(x), width - 1)), max(0, min(int(y), height - 1)), label="positive") for x, y in positive_points]
    points.extend(PromptPoint(max(0, min(int(x), width - 1)), max(0, min(int(y), height - 1)), label="negative") for x, y in negative_points)
    return build_object_prompt(size=(width, height), objects=[PromptObject(id=1, prompts=[FramePrompt(frame=frame_index, box=PromptBox(left, top, right, bottom), points=points)])])


async def segment_image_guided(image_data_url: str, width: int, height: int, box: tuple[float, float, float, float], positive_points: list[tuple[float, float]], negative_points: list[tuple[float, float]]) -> SamImageSegmentation:
    """Run the isolated single-image SAM access test with object id 1."""
    prompt = build_guided_prompt(0, width, height, box, positive_points, negative_points)
    try:
        async with _client() as client:
            stream = await client.responses.create(model="sam-3.1", stream=True, input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": image_data_url}]}], metadata={"mask_encoding": "one_bit", "include_confidence": "true"})
            async with stream, aclosing(_events(stream)) as events:
                parsed = parse_responses_stream(events, image_segmentation_format())
                async with parsed:
                    result = await parsed.final_result()
    except ResponsesStreamError as error:
        event = getattr(error, "event", None)
        diagnostics = {"prompt_mode": "guided_box_point_image", "prompt": prompt[:240], "stream_error_code": error.code, "stream_error": _error_detail(error), "stream_event_excerpt": str(event)[:1600]}
        raise SamUnavailableError(f"SAM 3.1 image test failed ({error.code}): {_error_detail(error)}", diagnostics) from error
    except (OpenAIError, OSError, RuntimeError, ValueError, KeyError, AttributeError, TypeError) as error:
        raise SamUnavailableError(f"SAM 3.1 image test request failed: {_error_detail(error)}", {"prompt_mode": "guided_box_point_image", "prompt": prompt[:240]}) from error
    masks = [record for record in result.records if record.kind == "mask"]
    boxes = [record for record in result.records if record.kind == "box"]
    diagnostics: dict[str, float | int | str] = {"prompt_mode": "guided_box_point_image", "prompt": prompt[:240], "outcome": str(result.outcome.status), "records": len(result.records), "boxes_returned": len(boxes), "masks_returned": len(masks), "raw_output_excerpt": result.raw_output[:1200], "parser_diagnostics": "; ".join(f"{item.code}: {item.message}" for item in result.diagnostics)[:1200]}
    selected = next((record for record in masks if record.object_id == "1"), None)
    if selected is None:
        diagnostics["object_ids_returned"] = ",".join(sorted({record.object_id for record in masks}))
        raise SamUnavailableError("SAM 3.1 image test returned no mask for object id 1.", diagnostics)
    raster = np.asarray(list(decode_mask_to_raster(selected.mask)), dtype=np.uint8).reshape(selected.mask.height, selected.mask.width)
    if not np.any(raster):
        raise SamUnavailableError("SAM 3.1 image test returned an empty mask for object id 1.", diagnostics)
    bounds = selected.bounds
    diagnostics["selected_object_id"] = "1"
    diagnostics["confidence"] = float(selected.confidence) if selected.confidence is not None else 0.0
    overlay = {"frame_index": 0, "left": bounds.left, "top": bounds.top, "right": bounds.right, "bottom": bounds.bottom, "width": selected.mask.width, "height": selected.mask.height, "raster_base64": base64.b64encode(raster.tobytes()).decode("ascii")}
    return SamImageSegmentation(diagnostics, overlay)


async def _segment_file(client: AsyncOpenAI, file_id: str, prompt: str, prompt_mode: str, propagation_direction: Literal["both", "forward", "backward"] = "forward") -> SamSegmentation:
    """Run one documented SAM request and derive mask centroids for the dominant track."""
    try:
        stream = await client.responses.create(model="sam-3.1", stream=True, input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}, {"type": "input_video", "file_id": file_id}]}], metadata={"mask_encoding": "one_bit", "include_confidence": "true", "propagation_direction": propagation_direction})
        async with stream, aclosing(_events(stream)) as events:
            parsed = parse_responses_stream(events, video_segmentation_format())
            async with parsed:
                result = await parsed.final_result()
    except ResponsesStreamError as error:
        event = getattr(error, "event", None)
        diagnostics = {"prompt_mode": prompt_mode, "prompt": prompt[:240], "propagation_direction": propagation_direction, "stream_error_code": error.code, "stream_error": _error_detail(error), "stream_event_excerpt": str(event)[:1600]}
        raise SamUnavailableError(f"SAM 3.1 stream failed ({error.code}): {_error_detail(error)}", diagnostics) from error
    except (OpenAIError, OSError, RuntimeError, ValueError, KeyError, AttributeError, TypeError) as error:
        raise SamUnavailableError(f"SAM 3.1 request failed: {_error_detail(error)}", {"prompt_mode": prompt_mode, "prompt": prompt[:240], "propagation_direction": propagation_direction}) from error
    base_diagnostics: dict[str, float | int | str] = {
        "prompt_mode": prompt_mode,
        "prompt": prompt[:240],
        "propagation_direction": propagation_direction,
        "outcome": str(result.outcome.status),
        "raw_output_excerpt": result.raw_output[:1200],
        "parser_diagnostics": "; ".join(f"{item.code}: {item.message}" for item in result.diagnostics)[:1200],
    }
    if result.outcome.status != "completed":
        raise SamUnavailableError("SAM 3.1 returned an incomplete segmentation.", base_diagnostics)
    boxes = [record for record in result.records if record.kind == "box" and record.frame is not None]
    masks = [record for record in result.records if record.kind == "mask" and record.frame is not None]
    diagnostics: dict[str, float | int | str] = {**base_diagnostics, "records": len(result.records), "boxes_returned": len(boxes), "masks_returned": len(masks), "frames_with_masks": len({record.frame.frame_index for record in masks})}
    if not masks:
        raise SamUnavailableError("SAM 3.1 returned no masks for the prompt.", diagnostics)
    counts: dict[str, int] = {}
    for record in masks:
        counts[record.object_id] = counts.get(record.object_id, 0) + 1
    if prompt_mode == "guided_box_point" and "1" not in counts:
        diagnostics["object_ids_returned"] = ",".join(sorted(counts))
        raise SamUnavailableError("SAM 3.1 did not preserve guided object id 1 across returned masks.", diagnostics)
    object_id = "1" if "1" in counts else max(counts, key=counts.get)
    tracks: list[dict[str, float]] = []
    decoded_masks: list[tuple[int, SegmentationMaskRecord, np.ndarray]] = []
    for mask_record in masks:
        if mask_record.object_id != object_id:
            continue
        raster = np.asarray(list(decode_mask_to_raster(mask_record.mask)), dtype=np.uint8).reshape(mask_record.mask.height, mask_record.mask.width)
        ys, xs = np.nonzero(raster)
        if len(xs) == 0:
            continue
        decoded_masks.append((mask_record.frame.frame_index, mask_record, raster))
        bounds = mask_record.bounds
        bounds_width = max(float(bounds.right - bounds.left), 1.0)
        bounds_height = max(float(bounds.bottom - bounds.top), 1.0)
        tracks.append({"time": float(mask_record.frame.frame_index), "x": float(bounds.left + (np.mean(xs) / mask_record.mask.width) * bounds_width), "y": float(bounds.top + (np.mean(ys) / mask_record.mask.height) * bounds_height), "confidence": float(mask_record.confidence if mask_record.confidence is not None else 0.0), "area": float(np.count_nonzero(raster))})
    diagnostics["selected_object_id"] = object_id
    diagnostics["frames_with_selected_mask"] = len({track["time"] for track in tracks})
    if len(tracks) < 12:
        raise SamUnavailableError("SAM 3.1 masks were returned but the selected object track was too short.", diagnostics)
    target_frame = sorted(frame for frame, _, _ in decoded_masks)[len(decoded_masks) // 2]
    overlay_record = next(record for frame, record, _ in decoded_masks if frame == target_frame)
    overlay_raster = next(raster for frame, _, raster in decoded_masks if frame == target_frame)
    bounds = overlay_record.bounds
    diagnostics["overlay_frame_index"] = target_frame
    return SamSegmentation(tracks, diagnostics, {"frame_index": target_frame, "left": bounds.left, "top": bounds.top, "right": bounds.right, "bottom": bounds.bottom, "width": overlay_record.mask.width, "height": overlay_record.mask.height, "raster_base64": base64.b64encode(overlay_raster.tobytes()).decode("ascii")})


async def segment_video(path: Path, concepts: tuple[str, ...] = ("colored marker on wool ball", "wool ball pendulum bob")) -> SamSegmentation:
    """Upload once and try precise then fallback noun-phrase prompts."""
    media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    if media_type not in {"video/mp4", "video/quicktime"}:
        raise SamUnavailableError(f"SAM 3.1 requires MP4 or MOV video; received {media_type}.", {"file_name": path.name, "media_type": media_type})
    try:
        async with _client() as client:
            uploaded = await client.files.create(file=(path.name, path.read_bytes(), media_type), purpose="user_data")
            last_error: SamUnavailableError | None = None
            for concept in concepts:
                try:
                    return await _segment_file(client, uploaded.id, concept, "text")
                except SamUnavailableError as error:
                    last_error = error
            if last_error is not None:
                raise SamUnavailableError(f"SAM 3.1 found no usable tracked pendulum bob after {len(concepts)} prompts.", last_error.diagnostics)
            raise SamUnavailableError("SAM 3.1 had no prompt attempts.")
    except (OpenAIError, OSError, RuntimeError) as error:
        if isinstance(error, SamUnavailableError):
            raise
        raise SamUnavailableError(f"SAM 3.1 request failed: {_error_detail(error)}") from error


async def segment_video_guided(path: Path, frame_index: int, width: int, height: int, box: tuple[float, float, float, float], positive_points: list[tuple[float, float]], negative_points: list[tuple[float, float]], propagation_direction: Literal["both", "forward", "backward"] = "forward") -> SamSegmentation:
    """Upload once and segment the user-selected object forward with stable id 1."""
    media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    if media_type not in {"video/mp4", "video/quicktime"}:
        raise SamUnavailableError(f"SAM 3.1 requires MP4 or MOV video; received {media_type}.", {"file_name": path.name, "media_type": media_type})
    prompt = build_guided_prompt(frame_index, width, height, box, positive_points, negative_points)
    try:
        async with _client() as client:
            uploaded = await client.files.create(file=(path.name, path.read_bytes(), media_type), purpose="user_data")
            return await _segment_file(client, uploaded.id, prompt, "guided_box_point", propagation_direction)
    except (OpenAIError, OSError, RuntimeError) as error:
        if isinstance(error, SamUnavailableError):
            raise
        raise SamUnavailableError(f"SAM 3.1 guided request failed: {_error_detail(error)}") from error
