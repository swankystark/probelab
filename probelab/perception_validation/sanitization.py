"""Server-side simulator-UI masking for temporary validation inputs."""

from __future__ import annotations

import base64
import math
from pathlib import Path

import cv2
import numpy as np

from .models import ExclusionRegion


def apply_exclusion_masks(
    frame: np.ndarray,
    regions: list[ExclusionRegion],
) -> np.ndarray:
    """Return a copy with every enabled mask region filled before model use."""
    height, width = frame.shape[:2]
    sanitized = frame.copy()
    for region in regions:
        if not region.enabled:
            continue
        if region.mode != "mask":
            raise ValueError("Only mask exclusions are supported for sanitized SAM input.")
        left = max(0, math.floor(region.left))
        top = max(0, math.floor(region.top))
        right = min(width, math.ceil(region.right))
        bottom = min(height, math.ceil(region.bottom))
        if left >= right or top >= bottom:
            raise ValueError("An exclusion region does not intersect the source frame.")
        sanitized[top:bottom, left:right] = 0
    return sanitized


def sanitized_frame_data_url(
    video_path: Path,
    frame_index: int,
    expected_width: int,
    expected_height: int,
    regions: list[ExclusionRegion],
) -> str:
    """Decode and sanitize one source frame for the browser preview."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("The validation video could not be opened for preview.")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError("The representative source frame could not be decoded.")
    height, width = frame.shape[:2]
    if (width, height) != (expected_width, expected_height):
        raise ValueError("Decoded frame dimensions do not match the prepared source frame.")
    sanitized = apply_exclusion_masks(frame, regions)
    ok, encoded = cv2.imencode(".jpg", sanitized)
    if not ok:
        raise ValueError("The sanitized preview frame could not be encoded.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
