"""Transparent OpenCV pendulum observation extraction."""

from pathlib import Path
import base64
import cv2
import numpy as np
from .models import ObservationResult, PreparedVideo, SamMaskOverlay


def prepare_video(path: Path, frame_index: int | None = None) -> PreparedVideo:
    """Extract the earliest or explicitly selected usable JPEG frame without calibration."""
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    selected_frame = 0 if frame_index is None else max(min(frame_index, max(frame_count - 1, 0)), 0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError("The uploaded video could not provide a representative frame.")
    height, width = frame.shape[:2]
    encoded_ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not encoded_ok:
        raise ValueError("The representative frame could not be encoded.")
    data_url = f"data:image/jpeg;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"
    return PreparedVideo(frame_data_url=data_url, width=width, height=height, fps=fps, frame_count=frame_count, frame_index=selected_frame)


def extract_from_sam_tracks(tracks: list[dict[str, float]], fps: float, calibration_px: float, calibration_m: float, sam_diagnostics: dict[str, float | int | str] | None = None, mask_overlay: dict[str, float | int | str] | None = None) -> ObservationResult:
    """Convert SAM bob centroids into an angle trajectory using a fitted pivot."""
    if len(tracks) < 12:
        return ObservationResult(source="video", method="SAM 3.1 tracked mask geometry", frames=len(tracks), duration_s=0.0, confidence=0.0, quality="insufficient", timestamps=[], angles_deg=[], message="SAM 3.1 returned too few tracked bob frames.", calibration_px=calibration_px, calibration_m=calibration_m, sam_diagnostics=sam_diagnostics or {})
    points = np.array([[track["x"], track["y"]] for track in tracks], dtype=float)
    pivot = _fit_pivot(points)
    gaps = np.diff([track["time"] for track in tracks])
    angles = np.degrees(np.unwrap(np.arctan2(points[:, 0] - pivot[0], points[:, 1] - pivot[1])))
    confidence = float(np.mean([track["confidence"] for track in tracks]))
    quality = "usable" if confidence >= 0.55 and float(np.max(gaps, initial=1.0)) <= 2.0 else "unreliable"
    radius_px = float(np.median(np.sqrt(np.sum((points - np.array(pivot)) ** 2, axis=1))))
    diagnostic_text = f" SAM diagnostics: masks={sam_diagnostics.get('masks_returned', 'unknown') if sam_diagnostics else 'unknown'}, frames_with_masks={sam_diagnostics.get('frames_with_masks', 'unknown') if sam_diagnostics else 'unknown'}, selected_frames={sam_diagnostics.get('frames_with_selected_mask', 'unknown') if sam_diagnostics else 'unknown'}."
    message = "SAM 3.1 supplied tracked mask geometry; numerical processing converted mask centroids into angular displacement." + diagnostic_text
    if quality == "unreliable":
        message += " Tracking gaps exceed two source frames; the trajectory was not treated as stationary during gaps."
    overlay = SamMaskOverlay.model_validate(mask_overlay) if mask_overlay is not None else None
    return ObservationResult(source="video", method=f"SAM 3.1 tracked mask geometry; calibration {calibration_m:.3f} m / {calibration_px:.1f} px", frames=len(tracks), duration_s=float((tracks[-1]["time"] - tracks[0]["time"]) / max(fps, 1.0)), confidence=confidence, quality=quality, timestamps=[float(track["time"] / max(fps, 1.0)) for track in tracks], angles_deg=[float(value) for value in angles], message=message, calibration_px=calibration_px, calibration_m=calibration_m, tracked_radius_m=radius_px * calibration_m / calibration_px, sam_diagnostics=sam_diagnostics or {}, tracked_points=[(track["time"], track["x"], track["y"], track.get("confidence", 0.0)) for track in tracks], sam_mask_overlay=overlay)


def _fit_pivot(points: np.ndarray) -> tuple[float, float]:
    """Estimate the fixed pivot as the center of the bob's observed arc."""
    x, y = points[:, 0], points[:, 1]
    design = np.column_stack((2 * x, 2 * y, np.ones(len(points))))
    target = x * x + y * y
    center_x, center_y, _ = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(center_x), float(center_y)


def extract_video(path: Path, calibration_px: float, calibration_m: float) -> ObservationResult:
    """Track the most plausible moving dark/bright object in an uploaded video."""
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    timestamps: list[float] = []
    centers: list[tuple[float, float]] = []
    frame_indices: list[int] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        _, mask = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 40:
                moments = cv2.moments(contour)
                if moments["m00"]:
                    centers.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))
                    timestamps.append(frame_index / fps)
                    frame_indices.append(frame_index)
        frame_index += 1
    capture.release()
    if len(centers) < 12:
        return ObservationResult(source="video", method="OpenCV threshold + contour centroid", frames=len(centers), duration_s=0.0, confidence=0.0, quality="insufficient", timestamps=timestamps, angles_deg=[], message="The video did not contain enough trackable frames. Improve lighting, contrast, or framing and try again.", calibration_px=calibration_px, calibration_m=calibration_m)
    points = np.array(centers, dtype=float)
    pivot = _fit_pivot(points)
    raw = np.degrees(np.arctan2(points[:, 0] - pivot[0], points[:, 1] - pivot[1]))
    angles = np.unwrap(np.radians(raw)) * 180 / np.pi
    confidence = min(0.99, len(centers) / max(len(centers) + 20, 1))
    gaps = np.diff(frame_indices)
    quality = "usable" if confidence > 0.6 and (len(gaps) == 0 or int(np.max(gaps)) <= 2) else "unreliable"
    radius_px = float(np.median(np.sqrt(np.sum((points - np.array(pivot)) ** 2, axis=1))))
    message = "Measurements are deterministic image-derived angles; no language model was used to invent coordinates."
    if quality == "unreliable":
        message += " Tracking gaps exceed two source frames; the trajectory was not treated as stationary during gaps."
    return ObservationResult(source="video", method=f"OpenCV contour centroid; calibration {calibration_m:.3f} m / {calibration_px:.1f} px", frames=len(centers), duration_s=float(timestamps[-1]), confidence=confidence, quality=quality, timestamps=timestamps, angles_deg=[float(value) for value in angles], message=message, calibration_px=calibration_px, calibration_m=calibration_m, tracked_radius_m=radius_px * calibration_m / calibration_px)
