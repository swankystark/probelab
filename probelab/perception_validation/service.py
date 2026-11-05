"""Quality-gated handoff from temporary validation into production science."""

from dataclasses import dataclass

import numpy as np

from ..meta_client import MetaMuseClient, NarrativeResult
from ..science import choose_experiment, fit_posterior
from .models import (
    DeterministicFitSummary,
    HandoffMeasurement,
    HandoffRequest,
    HandoffResponse,
    MuseSummary,
    QualityResult,
    SelectorSummary,
    SessionState,
)


@dataclass(frozen=True, slots=True)
class HandoffInputs:
    """Typed, comparison-independent inputs admitted to production science."""

    timestamps: tuple[float, ...]
    angles: tuple[float, ...]


def execute_handoff(
    state: SessionState,
    request: HandoffRequest,
    muse: MetaMuseClient,
) -> HandoffResponse:
    """Run production inference, EIG, and Muse only after the temporary quality gate."""
    quality = _quality_or_rejection(state.quality)
    unavailable = _unavailable_muse(
        "Muse Spark was not invoked because the validation handoff was not approved."
    )
    if quality.status != "approved":
        return HandoffResponse(
            session_id=state.session.session_id,
            status="blocked",
            reason="Quality approval is required before production handoff.",
            quality=quality,
            muse=unavailable,
        )
    if not _current_leakage_provenance_is_approved(state):
        return HandoffResponse(
            session_id=state.session.session_id,
            status="blocked",
            reason=(
                "Current production handoff requires a quality report from SAM tracks "
                "produced with the currently saved exclusion masks applied."
            ),
            quality=quality.model_copy(
                update={
                    "status": "rejected",
                    "production_handoff_approved": False,
                    "production_handoff_reasons":[
                        *quality.production_handoff_reasons,
                        "saved leakage masks do not match the sanitized tracking and quality report",
                    ],
                }
            ),
            muse=unavailable,
        )
    if state.geometry_quality is None or not state.geometry_quality.sendable:
        if state.geometry_quality is not None and not state.geometry_quality.calibration_available:
            return HandoffResponse(
                session_id=state.session.session_id,
                status="blocked",
                reason="Pixel-only tracking is allowed, but metric calibration is required for production handoff.",
                quality=quality,
                muse=unavailable,
            )
        return HandoffResponse(
            session_id=state.session.session_id,
            status="blocked",
            reason="The stored geometry quality result is not sendable.",
            quality=quality,
            muse=unavailable,
        )
    try:
        inputs = _approved_inputs(state, request)
    except ValueError as error:
        return HandoffResponse(
            session_id=state.session.session_id,
            status="blocked",
            reason=str(error),
            quality=quality,
            muse=unavailable,
        )
    try:
        fit = fit_posterior(np.asarray(inputs.timestamps), np.asarray(inputs.angles))
        angle, duration, eig, rationale = choose_experiment(
            fit.posterior, list(request.completed_angles)
        )
    except ValueError as error:
        return HandoffResponse(
            session_id=state.session.session_id,
            status="failed",
            reason=str(error),
            production_invoked=True,
            quality=quality,
            muse=_unavailable_muse("Muse Spark was not invoked because production fitting failed."),
        )

    fit_summary = DeterministicFitSummary(
        length_m=fit.length,
        damping_s_minus_1=fit.damping,
        length_interval_m=fit.length_interval,
        damping_interval_s_minus_1=fit.damping_interval,
        residual_rms_deg=fit.residual_rms,
        adequacy=fit.adequacy,
    )
    selector = SelectorSummary(
        angle_deg=angle,
        duration_s=duration,
        eig_nats=eig,
        rationale=rationale,
    )
    narrative = muse.reason(_muse_prompt(request.objective, fit_summary, selector))
    muse_summary = _muse_summary(narrative)
    return HandoffResponse(
        session_id=state.session.session_id,
        status="completed",
        reason="Quality-approved measurements were handed off to deterministic production science.",
        production_invoked=True,
        quality=quality,
        fit=fit_summary,
        selector=selector,
        muse=muse_summary,
    )


def _current_leakage_provenance_is_approved(state: SessionState) -> bool:
    """Require handoff quality to match the exact currently saved sanitized run."""
    quality = state.quality
    geometry = state.geometry_quality
    tracking = state.sanitized_sam_tracking
    return bool(
        quality is not None
        and quality.leakage_compliance_approved
        and geometry is not None
        and geometry.leakage_sanitization_applied
        and geometry.leakage_controls == state.leakage_controls
        and tracking is not None
        and tracking.input_sanitization == "exclusion_masks_applied"
        and tracking.applied_exclusion_regions == state.leakage_controls.regions
        and tracking.sampling == state.sampling
        and tracking.sanitized_frame_count > 0
        and state.sanitized_tracking_annotation == state.annotation
    )


def _quality_or_rejection(quality: QualityResult | None) -> QualityResult:
    if quality is not None:
        return quality
    return QualityResult(
        status="rejected",
        coverage_fraction=0.0,
        radius_stability="not_checked",
        discontinuity_count=0,
        reasons=["quality approval has not been recorded"],
    )


def _approved_inputs(state: SessionState, request: HandoffRequest) -> HandoffInputs:
    geometry = state.geometry_quality
    if geometry is None:
        raise ValueError("geometry quality must be calculated before handoff")
    if not geometry.calibration_available or not geometry.calibrated_geometry:
        raise ValueError("Pixel-only tracking is allowed, but metric calibration is required for production handoff.")
    expected = tuple(
        HandoffMeasurement(
            frame_index=item.frame_index,
            timestamp_s=item.timestamp_s,
            angle_deg=item.angle_deg,
        )
        for item in geometry.calibrated_geometry
    )
    if tuple(request.measurements) != expected:
        raise ValueError("handoff measurements must exactly match the approved calibrated geometry")
    return HandoffInputs(
        timestamps=tuple(item.timestamp_s for item in request.measurements),
        angles=tuple(item.angle_deg for item in request.measurements),
    )


def _muse_prompt(
    objective: str,
    fit: DeterministicFitSummary,
    selector: SelectorSummary,
) -> str:
    return (
        "You are Muse Spark interpreting a quality-approved scientific handoff. "
        f"Objective: {objective}. Deterministic production fit: length {fit.length_m:.6f} m; "
        f"interval {fit.length_interval_m}; damping {fit.damping_s_minus_1:.6f} s^-1; "
        f"interval {fit.damping_interval_s_minus_1}; residual RMS {fit.residual_rms_deg:.6f} degrees; "
        f"adequacy {fit.adequacy}. The deterministic EIG selector selected {selector.angle_deg:.6f} "
        f"degrees for {selector.duration_s:.6f} seconds with EIG {selector.eig_nats:.6f} nats. "
        f"Selector rationale: {selector.rationale} Explain these numerical results and uncertainty. "
        "Muse is interpretation-only: do not change parameters, intervals, adequacy, or action selection; "
        "do not invent measurements."
    )


def _muse_summary(narrative: NarrativeResult) -> MuseSummary:
    return MuseSummary(
        text=narrative.text,
        status=narrative.status,
        model=narrative.model,
        finish_reason=narrative.finish_reason,
        request_status=narrative.request_status,
        completion_status=narrative.completion_status,
        response_text_length=narrative.response_text_length,
        error_detail=narrative.error_detail,
    )


def _unavailable_muse(reason: str) -> MuseSummary:
    return MuseSummary(
        text=reason,
        status="Unavailable",
        model="muse-spark-1.3",
        finish_reason="not_called",
        request_status="not_invoked",
        completion_status="not_started",
        response_text_length=0,
        error_detail=reason,
    )
