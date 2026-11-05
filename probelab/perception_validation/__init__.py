"""Temporary, isolated perception-validation API boundary."""

from .router import router
from .geometry import attach_independent_accuracy, evaluate_geometry
from .sam_tracking import track_sampled_guided_video

__all__ = ["attach_independent_accuracy", "evaluate_geometry", "router", "track_sampled_guided_video"]
