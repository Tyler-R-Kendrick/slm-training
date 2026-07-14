"""Renderer-first ProgramSpec captures and visual-edit grounding."""

from slm_training.data.render.capture import CaptureConfig, capture_program
from slm_training.data.render.schema import (
    BoundingBox,
    CaptureVariant,
    GroundingResult,
    RenderCapture,
    RenderElement,
    ScrollTile,
    VisualMarkup,
    build_visual_edit_record,
    openui_node_id,
    resolve_markup,
)

__all__ = [
    "BoundingBox",
    "CaptureConfig",
    "CaptureVariant",
    "GroundingResult",
    "RenderCapture",
    "RenderElement",
    "ScrollTile",
    "VisualMarkup",
    "build_visual_edit_record",
    "capture_program",
    "openui_node_id",
    "resolve_markup",
]
