"""Stable render metadata and geometry-based visual-edit grounding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isclose
from typing import Any, Literal

from slm_training.data.edits import EditKind, EditPatch, apply_patch
from slm_training.data.progspec import ProgramSpec, emit_record
from slm_training.data.verify import RuntimeEvidence, VerificationContext, stamp_record
from slm_training.dsl.schema import ExampleRecord


Theme = Literal["light", "dark"]
RenderState = Literal["empty", "loading", "populated", "error"]
MarkupKind = Literal["node", "point", "box", "mask", "arrow"]


def openui_node_id(program_id: str, statement_name: str) -> str:
    """Return the stable capture identity for one ProgramSpec statement."""
    if not program_id.strip() or not statement_name.strip():
        raise ValueError("program_id and statement_name must be non-empty")
    return f"{program_id}::{statement_name}"


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError("bounding-box dimensions must be non-negative")

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.right and self.y <= y <= self.bottom

    def intersection_area(self, other: BoundingBox) -> float:
        width = max(0.0, min(self.right, other.right) - max(self.x, other.x))
        height = max(0.0, min(self.bottom, other.bottom) - max(self.y, other.y))
        return width * height

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BoundingBox:
        return cls(*(float(data[key]) for key in ("x", "y", "width", "height")))


@dataclass(frozen=True)
class CaptureVariant:
    width: int
    height: int
    theme: Theme
    render_state: RenderState
    interaction_state: str = "idle"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("viewport dimensions must be positive")
        if self.theme not in {"light", "dark"}:
            raise ValueError(f"unsupported theme: {self.theme}")
        if self.render_state not in {"empty", "loading", "populated", "error"}:
            raise ValueError(f"unsupported render state: {self.render_state}")
        if not self.interaction_state.strip():
            raise ValueError("interaction_state must be non-empty")

    @property
    def key(self) -> str:
        return (
            f"{self.width}x{self.height}.{self.theme}."
            f"{self.render_state}.{self.interaction_state}"
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CaptureVariant:
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            theme=str(data["theme"]),  # type: ignore[arg-type]
            render_state=str(data["render_state"]),  # type: ignore[arg-type]
            interaction_state=str(data.get("interaction_state") or "idle"),
        )


@dataclass(frozen=True)
class ScrollTile:
    screenshot: str
    scroll_x: int
    scroll_y: int
    width: int
    height: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScrollTile:
        return cls(
            screenshot=str(data["screenshot"]),
            scroll_x=int(data["scroll_x"]),
            scroll_y=int(data["scroll_y"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )


@dataclass(frozen=True)
class RenderElement:
    openui_node_id: str
    statement_name: str
    parent_node_id: str | None
    bounding_box: BoundingBox
    visible_clip: BoundingBox
    z_order: int
    semantic_role: str
    accessible_name: str
    interaction_target: bool
    render_state: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RenderElement:
        return cls(
            openui_node_id=str(data["openui_node_id"]),
            statement_name=str(data["statement_name"]),
            parent_node_id=(
                None
                if data.get("parent_node_id") is None
                else str(data["parent_node_id"])
            ),
            bounding_box=BoundingBox.from_dict(data["bounding_box"]),
            visible_clip=BoundingBox.from_dict(data["visible_clip"]),
            z_order=int(data.get("z_order") or 0),
            semantic_role=str(data.get("semantic_role") or "generic"),
            accessible_name=str(data.get("accessible_name") or ""),
            interaction_target=bool(data.get("interaction_target")),
            render_state=str(data["render_state"]),
        )


@dataclass(frozen=True)
class RenderCapture:
    program_id: str
    variant: CaptureVariant
    fixed_screenshot: str
    full_page_screenshot: str
    scroll_tiles: tuple[ScrollTile, ...]
    elements: tuple[RenderElement, ...]
    interaction_trace: tuple[str, ...] = ()
    console_errors: tuple[str, ...] = ()
    behavior_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ids = [element.openui_node_id for element in self.elements]
        statements = [element.statement_name for element in self.elements]
        if len(ids) != len(set(ids)) or len(statements) != len(set(statements)):
            raise ValueError("capture elements must have unique node and statement IDs")

    def element_for_statement(self, statement_name: str) -> RenderElement:
        return next(
            element
            for element in self.elements
            if element.statement_name == statement_name
        )

    def element_for_node(self, node_id: str) -> RenderElement:
        return next(
            element for element in self.elements if element.openui_node_id == node_id
        )

    def statement_at(self, x: float, y: float) -> str:
        candidates = [
            element for element in self.elements if element.visible_clip.contains(x, y)
        ]
        if not candidates:
            raise LookupError(f"no OpenUI statement at ({x}, {y})")
        candidates.sort(
            key=lambda element: (
                -element.z_order,
                element.visible_clip.area,
                element.openui_node_id,
            )
        )
        return candidates[0].statement_name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RenderCapture:
        return cls(
            program_id=str(data["program_id"]),
            variant=CaptureVariant.from_dict(data["variant"]),
            fixed_screenshot=str(data["fixed_screenshot"]),
            full_page_screenshot=str(data["full_page_screenshot"]),
            scroll_tiles=tuple(
                ScrollTile.from_dict(item) for item in data.get("scroll_tiles") or ()
            ),
            elements=tuple(
                RenderElement.from_dict(item) for item in data.get("elements") or ()
            ),
            interaction_trace=tuple(
                str(item) for item in data.get("interaction_trace") or ()
            ),
            console_errors=tuple(
                str(item) for item in data.get("console_errors") or ()
            ),
            behavior_errors=tuple(
                str(item) for item in data.get("behavior_errors") or ()
            ),
        )


@dataclass(frozen=True)
class VisualMarkup:
    kind: MarkupKind
    points: tuple[tuple[float, float], ...] = ()
    node_id: str | None = None

    def __post_init__(self) -> None:
        expected = {"node": 0, "point": 1, "box": 2, "arrow": 2}
        if self.kind == "node" and not self.node_id:
            raise ValueError("node markup requires node_id")
        if self.kind in expected and len(self.points) != expected[self.kind]:
            raise ValueError(
                f"{self.kind} markup requires {expected[self.kind]} points"
            )
        if self.kind == "mask" and len(self.points) < 3:
            raise ValueError("mask markup requires at least three points")

    @property
    def bounds(self) -> BoundingBox:
        if not self.points:
            raise ValueError("node markup has no geometric bounds")
        xs, ys = zip(*self.points)
        return BoundingBox(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "points": [list(point) for point in self.points],
            "node_id": self.node_id,
        }


@dataclass(frozen=True)
class GroundingResult:
    primary_node_id: str
    statement_name: str
    candidate_node_ids: tuple[str, ...]
    ambiguous: bool


def resolve_markup(capture: RenderCapture, markup: VisualMarkup) -> GroundingResult:
    """Resolve node, point, box, mask, or arrow markup to stable OpenUI IDs."""
    if markup.kind == "node":
        element = capture.element_for_node(str(markup.node_id))
        return GroundingResult(
            element.openui_node_id,
            element.statement_name,
            (element.openui_node_id,),
            False,
        )

    if markup.kind in {"point", "arrow"}:
        x, y = markup.points[-1]
        ranked = [
            (float(element.z_order), -element.visible_clip.area, element)
            for element in capture.elements
            if element.visible_clip.contains(x, y)
        ]
    else:
        bounds = markup.bounds
        ranked = [
            (
                element.visible_clip.intersection_area(bounds) / max(bounds.area, 1.0),
                float(element.z_order),
                element,
            )
            for element in capture.elements
            if element.visible_clip.intersection_area(bounds) > 0
        ]
    if not ranked:
        raise LookupError(
            f"{markup.kind} markup does not intersect a visible OpenUI node"
        )
    ranked.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            item[2].visible_clip.area,
            item[2].openui_node_id,
        )
    )
    primary = ranked[0][2]
    candidates = tuple(item[2].openui_node_id for item in ranked)
    ambiguous = (
        len(ranked) > 1
        and isclose(ranked[0][0], ranked[1][0])
        and isclose(ranked[0][1], ranked[1][1])
    )
    return GroundingResult(
        primary.openui_node_id, primary.statement_name, candidates, ambiguous
    )


def build_visual_edit_record(
    spec: ProgramSpec,
    *,
    capture: RenderCapture,
    markup: VisualMarkup,
    instruction: str,
    patch: EditPatch,
) -> ExampleRecord:
    """Create one grounded edit row after runtime, targeting, and G9 verification."""
    grounding = resolve_markup(capture, markup)
    if grounding.ambiguous:
        raise ValueError(f"ambiguous markup targets: {grounding.candidate_node_ids}")
    changed = tuple(
        dict.fromkeys(
            operation.name
            for operation in patch.operations
            if operation.kind is not EditKind.NOOP
        )
    )
    if changed != (grounding.statement_name,):
        raise ValueError(
            "visual edit must minimally change only the grounded statement "
            f"{grounding.statement_name!r}"
        )
    if patch.ast_operation_count != 1:
        raise ValueError("visual edit patch must contain exactly one AST operation")
    after = apply_patch(spec.canonical_openui, patch)
    record = emit_record(
        spec,
        prompt=instruction,
        task="edit",
        openui=after,
        source="visual_edit",
        parent_id=spec.id,
        meta={
            "edit": {
                "markup": markup.to_dict(),
                "target_node_id": grounding.primary_node_id,
                "target_statement_name": grounding.statement_name,
                "changed_statement_names": list(changed),
                "patch": patch.to_dict(),
                "screenshot": capture.fixed_screenshot,
            },
            "render": {
                "variant": asdict(capture.variant),
                "full_page_screenshot": capture.full_page_screenshot,
                "scroll_tiles": [asdict(tile) for tile in capture.scroll_tiles],
            },
        },
    )
    context = VerificationContext(
        source_kind="program-first",
        runtime=RuntimeEvidence(
            rendered=True,
            console_errors=capture.console_errors,
            behavior_errors=capture.behavior_errors,
            interaction_trace=capture.interaction_trace,
        ),
        require_runtime=True,
        patch_before=spec.canonical_openui,
        patch=patch,
        patch_after=after,
        patch_applier=apply_patch,
        provenance_complete=bool(
            capture.fixed_screenshot and capture.full_page_screenshot
        ),
    )
    stamped = stamp_record(record, context)
    if stamped.meta["failing_gate"] is not None:
        raise ValueError(
            f"visual edit failed verifier gate {stamped.meta['failing_gate']}"
        )
    return stamped
