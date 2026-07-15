"""Renderer capture matrix, grounding, and verified visual-edit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.data.progspec import ProgramSpec
from slm_training.data.edits import EditKind, EditOperation, EditPatch
from slm_training.data.render import (
    BoundingBox,
    CaptureConfig,
    CaptureVariant,
    RenderCapture,
    RenderElement,
    ScrollTile,
    VisualMarkup,
    build_visual_edit_record,
    openui_node_id,
    resolve_markup,
)
from slm_training.dsl import bridge_available
from slm_training.dsl.language_contract import contract_id


ROOT = Path(__file__).resolve().parents[2]


def _spec() -> ProgramSpec:
    data = json.loads(
        (ROOT / "src/slm_training/resources/render/sample_program.json").read_text(encoding="utf-8")
    )
    data["contract_id"] = contract_id()
    return ProgramSpec.from_dict(data)


def _element(
    spec: ProgramSpec,
    statement: str,
    box: BoundingBox,
    *,
    parent: str | None = "root",
    z: int = 0,
    role: str = "generic",
) -> RenderElement:
    return RenderElement(
        openui_node_id=openui_node_id(spec.id, statement),
        statement_name=statement,
        parent_node_id=None if parent is None else openui_node_id(spec.id, parent),
        bounding_box=box,
        visible_clip=box,
        z_order=z,
        semantic_role=role,
        accessible_name=statement,
        interaction_target=role == "button",
        render_state="populated",
    )


def _capture(spec: ProgramSpec, *, overlapping: bool = False) -> RenderCapture:
    cta_box = BoundingBox(20, 60, 100, 40)
    title_box = cta_box if overlapping else BoundingBox(20, 20, 200, 30)
    return RenderCapture(
        program_id=spec.id,
        variant=CaptureVariant(390, 844, "light", "populated"),
        fixed_screenshot="sample.fixed.png",
        full_page_screenshot="sample.full.png",
        scroll_tiles=(ScrollTile("sample.tile-0-0.png", 0, 0, 390, 844),),
        elements=(
            _element(spec, "root", BoundingBox(0, 0, 390, 200), parent=None),
            _element(spec, "title", title_box, z=1, role="text"),
            _element(
                spec, "cta", cta_box, z=2 if not overlapping else 1, role="button"
            ),
        ),
        interaction_trace=("click:button",),
    )


def test_capture_matrix_and_node_round_trip() -> None:
    spec = _spec()
    config = CaptureConfig(
        viewports=((390, 844),), interaction_states=("idle", "first-action")
    )
    variants = config.variants()
    assert len(variants) == 16
    assert len({variant.key for variant in variants}) == len(variants)

    capture = _capture(spec)
    node = capture.element_for_statement("cta")
    assert capture.element_for_node(node.openui_node_id) == node
    assert capture.statement_at(30, 70) == "cta"


def test_markup_resolves_node_point_box_mask_and_arrow() -> None:
    capture = _capture(_spec())
    cta_id = capture.element_for_statement("cta").openui_node_id
    markups = (
        VisualMarkup("node", node_id=cta_id),
        VisualMarkup("point", ((30, 70),)),
        VisualMarkup("box", ((20, 60), (120, 100))),
        VisualMarkup("mask", ((20, 60), (120, 60), (120, 100), (20, 100))),
        VisualMarkup("arrow", ((0, 0), (30, 70))),
    )
    assert {resolve_markup(capture, markup).statement_name for markup in markups} == {
        "cta"
    }

    ambiguous = resolve_markup(
        _capture(_spec(), overlapping=True), VisualMarkup("point", ((30, 70),))
    )
    assert ambiguous.ambiguous
    assert len(ambiguous.candidate_node_ids) >= 2


@pytest.mark.skipif(
    not bridge_available(), reason="OpenUI bridge dependencies unavailable"
)
def test_visual_edit_is_minimal_and_passes_runtime_patch_gates() -> None:
    spec = _spec()
    capture = _capture(spec)
    patch = EditPatch(
        (
            EditOperation(
                EditKind.REPLACE,
                "cta",
                before='Button(":hero.cta")',
                after='Button(":hero.secondary_cta")',
            ),
        )
    )

    record = build_visual_edit_record(
        spec,
        capture=capture,
        markup=VisualMarkup("point", ((30, 70),)),
        instruction="Change the call to action copy",
        patch=patch,
    )
    assert record.meta["task"] == "edit"
    assert record.meta["edit"]["target_statement_name"] == "cta"
    assert record.meta["verification_tier"] == "Silver"
    assert record.meta["failing_gate"] is None

    with pytest.raises(ValueError, match="minimally change"):
        build_visual_edit_record(
            spec,
            capture=capture,
            markup=VisualMarkup("point", ((30, 70),)),
            instruction="Change two nodes",
            patch=EditPatch(
                (
                    EditOperation(
                        EditKind.REPLACE,
                        "title",
                        before='TextContent(":hero.title")',
                        after='TextContent(":hero.heading")',
                    ),
                    *patch.operations,
                )
            ),
        )
