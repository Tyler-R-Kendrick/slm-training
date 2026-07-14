"""External-page projection evidence, tiering, and safety tests."""

from __future__ import annotations

import json

import pytest

from slm_training.data.deconstruct import (
    BrowserCapture,
    assess_projection,
    build_web_projection,
    fetch_approved_source,
    normalize_capture,
)
from slm_training.data.governance import SourceProvenance
from slm_training.data.leakage import (
    fingerprint_openui,
    fingerprint_openui_structure,
    fingerprint_pair,
    fingerprint_prompt,
)
from slm_training.data.verify import RuntimeEvidence

HTML = "<main><h1>Private launch</h1><button>Join now</button></main>"
OPENUI = """root = Stack([title, cta], "column")
title = TextContent(":page.title.text")
cta = Button(":page.cta.name")"""


def _capture() -> BrowserCapture:
    return BrowserCapture(
        source_url="https://example.com/launch",
        dom_snapshot=HTML,
        accessibility_tree=(
            {
                "node_id": "title",
                "role": "heading",
                "name": "Private launch",
                "text": "Private launch",
                "dom_path": "main > h1",
            },
            {
                "node_id": "cta",
                "role": "button",
                "name": "Join now",
                "parent_id": "root",
                "affordances": ["click"],
            },
        ),
        computed_layout=(
            {"node_id": "title", "bbox": [0, 0, 640, 80], "visible": True},
            {
                "node_id": "cta",
                "bbox": [0, 100, 120, 40],
                "visible": True,
                "screenshot_ref": "capture.png#cta",
            },
        ),
        screenshot_refs=("capture.png",),
        interaction_trace=("click:cta",),
        responsive_state="desktop",
    )


def _provenance() -> SourceProvenance:
    return SourceProvenance.from_content(
        source_url="https://example.com/launch",
        acquisition_date="2026-07-14",
        terms_policy_id="example-owner-approval-1",
        legal_basis="licensed by the source owner",
        license="CC-BY-4.0",
        attribution="Example, Inc.",
        asset_rights={
            "images": "excluded",
            "fonts": "excluded",
            "icons": "excluded",
            "embedded": "excluded",
        },
        robots_policy="allowed at acquisition time",
        deletion_procedure="remove by content hash and rebuild",
        content=HTML,
        transformation_history=("browser-capture", "ui-graph", "placeholderize"),
    )


def _record(**overrides):
    kwargs = {
        "projection_id": "page",
        "capture": _capture(),
        "candidate_openui": OPENUI,
        "element_statuses": {"title": "exact", "cta": "approximated"},
        "runtime_evidence": RuntimeEvidence(
            rendered=True, interaction_trace=("click:button",)
        ),
        "provenance": _provenance(),
        "candidate_links": {"title": "title", "cta": "cta"},
        "matched_behaviors": 1,
        "expected_behaviors": 1,
    }
    kwargs.update(overrides)
    return build_web_projection(**kwargs)


def test_projection_is_placeholderized_candidate_with_cross_modal_evidence() -> None:
    record = _record()
    serialized = json.dumps(record.meta["normalized_ui_graph"], sort_keys=True)
    assert "Private launch" not in serialized
    assert "Join now" not in serialized
    assert ":page.title.name" in serialized
    assert record.meta["label_kind"] == "candidate"
    assert record.meta["raw_html_is_ground_truth"] is False
    assert record.meta["projection"] == {
        "projection_status": "approximated",
        "element_statuses": {"cta": "approximated", "title": "exact"},
        "supported_element_ratio": 1.0,
        "behavior_coverage": 1.0,
        "review_required": True,
    }
    assert record.meta["verification_tier"] == "Bronze"
    assert record.meta["governance"]["status"] == "Complete"
    assert record.meta["capture_evidence"]["dom_sha256"] == _provenance().content_hash
    assert record.meta["contract_id"]
    assert record.meta["split_group_id"].startswith("web_")


def test_tiering_is_deterministic_and_missing_nodes_are_omitted() -> None:
    graph = normalize_capture(_capture(), namespace="page")
    first = assess_projection(graph, {"cta": "unsupported"})
    second = assess_projection(graph, {"cta": "unsupported"})
    assert first == second
    assert first.projection_status == "unsupported"
    assert first.element_statuses == (("cta", "unsupported"), ("title", "omitted"))
    assert first.supported_element_ratio == 0.0
    with pytest.raises(ValueError, match="invalid projection status"):
        assess_projection(graph, {"cta": "gold"})
    with pytest.raises(ValueError, match="unknown node"):
        assess_projection(graph, {"missing": "exact"})


def test_candidate_correspondence_must_name_a_real_dsl_binder() -> None:
    with pytest.raises(ValueError, match="unknown DSL node"):
        _record(candidate_links={"cta": "missing"})


def test_raw_dom_alone_is_rejected_and_unapproved_source_cannot_fetch() -> None:
    with pytest.raises(ValueError, match="raw DOM alone"):
        BrowserCapture(source_url="https://example.com", dom_snapshot=HTML)
    with pytest.raises(ValueError, match="not approved"):
        fetch_approved_source({"source_url": "https://example.com", "approved": False})


def test_missing_rights_or_eval_overlap_quarantines_or_rejects() -> None:
    quarantined = _record(provenance=None)
    assert quarantined.meta["verification_tier"] == "Quarantine"
    assert quarantined.meta["governance"]["status"] == "Quarantined"

    record = _record()
    eval_fingerprints = {
        "ids": {record.id},
        "split_group_ids": {record.meta["split_group_id"]},
        "prompts": {fingerprint_prompt(record.prompt)},
        "openuis": {fingerprint_openui(record.openui)},
        "structures": {fingerprint_openui_structure(record.openui)},
        "pairs": {fingerprint_pair(record.prompt, record.openui)},
        "design_mds": set(),
    }
    with pytest.raises(ValueError, match="overlaps eval"):
        _record(eval_fingerprints=eval_fingerprints)
