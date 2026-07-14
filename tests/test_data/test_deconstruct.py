"""External-page capture, projection evidence, and governance tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.data.deconstruct import (
    BrowserCapture,
    DeconstructConfig,
    ManifestEntry,
    ProjectionEvidence,
    assert_decontaminated,
    capture_page,
    load_manifest,
    normalize_capture,
    project_capture,
)
from slm_training.dsl.schema import ExampleRecord

OPENUI = 'root = Stack([cta])\ncta = Button(":external.cta")'


def _entry(**overrides: object) -> ManifestEntry:
    data: dict[str, object] = {
        "id": "licensed_page",
        "url": "https://example.com/licensed",
        "approved": True,
        "acquisition_date": "2026-07-14",
        "terms_policy_id": "owner-license-2026-07",
        "legal_basis": "licensed by source owner",
        "license": "CC-BY-4.0",
        "attribution": "Example, Inc.",
        "asset_rights": {
            "images": "excluded",
            "fonts": "excluded",
            "icons": "CC-BY-4.0",
            "embedded": "excluded",
        },
        "robots_policy": "allowed at acquisition time",
        "deletion_procedure": "delete by content hash and rebuild",
    }
    data.update(overrides)
    return ManifestEntry.from_dict(data)


def _capture(raw_text: str = "Licensed hero") -> BrowserCapture:
    return BrowserCapture.from_dict(
        {
            "source_url": "https://example.com/licensed",
            "dom_snapshot": f"<main><button>{raw_text}</button></main>",
            "accessibility_tree": f"- button {raw_text}",
            "viewport": {"width": 1440, "height": 900},
            "responsive_state": "desktop",
            "interaction_trace": ["observed:node_0:click"],
            "elements": [
                {
                    "id": "node_0",
                    "role": "button",
                    "accessible_name": raw_text,
                    "text": raw_text,
                    "bbox": {"x": 10, "y": 20, "width": 120, "height": 40},
                    "visible": True,
                    "form": {"type": "button", "value": raw_text},
                    "repeated_region": False,
                    "affordances": ["click"],
                    "responsive_state": "desktop",
                    "dom_path": "html > body > main > button",
                    "screenshot_ref": "node_0",
                    "dsl_node": "cta",
                }
            ],
        }
    )


def _eval_record() -> ExampleRecord:
    return ExampleRecord(
        id="held_text",
        prompt="Render a held text block",
        openui='root = TextContent(":held.text")',
        split="held_out",
    )


def test_manifest_is_stable_and_capture_is_fail_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({**_entry(id=value).__dict__, "asset_rights": dict(_entry().asset_rights or {})})
            for value in ("z", "a")
        )
        + "\n",
        encoding="utf-8",
    )
    assert [entry.id for entry in load_manifest(manifest)] == ["a", "z"]
    with pytest.raises(RuntimeError, match="live capture disabled"):
        capture_page(
            _entry(),
            DeconstructConfig(cache_dir=tmp_path / "cache", live=False),
        )
    with pytest.raises(PermissionError, match="not_approved"):
        capture_page(_entry(approved=False), DeconstructConfig(live=True))
    with pytest.raises(PermissionError, match="legal_basis_is_not_robots_policy"):
        capture_page(_entry(legal_basis="robots.txt"), DeconstructConfig(live=True))


def test_normalized_graph_contains_no_raw_user_content() -> None:
    graph = normalize_capture(_capture("Secret product name"), namespace="Page One")
    serialized = json.dumps(graph.to_dict())
    assert "Secret product name" not in serialized
    node = graph.nodes[0]
    assert node.accessible_name == ":page_one.node_0.name"
    assert node.text == ":page_one.node_0.text"
    assert node.form["value"] == ":page_one.node_0.value"
    assert node.dsl_node == "cta"
    assert node.screenshot_ref == "node_0"


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"matched_elements": 4, "total_source_elements": 4}, "exact"),
        ({"matched_elements": 3, "total_source_elements": 4}, "approximated"),
        ({"matched_elements": 1, "total_source_elements": 4}, "omitted"),
        ({"matched_elements": 0, "total_source_elements": 4}, "unsupported"),
    ],
)
def test_projection_status_is_deterministic(
    kwargs: dict[str, int], status: str
) -> None:
    evidence = ProjectionEvidence.from_counts(**kwargs)
    assert evidence.projection_status == status
    assert evidence.review_required is True
    assert evidence.confidence_tier == "Bronze"


def test_graph_comparison_derives_structure_behavior_and_visual_evidence() -> None:
    source = normalize_capture(_capture(), namespace="source")
    candidate = normalize_capture(_capture(), namespace="candidate")
    evidence = ProjectionEvidence.compare(
        source, candidate, appearance_similarity=0.95
    )
    assert evidence.projection_status == "exact"
    assert evidence.supported_element_ratio == 1.0
    assert evidence.behavior_coverage == 1.0
    reviewed = ProjectionEvidence.compare(source, candidate, human_reviewed=True)
    assert reviewed.review_required is False
    assert reviewed.confidence_tier == "Bronze"


def test_projection_is_weak_placeholderized_lineage_with_governance() -> None:
    evidence = ProjectionEvidence.from_counts(
        matched_elements=1,
        total_source_elements=1,
        supported_behaviors=1,
        total_behaviors=1,
        appearance_similarity=0.95,
    )
    record = project_capture(
        entry=_entry(),
        capture=_capture(),
        candidate_openui=OPENUI,
        evidence=evidence,
        eval_records=[_eval_record()],
    )
    assert record.source == "web_projection"
    assert record.meta["candidate_label"] == "weak_evidence_backed"
    assert record.meta["projection"]["projection_status"] == "exact"
    assert record.meta["projection"]["review_required"] is True
    assert record.meta["verification_tier"] == "Bronze"
    assert record.meta["governance"]["status"] == "Complete"
    assert record.meta["governance"]["source"]["legal_basis"].startswith("licensed")
    assert record.meta["contract_id"]
    assert record.meta["split_group_id"].startswith("web_projection:")
    assert record.meta["decontamination"] == {
        "eval_records_checked": 1,
        "exact_structural_split_group_leakage": 0,
    }
    assert "Licensed hero" not in record.prompt
    assert "Licensed hero" not in json.dumps(record.meta["normalized_ui_graph"])


def test_incomplete_rights_or_pii_capture_is_quarantined() -> None:
    incomplete = project_capture(
        entry=_entry(asset_rights={"images": "excluded"}),
        capture=_capture(),
        candidate_openui=OPENUI,
        evidence=ProjectionEvidence.from_counts(
            matched_elements=1, total_source_elements=1
        ),
        eval_records=[_eval_record()],
    )
    assert incomplete.meta["verification_tier"] == "Quarantine"
    assert "asset_rights.fonts" in incomplete.meta["governance"]["reasons"]

    pii = project_capture(
        entry=_entry(),
        capture=_capture("Contact jane@example.com"),
        candidate_openui=OPENUI,
        evidence=ProjectionEvidence.from_counts(
            matched_elements=1, total_source_elements=1
        ),
        eval_records=[_eval_record()],
    )
    assert pii.meta["verification_tier"] == "Quarantine"
    assert "pii.email" in pii.meta["governance"]["reasons"]


def test_eval_overlap_is_rejected() -> None:
    record = project_capture(
        entry=_entry(),
        capture=_capture(),
        candidate_openui=OPENUI,
        evidence=ProjectionEvidence.from_counts(
            matched_elements=1, total_source_elements=1
        ),
        eval_records=[_eval_record()],
    )
    with pytest.raises(ValueError, match="openui_structure"):
        assert_decontaminated(
            [record],
            [
                ExampleRecord(
                    id="held",
                    prompt="Different prompt",
                    openui='root = Stack([x])\nx = Button(":held.cta")',
                    split="held_out",
                )
            ],
        )


def test_raw_dom_is_never_the_only_projection_evidence() -> None:
    capture = BrowserCapture.from_dict(
        {
            "source_url": _entry().url,
            "dom_snapshot": "<main>raw only</main>",
        }
    )
    with pytest.raises(ValueError, match="raw DOM cannot be the sole"):
        project_capture(
            entry=_entry(),
            capture=capture,
            candidate_openui=OPENUI,
            evidence=ProjectionEvidence.from_counts(
                matched_elements=0, total_source_elements=0
            ),
            eval_records=[_eval_record()],
        )
