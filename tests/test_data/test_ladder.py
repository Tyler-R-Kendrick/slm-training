from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.data.frontier import artifact_path, gold_content_hash
from slm_training.data.house_style import resolve_target
from slm_training.data.ladder import (
    AbstractionLevel,
    FactContract,
    NoveltyBudget,
    NoveltyCandidate,
    NoveltyDimension,
    TargetDeterminacy,
    build_rung,
    check_grounding,
    make_counterfactual_pair,
    resolve_level,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.train_data.catalog import classify_source_family
from slm_training.harnesses.train_data.synth import FrozenArtifactSynthesizer


BUTTON = 'root = Stack([item], "column")\nitem = Button(":item.label")'
TEXT = 'root = Stack([item], "column")\nitem = TextContent(":item.text")'
ROW = 'root = Stack([item], "row")\nitem = Button(":item.label")'


@pytest.mark.parametrize(
    ("alias", "level", "determinacy", "family"),
    [
        ("dsl", AbstractionLevel.L0, TargetDeterminacy.EXACT, "frontier_semantic"),
        ("semantic", AbstractionLevel.L1, TargetDeterminacy.STRUCTURAL, "frontier_semantic"),
        ("detailed", AbstractionLevel.L2, TargetDeterminacy.STRUCTURAL, "frontier_semantic"),
        ("product", AbstractionLevel.L3, TargetDeterminacy.HOUSE_STYLE, "frontier_product"),
        ("user", AbstractionLevel.L4, TargetDeterminacy.HOUSE_STYLE, "frontier_user"),
        ("simplified", AbstractionLevel.L5, TargetDeterminacy.HOUSE_STYLE, "frontier_simplified"),
    ],
)
def test_l0_l5_aliases_and_determinacy(
    alias: str,
    level: AbstractionLevel,
    determinacy: TargetDeterminacy,
    family: str,
) -> None:
    assert resolve_level(alias) is level
    description = BUTTON if level is AbstractionLevel.L0 else "A vertical button layout."
    rung = build_rung(alias, description, BUTTON)
    assert rung.level is level
    assert rung.target_determinacy is determinacy
    assert rung.family == family
    assert 0.0 <= rung.constraint_coverage <= 1.0
    assert set(rung.to_meta()) >= {
        "required_facts",
        "optional_facts",
        "forbidden_facts",
        "unspecified_dimensions",
        "constraint_coverage",
        "target_determinacy",
    }


def test_grounding_catches_omission_invention_and_dsl_leak() -> None:
    contract = FactContract(
        required_facts=("component:Button",),
        forbidden_facts=("component:LineChart",),
    )
    omitted = check_grounding("Build an action area.", BUTTON, contract)
    assert {issue.code for issue in omitted.issues} == {"required_fact_omitted"}

    invented = check_grounding("Use a button and a line chart.", BUTTON, contract)
    assert {issue.code for issue in invented.issues} == {"forbidden_fact_invented"}

    mismatched = check_grounding("Use a button.", TEXT, contract)
    assert "target_missing_required_fact" in {issue.code for issue in mismatched.issues}

    leaked = check_grounding('Use Button(":item.label") in root = Stack([]).', BUTTON, contract)
    assert "dsl_leak" in {issue.code for issue in leaked.issues}


def test_house_style_selects_one_stable_target_for_vague_prompt() -> None:
    first = resolve_target("Build an action area.", (ROW, BUTTON), "L5")
    second = resolve_target("Build an action area.", (BUTTON, ROW), "simplified")
    assert first.target == second.target
    assert '"column"' in first.target
    assert first.level is AbstractionLevel.L5

    with pytest.raises(ValueError, match="one exact/structural target"):
        resolve_target("Detailed action area", (BUTTON, ROW), "L2")


def test_counterfactual_pair_changes_exactly_one_fact() -> None:
    shared = ("layout:column",)
    left = build_rung(
        "L2",
        "A vertical button layout.",
        BUTTON,
        contract=FactContract(required_facts=shared + ("component:Button",)),
    )
    right = build_rung(
        "L2",
        "A vertical text content layout.",
        TEXT,
        contract=FactContract(required_facts=shared + ("component:TextContent",)),
    )
    pair = make_counterfactual_pair(left, right)
    assert pair.changed_from == "component:Button"
    assert pair.changed_to == "component:TextContent"


def test_novelty_budget_caps_and_reports_variants() -> None:
    candidates = (
        NoveltyCandidate("a", "Create a card for an analyst", (NoveltyDimension.PERSONA,)),
        NoveltyCandidate("b", "Create a card for an analyst", (NoveltyDimension.FORM,)),
        NoveltyCandidate("c", "Write this for an operator", (NoveltyDimension.PERSONA,)),
        NoveltyCandidate("d", "Image then text", (NoveltyDimension.MODALITY,)),
        NoveltyCandidate("e", "Put the summary before details", (NoveltyDimension.ORDERING,)),
    )
    report = NoveltyBudget(
        max_variants=2,
        max_per_signature=1,
        near_duplicate_threshold=0.8,
    ).select(candidates)
    metrics = report.to_metrics()
    assert metrics == {"candidates": 5, "accepted": 2, "dropped": 3, "near_duplicates": 1}
    reasons = {decision.candidate.id: decision.reason for decision in report.decisions}
    assert reasons["b"] == "near_duplicate"
    assert reasons["c"] == "signature_cap"
    assert reasons["d"] == "online_modality_only"


def test_frozen_ladder_rows_get_canonical_metadata_and_families(tmp_path: Path) -> None:
    gold = ExampleRecord(
        id="gold-1",
        prompt="Build a button.",
        openui=BUTTON,
        placeholders=[":item.label"],
        source="fixture",
    )
    digest = gold_content_hash(gold.openui, gold.prompt)
    path = artifact_path(gold.id, digest, root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "gold_id": gold.id,
                "gold_content_hash": digest,
                "skeleton_openui": gold.openui,
                "ladder": {
                    "semantic": "A vertical button layout.",
                    "product": "An action area for a product.",
                    "user": "I need an action area.",
                    "simplified": "Help me take action.",
                },
            }
        ),
        encoding="utf-8",
    )
    rows = FrozenArtifactSynthesizer(root=tmp_path).expand(gold)
    assert len(rows) == 4
    assert {classify_source_family(row) for row in rows} == {
        "frontier_semantic",
        "frontier_product",
        "frontier_user",
        "frontier_simplified",
    }
    assert all("constraint_coverage" in row.meta for row in rows)
    assert all("target_determinacy" in row.meta for row in rows)
    assert sum("house_style_resolution" in row.meta for row in rows) == 3


def test_frozen_ladder_rejects_dsl_leak(tmp_path: Path) -> None:
    gold = ExampleRecord(id="gold-2", prompt="p", openui=BUTTON, source="fixture")
    digest = gold_content_hash(gold.openui, gold.prompt)
    path = artifact_path(gold.id, digest, root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "gold_id": gold.id,
                "gold_content_hash": digest,
                "skeleton_openui": gold.openui,
                "ladder": {"product": 'Use root = Stack([item]) and Button(":x").'},
            }
        ),
        encoding="utf-8",
    )
    assert FrozenArtifactSynthesizer(root=tmp_path).expand(gold) == []
