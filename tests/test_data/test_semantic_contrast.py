"""Regression tests for the semantic-contrast corpus builder (SPV2-01)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.data.progspec.generate import GeneratorConfig, ProgramGenerator
from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.contract import assert_canonical_template_markers
from slm_training.data.semantic_contrast import (
    ContrastFamily,
    SemanticContrastBuilder,
    generate_transforms,
)
from slm_training.data.semantic_plan.extract import OpenUISemanticPlanExtractor
from slm_training.data.semantic_plan.seed import PlanSeedBuilder
from slm_training.data.verify import VerificationContext, verify_record
from slm_training.dsl.pack import get_pack
from slm_training.dsl.schema import ExampleRecord


@pytest.fixture
def pack():
    return get_pack("openui")


@pytest.fixture
def sample_program():
    generator = ProgramGenerator(
        GeneratorConfig(
            max_depth=2,
            max_width=3,
            components=("Stack", "TextContent", "Button"),
            split="train",
        ),
        seed=7,
    )
    result = generator.generate(1)
    assert result.programs
    return result.programs[0]


@pytest.fixture
def sample_plan(sample_program, pack):
    return OpenUISemanticPlanExtractor().extract(sample_program, pack)


def _out(tmp_path: Path, dataset_id: str) -> Path:
    return tmp_path / "eval" / dataset_id


def test_generate_transforms_includes_positive_and_families(sample_plan):
    candidates = generate_transforms(sample_plan)
    assert any(c.family is ContrastFamily.POSITIVE for c in candidates)
    families = {c.family for c in candidates if c.family is not ContrastFamily.POSITIVE}
    assert ContrastFamily.CONTENT in families
    assert ContrastFamily.BINDING in families
    assert ContrastFamily.CONTRACT in families


def test_transform_candidates_are_rebuildable_plans(sample_plan):
    for candidate in generate_transforms(sample_plan):
        assert isinstance(candidate.plan, type(sample_plan))
        assert candidate.transform_id
        assert candidate.family.value
        assert candidate.severity.value


def test_builder_produces_required_artifacts(tmp_path):
    dataset_id = "semantic_contrast_test_v1"
    builder = SemanticContrastBuilder(
        output_root=tmp_path,
        dataset_id=dataset_id,
        seed=3,
        source_count=4,
        splits=("train", "held_out"),
        split_weights=(0.75, 0.25),
    )
    summary = builder.build()
    out = _out(tmp_path, dataset_id)
    assert (out / "pairs.jsonl").is_file()
    assert (out / "records.jsonl").is_file()
    assert (out / "rejected.jsonl").is_file()
    assert (out / "summary.json").is_file()
    assert (out / "manifest.json").is_file()
    assert summary["dataset_id"] == dataset_id
    assert summary["seed"] == 3
    assert summary["source_count"] == 4
    # Version stamp present and includes our component.
    stamp = summary["version_stamp"]
    assert stamp["stamp_schema"] == "version_stamp/v1"
    assert "data.semantic_contrast" in stamp["components"]


def test_positive_passes_negative_fails_and_surface_stays_valid(tmp_path):
    dataset_id = "semantic_contrast_test_v2"
    builder = SemanticContrastBuilder(
        output_root=tmp_path,
        dataset_id=dataset_id,
        seed=5,
        source_count=4,
        splits=("train",),
        split_weights=(1.0,),
    )
    builder.build()
    out = _out(tmp_path, dataset_id)
    pairs = [json.loads(line) for line in (out / "pairs.jsonl").read_text().splitlines()]
    assert pairs
    positives = [p["positive"] for p in pairs]
    negatives = [p["negative"] for p in pairs if p["family"] != "positive"]
    assert all(p["meaningful_report"]["verdict"] for p in positives)
    assert all(not n["meaningful_report"]["verdict"] for n in negatives)
    # Every record surface must pass the verifier.
    for rec in positives + negatives:
        record = ExampleRecord.from_dict(rec["record"])
        assert_canonical_template_markers(record)
        report = verify_record(record, VerificationContext(source_kind="program"))
        assert report.ok, f"{rec['record']['id']} failed verifier"


def test_split_isolation(tmp_path):
    dataset_id = "semantic_contrast_test_v3"
    builder = SemanticContrastBuilder(
        output_root=tmp_path,
        dataset_id=dataset_id,
        seed=11,
        source_count=8,
        splits=("train", "held_out"),
        split_weights=(0.5, 0.5),
    )
    builder.build()
    out = _out(tmp_path, dataset_id)
    records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    train_ids = {
        r["source_program_id"]
        for r in records
        if r["meta"]["split"] == "train"
    }
    test_ids = {
        r["source_program_id"]
        for r in records
        if r["meta"]["split"] == "held_out"
    }
    assert train_ids
    assert test_ids
    assert not train_ids.intersection(test_ids)


def test_scoreboard_reports_all_families(tmp_path):
    dataset_id = "semantic_contrast_test_v4"
    builder = SemanticContrastBuilder(
        output_root=tmp_path,
        dataset_id=dataset_id,
        seed=13,
        source_count=4,
        splits=("train",),
        split_weights=(1.0,),
    )
    summary = builder.build()
    families = {m["family"] for m in summary["scoreboard"]}
    assert "positive" in families
    assert any(f in families for f in ("content", "binding", "contract"))
    for metric in summary["scoreboard"]:
        assert "n_total" in metric
        assert "verifier_pass_rate" in metric
        assert "false_negative_rate" in metric


def test_strict_delta_ignores_the_whole_plan_hash(tmp_path):
    builder = SemanticContrastBuilder(
        output_root=tmp_path,
        dataset_id="semantic_contrast_strict_delta",
        source_count=1,
        splits=("train",),
        split_weights=(1.0,),
        wide_sources=True,
        strict_delta=True,
    )
    builder.build()
    pairs = [
        json.loads(line)
        for line in (tmp_path / "eval" / "semantic_contrast_strict_delta" / "pairs.jsonl")
        .read_text()
        .splitlines()
    ]
    negatives = [pair["negative"] for pair in pairs if pair["family"] != "positive"]
    assert negatives
    assert all(
        len(set(row["meta"]["semantic_delta"]) - {"exact"}) == 1
        for row in negatives
    )


def _leaf_content_plan(role_ids: list[str], edges: list[dict], families: dict) -> SemanticPlanV1:
    """Build a plan giving every leaf role a bound placeholder symbol.

    ``TextContent``/``Button`` require ``text``/``label`` respectively, so any
    hand-built plan whose leaves lack a binding fails to compile -- this
    mirrors what ``OpenUISemanticPlanExtractor`` would have produced.
    """
    from slm_training.data.progspec.semantic_plan import (
        PlanBinding,
        PlanIdentity,
        PlanSymbol,
        PlanTopology,
        RoleSlot,
    )

    parent_ids = {str(e["parent_role_id"]) for e in edges}
    leaf_ids = [rid for rid in role_ids if rid not in parent_ids]
    symbols = tuple(
        PlanSymbol(symbol_id=f"sym_{i:04d}", semantic_role="text")
        for i in range(len(leaf_ids))
    )
    bindings = tuple(
        PlanBinding(role_slot_id=rid, candidate_symbols=(sym.symbol_id,))
        for rid, sym in zip(leaf_ids, symbols)
    )
    return SemanticPlanV1(
        identity=PlanIdentity(pack_id="openui", provenance="gold"),
        role_slots=tuple(
            RoleSlot(role_id=rid, component_family=families[rid]) for rid in role_ids
        ),
        topology=PlanTopology(parent_relation_candidates=tuple(edges)),
        symbols=symbols,
        bindings=bindings,
    )


def _two_container_plan() -> SemanticPlanV1:
    """root(Stack) -> [containerA(Stack), leafX]; containerA -> [leafY, leafZ].

    containerA keeps a child after leafX moves in, and root keeps containerA
    after leafX moves out -- a reparent should fire cleanly.
    """
    edges = [
        {"parent_role_id": "root", "child_role_id": "containerA", "relation": "contains"},
        {"parent_role_id": "root", "child_role_id": "leafX", "relation": "contains"},
        {"parent_role_id": "containerA", "child_role_id": "leafY", "relation": "contains"},
        {"parent_role_id": "containerA", "child_role_id": "leafZ", "relation": "contains"},
    ]
    families = {
        "root": "Stack",
        "containerA": "Stack",
        "leafX": "TextContent",
        "leafY": "TextContent",
        "leafZ": "TextContent",
    }
    return _leaf_content_plan(
        ["root", "containerA", "leafX", "leafY", "leafZ"], edges, families
    )


def _chain_plan() -> SemanticPlanV1:
    """root(Stack) -> containerA(Stack) -> leafY: every parent has one child."""
    edges = [
        {"parent_role_id": "root", "child_role_id": "containerA", "relation": "contains"},
        {"parent_role_id": "containerA", "child_role_id": "leafY", "relation": "contains"},
    ]
    families = {"root": "Stack", "containerA": "Stack", "leafY": "TextContent"}
    return _leaf_content_plan(["root", "containerA", "leafY"], edges, families)


def test_topology_reparent_moves_a_role_into_a_sibling_container(pack):
    plan = _two_container_plan()
    candidates = generate_transforms(plan)
    reparent = next(
        (c for c in candidates if c.transform_id == "topology_reparent"), None
    )
    assert reparent is not None, "expected a reparent candidate for a two-container plan"
    edges = reparent.plan.topology.parent_relation_candidates
    # Neither container was emptied by the move.
    parents = [str(e["parent_role_id"]) for e in edges]
    assert parents.count("root") >= 1
    assert parents.count("containerA") >= 1
    # The mutated plan still compiles to a valid, non-empty seed distinct
    # from the original.
    seed_builder = PlanSeedBuilder(pack)
    result = seed_builder.build(reparent.plan)
    assert result.ok, result.reason
    assert result.seed
    assert result.seed != seed_builder.build(plan).seed


def test_topology_reparent_refuses_to_empty_a_container(pack):
    # Every parent in this chain has exactly one child; reparenting either
    # edge would leave PlanSeedBuilder rendering a childless container
    # without its required "children" prop (schema-invalid), so no
    # candidate is produced.
    plan = _chain_plan()
    candidates = generate_transforms(plan)
    assert not any(c.transform_id == "topology_reparent" for c in candidates)


def test_sibling_reorder_admits_as_a_positive_equivalence_control(tmp_path):
    dataset_id = "semantic_contrast_sibling_reorder"
    builder = SemanticContrastBuilder(
        output_root=tmp_path,
        dataset_id=dataset_id,
        seed=3,
        source_count=10,
        splits=("train",),
        split_weights=(1.0,),
        wide_sources=True,
        strict_delta=True,
    )
    builder.build()
    pairs = [
        json.loads(line)
        for line in (tmp_path / "eval" / dataset_id / "pairs.jsonl").read_text().splitlines()
    ]
    reorder_pairs = [p for p in pairs if p["transform_id"] == "positive_control_sibling_reorder"]
    assert reorder_pairs, "expected at least one admitted sibling-reorder control"
    for pair in reorder_pairs:
        assert pair["family"] == "positive"
        assert pair["positive"]["meaningful_report"]["verdict"] is True
        assert pair["negative"]["meaningful_report"]["verdict"] is True
        # The compiled surface must actually differ -- otherwise this would
        # be indistinguishable from the identity control.
        assert pair["positive"]["record"]["openui"] != pair["negative"]["record"]["openui"]
        # The reorder is a single-factor delta on the declared topology
        # factor (sibling order is not alpha/order-normalized for
        # gold-provenance plans -- see the transform's docstring).
        delta = set(pair["negative"]["meta"]["semantic_delta"]) - {"exact"}
        assert delta == {"topology"}


def test_prompt_requirements_are_attached_to_every_record(tmp_path):
    dataset_id = "semantic_contrast_prompt_requirements"
    builder = SemanticContrastBuilder(
        output_root=tmp_path,
        dataset_id=dataset_id,
        seed=3,
        source_count=4,
        splits=("train",),
        split_weights=(1.0,),
    )
    builder.build()
    records = [
        json.loads(line)
        for line in (tmp_path / "eval" / dataset_id / "records.jsonl").read_text().splitlines()
    ]
    assert records
    for record in records:
        requirements = record["meta"]["prompt_requirements"]
        assert requirements["requirements_version"] == "1"
        fingerprints = record["meta"]["prompt_requirements_fingerprints"]
        assert {"exact", "fact_set", "authority", "ambiguity_groups"} <= set(fingerprints)
        # Never gold-provenance: this is a conservative, gold-blind read of
        # the shared prompt (SGS-004), not derived from the AST.
        assert all(
            fact["provenance"] != "gold" for fact in requirements["facts"]
        )
