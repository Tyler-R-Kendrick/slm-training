"""Tiny v2 plan build: unique roots ≠ rows, no OpenUI in simplified fallback."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.staged import Capability
from slm_training.harnesses.synthesis_plan import (
    tiny_corpus_generation_policy,
)
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data
from slm_training.harnesses.train_data.pipeline import (
    _ProgramspecPrompt,
    _load_progspecs,
    _prompt_for_programspec,
    _records_from_progspec,
)
from slm_training.harnesses.train_data.semantic_prompts import (
    PromptCandidate,
    PromptRenderResult,
)
from slm_training.data.progspec.generate import generate_program_pool
from dataclasses import replace

REPO_ROOT = Path(__file__).resolve().parents[3]
V2_PLAN_PATH = (
    REPO_ROOT / "src/slm_training/resources/synthesis_plans/corpus/cap0_tiny_v2.json"
)


def test_pool_roots_are_not_inflated_by_prompt_or_repair_counts() -> None:
    policy = tiny_corpus_generation_policy(
        capability=Capability.CAP0_GRAMMAR, unique_root_targets=(3,), seed=17
    )
    policy = replace(
        policy,
        generator=replace(
            policy.generator,
            components=("TextContent", "Button", "Separator"),
            max_depth=2,
            max_width=3,
        ),
        max_attempts=48,
    )
    pool = generate_program_pool(policy)
    assert len(pool.root_ids) == 3
    assert len(set(pool.root_ids)) == 3
    # Prompts-per-root and repairs are not roots.
    assert policy.prompts.prompts_per_root >= 2
    assert policy.derivatives.repairs_per_root >= 1


def test_v2_plan_loader_does_not_require_staged_graph(tmp_path: Path) -> None:
    plan = V2_PLAN_PATH
    result = build_train_data(
        TrainDataConfig(
            profile="permissive",
            source="programspec",
            synthesis_plan_path=plan,
            output_root=tmp_path / "out",
            version="tiny",
            include_language_contract=False,
            include_scope_corpus=False,
            include_frontier_artifacts=False,
            include_edit_derivatives=False,
            repairs_per_program=0,
            require_design_md=False,
            programspec_count=4,
            programspec_seed=17,
        )
    )
    manifest = result["manifest"]
    assert manifest["corpus_generation"]["unique_root_targets"] == [4]
    assert result["quality_report"]["claim_class"] == "fixture_wiring"
    assert result["quality_report"]["capability_certificate"] is False
    assert (
        result["quality_report"]["unique_roots"]["prompt_provider_authoritative"]
        is False
    )
    assert result["quality_report"]["unique_roots"]["admitted"] == 4
    assert "pair_quality" in result["quality_report"]
    assert "source_anchors" in result["quality_report"]
    assert result["synthesis_feedback"]["capability_certificate"] is False
    assert "findings" in result["synthesis_feedback"]
    records = result["stats"]["record_count"]
    assert records >= 1


def test_corpus_policy_ignores_committed_programspec_jsonl(
    tmp_path: Path, monkeypatch
) -> None:
    dummy = tmp_path / "programs.jsonl"
    dummy.write_text('{"id": "DUMMY-SHOULD-NOT-LOAD"}\n', encoding="utf-8")

    class Pool:
        programs = [SimpleNamespace(id="generated-root")]

    seen: dict[str, object] = {}

    def fake_pool(policy):
        seen["policy"] = policy
        return Pool()

    monkeypatch.setattr(
        "slm_training.data.progspec.generate.generate_program_pool", fake_pool
    )
    policy = tiny_corpus_generation_policy(capability=Capability.CAP0_GRAMMAR)
    specs, errors = _load_progspecs(
        TrainDataConfig(
            programspec_path=dummy,
            corpus_generation=policy,
            programspec_count=4,
        )
    )
    assert seen["policy"] is policy
    assert [item.id for item in specs] == ["generated-root"]
    assert not any("DUMMY" in str(item) for item in errors)
    assert not any("programspec:1" in str(item.get("id", "")) for item in errors)


def _spec() -> SimpleNamespace:
    return SimpleNamespace(
        id="root-1",
        program_family_id="hero_card",
        split_group_id="fam-1",
        split="train",
        canonical_openui="unused",
    )


def test_v2_prompt_render_failure_skips_generic(monkeypatch) -> None:
    monkeypatch.setattr("slm_training.dsl.parser.validate", lambda source: object())
    monkeypatch.setattr(
        "slm_training.dsl.semantic_frame.derive_semantic_frame",
        lambda program, schema=None: object(),
    )
    monkeypatch.setattr(
        "slm_training.harnesses.train_data.semantic_prompts.render_prompt_candidates",
        lambda *args, **kwargs: PromptRenderResult(
            (), ({"reason": "zero_candidates", "detail": "none"},)
        ),
    )
    rendered = _prompt_for_programspec(
        _spec(),
        TrainDataConfig(
            corpus_generation=tiny_corpus_generation_policy(
                capability=Capability.CAP0_GRAMMAR
            )
        ),
    )
    assert rendered.prompt is None
    assert rendered.error == "prompt renderer produced zero candidates"
    assert rendered.rejections[0]["reason"] == "zero_candidates"
    assert "Need a compact screen" not in (rendered.error or "")


def test_v2_prompt_stamps_pair_quality_provenance(monkeypatch) -> None:
    chosen = PromptCandidate(
        prompt_text="Need a title and a submit action.",
        renderer_family="concise_intent",
        renderer_version="v1",
        surface="simplified_nl",
        required_fact_ids=("title",),
        house_style_fact_ids=(),
        forbidden_fact_ids=("sidebar",),
        cue_intervention=None,
        invariance_group_id="inv-1",
        provider_id="offline",
        provenance={
            "required_fact_phrases": ["title"],
            "target_fact_phrases": ["title"],
            "forbidden_fact_phrases": ["sidebar"],
        },
    )
    monkeypatch.setattr("slm_training.dsl.parser.validate", lambda source: object())
    monkeypatch.setattr(
        "slm_training.dsl.semantic_frame.derive_semantic_frame",
        lambda program, schema=None: object(),
    )
    monkeypatch.setattr(
        "slm_training.harnesses.train_data.semantic_prompts.render_prompt_candidates",
        lambda *args, **kwargs: PromptRenderResult((chosen,), ()),
    )
    rendered = _prompt_for_programspec(
        _spec(),
        TrainDataConfig(
            corpus_generation=tiny_corpus_generation_policy(
                capability=Capability.CAP1_SEMANTICS
            )
        ),
    )
    assert rendered.prompt == chosen.prompt_text
    assert rendered.provenance["renderer_family"] == "concise_intent"
    assert rendered.provenance["template_id"] == "inv-1"
    assert rendered.provenance["required_fact_ids"] == ["title"]
    assert rendered.provenance["forbidden_fact_ids"] == ["sidebar"]
    assert rendered.provenance["pair_required_facts"] == ["title"]
    assert rendered.provenance["pair_target_facts"] == ["title"]
    assert rendered.provenance["pair_forbidden_facts"] == ["sidebar"]
    assert "required_facts" not in rendered.provenance
    assert "forbidden_facts" not in rendered.provenance
    assert len(rendered.items) == 1


def test_v2_emits_every_prompt_candidate_once(monkeypatch) -> None:
    first = PromptCandidate(
        prompt_text="Need a title and a submit action.",
        renderer_family="concise_intent",
        renderer_version="v1",
        surface="simplified_nl",
        required_fact_ids=("title",),
        house_style_fact_ids=(),
        forbidden_fact_ids=(),
        cue_intervention=None,
        invariance_group_id="inv-1",
        provider_id="offline",
        provenance={},
    )
    second = PromptCandidate(
        prompt_text="A compact screen needs a heading and a confirm control.",
        renderer_family="task_context",
        renderer_version="v1",
        surface="simplified_nl",
        required_fact_ids=("title",),
        house_style_fact_ids=(),
        forbidden_fact_ids=(),
        cue_intervention=None,
        invariance_group_id="inv-2",
        provider_id="offline",
        provenance={},
    )
    emitted: list[str] = []

    def fake_emit(spec, prompt, task, record_id, source, meta):
        emitted.append(record_id)
        return ExampleRecord(
            id=record_id,
            prompt=prompt,
            openui=spec.canonical_openui,
            placeholders=[],
            split="train",
            meta=meta,
        )

    monkeypatch.setattr("slm_training.dsl.parser.validate", lambda source: object())
    monkeypatch.setattr(
        "slm_training.dsl.semantic_frame.derive_semantic_frame",
        lambda program, schema=None: object(),
    )
    monkeypatch.setattr(
        "slm_training.harnesses.train_data.semantic_prompts.render_prompt_candidates",
        lambda *args, **kwargs: PromptRenderResult((first, second), ()),
    )
    monkeypatch.setattr("slm_training.data.progspec.emit_record", fake_emit)
    monkeypatch.setattr(
        "slm_training.data.verify.stamp_record", lambda record, ctx: record
    )
    monkeypatch.setattr(
        "slm_training.evals.learnability_diagnostics.make_interventions",
        lambda *a, **k: (),
    )
    monkeypatch.setattr(
        "slm_training.harnesses.train_data.pipeline._counterfactual_records",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "slm_training.harnesses.train_data.pipeline._program_repair_records",
        lambda spec, limit: [],
    )
    records, errors, _rejected = _records_from_progspec(
        TrainDataConfig(
            corpus_generation=tiny_corpus_generation_policy(
                capability=Capability.CAP1_SEMANTICS
            ),
            repairs_per_program=0,
            include_edit_derivatives=False,
            include_scope_derivatives=False,
        ),
        preloaded=([_spec()], []),
    )
    assert not errors
    assert emitted == ["root-1", "root-1-p1"]
    assert [item.prompt for item in records] == [
        first.prompt_text,
        second.prompt_text,
    ]
    assert records[0].meta["root_id"] == "root-1"
    assert records[1].meta["prompt_index"] == 1


def test_counterfactual_failure_keeps_repairs(monkeypatch) -> None:
    gold = ExampleRecord(
        id="root-1",
        prompt="Need a title.",
        openui="root = Stack([x])",
        placeholders=[],
        split="train",
        meta={},
    )
    repair = ExampleRecord(
        id="repair-1",
        prompt="fix",
        openui="root = Stack([x])",
        placeholders=[],
        split="train",
    )
    monkeypatch.setattr(
        "slm_training.harnesses.train_data.pipeline._prompt_for_programspec",
        lambda spec, config: _ProgramspecPrompt("Need a title.", {}),
    )
    monkeypatch.setattr("slm_training.data.progspec.emit_record", lambda *a, **k: gold)
    monkeypatch.setattr(
        "slm_training.data.verify.stamp_record", lambda record, ctx: record
    )
    monkeypatch.setattr(
        "slm_training.evals.learnability_diagnostics.make_interventions",
        lambda *a, **k: (),
    )
    monkeypatch.setattr(
        "slm_training.harnesses.train_data.pipeline._counterfactual_records",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("cf exploded")),
    )
    monkeypatch.setattr(
        "slm_training.harnesses.train_data.pipeline._program_repair_records",
        lambda spec, limit: [repair],
    )
    records, errors, _rejected = _records_from_progspec(
        TrainDataConfig(
            corpus_generation=tiny_corpus_generation_policy(
                capability=Capability.CAP0_GRAMMAR
            ),
            repairs_per_program=1,
            include_edit_derivatives=False,
            include_scope_derivatives=False,
        ),
        preloaded=([_spec()], []),
    )
    assert any(item.id == "repair-1" for item in records)
    assert any(
        item.get("family") == "semantic_counterfactual"
        and "cf exploded" in str(item.get("error"))
        for item in errors
    )
