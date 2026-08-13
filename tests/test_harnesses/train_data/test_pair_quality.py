"""Prompt/target integrity and nuisance-cue corpus diagnostics."""

from __future__ import annotations

from typing import Any

from slm_training.harnesses.staged import Capability
from slm_training.harnesses.synthesis_plan import tiny_corpus_generation_policy
from slm_training.harnesses.train_data.pair_quality import (
    SCHEMA_VERSION,
    CorpusAudit,
    PairAudit,
    audit_corpus,
    audit_pair,
)

_RENDERERS = (
    "concise_intent",
    "task_context",
    "compositional_requirement",
    "paraphrase",
)
_PROVIDERS = ("prov_a", "prov_b")
_TOPOS = ("stack_column", "stack_row")
_SEM_A = "login credentials email password"
_SEM_B = "photo gallery image tiles"


def _policy():
    return tiny_corpus_generation_policy(capability=Capability.CAP0_GRAMMAR).quality


def _record(
    *,
    prompt: str,
    root_id: str,
    split_family_id: str,
    topology_id: str = _TOPOS[0],
    template_id: str | None = None,
    source: str = "fixture",
    source_family: str = "human_curated",
    provider_id: str = _PROVIDERS[0],
    renderer_family: str = _RENDERERS[0],
    target: str = "Card with title",
    required_facts: list[str] | None = None,
    target_facts: list[str] | None = None,
    forbidden_facts: list[str] | None = None,
    component_inventory: list[str] | None = None,
    semantic_frame_id: str = "frame.login",
    invariance_group_id: str | None = None,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "target": target,
        "source": source,
        "meta": {
            "renderer_family": renderer_family,
            "source_family": source_family,
            "provider_id": provider_id,
            "root_id": root_id,
            "topology_id": topology_id,
            "split_family_id": split_family_id,
            "template_id": template_id or f"tpl-{root_id}",
            "semantic_frame_id": semantic_frame_id,
            "required_facts": list(required_facts or ()),
            "target_facts": list(target_facts or ()),
            "forbidden_facts": list(forbidden_facts or ()),
            "component_inventory": list(component_inventory or ()),
            "invariance_group_id": invariance_group_id,
        },
    }


def _family_prompt(index: int, family: str, *, openui: bool = False) -> str:
    stem = _SEM_A if family == "family_a" else _SEM_B
    marker = "openui " if openui else ""
    return f"tok{index:02d} {marker}{stem}"


def _assign(index: int) -> tuple[str, str, str, str]:
    topology = _TOPOS[index % 2]
    provider = _PROVIDERS[(index // 2) % 2]
    renderer = _RENDERERS[(index // 2) % 4]
    source_family = "human_curated" if (index // 2) % 2 == 0 else "rico_real"
    return renderer, provider, topology, source_family


def _family_records(
    n_per_family: int,
    *,
    variants: int = 1,
    contaminate_openui: bool = False,
    source_predicts_topology: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family_index, family in enumerate(("family_a", "family_b")):
        for index in range(n_per_family):
            renderer, provider, topology, source_family = _assign(index)
            if source_predicts_topology:
                provider = f"prov_{topology}"
            root_id = f"{family}-root-{index:02d}"
            for variant in range(variants):
                prompt_index = index + variant * 50 + family_index * 100
                records.append(
                    _record(
                        prompt=_family_prompt(
                            prompt_index,
                            family,
                            openui=contaminate_openui and family == "family_a",
                        ),
                        root_id=root_id,
                        split_family_id=family,
                        topology_id=topology,
                        template_id=f"tpl-{root_id}-v{variant}",
                        provider_id=provider,
                        renderer_family=renderer,
                        source_family=source_family,
                    )
                )
    return records


def test_contaminated_openui_cue_only_lift_fails() -> None:
    policy = _policy()
    records = _family_records(12, contaminate_openui=True)
    audit = audit_corpus(records, policy)
    assert audit.cue_only["support"] >= 20
    assert audit.cue_only["disposition"] == "fail"
    assert audit.cue_only["lift"] > policy.max_cue_only_accuracy_lift
    assert audit.passed is False
    assert "cue_only_lift" in audit.blocking_reasons


def test_source_provider_predicts_topology() -> None:
    policy = _policy()
    records = _family_records(12, source_predicts_topology=True)
    audit = audit_corpus(records, policy)
    assert audit.source_only["support"] >= 20
    assert audit.source_only["disposition"] == "fail"
    assert audit.source_only["lift"] > policy.max_source_only_accuracy_lift
    assert audit.passed is False
    assert "source_only_lift" in audit.blocking_reasons


def test_balanced_semantic_content_does_not_fail_statistical_gates() -> None:
    policy = _policy()
    records = _family_records(12)
    audit = audit_corpus(records, policy)
    assert audit.cue_only["support"] >= 20
    assert audit.cue_only["disposition"] == "pass"
    assert audit.source_only["disposition"] == "pass"
    assert audit.cue_only["lift"] <= policy.max_cue_only_accuracy_lift
    assert audit.source_only["lift"] <= policy.max_source_only_accuracy_lift
    assert "cue_only_lift" not in audit.blocking_reasons
    assert "source_only_lift" not in audit.blocking_reasons
    assert audit.passed is True


def test_tiny_sample_reports_insufficient_support() -> None:
    policy = _policy()
    records = _family_records(2)
    audit = audit_corpus(records, policy)
    assert audit.cue_only["support"] < 20
    assert audit.cue_only["disposition"] == "insufficient_support"
    assert audit.source_only["disposition"] == "insufficient_support"
    assert audit.cue_only["disposition"] != "pass"
    assert "cue_only_lift" not in audit.blocking_reasons
    assert "source_only_lift" not in audit.blocking_reasons


def test_prompt_target_contradiction_fails_pair_audit() -> None:
    record = _record(
        prompt="Need a sidebar next to the article.",
        root_id="root-contradiction",
        split_family_id="family_a",
        required_facts=["sidebar"],
        target_facts=["footer"],
        forbidden_facts=[],
    )
    audit = audit_pair(record, record["meta"])
    assert audit.accepted is False
    assert audit.contradiction_count >= 1
    assert "prompt_target_contradiction" in audit.reasons


def test_complete_inventory_prose_leak_fails_pair_audit() -> None:
    record = _record(
        prompt="Use Card, Button, and Avatar together.",
        root_id="root-inventory",
        split_family_id="family_a",
        required_facts=["Card"],
        target_facts=["Card", "Button", "Avatar"],
        component_inventory=["Card", "Button", "Avatar"],
    )
    audit = audit_pair(record, record["meta"])
    assert audit.accepted is False
    assert audit.complete_inventory_leak_count >= 2
    assert "complete_inventory_leak" in audit.reasons


def test_full_prompt_signal_is_reported_not_a_shortcut_gate() -> None:
    policy = _policy()
    records = _family_records(12)
    audit = audit_corpus(records, policy)
    assert audit.full_prompt["accuracy"] > audit.cue_only["accuracy"]
    assert audit.full_prompt["lift"] > audit.cue_only["lift"]
    assert audit.full_prompt["disposition"] == "report_only"
    assert "full_prompt" not in audit.blocking_reasons
    assert all("full_prompt" not in reason for reason in audit.blocking_reasons)


def test_root_grouped_folds_never_cross_variants() -> None:
    records = _family_records(12, variants=2)
    audit = audit_corpus(records, _policy())
    folds = audit.cue_only["record_folds"]
    assert len(folds) == len(audit.pair_audits)
    by_root: dict[str, set[int]] = {}
    for pair, fold in zip(audit.pair_audits, folds):
        by_root.setdefault(pair.root_id, set()).add(fold)
    assert all(len(seen) == 1 for seen in by_root.values())
    assert len(audit.cue_only["fold_assignments"]) == 24


def test_thresholds_are_echoed_from_policy() -> None:
    policy = _policy()
    audit = audit_corpus(_family_records(12), policy)
    payload = policy.to_dict()
    for key, value in payload.items():
        assert audit.thresholds[key] == value
    assert audit.thresholds["min_support_roots"] == 20
    assert audit.to_dict()["schema_version"] == SCHEMA_VERSION


def test_ordering_and_sha_are_deterministic() -> None:
    records = _family_records(12)
    first = audit_corpus(records, _policy())
    second = audit_corpus(list(reversed(records)), _policy())
    assert first.sha == second.sha
    assert first.to_dict() == second.to_dict()
    assert [item.root_id for item in first.pair_audits] == [
        item.root_id for item in second.pair_audits
    ]
    assert isinstance(first, CorpusAudit)
    assert isinstance(first.pair_audits[0], PairAudit)
    assert audit_corpus(records, _policy()).sha == first.sha
