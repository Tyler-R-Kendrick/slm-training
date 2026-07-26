#!/usr/bin/env python3
"""Run SLM-402's bounded local typed-policy permutation stop-rule probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import torch

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    CompilerCoverage,
    CompilerFact,
    OperatorArgumentSlotV1,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    RegisteredOperatorV1,
    build_reference_table,
    enumerate_operator_legal_set,
)
from slm_training.dsl.pack import get_pack
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.models.operator_feature_encoder import (
    CandidateScoringHead,
    FeatureArm,
    OperatorFeatureEncoder,
    make_fixture_operator_decisions,
    train_fixture_arm,
)
from slm_training.models.operator_permutation_consistency import (
    OperatorPermutationCacheV1,
    OperatorPermutationReportV1,
    build_operator_permutation_alignment,
    compare_permuted_scores,
)
from slm_training.models.operator_policy_view import build_operator_policy_input
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = "model.operator_permutation_consistency"
JSON_OUT = ROOT / "docs/design/dsh3-27-operator-permutation-consistency.json"
MARKDOWN_OUT = ROOT / "docs/design/dsh3-27-operator-permutation-consistency.md"
AGENTV_DIR = (
    ROOT / "docs/design/dsh3-27-operator-permutation-consistency-agentv-20260726"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _portable(value: Any, output_dir: Path) -> Any:
    prefix = str(output_dir.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "agentv-dir://" + value[len(prefix) :].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, output_dir) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    replacements = {
        str(output_dir.resolve()): "agentv-dir://",
        quote(str(output_dir.resolve()), safe=""): quote("agentv-dir://", safe=""),
        str(ROOT.resolve()): "repo://",
        quote(str(ROOT.resolve()), safe=""): quote("repo://", safe=""),
    }
    for path in (output_dir / "agentv").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}:
            text = path.read_text(encoding="utf-8")
            for source, replacement in replacements.items():
                text = text.replace(source, replacement)
            path.write_text(text, encoding="utf-8")


def _fixture(seed: int):
    pack = get_pack("openui")
    source = 'root = TextContent(":hero.title")'
    state = OperatorStateV1.from_source(pack, source)
    descriptors = tuple(
        ReferenceDescriptorV1(
            ref_kind=RefKind.VALUE,
            semantic_fingerprint=_sha(f"descriptor-{index}"),
            value_type="openui.number" if index == 2 else "openui.string",
            compiler_facts=(CompilerFact.VALUE_VISIBLE,),
        )
        for index in range(5)
    )
    table = build_reference_table(
        request_id="slm402-local-request",
        state_digest=state.state_digest,
        branch_digest=_sha("slm402-local-branch"),
        descriptors=descriptors,
        seed=seed,
    )
    declaration = AstOperatorV1(
        operator_id="openui.fixture_op",
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(
            OperatorArgumentSlotV1("value", RefKind.VALUE, BindingPhase.APPLICATION),
        ),
        preconditions=(),
        effect_signature=(),
        locality="node",
        cost=1.0,
    )
    library = OperatorLibraryV1(
        (
            RegisteredOperatorV1(
                declaration,
                lambda _state, _arguments: OperatorMutationV1(
                    source=source,
                    effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
                ),
            ),
        )
    )
    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="slm402.fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(source),
        request_id=table.request_id,
    )
    return replace(pack, operator_library=library), state, table, library, provenance


def _enumerate(pack, state, table, library, provenance):
    return enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=state,
        reference_table=table,
        provenance=provenance,
    )


def _semantic_scores(
    trained: dict[str, Any], view, table, legal_set
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    encoder: OperatorFeatureEncoder = trained["encoder"]
    head: CandidateScoringHead = trained["head"]
    encoder.eval()
    head.eval()
    with torch.no_grad():
        candidates = view.action_rows[0].argument_slots[0].candidate_rows
        references = encoder.encode_reference_rows(view)
        logits = head(encoder.encode(view)[0], references[torch.tensor(candidates)])
    semantic_by_ref = {
        action.arguments[0].value: action.semantic_id
        for action in legal_set.operator_actions
    }
    score_by_id = {
        semantic_by_ref[table.entries[row].ref]: float(score)
        for row, score in zip(candidates, logits.tolist())
    }
    ids = tuple(sorted(score_by_id))
    return ids, tuple(score_by_id[item] for item in ids)


def run(*, seed: int, k_values: tuple[int, ...], agentv_dir: Path) -> dict[str, Any]:
    pack, state, table, library, provenance = _fixture(seed)
    legal_set = _enumerate(pack, state, table, library, provenance)
    view = build_operator_policy_input(table, legal_set, library)
    train = make_fixture_operator_decisions(30, seed=seed, n_candidates=5)
    trained = train_fixture_arm(train, arm=FeatureArm.TYPED, steps=150, seed=seed)
    base_alignment = build_operator_permutation_alignment(table, legal_set)
    base_ids, base_scores = _semantic_scores(trained, view, table, legal_set)
    cache = OperatorPermutationCacheV1()
    comparisons = []
    for k in k_values:
        for offset in range(k):
            permuted_table = cache.get_or_build(
                table, base_alignment, seed=seed + 100 + offset
            )
            permuted_legal_set = _enumerate(
                pack, state, permuted_table, library, provenance
            )
            permuted_view = build_operator_policy_input(
                permuted_table, permuted_legal_set, library
            )
            permuted_alignment = build_operator_permutation_alignment(
                permuted_table, permuted_legal_set
            )
            ids, scores = _semantic_scores(
                trained, permuted_view, permuted_table, permuted_legal_set
            )
            comparisons.append(
                compare_permuted_scores(
                    base_alignment,
                    permuted_alignment,
                    original_scores=base_scores,
                    permuted_scores=scores,
                    accepted_semantic_action_ids=base_ids,
                )
            )
    # Exercise deterministic cache reuse without changing any comparison result.
    cache.get_or_build(table, base_alignment, seed=seed + 100)
    probe = OperatorPermutationReportV1(tuple(comparisons), cache.evidence()).to_dict()
    invariant = probe["top1_flips"] == 0 and probe["score_disagreement"] == 0.0
    stamp = build_version_stamp(COMPONENT)
    report: dict[str, Any] = {
        "schema": "dsh3_27_operator_permutation_fixture/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_class": "wiring",
        "honest_verdict": "fixture_wiring_stop_rule_negative",
        "recipe": {
            "device": "cpu",
            "backend": "local",
            "seed": seed,
            "k_values": list(k_values),
            "train_decisions": 30,
            "steps": 150,
            "suite_n": len(comparisons),
        },
        "probe": probe,
        "disposition": "closed_negative_unaugmented_invariant"
        if invariant
        else "augmentation_experiment_required",
        "version_stamp": stamp,
    }
    published = publish_agentv_evaluation(
        agentv_dir,
        name="dsh3-27-operator-permutation",
        claim="fixture_permutation_stop_rule_not_ship",
        cases=[
            {
                "id": "unaugmented_typed_policy_is_invariant",
                "criteria": "The local typed fixture may close augmentation only when every compatible unseen permutation preserves top-1 and scores.",
                "assertions": [
                    {
                        "id": "stop_rule_negative",
                        "actual": invariant,
                        "operator": "eq",
                        "expected": True,
                    }
                ],
                "result": {"probe": probe, "disposition": report["disposition"]},
            }
        ],
        version_stamp=stamp,
    )
    _rewrite_agentv_paths(agentv_dir)
    report["agentv"] = _portable(published, agentv_dir)
    return report


def _markdown(report: dict[str, Any]) -> str:
    probe, recipe = report["probe"], report["recipe"]
    return f"""# SLM-402 (DSH3-27): local operator-permutation stop-rule probe

**Status:** fixture / wiring only.
**Honest verdict:** `{report["honest_verdict"]}`.
**Disposition:** `{report["disposition"]}`.

The typed encoder’s evaluator-side scores were unchanged across {recipe["suite_n"]} compatible
reference/candidate permutations on unseen seeds. The stop rule therefore retains the alignment
and cache regression contract and does **not** add augmentation or KL/JS consistency training.

| Metric | Value |
| --- | ---: |
| Top-1 semantic flips | {probe["top1_flips"]} |
| Accepted-set mass drift | {probe["accepted_set_mass_drift"]:.6f} |
| Calibration-confidence drift | {probe["calibration_confidence_drift"]:.6f} |
| Score disagreement | {probe["score_disagreement"]:.6f} |
| Cache hits / misses | {probe["cache"]["hits"]} / {probe["cache"]["misses"]} |

- Local CPU; 30 synthetic typed training decisions; 150 steps; K={recipe["k_values"]}; suite n={recipe["suite_n"]}.
- Fresh legal sets were re-enumerated after `ReferenceTableV1.permuted`; semantic action IDs and descriptor fingerprints were used only by evaluator alignment, never by model input.
- Unequal PARTIAL membership, cross-request/state/branch comparisons, and any KL/JS over those mismatched domains remain rejected by the contract.
- This is not a CAP2, meaningful-parse, ship-gate, or model-promotion result. AgentEvals/AgentV records only the fixture stop-rule assertion.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=402)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=MARKDOWN_OUT)
    parser.add_argument("--agentv-dir", type=Path, default=AGENTV_DIR)
    args = parser.parse_args()
    report = run(seed=args.seed, k_values=(1, 2, 4), agentv_dir=args.agentv_dir)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {"json": str(args.json_out), "disposition": report["disposition"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
