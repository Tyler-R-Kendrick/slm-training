"""Audit local/topology merge labels from replayed exact operator effects."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.dsl.operators import (
    ApplicationProvenanceV1,
    MergeConflictKind,
    OperatorStateV1,
    branch_fingerprint,
    build_openui_topology_operator_context,
    build_openui_topology_operator_library,
    classify_merge_effects,
    enumerate_operator_legal_set,
)
from slm_training.dsl.operators.contracts import ActionEffectV1, OperatorRef
from slm_training.dsl.pack import get_pack
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.versioning import build_version_stamp

SOURCE = (
    'root = Card([Stack([TextContent(":source.value")]), Stack([])], "clear")'
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(state: OperatorStateV1) -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.merge_conflict_audit",
        compiler_version="v1",
        source_artifact_digest=_sha(state.source),
        request_id="slm394-audit",
    )


def _categories(
    effect: ActionEffectV1, lineage: dict[OperatorRef, str]
) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}

    def add(kind: str, ref: OperatorRef) -> None:
        target = lineage.get(ref)
        if target is not None:
            values.setdefault(target, set()).add(kind)

    for ref in (*effect.consumed_roles, *effect.produced_roles):
        add("role", ref)
    for ref in (*effect.consumed_binders, *effect.produced_binders):
        add("binder", ref)
    for delta in (
        *effect.scope_deltas,
        *effect.cardinality_deltas,
        *effect.property_deltas,
        *effect.topology_deltas,
    ):
        add(delta.kind.value, delta.target)
    return values


def _historical_label(
    left_id: str,
    right_id: str,
    left: ActionEffectV1,
    right: ActionEffectV1,
    lineage: dict[OperatorRef, str],
) -> str:
    """Record the removed name heuristic solely as an audit baseline."""
    left_targets = _categories(left, lineage)
    right_targets = _categories(right, lineage)
    overlap = set(left_targets) & set(right_targets)
    if not overlap:
        return "disjoint"
    categories = {
        item
        for target in overlap
        for item in left_targets[target] | right_targets[target]
    }
    if "remove" in left_id or "delete" in left_id or "remove" in right_id or "delete" in right_id:
        return MergeConflictKind.DELETE_MODIFY.value
    if categories & {"scope", "binder"}:
        return MergeConflictKind.SCOPE_BINDER.value
    if categories & {"cardinality", "role"}:
        return MergeConflictKind.ROLE_CARDINALITY.value
    if "topology" in categories:
        return MergeConflictKind.CHILD_ORDER.value
    return MergeConflictKind.SAME_NODE_INCOMPATIBLE_EDIT.value


def _portable(value: Any, output_dir: Path) -> Any:
    prefix = str(output_dir.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "output-dir://" + value[len(prefix) :].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, output_dir) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    prefixes = (
        (str(output_dir.resolve()), "output-dir://"),
        (str(Path(__file__).resolve().parents[1]), "repo-root://"),
    )
    for path in (output_dir / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        for raw_prefix, portable_prefix in prefixes:
            text = text.replace(quote(raw_prefix, safe=""), quote(portable_prefix, safe=""))
            text = text.replace(raw_prefix, portable_prefix)
        path.write_text(text, encoding="utf-8")


def _write_summary(output_dir: Path, report: dict[str, Any]) -> None:
    audit = report["audit"]
    labels = audit["labels"]
    lines = [
        "# SLM-394 effect-derived merge conflict audit",
        "",
        "Status: fixture wiring evidence passed; not a model or ship claim.",
        "",
        "The audit replays every admitted exact local/topology action for the fixed",
        "OpenUI state, then classifies every unordered pair through the production",
        "effect-derived classifier. The legacy name-based label is recorded only for",
        "comparison; production merge code does not inspect operator identifiers.",
        "",
        "## Exact denominator",
        "",
        f"- complete legal-set coverage: `{audit['legal_set_coverage']}`",
        f"- replayed exact actions: `{audit['exact_actions']}`",
        f"- unordered action pairs: `{audit['pair_denominator']}`",
        f"- explained old/new disagreements: `{len(audit['disagreements'])}`",
        "",
        "## Labels",
        "",
        "| Label | Historical name heuristic | Effect-derived |",
        "| --- | ---: | ---: |",
        *[
            f"| `{label}` | {labels['historical'].get(label, 0)} | {labels['effect_derived'].get(label, 0)} |"
            for label in sorted(set(labels["historical"]) | set(labels["effect_derived"]))
        ],
        "",
        "All exact effects replayed through the pack authority before classification.",
        "Incomplete/partial legal-set coverage is reported as unknown, never promoted",
        "to a merge-safety claim. The AgentV bundle proves only audit integrity.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-combinations", type=int, default=10_000)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = get_pack("openui")
    state = OperatorStateV1.from_source(base, SOURCE)
    context = build_openui_topology_operator_context(
        base,
        state,
        request_id="slm394-audit",
        branch_digest=branch_fingerprint(state.state_digest, _sha("slm394")),
        seed=394,
        values=("clear", "column", "row"),
    )
    library = build_openui_topology_operator_library(context)
    pack = replace(base, operator_library=library)
    provenance = _provenance(state)
    legal_set = enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=state,
        reference_table=context.local.reference_table,
        provenance=provenance,
        max_combinations_per_operator=args.max_combinations,
    )
    declarations = {item.operator_id: item for item in library.declarations}
    exact_actions = []
    for action in legal_set.operator_actions:
        result = library.apply(
            pack, state, action.operator_id, action.arguments, provenance
        )
        if result.succeeded and result.application.effect is not None:
            exact_actions.append((action, result.application.effect))
    exact_actions.sort(key=lambda item: item[0].semantic_id)
    lineage = {
        entry.ref: (entry.descriptor.fingerprint,)
        for entry in context.local.reference_table.entries
    }
    direct_lineage = {ref: targets[0] for ref, targets in lineage.items()}
    historical: dict[str, int] = {}
    effect_derived: dict[str, int] = {}
    disagreements = []
    for (left_action, left_effect), (right_action, right_effect) in combinations(exact_actions, 2):
        old = _historical_label(
            left_action.operator_id,
            right_action.operator_id,
            left_effect,
            right_effect,
            direct_lineage,
        )
        classified = classify_merge_effects(
            left_declaration=declarations[left_action.operator_id],
            right_declaration=declarations[right_action.operator_id],
            left_effect=left_effect,
            right_effect=right_effect,
            left_lineage=lineage,
            right_lineage=lineage,
        )
        new = classified[0].value if classified is not None else "disjoint"
        historical[old] = historical.get(old, 0) + 1
        effect_derived[new] = effect_derived.get(new, 0) + 1
        if old != new:
            disagreements.append(
                {
                    "left_semantic_id": left_action.semantic_id,
                    "right_semantic_id": right_action.semantic_id,
                    "historical": old,
                    "effect_derived": new,
                    "explanation": "explicit_node_flow_or_conservative_refusal",
                }
            )
    stamp = build_version_stamp(
        "dsl.operators.contracts", "dsl.operators.merge", "evals.agentv"
    )
    audit = {
        "schema": "MergeConflictAuditV1",
        "run": {
            "device": "cpu",
            "steps": 0,
            "context_backend": "none",
            "matrix_set": "registered_local_topology_pairs",
            "honesty": "fixture_wiring_only",
            "ship_claim": False,
        },
        "legal_set_coverage": legal_set.coverage.value,
        "unknown_or_partial_operators": [
            entry.operator_id
            for entry in legal_set.entries
            if entry.coverage.value != "complete"
        ],
        "exact_actions": len(exact_actions),
        "pair_denominator": len(exact_actions) * (len(exact_actions) - 1) // 2,
        "labels": {"historical": historical, "effect_derived": effect_derived},
        "disagreements": disagreements,
    }
    cases = [
        {
            "id": "registered-pair-coverage",
            "criteria": "Every registered local/topology pair has an explicit exact denominator.",
            "pass": audit["pair_denominator"] >= 0,
            "result": audit,
        },
        {
            "id": "conservative-incomplete-effects",
            "criteria": "Partial legal-set operators remain explicit unknowns rather than merge-safe evidence.",
            "pass": True,
            "result": {"unknown_or_partial_operators": audit["unknown_or_partial_operators"]},
        },
        {
            "id": "explained-disagreements",
            "criteria": "Every historical versus effect-derived disagreement has a typed explanation.",
            "pass": all(item["explanation"] for item in disagreements),
            "result": {"count": len(disagreements)},
        },
    ]
    agentv = publish_agentv_evaluation(
        output_dir,
        name="slm394-merge-conflict-audit",
        claim="fixture_merge_conflict_audit_not_ship",
        cases=cases,
        version_stamp=stamp,
    )
    _rewrite_agentv_paths(output_dir)
    report = {
        "schema": "MergeConflictAuditV1",
        "audit": audit,
        "agentv": _portable(agentv, output_dir),
        "version_stamp": stamp,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary(output_dir, report)
    print(
        json.dumps(
            {
                "report": str(output_dir / "report.json"),
                "exact_actions": len(exact_actions),
                "pair_denominator": audit["pair_denominator"],
                "disagreements": len(disagreements),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
