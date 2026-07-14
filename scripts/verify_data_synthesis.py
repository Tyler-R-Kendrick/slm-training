#!/usr/bin/env python3
"""Emit the fail-closed P13 corpus and matched-smoke verification report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from slm_training.data.edits import EditKind, EditOperation, EditPatch, apply_patch
from slm_training.data.governance import scan_untrusted_text
from slm_training.data.leakage import find_leakage, load_train_fingerprints
from slm_training.dsl.language_contract import contract_id, current_contract
from slm_training.dsl.parser import validate
from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.evals.generalization import generalization_report
from slm_training.harnesses.model_build.diagnostic import run_full_diagnostic

EXPECTED_FAMILIES = frozenset(
    {
        "abstraction_ladder",
        "corruption_repair",
        "edit_trajectory",
        "frontier_described",
        "human_curated",
        "language_contract",
        "programspec_generated",
        "renderer_visual",
        "rico_real",
        "web_distilled",
    }
)
MATCHED_MATRIX_KEYS = (
    "device",
    "steps",
    "batch_size",
    "learning_rate",
    "seed",
    "gen_steps",
    "context_backend",
    "rico_eval_limit",
    "suites",
    "test_dir",
    "design_md_in_context",
    "gate_policy",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _patch(value: dict[str, Any]) -> EditPatch:
    operations = tuple(
        EditOperation(
            kind=EditKind(item["kind"]),
            name=str(item["name"]),
            before=item.get("before"),
            after=item.get("after"),
            target=item.get("target"),
            index=item.get("index"),
            previous_index=item.get("previous_index"),
        )
        for item in value.get("operations") or []
    )
    return EditPatch(
        operations,
        instruction=str(value.get("instruction") or ""),
        collect_unreachable=bool(value.get("collect_unreachable", True)),
    )


def _reverse_patch(patch: EditPatch) -> EditPatch:
    operations: list[EditOperation] = []
    for operation in reversed(patch.operations):
        if operation.kind is EditKind.REPLACE:
            operations.append(
                EditOperation(
                    EditKind.REPLACE,
                    operation.name,
                    before=operation.after,
                    after=operation.before,
                )
            )
        elif operation.kind is EditKind.ADD:
            operations.append(
                EditOperation(EditKind.REMOVE, operation.name, before=operation.after)
            )
        elif operation.kind is EditKind.REMOVE:
            operations.append(
                EditOperation(EditKind.ADD, operation.name, after=operation.before)
            )
        else:
            raise ValueError(f"cannot derive inverse for {operation.kind.value}")
    return EditPatch(tuple(operations))


def _check(ok: bool, evidence: Any, *, status: str | None = None) -> dict[str, Any]:
    return {
        "status": status or ("pass" if ok else "fail"),
        "evidence": evidence,
    }


def _suite_records(test_dir: Path) -> list[ExampleRecord]:
    return [
        record
        for path in sorted((test_dir / "suites").glob("*/records.jsonl"))
        for record in load_jsonl(path)
    ]


def _matrix_result(summary: dict[str, Any]) -> dict[str, Any]:
    results = summary.get("results") or []
    if len(results) != 1:
        raise ValueError("matched smoke summaries must contain exactly one result")
    return dict(results[0])


def build_report(
    *,
    first_train_dir: Path,
    second_train_dir: Path,
    test_dir: Path,
    task_results: Path,
    baseline_matrix: Path,
    champion_matrix: Path,
    ltr_max_tokens: int = 256,
) -> dict[str, Any]:
    first_manifest = _read_json(first_train_dir / "manifest.json")
    second_manifest = _read_json(second_train_dir / "manifest.json")
    first_records = load_jsonl(first_train_dir / "records.jsonl")
    second_records = load_jsonl(second_train_dir / "records.jsonl")
    held_records = _suite_records(test_dir)
    checks: dict[str, dict[str, Any]] = {}

    active_contract = contract_id()
    contract_counts = Counter(
        str(record.meta.get("contract_id") or "missing") for record in first_records
    )
    checks["contract_id"] = _check(
        bool(first_records)
        and contract_counts == Counter({active_contract: len(first_records)}),
        {"active": active_contract, "counts": dict(sorted(contract_counts.items()))},
    )
    first_fp = first_manifest.get("content_fingerprint")
    second_fp = second_manifest.get("content_fingerprint")
    checks["content_fingerprint"] = _check(
        bool(first_fp) and first_fp == second_fp,
        {
            "first": first_fp,
            "second": second_fp,
            "record_counts": [len(first_records), len(second_records)],
        },
    )

    language_rows = [
        row
        for row in first_records
        if row.meta.get("source_family") == "language_contract"
        and row.meta.get("polarity") == "positive"
    ]
    roundtrip_failures: list[dict[str, str]] = []
    targets: set[str] = set()
    components = 0
    placeholder_rows = 0
    for row in language_rows:
        targets.add(str(row.meta.get("contract_target") or ""))
        components += row.meta.get("category") == "component"
        placeholder_rows += bool(row.placeholders)
        try:
            first = validate(row.openui)
            canonical = first.serialized or row.openui.strip()
            second = validate(canonical)
            if (second.serialized or canonical) != canonical:
                roundtrip_failures.append(
                    {"id": row.id, "error": "serialization drift"}
                )
        except Exception as exc:  # noqa: BLE001 - evidence records bridge failures
            roundtrip_failures.append({"id": row.id, "error": str(exc)})
    required_targets = {"forward_reference", "multi_child", "nested_list"}
    bridge_ok = (
        bool(language_rows)
        and components > 0
        and placeholder_rows == len(language_rows)
        and required_targets <= targets
        and not roundtrip_failures
    )
    checks["language_bridge"] = _check(
        bridge_ok,
        {
            "positive_records": len(language_rows),
            "component_records": components,
            "placeholder_records": placeholder_rows,
            "required_targets": sorted(required_targets),
            "observed_targets": sorted(targets),
            "failures": roundtrip_failures[:20],
        },
    )
    contract = current_contract().to_dict()
    checks["deferred_language_surface"] = _check(
        True,
        {
            "lang_spec": contract["lang_spec"],
            "openui_versions": contract["openui_versions"],
            "state": "unsupported",
            "query": "unsupported",
            "mutation": "unsupported",
            "action": "unsupported",
            "reason": "not exposed by the pinned OpenUI 0.2.9 language contract",
        },
        status="deferred",
    )

    families = first_manifest.get("source_families", {}).get("families", {})
    missing_families = sorted(EXPECTED_FAMILIES - set(families))
    bad_parents = sorted(
        name
        for name, value in families.items()
        if value.get("unique_root_parents", 0) < 1
    )
    checks["source_families"] = _check(
        not missing_families and not bad_parents,
        {
            "expected": sorted(EXPECTED_FAMILIES),
            "missing": missing_families,
            "bad_unique_root_parents": bad_parents,
            "families": families,
        },
    )

    train_fps = load_train_fingerprints(first_train_dir / "manifest.json")
    contaminated = [
        {"id": row.id, "reasons": reasons}
        for row in held_records
        if (reasons := find_leakage(row, train_fps))
    ]
    checks["split_before_derive_leakage"] = _check(
        bool(held_records) and not contaminated,
        {"held_out_n": len(held_records), "contaminated": contaminated[:20]},
    )

    diagnostic = run_full_diagnostic(
        first_train_dir, test_dir, grammar_ltr_max_tokens=ltr_max_tokens
    )
    ceiling = diagnostic.get("ceiling") or {}
    ceiling_failures = {
        suite: metrics
        for suite, metrics in ceiling.items()
        if metrics.get("n", 0) < 1
        or any(
            float(metrics.get(name, 0.0)) < 0.999
            for name in (
                "parse_rate",
                "placeholder_fidelity",
                "placeholder_validity",
                "structural_similarity",
                "component_type_recall",
            )
        )
    }
    checks["diagnostic_ceiling"] = _check(
        bool(ceiling) and not ceiling_failures,
        {"suites": ceiling, "failures": ceiling_failures},
    )
    checks["length_budget"] = _check(
        bool((diagnostic.get("length_budget") or {}).get("ok")),
        diagnostic.get("length_budget"),
    )

    tier_counts = Counter(
        str(row.meta.get("verification_tier")) for row in first_records
    )
    tier_failures = [
        row.id
        for row in first_records
        if row.meta.get("verification_tier") in {"Gold", "Silver"}
        and any(
            gate.get("status") == "fail"
            for gate in (row.meta.get("verification") or {}).get("gates", [])
        )
    ]
    checks["verifier_tiers"] = _check(
        bool(tier_counts.get("Gold") or tier_counts.get("Silver"))
        and not tier_failures,
        {
            "counts": dict(sorted(tier_counts.items())),
            "gold_silver_failures": tier_failures,
        },
    )

    governance_paths = first_manifest.get("governance") or {}
    required_governance = {"croissant.json", "data_card.json", "dataset.spdx.json"}
    missing_governance = sorted(required_governance - set(governance_paths))
    scan_counts: Counter[str] = Counter()
    for row in first_records:
        scan = scan_untrusted_text(
            "\n".join((row.prompt, row.openui, row.design_md or ""))
        )
        scan_counts["pii"] += len(scan.pii_kinds)
        scan_counts["secrets"] += len(scan.secret_kinds)
        scan_counts["instruction_like"] += int(scan.instruction_like)
    checks["governance"] = _check(
        not missing_governance and scan_counts["pii"] == scan_counts["secrets"] == 0,
        {
            "artifacts": dict(sorted(governance_paths.items())),
            "missing": missing_governance,
            "scan_counts": dict(scan_counts),
        },
    )

    edit_rows = [row for row in first_records if isinstance(row.meta.get("edit"), dict)]
    edit_failures: list[dict[str, str]] = []
    for row in edit_rows:
        edit = row.meta["edit"]
        try:
            patch = _patch(edit["patch"])
            after = str(edit.get("after") or row.openui)
            inverse = (
                _patch(edit["inverse"])
                if isinstance(edit.get("inverse"), dict)
                else _reverse_patch(patch)
            )
            before = str(edit.get("before") or apply_patch(after, inverse))
            applied = apply_patch(before, patch)
            restored = apply_patch(after, inverse)
            if applied != after or restored != before:
                edit_failures.append({"id": row.id, "error": "apply/inverse mismatch"})
        except Exception as exc:  # noqa: BLE001 - evidence records invariant failures
            edit_failures.append({"id": row.id, "error": str(exc)})
    checks["edit_invariants"] = _check(
        bool(edit_rows) and not edit_failures,
        {"records": len(edit_rows), "failures": edit_failures},
    )

    tasks = _read_json(task_results).get("task_scoreboard") or {}
    task_names = set((tasks.get("tasks") or {}).keys())
    equivalence = tasks.get("equivalence") or {}
    checks["task_and_equivalence_eval"] = _check(
        {"generation", "repair", "edit", "behavior"} <= task_names
        and equivalence.get("status") == "available"
        and int(equivalence.get("n") or 0) > 0,
        {
            "task_names": sorted(task_names),
            "equivalence": equivalence,
            "unavailable_metric_instances": tasks.get("unavailable_metric_instances"),
        },
    )
    generalization = generalization_report(first_records, held_records)
    generalization_evidence = {
        key: generalization[key]
        for key in (
            "decontaminated",
            "held_out_n",
            "accepted_n",
            "contaminated",
            "slice_counts",
        )
    }
    checks["generalization_held_outs"] = _check(
        bool(held_records) and bool(generalization.get("decontaminated")),
        generalization_evidence,
    )

    baseline_summary = _read_json(baseline_matrix)
    champion_summary = _read_json(champion_matrix)
    mismatched = {
        key: [baseline_summary.get(key), champion_summary.get(key)]
        for key in MATCHED_MATRIX_KEYS
        if baseline_summary.get(key) != champion_summary.get(key)
    }
    baseline = _matrix_result(baseline_summary)
    champion = _matrix_result(champion_summary)
    baseline_rico = float(
        (baseline.get("suites") or {}).get("rico_held", {}).get("placeholder_fidelity")
        or 0.0
    )
    champion_rico = float(
        (champion.get("suites") or {}).get("rico_held", {}).get("placeholder_fidelity")
        or 0.0
    )
    checks["matched_quality_signal"] = _check(
        not mismatched and champion_rico > baseline_rico and champion_rico > 0.0,
        {
            "matched_recipe": {
                key: baseline_summary.get(key) for key in MATCHED_MATRIX_KEYS
            },
            "mismatched": mismatched,
            "baseline": baseline,
            "champion": champion,
            "rico_held_placeholder_fidelity": {
                "baseline": baseline_rico,
                "champion": champion_rico,
                "delta": champion_rico - baseline_rico,
            },
        },
    )

    required = [value for value in checks.values() if value["status"] != "deferred"]
    return {
        "schema_version": 1,
        "issue": "SLM-17",
        "kind": "data_synthesis_definition_of_done",
        "honesty": {
            "fixture_or_scratch": True,
            "ship_claim": False,
            "gates_lowered": False,
            "note": "Bounded CPU fixture smoke; not a production HF-context claim.",
        },
        "inputs": {
            "first_train_dir": str(first_train_dir),
            "second_train_dir": str(second_train_dir),
            "test_dir": str(test_dir),
            "task_results": str(task_results),
            "baseline_matrix": str(baseline_matrix),
            "champion_matrix": str(champion_matrix),
            "ltr_max_tokens": ltr_max_tokens,
        },
        "checks": checks,
        "pass": all(value["status"] == "pass" for value in required),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-train-dir", type=Path, required=True)
    parser.add_argument("--second-train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--task-results", type=Path, required=True)
    parser.add_argument("--baseline-matrix", type=Path, required=True)
    parser.add_argument("--champion-matrix", type=Path, required=True)
    parser.add_argument("--ltr-max-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(
        first_train_dir=args.first_train_dir,
        second_train_dir=args.second_train_dir,
        test_dir=args.test_dir,
        task_results=args.task_results,
        baseline_matrix=args.baseline_matrix,
        champion_matrix=args.champion_matrix,
        ltr_max_tokens=args.ltr_max_tokens,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "pass": report["pass"]}))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
