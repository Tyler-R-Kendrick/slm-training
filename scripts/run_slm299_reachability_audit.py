#!/usr/bin/env python3
"""SLM-299 (LAR1-03): audit X22 tree-edit reachability across suite corpora.

Runs the real-space reachability analyzer
(``slm_training.harnesses.experiments.slm299_edit_reachability``) from the
standard X22 minimal seed over train / smoke / held_out / adversarial / ood /
rico corpora and writes the durable design artifacts:

  docs/design/iter-slm299-edit-reachability-20260724.json
  docs/design/iter-slm299-edit-reachability-20260724.md

UNKNOWN_BUDGET cases are reported separately and are NEVER counted as
unreachable; ``reachable_fraction`` is computed over decided cases only.
Suites with no available corpus are reported as ``corpus_unavailable`` —
never as zero-reachable.

Example:
  python -m scripts.run_slm299_reachability_audit
  python -m scripts.run_slm299_reachability_audit --max-edits 6 --node-budget 400
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.harnesses.experiments.slm299_edit_reachability import (
    DEFAULT_SEED_SOURCE,
    EXPERIMENT_ID,
    ReachabilityCase,
    Verdict,
    analyze_reachability,
)
from slm_training.versioning import build_version_stamp

TEST_SEEDS = Path("src/slm_training/resources/test_seeds.jsonl")
RICO_CORPUS = Path("src/slm_training/resources/evals/slm294_frozen_requests_v1.jsonl")
TRAIN_CORPUS = Path("outputs/data/train/slm230_symbol_only_v1/records.jsonl")

DEFAULT_JSON_OUT = Path("docs/design/iter-slm299-edit-reachability-20260724.json")
DEFAULT_MD_OUT = Path("docs/design/iter-slm299-edit-reachability-20260724.md")

# Append-only annotations on these earlier X22 readings (never edited).
X22_EVIDENCE_DOCS = (
    "docs/design/iter-x22-d3-kapur-tree-edit-20260717.md",
    "docs/design/iter-efs0-04-x22-reproduction-20260717.md",
)

COMPONENT = "harness.experiments.slm299_edit_reachability"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _suite_label(record: dict[str, Any]) -> str:
    meta = record.get("meta") or {}
    return str(meta.get("suite") or record.get("split") or "")


def load_corpora() -> dict[str, list[dict[str, Any]]]:
    """Suite name → records; an empty list means corpus unavailable."""
    seeds = _load_jsonl(TEST_SEEDS)
    corpora: dict[str, list[dict[str, Any]]] = {
        "train": _load_jsonl(TRAIN_CORPUS),
        "smoke": [r for r in seeds if _suite_label(r) == "smoke"],
        "held_out": [r for r in seeds if _suite_label(r) == "held_out"],
        "adversarial": [r for r in seeds if _suite_label(r) == "adversarial"],
        "ood": [r for r in seeds if _suite_label(r) == "ood"],
        "rico": _load_jsonl(RICO_CORPUS),
    }
    return corpora


def analyze_record(
    record: dict[str, Any], *, max_edits: int, node_budget: int, mode: str = "extended"
) -> ReachabilityCase:
    target = str(record.get("openui") or "")
    placeholders = record.get("placeholders") or extract_placeholders(target)
    return analyze_reachability(
        DEFAULT_SEED_SOURCE,
        target,
        slot_inventory=[str(p) for p in placeholders],
        max_edits=max_edits,
        node_budget=node_budget,
        mode=mode,
    )


def summarize_suite(
    cases: list[tuple[str, ReachabilityCase]],
) -> dict[str, Any]:
    decided = [c for _, c in cases if c.verdict is not Verdict.UNKNOWN_BUDGET]
    reachable = [c for c in decided if c.verdict is Verdict.PROVEN_REACHABLE]
    reasons = Counter(c.reason_code for _, c in cases)
    actions: Counter[str] = Counter()
    components: Counter[str] = Counter()
    for case in reachable:
        for step in case.path:
            actions[step["action"]] += 1
            if "comp" in step:
                components[step["comp"]] += 1
    bounds = [c.edit_lower_bound for c in reachable if c.edit_lower_bound is not None]
    return {
        "n_cases": len(cases),
        "n_decided": len(decided),
        "n_unknown_budget": len(cases) - len(decided),
        "n_reachable": len(reachable),
        # UNKNOWN_BUDGET is excluded from the denominator — never counted
        # as unreachable. None when nothing was decided.
        "reachable_fraction": (
            round(len(reachable) / len(decided), 6) if decided else None
        ),
        "reason_histogram": dict(sorted(reasons.items())),
        "action_coverage": dict(sorted(actions.items())),
        "component_coverage": dict(sorted(components.items())),
        "edit_lower_bound": (
            {
                "min": min(bounds),
                "median": statistics.median(bounds),
                "max": max(bounds),
                "mean": round(statistics.mean(bounds), 4),
            }
            if bounds
            else None
        ),
        "cases": [
            {
                "id": case_id,
                "verdict": case.verdict.value,
                "reason_code": case.reason_code,
                "edit_lower_bound": case.edit_lower_bound,
                "path_length": len(case.path),
            }
            for case_id, case in cases
        ],
    }


def build_x22_evidence_annotations(
    suites: dict[str, Any], *, generated_at: str
) -> list[dict[str, Any]]:
    """Append-only annotations stating which suite-level quality readings in
    the earlier X22 docs are bounded by the measured reachable fractions."""
    annotations: list[dict[str, Any]] = []
    for suite, summary in sorted(suites.items()):
        if summary.get("status") == "corpus_unavailable":
            continue
        fraction = summary["reachable_fraction"]
        if fraction is None:
            continue
        for doc in X22_EVIDENCE_DOCS:
            annotations.append(
                {
                    "date": generated_at,
                    "target_doc": doc,
                    "suite": suite,
                    "annotation": (
                        f"SLM-299 edit-reachability audit: from the standard X22 "
                        f"minimal seed, reachable_fraction={fraction} over "
                        f"{summary['n_decided']} decided {suite} cases "
                        f"({summary['n_unknown_budget']} UNKNOWN_BUDGET, never "
                        f"counted unreachable). Suite-level quality readings of "
                        f"the X22 tree-edit decode on {suite} in this document "
                        f"are bounded by that fraction: the unreachable share of "
                        f"gold programs cannot be produced by the decode space at "
                        f"all, so measured quality on those cases reflects space "
                        f"coverage, not model quality."
                    ),
                }
            )
    return annotations


def _summarize_records(
    records: list[dict[str, Any]], *, max_edits: int, node_budget: int, mode: str
) -> dict[str, Any]:
    cases = [
        (
            str(record.get("id", f"case_{index}")),
            analyze_record(record, max_edits=max_edits, node_budget=node_budget, mode=mode),
        )
        for index, record in enumerate(records)
    ]
    return {"status": "ok", **summarize_suite(cases)}


def _compare_summaries(v1: dict[str, Any], extended: dict[str, Any]) -> dict[str, Any]:
    """Old-vs-extended comparison: verdict flips and action-cost deltas over
    matched cases (never a quality claim — reachability only)."""
    v1_cases = {c["id"]: c for c in v1.get("cases", [])}
    ext_cases = {c["id"]: c for c in extended.get("cases", [])}
    deltas: list[int] = []
    flips: list[dict[str, Any]] = []
    for case_id, ext in sorted(ext_cases.items()):
        old = v1_cases.get(case_id)
        if old is None:
            continue
        if old["verdict"] != ext["verdict"]:
            flips.append(
                {"id": case_id, "v1": old["verdict"], "extended": ext["verdict"]}
            )
        if old["edit_lower_bound"] is not None and ext["edit_lower_bound"] is not None:
            deltas.append(ext["edit_lower_bound"] - old["edit_lower_bound"])
    return {
        "n_verdict_flips": len(flips),
        "verdict_flips": flips,
        "reachable_fraction_v1": v1.get("reachable_fraction"),
        "reachable_fraction_extended": extended.get("reachable_fraction"),
        "action_cost_delta": (
            {
                "min": min(deltas),
                "median": statistics.median(deltas),
                "max": max(deltas),
                "mean": round(statistics.mean(deltas), 4),
            }
            if deltas
            else None
        ),
    }


def build_report(
    corpora: dict[str, list[dict[str, Any]]],
    *,
    max_edits: int,
    node_budget: int,
    generated_at: str,
    limit: int | None = None,
    mode: str = "extended",
    compare: bool = False,
) -> dict[str, Any]:
    suites: dict[str, Any] = {}
    suites_v1: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for suite, records in corpora.items():
        if not records:
            suites[suite] = {
                "status": "corpus_unavailable",
                "n_cases": 0,
                "reachable_fraction": None,
            }
            if compare:
                suites_v1[suite] = dict(suites[suite])
            continue
        if limit is not None:
            records = records[:limit]
        suites[suite] = _summarize_records(
            records, max_edits=max_edits, node_budget=node_budget, mode=mode
        )
        if compare:
            suites_v1[suite] = _summarize_records(
                records, max_edits=max_edits, node_budget=node_budget, mode="v1"
            )
            comparisons[suite] = _compare_summaries(suites_v1[suite], suites[suite])

    payload: dict[str, Any] = {
        "schema": "slm299_edit_reachability_audit/v2",
        "experiment_id": EXPERIMENT_ID,
        "seed_source": DEFAULT_SEED_SOURCE,
        "mode": mode,
        "max_edits": max_edits,
        "node_budget": node_budget,
        "generated_at": generated_at,
        "verdict_policy": (
            "reachable_fraction is computed over decided cases only; "
            "UNKNOWN_BUDGET cases are reported separately and are never counted "
            "as unreachable; suites without a corpus are corpus_unavailable, "
            "never zero-reachable. Reachability is a space-coverage proof, "
            "never a model-quality claim."
        ),
        "suites": suites,
        "version_stamp": build_version_stamp(COMPONENT),
    }
    if compare:
        payload["suites_v1"] = suites_v1
        payload["old_vs_extended"] = comparisons
    payload["x22_evidence_annotations"] = build_x22_evidence_annotations(
        suites, generated_at=generated_at
    )
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SLM-299 (LAR1-03): X22 edit-space reachability audit",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- seed: `{payload['seed_source']}`",
        f"- mode: `{payload.get('mode', 'extended')}`",
        f"- max_edits: {payload['max_edits']}, node_budget: {payload['node_budget']}",
        f"- verdict policy: {payload['verdict_policy']}",
        "",
        "> Reachability is space coverage, not model quality: no quality claim",
        "> follows from these proofs alone.",
        "",
        "## Suite summary",
        "",
        "| suite | n | decided | unknown | reachable_fraction | min/med/max edits |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for suite, summary in payload["suites"].items():
        if summary.get("status") == "corpus_unavailable":
            lines.append(f"| {suite} | corpus_unavailable | — | — | — | — |")
            continue
        bounds = summary["edit_lower_bound"] or {}
        bound_txt = (
            f"{bounds.get('min')}/{bounds.get('median')}/{bounds.get('max')}"
            if bounds
            else "—"
        )
        lines.append(
            f"| {suite} | {summary['n_cases']} | {summary['n_decided']} | "
            f"{summary['n_unknown_budget']} | {summary['reachable_fraction']} | "
            f"{bound_txt} |"
        )
    lines += ["", "## Reason-code histograms", ""]
    for suite, summary in payload["suites"].items():
        if summary.get("status") == "corpus_unavailable":
            continue
        lines.append(f"- **{suite}**: `{json.dumps(summary['reason_histogram'], sort_keys=True)}`")
    lines += ["", "## Action / component coverage (reachable paths)", ""]
    for suite, summary in payload["suites"].items():
        if summary.get("status") == "corpus_unavailable":
            continue
        lines.append(
            f"- **{suite}**: actions `{json.dumps(summary['action_coverage'], sort_keys=True)}` "
            f"components `{json.dumps(summary['component_coverage'], sort_keys=True)}`"
        )
    lines += ["", "## Per-case verdicts", ""]
    for suite, summary in payload["suites"].items():
        if summary.get("status") == "corpus_unavailable":
            lines.append(f"- **{suite}**: corpus unavailable")
            continue
        lines.append(f"- **{suite}**:")
        for case in summary["cases"]:
            lines.append(
                f"  - `{case['id']}` {case['verdict']} "
                f"({case['reason_code']}, lower_bound={case['edit_lower_bound']})"
            )
    lines += ["", "## X22 evidence annotations (append-only)", ""]
    for note in payload["x22_evidence_annotations"]:
        lines.append(f"- `{note['target_doc']}` [{note['suite']}]: {note['annotation']}")
    if "old_vs_extended" in payload:
        lines += ["", "## Old (v1) vs extended (SLM-305) reachability", ""]
        lines.append(
            "| suite | reachable v1 | reachable extended | verdict flips | "
            "action-cost delta min/med/max |"
        )
        lines.append("| --- | --- | --- | --- | --- |")
        for suite, comp in payload["old_vs_extended"].items():
            delta = comp["action_cost_delta"] or {}
            delta_txt = (
                f"{delta.get('min')}/{delta.get('median')}/{delta.get('max')}"
                if delta
                else "—"
            )
            lines.append(
                f"| {suite} | {comp['reachable_fraction_v1']} | "
                f"{comp['reachable_fraction_extended']} | "
                f"{comp['n_verdict_flips']} | {delta_txt} |"
            )
        for suite, comp in payload["old_vs_extended"].items():
            for flip in comp["verdict_flips"]:
                lines.append(
                    f"- flip `{suite}/{flip['id']}`: {flip['v1']} → {flip['extended']}"
                )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-edits", type=int, default=8)
    parser.add_argument("--node-budget", type=int, default=800)
    parser.add_argument("--limit", type=int, default=None, help="per-suite case cap (debug)")
    parser.add_argument(
        "--mode",
        choices=["v1", "extended"],
        default="extended",
        help="action space to audit (extended is the deployed SLM-305 space)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also run the v1 space and publish old-vs-extended deltas",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = build_report(
        load_corpora(),
        max_edits=args.max_edits,
        node_budget=args.node_budget,
        generated_at=generated_at,
        limit=args.limit,
        mode=args.mode,
        compare=args.compare,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    for suite, summary in payload["suites"].items():
        if summary.get("status") == "corpus_unavailable":
            print(f"{suite}: corpus_unavailable")
        else:
            print(
                f"{suite}: reachable_fraction={summary['reachable_fraction']} "
                f"(decided={summary['n_decided']}, unknown={summary['n_unknown_budget']})"
            )
    print(f"wrote {args.json_out} and {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
