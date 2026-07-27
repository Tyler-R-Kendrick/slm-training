#!/usr/bin/env python3
"""VAR0-02 (SLM-423): per-variant edit-space reachability matrix.

Re-scopes ``reachable_fraction`` from a program-wide scalar to a
``(variant x suite)`` matrix, because reachability is a function of
``(action alphabet x seed x target set)`` -- the SLM-299/305 audit measured
exactly one variant's (``tree_edit_diffusion``) alphabet from one seed, and
that number was being read as a program-level verdict on patch-based
generation. This script attributes it correctly and reports the other two
registered variants (VAR0-01) as either genuinely unmeasured or genuinely
inapplicable -- never inferred from the tree-edit row.

  docs/design/var0-02-reachability-matrix-20260726.json
  docs/design/var0-02-reachability-matrix-20260726.md

Example:
  python -m scripts.run_var0_02_reachability_matrix
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.variants import VARIANTS
from slm_training.harnesses.experiments.slm299_edit_reachability import (
    TREE_EDIT_VARIANT_ID,
)
from scripts.run_slm299_reachability_audit import build_report, load_corpora
from slm_training.versioning import build_version_stamp

DEFAULT_JSON_OUT = Path("docs/design/var0-02-reachability-matrix-20260726.json")
DEFAULT_MD_OUT = Path("docs/design/var0-02-reachability-matrix-20260726.md")

COMPONENT = "harness.experiments.var0_02_reachability_matrix"

# Non-tree-edit registered variants: the recorded reason each is NOT_MEASURED
# today, distinguishing "deferred but feasible" from "structurally
# inapplicable." Conflating the two would be exactly the failure mode this
# issue exists to prevent one level down -- never infer a variant's row from
# another variant's engine, and never claim a structural non-goal is a mere
# TODO or vice versa.
NOT_MEASURED_REASONS: dict[str, tuple[str, str]] = {
    "repl_operators": (
        "not_measured_deferred",
        "Enumeration is feasible with existing primitives -- "
        "dsl/operators/legal_set.py::enumerate_operator_legal_set plus "
        "registry.py::OperatorLibraryV1.apply and "
        "registry.py::OperatorStateV1.from_source already compose exactly "
        "this pipeline in harnesses/train_data/operator_corpus.py:494-660 "
        "(source -> state -> context/library -> legal set -> per-action "
        "apply -> successor source, re-validated via "
        "validate_with_pack_authority at every step). A standalone "
        "enumerate_operator_successors() BFS engine analogous to "
        "_enumerate_children is out of this issue's scope (measurement and "
        "attribution only, per its own non-goals: 'changing any action "
        "alphabet' and building a new search engine are not the same "
        "thing, but a new engine is still new capability work) and is "
        "deferred to a follow-up issue.",
    ),
    "twotower_prompt_ast": (
        "not_applicable",
        "This variant has no discrete edit-action space to search: it "
        "denoises a fully-masked target directly via iterative MaskGIT-style "
        "denoising (models/twotower.py) rather than applying a bounded "
        "sequence of edits to a seed program. 'Bounded seed-to-target "
        "edit-path search' is not a meaningful measurement for this variant "
        "at all, independent of engineering effort -- this is a structural "
        "non-goal, not a deferred one.",
    ),
}


def _measured_cell(variant, summary: dict[str, Any]) -> dict[str, Any]:
    base = {
        "seed_id": variant.seed_policy_id,
        "alphabet_fingerprint": variant.action_alphabet_fingerprint,
    }
    if summary.get("status") == "corpus_unavailable":
        return {
            **base,
            "status": "corpus_unavailable",
            "reachable_fraction": None,
            "decided": None,
            "unknown_budget": None,
            "reason_histogram": {},
        }
    return {
        **base,
        "status": "measured",
        "reachable_fraction": summary["reachable_fraction"],
        "decided": summary["n_decided"],
        "unknown_budget": summary["n_unknown_budget"],
        "reason_histogram": summary["reason_histogram"],
    }


def _not_measured_cell(variant, status: str, reason: str) -> dict[str, Any]:
    return {
        "seed_id": variant.seed_policy_id,
        "alphabet_fingerprint": variant.action_alphabet_fingerprint,
        "status": status,
        "reason": reason,
        "reachable_fraction": None,
        "decided": None,
        "unknown_budget": None,
        "reason_histogram": {},
    }


def build_matrix(
    *, max_edits: int, node_budget: int, generated_at: str
) -> dict[str, Any]:
    corpora = load_corpora()
    tree_edit_report = build_report(
        corpora,
        max_edits=max_edits,
        node_budget=node_budget,
        generated_at=generated_at,
        mode="extended",
    )
    suites = sorted(corpora)

    matrix: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        if variant.variant_id == TREE_EDIT_VARIANT_ID:
            matrix[variant.variant_id] = {
                suite: _measured_cell(variant, tree_edit_report["suites"][suite])
                for suite in suites
            }
            continue
        status, reason = NOT_MEASURED_REASONS[variant.variant_id]
        matrix[variant.variant_id] = {
            suite: _not_measured_cell(variant, status, reason) for suite in suites
        }

    return {
        "schema": "reachability_matrix/v1",
        "generated_at": generated_at,
        "seed_source": tree_edit_report["seed_source"],
        "mode": tree_edit_report["mode"],
        "max_edits": max_edits,
        "node_budget": node_budget,
        "verdict_policy": (
            "reachable_fraction is computed over decided cases only; "
            "UNKNOWN_BUDGET cases are reported separately and are never "
            "counted as unreachable; suites without a corpus are "
            "corpus_unavailable, never zero-reachable. Reachability is a "
            "space-coverage proof, never a model-quality claim, and is "
            "scoped to one (variant, suite) cell -- never cited as a "
            "program-wide status (decode-invariants.md I12/I14)."
        ),
        "matrix": matrix,
        "version_stamp": build_version_stamp(COMPONENT),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# VAR0-02 (SLM-423): per-variant edit-space reachability matrix",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- seed: `{payload['seed_source']}`",
        f"- mode: `{payload['mode']}`, max_edits: {payload['max_edits']}, "
        f"node_budget: {payload['node_budget']}",
        f"- {payload['verdict_policy']}",
        "",
        "> Reachability is space coverage, not model quality, and is scoped",
        "> to one (variant, suite) cell. A `reachable_fraction=0.0` for one",
        "> variant says nothing about any other registered variant.",
        "",
    ]
    suites = sorted({s for rows in payload["matrix"].values() for s in rows})
    for variant_id, rows in payload["matrix"].items():
        lines += [f"## `{variant_id}`", ""]
        any_row = next(iter(rows.values()))
        lines.append(f"- alphabet_fingerprint: `{any_row['alphabet_fingerprint']}`")
        lines.append(f"- seed_id: `{any_row['seed_id']}`")
        lines.append("")
        lines.append("| suite | status | reachable_fraction | decided | unknown_budget |")
        lines.append("| --- | --- | --- | --- | --- |")
        for suite in suites:
            cell = rows[suite]
            lines.append(
                f"| {suite} | {cell['status']} | {cell['reachable_fraction']} | "
                f"{cell['decided']} | {cell['unknown_budget']} |"
            )
        reasons = {rows[s].get("reason") for s in suites if rows[s].get("reason")}
        for reason in sorted(reasons):
            lines.append("")
            lines.append(f"> {reason}")
        histograms = {
            s: rows[s]["reason_histogram"] for s in suites if rows[s]["reason_histogram"]
        }
        if histograms:
            lines.append("")
            lines.append("Reason-code histograms:")
            for suite, hist in histograms.items():
                lines.append(f"- **{suite}**: `{json.dumps(hist, sort_keys=True)}`")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-edits", type=int, default=8)
    parser.add_argument("--node-budget", type=int, default=120)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = build_matrix(
        max_edits=args.max_edits, node_budget=args.node_budget, generated_at=generated_at
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    for variant_id, rows in payload["matrix"].items():
        statuses = {suite: cell["status"] for suite, cell in rows.items()}
        print(f"{variant_id}: {statuses}")
    print(f"wrote {args.json_out} and {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
