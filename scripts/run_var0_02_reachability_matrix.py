#!/usr/bin/env python3
"""SLM-423 (VAR0-02): per-variant x suite edit-reachability matrix.

Re-scopes the SLM-299/SLM-305 reachability audit from a single program-wide
scalar to a matrix keyed by *(registered variant, suite)*. The SLM-299
analyzer's ``reachable_fraction = 0.0`` result was measured for exactly one
variant -- the tree-edit diffusion alphabet (v1/extended generations of
``TreeEditSpace``), started from the standard X22 minimal seed. It was never
measured for ``repl_operators`` (the operator/REPL surface enumerable through
``dsl/operators/legal_set.py``) or for ``twotower_prompt_ast``. Citing that
one variant's number as a program-wide verdict on patch-based generation is
exactly the failure mode I14 exists to prevent, one level up: I14 stops an
approach's rejection from closing a goal; this matrix stops a variant's
measurement from closing an approach.

Rows = every variant in ``slm_training.dsl.variants.VARIANTS`` (SLM-422,
VariantContractV1). Columns = the six standard suites (train / smoke /
held_out / adversarial / ood / rico). Each cell carries ``reachable_fraction,
decided, unknown_budget, reason_histogram, seed_id, alphabet_fingerprint`` and
a ``status`` of ``measured`` / ``corpus_unavailable`` / ``NOT_MEASURED``.

Verdict policy is unchanged from SLM-299: ``UNKNOWN_BUDGET`` is never counted
as unreachable, and a suite with no corpus is ``corpus_unavailable``, never
zero-reachable. This issue adds one more never: a variant whose enumeration
engine does not exist yet is ``NOT_MEASURED``, never inferred from another
variant's row and never a fabricated number.

Only the ``tree_edit_diffusion`` row is measured here -- by reusing the
existing SLM-299 audit's ``build_report`` (same seed, same verdict policy,
same corpora) rather than re-implementing it. ``repl_operators`` and
``twotower_prompt_ast`` are ``NOT_MEASURED`` with a real, specific reason each
(see ``NOT_MEASURED_REASONS`` below); this issue is measurement/attribution
only and explicitly does not implement a new enumeration engine or assert
either unmeasured alphabet has better or worse coverage than the tree-edit
row.

Example:
  python -m scripts.run_var0_02_reachability_matrix
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.variants import VARIANTS, VariantContractV1
from slm_training.harnesses.experiments.slm299_edit_reachability import (
    DEFAULT_SEED_SOURCE,
    TREE_EDIT_VARIANT,
)
from slm_training.versioning import build_version_stamp
from scripts.run_slm299_reachability_audit import build_report, load_corpora

EXPERIMENT_ID = "var0-02-reachability-matrix"
SCHEMA = "reachability_matrix/v1"
COMPONENT = "harness.experiments.var0_02_reachability_matrix"

SUITES: tuple[str, ...] = ("train", "smoke", "held_out", "adversarial", "ood", "rico")

STATUS_MEASURED = "measured"
STATUS_CORPUS_UNAVAILABLE = "corpus_unavailable"
STATUS_NOT_MEASURED = "NOT_MEASURED"

DEFAULT_JSON_OUT = Path("docs/design/var0-02-reachability-matrix-20260725.json")
DEFAULT_MD_OUT = Path("docs/design/var0-02-reachability-matrix-20260725.md")

# Real, specific reasons -- never a placeholder, never inferred from the
# tree-edit row. Each names the concrete engineering gap that keeps this
# variant's enumeration out of this issue's scope.
NOT_MEASURED_REASONS: dict[str, str] = {
    "repl_operators": (
        "successor enumeration for this variant requires "
        "dsl/operators/legal_set.py's enumerate_operator_legal_set, which "
        "operates over OperatorStateV1 / ReferenceTableV1 / "
        "ApplicationProvenanceV1 built from the live operator registry -- a "
        "different state representation and search than TreeEditSpace's "
        "statement list. Wiring a seed->target BFS over that alphabet (build "
        "an OperatorStateV1 per corpus record, enumerate legal operator "
        "actions, apply each to get a successor source, compare canonical "
        "forms against the target) is a new enumeration engine, which this "
        "issue's explicit scope excludes ('changing any action alphabet... "
        "measurement and attribution only'; 'if it is not measured in this "
        "issue, it is NOT_MEASURED'). Recorded here rather than left blank "
        "or inferred from the tree-edit row."
    ),
    "twotower_prompt_ast": (
        "this variant has no edit-action alphabet to measure: it is a "
        "full-AST prompt->AST denoiser over structural tokens, not a "
        "seed->target edit-transition system, so 'edit reachability' as this "
        "analyzer defines it (bounded BFS over an action alphabet from a "
        "minimal seed) does not apply to it. Recorded as NOT_MEASURED rather "
        "than a fabricated number or a blank cell."
    ),
}

VERDICT_POLICY = (
    "reachable_fraction is computed over decided cases only; UNKNOWN_BUDGET "
    "cases are never counted as unreachable; a suite with no corpus is "
    "corpus_unavailable, never zero-reachable; a variant with no enumeration "
    "engine wired in this issue is NOT_MEASURED, never inferred from another "
    "variant's row and never a fabricated number. Reachability is a "
    "space-coverage proof, never a model-quality claim."
)


def _seed_id(seed_source: str) -> str:
    return hashlib.sha256(seed_source.encode("utf-8")).hexdigest()[:16]


def _not_measured_cell(variant: VariantContractV1, suite: str) -> dict[str, Any]:
    return {
        "status": STATUS_NOT_MEASURED,
        "variant_id": variant.variant_id,
        "suite": suite,
        "reachable_fraction": None,
        "decided": None,
        "unknown_budget": None,
        "reason_histogram": None,
        "seed_id": None,
        "alphabet_fingerprint": variant.action_alphabet_fingerprint,
        "reason": NOT_MEASURED_REASONS[variant.variant_id],
    }


def _tree_edit_row(
    *,
    max_edits: int,
    node_budget: int,
    generated_at: str,
    corpora: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """The only measured row: reuse the SLM-299 audit's own ``build_report``
    (same seed, corpora, and verdict policy) rather than a parallel
    reimplementation. ``corpora`` defaults to the live suite corpora
    (``load_corpora()``); tests pass a frozen fixture corpus instead so the
    matrix's row-building logic can be checked without depending on
    whatever happens to be on disk."""
    if corpora is None:
        corpora = load_corpora()
    slm299_report = build_report(
        corpora,
        max_edits=max_edits,
        node_budget=node_budget,
        generated_at=generated_at,
        mode="extended",
    )
    seed_id = _seed_id(DEFAULT_SEED_SOURCE)
    fingerprint = TREE_EDIT_VARIANT.action_alphabet_fingerprint
    row: dict[str, dict[str, Any]] = {}
    for suite in SUITES:
        summary = slm299_report["suites"][suite]
        if summary.get("status") == STATUS_CORPUS_UNAVAILABLE:
            row[suite] = {
                "status": STATUS_CORPUS_UNAVAILABLE,
                "variant_id": TREE_EDIT_VARIANT.variant_id,
                "suite": suite,
                "reachable_fraction": None,
                "decided": 0,
                "unknown_budget": 0,
                "reason_histogram": {},
                "seed_id": seed_id,
                "alphabet_fingerprint": fingerprint,
                "reason": None,
            }
            continue
        row[suite] = {
            "status": STATUS_MEASURED,
            "variant_id": TREE_EDIT_VARIANT.variant_id,
            "suite": suite,
            "reachable_fraction": summary["reachable_fraction"],
            "decided": summary["n_decided"],
            "unknown_budget": summary["n_unknown_budget"],
            "reason_histogram": summary["reason_histogram"],
            "seed_id": seed_id,
            "alphabet_fingerprint": fingerprint,
            "reason": None,
        }
    return row, slm299_report


def build_matrix(
    *,
    max_edits: int = 8,
    node_budget: int = 120,
    generated_at: str,
    corpora: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build ``ReachabilityMatrixV1``: rows = every registered variant,
    columns = the six standard suites. Only ``tree_edit_diffusion`` is
    measured; every other registered variant is ``NOT_MEASURED`` with a real
    reason (see module docstring). ``corpora`` defaults to the live suite
    corpora; pass a fixture dict for deterministic, disk-independent tests."""
    matrix: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        if variant.variant_id == TREE_EDIT_VARIANT.variant_id:
            row, _slm299_report = _tree_edit_row(
                max_edits=max_edits,
                node_budget=node_budget,
                generated_at=generated_at,
                corpora=corpora,
            )
        else:
            row = {suite: _not_measured_cell(variant, suite) for suite in SUITES}
        matrix[variant.variant_id] = row

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "generated_at": generated_at,
        "seed_source": DEFAULT_SEED_SOURCE,
        "max_edits": max_edits,
        "node_budget": node_budget,
        "columns": list(SUITES),
        "rows": [entry.variant_id for entry in VARIANTS],
        "verdict_policy": VERDICT_POLICY,
        "matrix": matrix,
        "source_command": (
            "python -m scripts.run_slm299_reachability_audit "
            f"--max-edits {max_edits} --node-budget {node_budget}"
        ),
        "version_stamp": build_version_stamp(COMPONENT),
    }


def _fmt_cell(cell: dict[str, Any]) -> str:
    status = cell["status"]
    if status == STATUS_MEASURED:
        return f"{cell['reachable_fraction']} (n={cell['decided']}, unk={cell['unknown_budget']})"
    if status == STATUS_CORPUS_UNAVAILABLE:
        return "corpus_unavailable"
    return "NOT_MEASURED"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# VAR0-02 (SLM-423): per-variant x suite edit-reachability matrix",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- seed: `{payload['seed_source']}`",
        f"- max_edits: {payload['max_edits']}, node_budget: {payload['node_budget']}",
        f"- verdict policy: {payload['verdict_policy']}",
        "",
        "> `reachable_fraction` is a property of a *(variant, suite)* pair, "
        "not of the program. A number measured for one variant's action "
        "alphabet may never be cited as a verdict on any other variant, or "
        "on patch-based generation in general.",
        "",
        "## Matrix",
        "",
        "| variant | " + " | ".join(SUITES) + " |",
        "| --- | " + " | ".join(["---"] * len(SUITES)) + " |",
    ]
    for variant_id in payload["rows"]:
        row = payload["matrix"][variant_id]
        cells = [_fmt_cell(row[suite]) for suite in SUITES]
        lines.append(f"| `{variant_id}` | " + " | ".join(cells) + " |")

    lines += ["", "## Alphabet fingerprints", ""]
    for variant_id in payload["rows"]:
        row = payload["matrix"][variant_id]
        fingerprint = next(iter(row.values()))["alphabet_fingerprint"]
        lines.append(f"- `{variant_id}`: `{fingerprint}`")

    lines += ["", "## NOT_MEASURED reasons", ""]
    for variant_id in payload["rows"]:
        row = payload["matrix"][variant_id]
        reasons = {cell["reason"] for cell in row.values() if cell["reason"]}
        for reason in sorted(reasons):
            lines.append(f"- `{variant_id}`: {reason}")
    if not any(
        cell["reason"]
        for row in payload["matrix"].values()
        for cell in row.values()
    ):
        lines.append("- (none -- every row measured)")

    lines += [
        "",
        "## Reason-code histograms (measured rows only)",
        "",
    ]
    for variant_id in payload["rows"]:
        row = payload["matrix"][variant_id]
        if all(cell["status"] != STATUS_MEASURED for cell in row.values()):
            continue
        lines.append(f"- **{variant_id}**:")
        for suite in SUITES:
            cell = row[suite]
            if cell["status"] != STATUS_MEASURED:
                continue
            lines.append(
                f"  - {suite}: `{json.dumps(cell['reason_histogram'], sort_keys=True)}`"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-edits", type=int, default=8)
    # 120, not the analyzer's own 800 default: at the current corpus sizes
    # (rico grew to 35 records since 20260724) node_budget=800 takes ~2m20s,
    # too close to MAX_RUN_MINUTES=3 for comfort; 120 (the same budget
    # iter-slm305-edit-language-20260724.md was generated at) finishes in
    # ~20s with an identical verdict policy.
    parser.add_argument("--node-budget", type=int, default=120)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = build_matrix(
        max_edits=args.max_edits, node_budget=args.node_budget, generated_at=generated_at
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    for variant_id in payload["rows"]:
        row = payload["matrix"][variant_id]
        statuses = {cell["status"] for cell in row.values()}
        print(f"{variant_id}: {sorted(statuses)}")
    print(f"wrote {args.json_out} and {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
