#!/usr/bin/env python3
"""Publish the DSH3-18 CAP2 operator-policy rebase addendum (SLM-393).

Design-only: reads the immutable frozen DSH3-13 CAP2 report and this
session's reproducibility check, and writes the addendum JSON + narrative
Markdown named exactly as the issue's implementation contract requires:
``docs/design/cap2-operator-policy-rebase.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_training.evals.cap2_operator_policy_rebase import (
    build_cap2_operator_policy_rebase,
)
from slm_training.harness_core.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
CAP2_REPORT = ROOT / "docs/design/dsh3-13-cap2-operator-eval-20260723/report.json"

#: This session's honestly-reported attempt to re-run the canonical DSH3-13
#: CAP2 suite (required by DSH3-18's "Required tests"). Two environment-only
#: blockers were found and fixed (missing openui_bridge node_modules; a
#: sandbox-global NODE_OPTIONS that breaks the bridge subprocess). After
#: fixing both, regeneration still does not reproduce the frozen operator
#: corpus fingerprint. This is recorded as an open, unresolved reproducibility
#: gap -- not fixed here, since fixing corpus-generation determinism would be
#: a runtime behavior change, which is one of DSH3-18's explicit non-goals.
REPRODUCIBILITY_NOTE: dict[str, Any] = {
    "attempted": True,
    "reproduced": False,
    "commands": [
        "cd src/apps/openui_bridge && npm ci",
        "NODE_OPTIONS= python -m pytest -q tests/test_evals/test_cap2_operator.py",
    ],
    "blockers_fixed_this_session": [
        "src/apps/openui_bridge/node_modules were not installed",
        "a sandbox-global NODE_OPTIONS=--import tsx breaks the bridge "
        "subprocess invocation in src/slm_training/dsl/operators/registry.py",
    ],
    "residual_error": (
        "ValueError: CAP2 generated operator corpus drifted "
        "(src/slm_training/evals/cap2_operator.py:365) -- the regenerated "
        "content_fingerprint does not match the frozen manifest fingerprint "
        "even after both environment blockers are fixed"
    ),
    "disposition": (
        "Treated as an open reproducibility gap of the DSH3-13 authority "
        "itself, not a DSH3-18 finding to fix. The frozen "
        "docs/design/dsh3-13-cap2-operator-eval-20260723/report.json remains "
        "the evidence of record (adversarial control: prove with behavior, "
        "never mutate the frozen suite). A dedicated follow-up should bisect "
        "why build_symbolic_operator_corpus no longer reproduces its frozen "
        "fingerprint in a clean environment."
    ),
}


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# DSH3-18 CAP2 operator-policy rebase addendum (SLM-393)",
        "",
        "Date: 2026-07-25",
        "Status: design-only addendum; no runtime behavior change; no E-number allocated",
        (
            "Scope: inventory, issue-disposition, and machine-readable "
            "dependency map over the frozen DSH3-13 CAP2 suite (SLM-381) and "
            "the terminal DSH3-17 disposition (SLM-385)"
        ),
        "",
        "## Decision",
        "",
        (
            "Kimi found the prior DSH3-14/15/17 plan stale by roughly 190 "
            "SLM-numbered experiments: much of the proposed scorer/eval "
            "surface already exists. This addendum inventories exactly what "
            "exists today (state-local heads, candidate representations, "
            "singleton bypass, the SLM-127 selector, collapse hard "
            "negatives, coverage semantics, telemetry), records update "
            "proposals for DSH3-14/15/17, and publishes a dependency map for "
            "every new DSH3 M5/M6/M7 rebase issue (DSH3-19 .. DSH3-33) so "
            "none of them start ahead of an unmet blocker or duplicate an "
            "existing callable."
        ),
        "",
        "## Stop-rule check",
        "",
        payload["disposition"]["stop_rule"]["reason"],
        "",
        "## Frozen suite identity (read-only; not regenerated or mutated)",
        "",
        "```json",
        json.dumps(payload["disposition"]["frozen_suite_identity"], indent=2, sort_keys=True),
        "```",
        "",
        "## Reproducibility check (DSH3-18 \"Required tests\")",
        "",
        (
            "This session ran `pytest -q tests/test_evals/test_cap2_operator.py` "
            "against the frozen DSH3-13 suite. Two environment-only blockers "
            "were found and fixed first:"
        ),
        "",
    ]
    for blocker in payload["disposition"]["reproducibility_note"][
        "blockers_fixed_this_session"
    ]:
        lines.append(f"- {blocker}")
    lines.extend(
        [
            "",
            (
                "After fixing both, the suite still does not reproduce: "
                f"`{payload['disposition']['reproducibility_note']['residual_error']}`"
            ),
            "",
            payload["disposition"]["reproducibility_note"]["disposition"],
            "",
            "## Inventory",
            "",
            "| Symbol | Status | Anchor |",
            "| --- | --- | --- |",
        ]
    )
    for item in payload["disposition"]["inventory"]:
        lines.append(
            f"| `{item['symbol_id']}` | `{item['status']}` | "
            f"`{item['code_anchor']}` |"
        )
    lines.extend(["", "Notes:", ""])
    for item in payload["disposition"]["inventory"]:
        lines.append(f"- **`{item['symbol_id']}`**: {item['note']}")
    lines.extend(
        [
            "",
            "## Update proposals for DSH3-14/15/17",
            "",
            "| Alias | Issue | Action | Superseded by |",
            "| --- | --- | --- | --- |",
        ]
    )
    for note in payload["disposition"]["issue_notes"]:
        lines.append(
            f"| `{note['alias']}` | {note['issue_id']} | `{note['action']}` | "
            f"{note['superseded_by'] or '-'} |"
        )
    lines.extend(["", "Notes:", ""])
    for note in payload["disposition"]["issue_notes"]:
        lines.append(f"- **`{note['alias']}`**: {note['note']}")
    lines.extend(
        [
            "",
            "## Dependency map (DSH3-19 .. DSH3-33)",
            "",
            (
                "`blocked_by` lists only directly-observed Linear "
                "`blockedBy` edges (never a guessed transitive closure). "
                "`ready` is true only when every direct blocker is Done, In "
                "Review, or one of the external DSH3-13/DSH3-18 "
                "prerequisites."
            ),
            "",
            "| Alias | Issue | Status | Blocked by | Ready |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for node in payload["disposition"]["dependency_map"]:
        blocked = ", ".join(node["blocked_by"]) or "-"
        lines.append(
            f"| `{node['alias']}` | {node['issue_id']} | `{node['status']}` | "
            f"{blocked} | {'yes' if node['ready'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            (
                "As of this addendum, only `DSH3-19` and `DSH3-21` are "
                "ready; `DSH3-21` (SLM-396) had already started and reached "
                "In Review (PR #897) before `DSH3-18` was claimed in this "
                "session -- recorded here rather than silently corrected, "
                "since Linear state is the source of truth, not this doc."
            ),
            "",
            "## Non-goals confirmed",
            "",
            (
                "No runtime behavior changed. No frozen-suite edit. No "
                "quality/efficiency claim. No E-number allocated."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json", type=Path, default=ROOT / "docs/design/cap2-operator-policy-rebase.json"
    )
    parser.add_argument(
        "--output-md", type=Path, default=ROOT / "docs/design/cap2-operator-policy-rebase.md"
    )
    args = parser.parse_args(argv)

    stamp = build_version_stamp("evals.cap2_operator_policy_rebase")
    cap2_report = json.loads(CAP2_REPORT.read_text(encoding="utf-8"))
    rebase = build_cap2_operator_policy_rebase(
        repo_root=ROOT,
        cap2_report=cap2_report,
        reproducibility_note=REPRODUCIBILITY_NOTE,
        version_stamp=stamp,
    )
    disposition = rebase.to_dict()
    payload = {
        "schema": "cap2_operator_policy_rebase_report/v1",
        "issue": {"alias": "DSH3-18", "id": "SLM-393"},
        "disposition": disposition,
    }
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "inventory_items": len(disposition["inventory"]),
                "dependency_nodes": len(disposition["dependency_map"]),
                "ready_now": [
                    node["alias"]
                    for node in disposition["dependency_map"]
                    if node["ready"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
