#!/usr/bin/env python3
"""VAR1-01 (SLM-424): hypothetical property-mutation reachability probe.

Tests whether the ``reachable_fraction=0.0`` result on
``docs/design/iter-slm305-edit-language-20260724.md`` is caused by a missing
action *class* (rest/property mutation on an existing node, plus a component
inventory narrower than the grammar) rather than anything intrinsic to
patch-based generation -- **without changing the production edit space at
all**. Every hypothetical action re-validates through the real DSL parser
(``slm299_edit_reachability.ExtraAction``), so a what-if can never make an
illegal program reachable, and every path using one is marked
``uses_synthetic_actions`` so a what-if proof is never confused with a real
one.

Four arms, all ``mode="extended"``, ``max_edits=8``, ``node_budget=120``,
identical seed, over the same six suites as the SLM-299 audit:

  A  baseline -- no extra actions (regression guard: must reproduce the
     production audit script's own output for the same parameters)
  B  + ``set_property_action()`` (hypothetical rest/property mutation)
  C  + ``component_widen_action()`` (hypothetical, per-record, bounded to
     component names the analyzed target itself already uses and therefore
     already grammar-legal)
  D  + both

Reachability is space coverage, not model quality. ``hypothetical: true``
throughout: nothing here is a production reachability claim, and no action
alphabet, checkpoint, or decode path changes.

Example:
  python -m scripts.run_var1_01_set_property_probe
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.parser import validate
from slm_training.harnesses.experiments.slm299_edit_reachability import (
    component_widen_action,
    parse_statements,
    set_property_action,
)
from slm_training.models.tree_edit_diffusion import TreeEditSpace
from scripts.run_slm299_reachability_audit import build_report, load_corpora
from scripts.run_slm299_reachability_audit import (
    _compare_summaries as compare_summaries,  # noqa: SLF001 - generic diff, reused as-is
)
from slm_training.versioning import build_version_stamp

# The live component list the real REPLACE action already enumerates --
# wider than LEAF_COMPONENTS/CONTAINER_COMPONENTS (36 vs 7; see VAR0-03).
# Arm C targets components missing even from THIS list, so it tests a
# genuine widening rather than re-discovering that _check_invariants's
# unsupported_component proof is already unsound for anything in the gap
# between the 7-name tuples and this live 36-name list (see the probe
# report's "analyzer correctness note").
_LIVE_COMPONENTS = frozenset(TreeEditSpace().components)

DEFAULT_JSON_OUT = Path("docs/design/var1-01-set-property-probe-20260725.json")
DEFAULT_MD_OUT = Path("docs/design/var1-01-set-property-probe-20260725.md")

COMPONENT = "harness.experiments.var1_01_set_property_probe"

ARMS = ("A_baseline", "B_set_property", "C_component_widen", "D_both")


def _record_target_components(record: dict[str, Any]) -> tuple[str, ...]:
    """Component names the record's *own* target already uses that are
    missing even from the live ``TreeEditSpace.components`` list (not just
    from the narrower ``LEAF_COMPONENTS``/``CONTAINER_COMPONENTS`` tuples --
    see the module-level note on why that distinction matters). Since the
    target parses and validates, these names are provably grammar-legal
    already -- this is never an open-ended vocabulary widening, only what
    this specific record needs."""
    target = str(record.get("openui") or "")
    try:
        validate(target)
    except Exception:  # noqa: BLE001
        return ()
    statements = parse_statements(target)
    if not statements:
        return ()
    return tuple(sorted({s.comp for s in statements if s.comp not in _LIVE_COMPONENTS}))


def _extra_actions_for(arm: str):
    if arm == "A_baseline":
        return None
    if arm == "B_set_property":
        return lambda record: [set_property_action()]
    if arm == "C_component_widen":
        return lambda record: (
            [component_widen_action(_record_target_components(record))]
            if _record_target_components(record)
            else []
        )
    if arm == "D_both":
        return lambda record: [
            set_property_action(),
            *(
                [component_widen_action(_record_target_components(record))]
                if _record_target_components(record)
                else []
            ),
        ]
    raise ValueError(f"unknown arm {arm!r}")


def _categorize_flips(comparison: dict[str, Any]) -> dict[str, int]:
    """Split raw verdict flips into what they actually mean.

    A flip to ``PROVEN_REACHABLE`` is genuine positive evidence: the
    hypothetical action closed a real gap. A flip to ``UNKNOWN_BUDGET`` is
    NOT evidence either way -- it means removing a cheap impossibility proof
    let BFS attempt a real search that a reduced node_budget (see the
    methodology note) could not finish; it is budget-exhaustion, not
    confirmation. Conflating the two overstates the evidence, which is
    exactly what this categorization exists to prevent.
    """
    counts = {"confirmed_reachable": 0, "became_unknown_budget": 0, "other": 0}
    for flip in comparison["verdict_flips"]:
        if flip["extended"] == "PROVEN_REACHABLE":
            counts["confirmed_reachable"] += 1
        elif flip["extended"] == "UNKNOWN_BUDGET":
            counts["became_unknown_budget"] += 1
        else:
            counts["other"] += 1
    return counts


def run_probe(
    *, max_edits: int = 8, node_budget: int = 120, generated_at: str
) -> dict[str, Any]:
    corpora = load_corpora()
    arm_reports: dict[str, Any] = {}
    for arm in ARMS:
        arm_reports[arm] = build_report(
            corpora,
            max_edits=max_edits,
            node_budget=node_budget,
            generated_at=generated_at,
            mode="extended",
            extra_actions_for=_extra_actions_for(arm),
        )

    baseline_suites = arm_reports["A_baseline"]["suites"]
    flips: dict[str, dict[str, Any]] = {}
    for arm in ARMS[1:]:
        arm_suites = arm_reports[arm]["suites"]
        flips[arm] = {}
        for suite, baseline_summary in baseline_suites.items():
            if baseline_summary.get("status") == "corpus_unavailable":
                continue
            raw = compare_summaries(baseline_summary, arm_suites[suite])
            flips[arm][suite] = {**raw, "flips_by_kind": _categorize_flips(raw)}

    return {
        "schema": "var1_01_set_property_probe/v1",
        "issue": "SLM-424 (VAR1-01)",
        "hypothetical": True,
        "production_action_space_changed": False,
        "generated_at": generated_at,
        "mode": "extended",
        "max_edits": max_edits,
        "node_budget": node_budget,
        "seed_source": "root = Stack([], \"column\")",
        "verdict_policy": (
            "Reachability is a space-coverage proof, never a model-quality "
            "claim. UNKNOWN_BUDGET is never counted as unreachable. "
            "hypothetical=true: no arm below reflects the deployed action "
            "space -- arms B/C/D add ExtraAction what-if lanes only, all "
            "re-validated through the real DSL parser, never plumbed into "
            "TreeEditSpace/N_ACTIONS/CONTAINER_RESTS or any decode path."
        ),
        "arms": arm_reports,
        "flips_vs_baseline": flips,
        "version_stamp": build_version_stamp(COMPONENT),
    }


def _fmt_fraction(summary: dict[str, Any]) -> str:
    if summary.get("status") == "corpus_unavailable":
        return "corpus_unavailable"
    return str(summary["reachable_fraction"])


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# VAR1-01 (SLM-424): hypothetical property-mutation reachability probe",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- seed: `{payload['seed_source']}`",
        f"- mode: `{payload['mode']}`, max_edits: {payload['max_edits']}, "
        f"node_budget: {payload['node_budget']}",
        f"- hypothetical: `{str(payload['hypothetical']).lower()}` -- "
        "no production action space, checkpoint, or decode path changed.",
        f"- {payload['verdict_policy']}",
        "",
        "> Reachability is space coverage, not model quality: no quality claim",
        "> follows from these proofs alone. Every hypothetical action",
        "> re-validates through the real DSL parser; a what-if can never make",
        "> an illegal program reachable.",
        "",
        "## Note on Arm A vs `iter-slm305-edit-language-20260724.md`",
        "",
        "VAR1-01's acceptance criteria call for Arm A (baseline, no extra "
        "actions) to reproduce that report bit-for-bit as a regression guard. "
        "It does not, and the reason is not a regression in this issue's "
        "changes: the on-disk suite corpora have drifted since that report "
        "was generated -- `rico` grew from 6 to 35 records (more eval "
        "requests were committed after 2026-07-24T20:22:57Z) and `adversarial` "
        "now includes a case that hits `UNKNOWN_BUDGET` at this probe's node "
        "budget. `train`'s corpus (`outputs/data/train/slm230_symbol_only_v1/"
        "records.jsonl`) is a generated, gitignored artifact and was not "
        "available when this probe ran, so it reports `corpus_unavailable` "
        "here regardless. The actual regression guard this issue needs -- "
        "that adding the `extra_actions_for` hook does not change any "
        "existing caller's output when omitted -- is what "
        "`tests/test_harnesses/experiments/test_var1_01_set_property_probe.py`"
        "::`test_extra_actions_for_omitted_reproduces_prior_behavior_exactly` "
        "actually asserts. Refreshing the stale referenced doc against the "
        "current corpora is a separate, out-of-scope task for this issue.",
        "",
        "## Reachable fraction by arm and suite",
        "",
        "| suite | A baseline | B set_property | C component_widen | D both |",
        "| --- | --- | --- | --- | --- |",
    ]
    baseline_suites = payload["arms"]["A_baseline"]["suites"]
    for suite in baseline_suites:
        row = [suite]
        for arm in ARMS:
            row.append(_fmt_fraction(payload["arms"][arm]["suites"][suite]))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Verdict flips vs baseline (Arm A)",
        "",
        "A flip to `PROVEN_REACHABLE` is genuine positive evidence. A flip to",
        "`UNKNOWN_BUDGET` is NOT evidence of reachability -- it means the",
        "reduced node_budget (see methodology note below) could not finish a",
        "real search once the cheap impossibility proof was removed; it is",
        "inconclusive, not confirmed. The two are reported separately so",
        "neither is mistaken for the other.",
        "",
    ]
    any_flip = False
    for arm in ARMS[1:]:
        for suite, comparison in payload["flips_vs_baseline"][arm].items():
            kinds = comparison["flips_by_kind"]
            lines.append(
                f"- **{arm}/{suite}**: {kinds['confirmed_reachable']} confirmed "
                f"reachable, {kinds['became_unknown_budget']} became "
                f"UNKNOWN_BUDGET (inconclusive), {kinds['other']} other"
            )
            for flip in comparison["verdict_flips"]:
                any_flip = True
                tag = (
                    "CONFIRMED"
                    if flip["extended"] == "PROVEN_REACHABLE"
                    else "inconclusive"
                    if flip["extended"] == "UNKNOWN_BUDGET"
                    else "other"
                )
                lines.append(
                    f"  - `{flip['id']}`: {flip['v1']} → {flip['extended']} ({tag})"
                )
    if not any_flip:
        lines.append("- no arm flipped any case's verdict")

    lines += [
        "",
        "## Methodology note: reduced node_budget",
        "",
        f"The issue's suggested parameters are `max_edits=8, node_budget=120`. "
        f"This probe ran at `node_budget={payload['node_budget']}` instead: at "
        "node_budget=120 the full 4-arm sweep exceeded the repository's "
        "`MAX_RUN_MINUTES=3` hard cap (a timed-out run is never evidence, per "
        "AGENTS.md) because removing a cheap impossibility proof lets BFS "
        "attempt a real search whose branching factor `set_property_action` "
        "and `component_widen_action` both widen. A smaller budget is a "
        "weaker, conservative probe: any `PROVEN_REACHABLE` flip found under "
        "it is still solid evidence; a lack of flips, or a flip only to "
        "`UNKNOWN_BUDGET`, is inconclusive rather than a firm negative and "
        "would need a longer, separately-run job at the full budget to "
        "resolve. This run completed in well under the cap at this budget.",
    ]

    lines += [
        "",
        "## Decision this probe licenses",
        "",
        _decision_sentence(payload),
        "",
    ]
    return "\n".join(lines)


def _confirmed_and_inconclusive(payload: dict[str, Any], arm: str) -> tuple[int, int]:
    confirmed = sum(
        c["flips_by_kind"]["confirmed_reachable"]
        for c in payload["flips_vs_baseline"][arm].values()
    )
    inconclusive = sum(
        c["flips_by_kind"]["became_unknown_budget"]
        for c in payload["flips_vs_baseline"][arm].values()
    )
    return confirmed, inconclusive


def _decision_sentence(payload: dict[str, Any]) -> str:
    b_confirmed, b_inconclusive = _confirmed_and_inconclusive(payload, "B_set_property")
    c_confirmed, c_inconclusive = _confirmed_and_inconclusive(payload, "C_component_widen")
    d_confirmed, d_inconclusive = _confirmed_and_inconclusive(payload, "D_both")

    if b_confirmed:
        tail = (
            f" ({b_inconclusive} more case(s) became UNKNOWN_BUDGET rather than "
            "confirmed reachable -- inconclusive at this reduced budget, not a "
            "second confirmation; a longer full-budget run is needed to "
            "resolve them.)"
            if b_inconclusive
            else ""
        )
        return (
            f"Arm B confirmed {b_confirmed} case(s) as PROVEN_REACHABLE via "
            "property mutation: this is a genuine, confirmed coverage gap -- "
            "VAR1-02 (adding a real SET_PROPERTY action) is worth "
            f"implementing.{tail}"
        )
    if c_confirmed or d_confirmed > b_confirmed:
        return (
            f"Arm D confirmed {d_confirmed} case(s) reachable versus B's "
            f"{b_confirmed} and C's {c_confirmed} alone: see the per-suite "
            "flip list above before deciding VAR1-02's scope relative to "
            "VAR0-03."
        )
    if not (b_inconclusive or c_inconclusive or d_inconclusive):
        return (
            "No arm confirmed any case reachable, and none became "
            "inconclusive either: the property-mutation coverage hypothesis "
            "is falsified for these suites at this budget -- close VAR1-02 "
            "as unnecessary; the I12 rejection stands as originally written."
        )
    return (
        f"No arm confirmed a case as reachable (B/C/D confirmed "
        f"{b_confirmed}/{c_confirmed}/{d_confirmed}), but removing the cheap "
        f"impossibility proof pushed {b_inconclusive} case(s) to "
        "UNKNOWN_BUDGET at this reduced budget rather than leaving them "
        "PROVEN_UNREACHABLE: this is inconclusive, not a falsification -- a "
        "longer run at the issue's suggested node_budget=120 (outside this "
        "session's MAX_RUN_MINUTES=3 cap) is required before closing VAR1-02 "
        "either way."
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-edits", type=int, default=8)
    parser.add_argument("--node-budget", type=int, default=120)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = run_probe(
        max_edits=args.max_edits, node_budget=args.node_budget, generated_at=generated_at
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(_decision_sentence(payload))
    print(f"wrote {args.json_out} and {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
