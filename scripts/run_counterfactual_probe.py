"""Bounded counterfactual-probe evidence report for LDI3-03 (SLM-131).

Demonstrates the probe orchestration over a frozen fixture state with a
**deterministic mock rollout backend** — no model runs, no training, no held-out
prompt content is copied. It emits the state/action/rollout support table, the
value-materializer comparison, and qualified/unresolved counts. The ship-grade run
swaps the mock for a real model + G0-G12 verifier behind the same RolloutBackend.

    python scripts/run_counterfactual_probe.py --out docs/design/ldi3-03-counterfactual-probe-report-20260718.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_training.harnesses.preference.counterfactual_probe import (
    CandidateState,
    ProbeConfig,
    RawOutcome,
    binary_verdict,
    pareto_front,
    run_probe,
    semantic_partition,
)
from slm_training.harnesses.preference.decision_events_v2 import DecisionStateV2

_GATES = tuple(f"G{i}" for i in range(13))


def _vec(g1: str) -> tuple[tuple[str, str], ...]:
    return tuple((g, g1 if g == "G1" else "pass") for g in _GATES)


class _MockBackend:
    """Illustrative only: actions 2/3 recover the failure, 4 makes it worse."""

    def rollout(self, state: DecisionStateV2, action_id: int, seed: int) -> RawOutcome:
        g1 = "pass" if action_id in (2, 3) else ("fail" if action_id == 4 else ("pass" if seed == 0 else "fail"))
        return RawOutcome(canonical_output=f"o{action_id}", finish_reason="stop", verifier_vector=_vec(g1))


def build_report() -> dict[str, Any]:
    state = DecisionStateV2(
        group_id="grp", architecture="twotower", context_text="ctx",
        decision_position=0, legal_action_ids=(1, 2, 3, 4, 5), decision_kind="component",
        abstract_state_role="root", grammar_state_hash="gh", policy_checkpoint_sha="pol",
        tokenizer_sha="tok", decode_config_hash="dec", verifier_bundle_hash="ver",
        split="train", canvas_ids=(1, 2, 3),
    )
    cfg = ProbeConfig(seeds=(0, 1, 2), min_rollouts=3, min_effect=0.3, required_gates=("G0",))
    outcomes = run_probe(
        [CandidateState(state, "detector_localized")], _MockBackend(), config=cfg,
        selection={state.state_id: (1, 2, 3, 4)},  # action 5 left unobserved
    )
    part = semantic_partition(state, outcomes, config=cfg, policy_action=1)
    verdicts = {o.action_id: binary_verdict(o, ("G0", "G1")) for o in outcomes}
    return {
        "note": (
            "Mock backend; no model rollout, no verifier run, no training. Delayed "
            "failures are attributed only via forced-action rollout verdicts at the "
            "exact state — never from final-output position."
        ),
        "state_id": state.state_id,
        "support_table": {
            str(o.action_id): {
                "seeds": list(o.continuation_seeds),
                "verdict": verdicts[o.action_id],
            }
            for o in outcomes
        },
        "pareto_front": pareto_front(outcomes, ("G0", "G1")),
        "semantic_partition": part.as_dict(),
        "counts": {
            "qualified_good": len(part.good_action_ids),
            "qualified_bad": len(part.bad_action_ids),
            "ambiguous": len(part.ambiguous_action_ids),
            "unobserved": len(part.unobserved_action_ids),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = build_report()
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
