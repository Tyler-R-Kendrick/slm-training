#!/usr/bin/env python3
"""SLM-317 (LAR2-06): valid-state repair advancement screen (AR→repair hybrid).

Runs the preregistered do-no-harm screen over matched arms on the frozen
SLM-155 fixture decision corpus (identical examples + seeds + budgets):

- ``ar_only`` — the SLM-155 AR legal-action path; final = AR program.
- ``repair_only`` — X22 value-guided beam decode from the minimal seed.
- ``ar_repair_historical`` — AR source → repair decode under the pre-LAR2
  config (``mutation_count`` value labels, legacy STOP-slot accounting). The
  pre-SLM-305 4-action edit space is NOT reproducible on this branch (the
  space was extended in place; no legacy-subset knob exists) — declared
  deviation, the arm keeps the historical value/stop config only.
- ``ar_repair_improved`` — AR source → repair decode under this branch's
  config (``bounded_distance`` oracle value labels + corrected STOP
  accounting), then the do-no-harm commit rule with the policy value head as
  the calibrated learned score.
- ``oracle_commit`` — commits exactly when the candidate's hard evidence is
  strictly better (non-promotable upper bound).

Every final is scored on the LAR1-02 unified ladder (official parse → output
contract → meaningful_program_v1). Paired per-example outcomes key on
(record_id, seed); invalid-over-valid counts are reported per arm and never
aggregated away. Advancement gates: Safety = 0 invalid-over-valid on the
frozen safety set; Value = paired improvement over AR with Wilson lower bound
above the preregistered minimum useful effect (0.05); Reachability/provenance
= LAR2 artifacts present (slm299 reachability, slm291 evidence bundles).
Exactly one disposition: repair_positive | repair_negative | inconclusive,
with an explicit LAR3 open/close statement.

Writes ``docs/design/iter-slm317-repair-hybrid-20260724.{json,md}`` and
durable commit decisions under ``outputs/slm317/``.

Example:
  python -m scripts.run_slm317_repair_hybrid --steps 8
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.evals.power_protocol import wilson_interval
from slm_training.harnesses.experiments.slm155_factorization_comparison import (
    _program_from_actions,
    paired_validity_table,
)
from slm_training.harnesses.experiments.slm317_repair_hybrid import (
    DISPOSITION_INCONCLUSIVE,
    DISPOSITION_NEGATIVE,
    DISPOSITION_POSITIVE,
    EXPERIMENT_ID,
    CommitDecision,
    commit,
    hard_evidence,
    oracle_commit,
    repair_decode,
    state_value,
)
from slm_training.models.legal_action_scorer import (
    LegalActionScorerConfig,
    make_fixture_decisions,
    train_fixture_scorer,
)
from slm_training.versioning import build_version_stamp

DEFAULT_JSON_OUT = Path("docs/design/iter-slm317-repair-hybrid-20260724.json")
DEFAULT_MD_OUT = Path("docs/design/iter-slm317-repair-hybrid-20260724.md")
DEFAULT_DECISIONS_OUT = Path("outputs/slm317/commit_decisions.jsonl")

ARMS = (
    "ar_only",
    "repair_only",
    "ar_repair_historical",
    "ar_repair_improved",
    "oracle_commit",
)
PROMOTABLE_ARMS = {"ar_only", "repair_only", "ar_repair_historical", "ar_repair_improved"}

# LAR2 reachability/provenance artifacts this screen builds on.
REACHABILITY_ARTIFACTS = (
    "docs/design/iter-slm299-edit-reachability-20260724.json",
    "docs/design/iter-slm299-edit-reachability-20260724.md",
    "docs/design/iter-slm291-evidence-bundles-20260724.json",
    "docs/design/iter-slm291-evidence-bundles-20260724.md",
)

# PREREGISTERED (locked before any run; deviations append-only/exploratory).
PREREGISTERED = {
    "min_useful_effect": 0.05,
    "safety_rule": (
        "invalid-over-valid count of ar_repair_improved vs ar_only on the "
        "frozen safety set (all eval examples) must be exactly 0"
    ),
    "value_rule": (
        "paired hard-valid improvement of ar_repair_improved over ar_only: "
        "Wilson 95% lower bound of improvements/n_paired must exceed 0.05 "
        "with zero damages"
    ),
    "disposition_rule": (
        "repair_positive iff safety AND value AND reachability gates pass; "
        "repair_negative iff safety fails OR the value Wilson UPPER bound is "
        "below the minimum useful effect (effect ruled out); else inconclusive"
    ),
    "historical_deviation": (
        "pre-SLM-305 4-action edit space is not reproducible on this branch "
        "(edit space extended in place, no legacy-subset knob); the "
        "historical arm keeps mutation_count value labels + legacy STOP-slot "
        "accounting only"
    ),
}

FIXTURE_CONFIG_OVERRIDES = {
    "d_model": 64,
    "denoiser_layers": 2,
    "context_layers": 1,
    "max_chain": 3,
    "max_search_steps": 6,
    "beam_width": 3,
    "expand_per_state": 3,
}


@dataclass(frozen=True)
class ArmOutcome:
    """Paired per-example final outcome for one arm (duck-types slm155's)."""

    record_id: str
    arm_id: str
    seed: int
    final_source: str
    v1_verdict: bool
    hard_rank: int
    commit_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "arm_id": self.arm_id,
            "seed": self.seed,
            "final_source": self.final_source,
            "v1_verdict": self.v1_verdict,
            "hard_rank": self.hard_rank,
            "commit_reason": self.commit_reason,
        }


def _inventory_for(source: str) -> list[str]:
    import re

    slots = sorted(set(re.findall(r":[A-Za-z][\w.]*", source)))
    return slots or [":content"]


def _prompt_for(decision_id: str) -> str:
    return f"fixture decision {decision_id}: render the accepted content"


def _build_model(records, *, seed: int, value_label_mode: str, stop_accounting: str):
    import torch

    from slm_training.models.tree_edit_diffusion import (
        TreeEditDiffusionConfig,
        TreeEditDiffusionModel,
    )

    torch.manual_seed(seed)
    config = TreeEditDiffusionConfig(
        seed=seed,
        value_label_mode=value_label_mode,
        stop_slot_accounting=stop_accounting,
        context_backend="scratch",
        **FIXTURE_CONFIG_OVERRIDES,
    )
    return TreeEditDiffusionModel.from_records(records, config=config, device="cpu")


def _train(model, records, *, steps: int, lr: float):
    import torch

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        loss = model.training_loss(records)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def run_screen(
    *,
    n_train: int,
    n_eval: int,
    seeds: tuple[int, ...],
    steps: int,
    lr: float,
    scorer_steps: int,
    decisions_out: Path,
) -> dict[str, Any]:
    import torch

    from slm_training.dsl.schema import ExampleRecord

    torch.set_num_threads(1)  # tiny CPU fixture; identical for every arm
    start = time.perf_counter()

    train_decisions = make_fixture_decisions(n=n_train, seed=0)
    eval_decisions = make_fixture_decisions(n=n_eval, seed=1)
    scorer_result = train_fixture_scorer(
        train_decisions,
        config=LegalActionScorerConfig(variant="mlp", seed=0),
        steps=scorer_steps,
        lr=0.05,
    )
    scorer = scorer_result["scorer"]

    def ar_source_for(decision) -> str:
        scores = scorer.score(
            decision.context_features,
            decision.state_features,
            decision.legal_actions,
            plan_features=decision.plan_features,
            plan_action_features=decision.plan_action_features,
            pack_id=decision.pack_id,
        )
        chosen = scorer.decode(scores, decision.legal_actions).action_identity
        return _program_from_actions([chosen] if chosen else [])

    train_records = [
        ExampleRecord(
            id=d.decision_id,
            prompt=_prompt_for(d.decision_id),
            openui=_program_from_actions(
                [d.accepted_action_ids[0]] if d.accepted_action_ids else []
            ),
            placeholders=_inventory_for(
                _program_from_actions(
                    [d.accepted_action_ids[0]] if d.accepted_action_ids else []
                )
            ),
        )
        for d in train_decisions
    ]

    model_hist = _train(
        _build_model(
            train_records,
            seed=0,
            value_label_mode="mutation_count",
            stop_accounting="legacy",
        ),
        train_records,
        steps=steps,
        lr=lr,
    )
    model_imp = _train(
        _build_model(
            train_records,
            seed=0,
            value_label_mode="bounded_distance",
            stop_accounting="corrected",
        ),
        train_records,
        steps=steps,
        lr=lr,
    )

    outcomes: list[ArmOutcome] = []
    commit_rows: list[CommitDecision] = []

    def record(arm: str, record_id: str, seed: int, final: str, reason: str | None):
        ev = hard_evidence(final)
        outcomes.append(
            ArmOutcome(
                record_id=record_id,
                arm_id=arm,
                seed=seed,
                final_source=final,
                v1_verdict=ev.v1_verdict,
                hard_rank=ev.rank,
                commit_reason=reason,
            )
        )

    for seed in seeds:
        for decision in eval_decisions:
            rid = decision.decision_id
            prompt = _prompt_for(rid)
            ar_source = ar_source_for(decision)
            inventory = _inventory_for(ar_source)

            # Arm 1: AR only.
            record("ar_only", rid, seed, ar_source, None)

            # Arm 2: repair-only from the minimal seed (improved model decode).
            ctx, ctx_pad = model_imp._encode_context(
                [model_imp._format_context(prompt, slot_contract=inventory)]
            )
            seed_final, _ = model_imp._decode_one(ctx, ctx_pad, inventory)
            record("repair_only", rid, seed, seed_final, None)

            # Arm 3: AR → historical repair (pre-LAR2 value/stop config).
            cand_hist, _ = repair_decode(model_hist, ar_source, inventory, prompt)
            decision_hist = commit(
                ar_source,
                cand_hist,
                evidence={
                    "learned_score_source": state_value(
                        model_hist, ar_source, inventory, prompt
                    ),
                    "learned_score_candidate": state_value(
                        model_hist, cand_hist, inventory, prompt
                    ),
                },
                record_id=rid,
                arm_id="ar_repair_historical",
                seed=seed,
            )
            commit_rows.append(decision_hist)
            record(
                "ar_repair_historical", rid, seed, decision_hist.final, decision_hist.reason
            )

            # Arm 4: AR → improved repair (this branch's config).
            cand_imp, _ = repair_decode(model_imp, ar_source, inventory, prompt)
            decision_imp = commit(
                ar_source,
                cand_imp,
                evidence={
                    "learned_score_source": state_value(
                        model_imp, ar_source, inventory, prompt
                    ),
                    "learned_score_candidate": state_value(
                        model_imp, cand_imp, inventory, prompt
                    ),
                },
                record_id=rid,
                arm_id="ar_repair_improved",
                seed=seed,
            )
            commit_rows.append(decision_imp)
            record(
                "ar_repair_improved", rid, seed, decision_imp.final, decision_imp.reason
            )

            # Arm 5: oracle commit selector (non-promotable upper bound).
            decision_oracle = oracle_commit(
                ar_source, cand_imp, record_id=rid, arm_id="oracle_commit", seed=seed
            )
            commit_rows.append(decision_oracle)
            record("oracle_commit", rid, seed, decision_oracle.final, decision_oracle.reason)

    decisions_out.parent.mkdir(parents=True, exist_ok=True)
    with decisions_out.open("w", encoding="utf-8") as fh:
        for row in commit_rows:
            fh.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")

    by_arm: dict[str, list[ArmOutcome]] = {arm: [] for arm in ARMS}
    for outcome in outcomes:
        by_arm[outcome.arm_id].append(outcome)

    ar_outcomes = by_arm["ar_only"]
    paired: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        if arm == "ar_only":
            continue
        paired[arm] = paired_validity_table(ar_outcomes, by_arm[arm])

    def valid_rate(arm: str) -> float:
        rows = by_arm[arm]
        return sum(o.v1_verdict for o in rows) / max(len(rows), 1)

    n_paired = sum(paired["ar_repair_improved"].values()) - paired["ar_repair_improved"]["unpaired"]
    improvements = paired["ar_repair_improved"]["a_invalid_b_valid"]
    damages = paired["ar_repair_improved"]["a_valid_b_invalid"]
    wilson = wilson_interval(improvements, max(n_paired, 0))
    wilson_damage = wilson_interval(damages, max(n_paired, 0))

    reachability = {p: Path(p).exists() for p in REACHABILITY_ARTIFACTS}
    gates = {
        "safety": {
            "rule": PREREGISTERED["safety_rule"],
            "invalid_over_valid": damages,
            "pass": damages == 0,
        },
        "value": {
            "rule": PREREGISTERED["value_rule"],
            "improvements": improvements,
            "n_paired": n_paired,
            "wilson": wilson,
            "min_useful_effect": PREREGISTERED["min_useful_effect"],
            "pass": bool(
                wilson["low"] is not None
                and wilson["low"] > PREREGISTERED["min_useful_effect"]
                and damages == 0
            ),
        },
        "reachability": {
            "artifacts": reachability,
            "pass": all(reachability.values()),
        },
    }

    safety_pass = gates["safety"]["pass"]
    value_pass = gates["value"]["pass"]
    reach_pass = gates["reachability"]["pass"]
    if safety_pass and value_pass and reach_pass:
        disposition = DISPOSITION_POSITIVE
    elif not safety_pass or (
        wilson["high"] is not None
        and wilson["high"] < PREREGISTERED["min_useful_effect"]
    ):
        disposition = DISPOSITION_NEGATIVE
    else:
        disposition = DISPOSITION_INCONCLUSIVE

    lar3_statement = {
        DISPOSITION_POSITIVE: (
            "LAR3 (learned repair integration) is OPEN: the do-no-harm hybrid "
            "cleared safety, value, and reachability gates on the frozen screen."
        ),
        DISPOSITION_NEGATIVE: (
            "LAR3 (learned repair integration) is CLOSED: repair damaged valid "
            "AR programs or its paired value was ruled out below the minimum "
            "useful effect on the frozen screen."
        ),
        DISPOSITION_INCONCLUSIVE: (
            "LAR3 (learned repair integration) remains OPEN but NOT advanced: "
            "the screen neither cleared the value gate nor ruled the effect "
            "out at fixture power; a powered rerun is required before LAR3 "
            "can open or close."
        ),
    }[disposition]

    payload = {
        "experiment": EXPERIMENT_ID,
        "issue": "SLM-317",
        "question": (
            "Does a do-no-harm AR→repair hybrid (deterministic commit rule) "
            "improve paired hard-valid outcomes over AR only without ever "
            "damaging a valid AR program?"
        ),
        "preregistered": PREREGISTERED,
        "arms": list(ARMS),
        "promotable_arms": sorted(PROMOTABLE_ARMS),
        "config": {
            "n_train_decisions": n_train,
            "n_eval_decisions": n_eval,
            "seeds": list(seeds),
            "steps": steps,
            "lr": lr,
            "scorer_steps": scorer_steps,
            "fixture_config_overrides": FIXTURE_CONFIG_OVERRIDES,
            "torch_num_threads": 1,
        },
        "arm_summaries": {
            arm: {
                "n": len(by_arm[arm]),
                "valid_rate": valid_rate(arm),
                "mean_hard_rank": (
                    sum(o.hard_rank for o in by_arm[arm]) / max(len(by_arm[arm]), 1)
                ),
            }
            for arm in ARMS
        },
        "paired_vs_ar_only": paired,
        "invalid_over_valid_per_arm": {
            arm: paired[arm]["a_valid_b_invalid"] for arm in paired
        },
        "commit_reason_counts": {
            arm: {
                reason: sum(1 for o in by_arm[arm] if o.commit_reason == reason)
                for reason in sorted(
                    {o.commit_reason for o in by_arm[arm] if o.commit_reason}
                )
            }
            for arm in ("ar_repair_historical", "ar_repair_improved", "oracle_commit")
        },
        "gates": gates,
        "wilson_damage": wilson_damage,
        "oracle_upper_bound": {
            "valid_rate": valid_rate("oracle_commit"),
            "sanity": valid_rate("oracle_commit") >= valid_rate("ar_repair_improved"),
        },
        "per_example_outcomes": [o.to_dict() for o in outcomes],
        "decisions_path": str(decisions_out),
        "disposition": disposition,
        "lar3_statement": lar3_statement,
        "wall_seconds": time.perf_counter() - start,
        "honesty": (
            "Fixture-scale wiring screen: the frozen corpus is the SLM-155 "
            "synthetic decision fixture (n eval decisions below any ship-gate "
            "prompt count), models are tiny CPU fixtures trained for a handful "
            "of steps, and no ship-gate claim is made. The commit rule is "
            "deterministic and metamorphism-invariant by construction; the "
            "screen measures the mechanism, not production quality. The "
            "historical arm's pre-SLM-305 4-action space is declared "
            "non-reproducible on this branch (see preregistered deviation)."
        ),
    }
    payload["version_stamp"] = build_version_stamp(
        "harness.experiments.slm317_repair_hybrid",
        "harness.experiments.slm155_factorization_comparison",
        "harness.experiments.slm299_edit_reachability",
    )
    return payload


def render_markdown(payload: dict) -> str:
    def f(x: float) -> str:
        return f"{x:.3f}"

    lines = [
        "# SLM-317 (LAR2-06): do-no-harm AR→repair hybrid screen",
        "",
        f"**Disposition: `{payload['disposition']}`** — {payload['lar3_statement']}",
        "",
        "Fixture-scale mechanism screen; **not a ship claim**.",
        "",
        "## Preregistered (locked before results)",
        "",
        f"- safety: {payload['preregistered']['safety_rule']}",
        f"- value: {payload['preregistered']['value_rule']}",
        f"- disposition: {payload['preregistered']['disposition_rule']}",
        f"- declared deviation: {payload['preregistered']['historical_deviation']}",
        "",
        "## Arms (matched examples + seeds + budgets)",
        "",
        "| arm | n | hard-valid rate | mean hard rank | invalid-over-valid vs AR |",
        "| --- | --- | --- | --- | --- |",
    ]
    iov = payload["invalid_over_valid_per_arm"]
    for arm in payload["arms"]:
        s = payload["arm_summaries"][arm]
        lines.append(
            f"| {arm} | {s['n']} | {f(s['valid_rate'])} | "
            f"{f(s['mean_hard_rank'])} | {iov.get(arm, '—')} |"
        )
    lines += [
        "",
        "## Paired outcomes vs `ar_only` (per-example, nothing aggregated away)",
        "",
        "| arm | both valid | AR valid, arm invalid | AR invalid, arm valid | both invalid | unpaired |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm, counts in payload["paired_vs_ar_only"].items():
        lines.append(
            f"| {arm} | {counts['both_valid']} | {counts['a_valid_b_invalid']} | "
            f"{counts['a_invalid_b_valid']} | {counts['both_invalid']} | {counts['unpaired']} |"
        )
    gates = payload["gates"]
    wilson = gates["value"]["wilson"]
    lines += [
        "",
        "## Advancement gates",
        "",
        f"- **Safety**: invalid-over-valid = {gates['safety']['invalid_over_valid']} "
        f"(must be 0) → {'PASS' if gates['safety']['pass'] else 'FAIL'}",
        f"- **Value**: improvements {gates['value']['improvements']}/{gates['value']['n_paired']}, "
        f"Wilson 95% [{wilson['low']}, {wilson['high']}] vs minimum useful effect "
        f"{gates['value']['min_useful_effect']} → {'PASS' if gates['value']['pass'] else 'FAIL'}",
        f"- **Reachability/provenance**: slm299/291 artifacts present → "
        f"{'PASS' if gates['reachability']['pass'] else 'FAIL'}",
        "",
        "## Commit reasons",
        "",
        "| arm | reason | count |",
        "| --- | --- | --- |",
    ]
    for arm, reasons in payload["commit_reason_counts"].items():
        for reason, count in reasons.items():
            lines.append(f"| {arm} | {reason} | {count} |")
    lines += [
        "",
        f"Oracle commit upper bound hard-valid rate: "
        f"{f(payload['oracle_upper_bound']['valid_rate'])} "
        f"(sanity ≥ improved hybrid: {payload['oracle_upper_bound']['sanity']}).",
        "",
        f"Durable per-example commit decisions: `{payload['decisions_path']}`.",
        "",
        "## Honesty",
        "",
        payload["honesty"],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--n-train", type=int, default=16)
    parser.add_argument("--n-eval", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--scorer-steps", type=int, default=20)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--decisions-out", type=Path, default=DEFAULT_DECISIONS_OUT)
    args = parser.parse_args(argv)

    payload = run_screen(
        n_train=args.n_train,
        n_eval=args.n_eval,
        seeds=tuple(args.seeds),
        steps=args.steps,
        lr=args.lr,
        scorer_steps=args.scorer_steps,
        decisions_out=args.decisions_out,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"disposition={payload['disposition']} "
        f"safety={payload['gates']['safety']['pass']} "
        f"value={payload['gates']['value']['pass']} "
        f"iov={payload['invalid_over_valid_per_arm'].get('ar_repair_improved')} "
        f"-> {args.json_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
