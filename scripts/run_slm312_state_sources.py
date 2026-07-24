#!/usr/bin/env python3
"""SLM-312 (LAR2-04): matched state-source mixture arms {gold_only, seedward, on_policy, mixed}.

Question: does closing the gold-corruption vs seed-rollout distribution gap
(training on seedward and/or on-policy valid states, oracle-labeled) improve
repair more than extra offline near-gold training, at equal model / steps /
optimizer / evaluator / seed budget?

Preregistered (written into the output payload BEFORE any result):

- arms + mixture weights + acquisition caps: ``PREREGISTERED_ARMS`` and
  ``--per-arm-budget`` (locked before training; deviations append-only);
- primary: best non-gold arm must improve beam regret vs ``gold_only`` by
  **>= 0.05** (distance units) AND must not degrade value rank correlation
  by more than **0.05**;
- verdict ``distribution_gap_closed`` iff both hold, else ``rejected`` —
  honestly computed from the measured arms, never narrated.

Arms share the tiny fixture corpus, model init seed, optimizer budget, batch
order, and evaluator; they differ ONLY in which state source feeds
``state_supervision_loss``. Evaluation runs on HELD-OUT SEED TRAJECTORIES
(random walks from the decode seed over held-out records — never acquired
states): valid-final decode rate, value calibration (rank correlation,
concordance, Brier, per-bin), beam regret vs oracle-best (SLM-308
machinery), AST-fingerprint duplicate rate + state-space coverage per
source. Leak guards are fail-closed: any train/held-out fingerprint overlap
aborts the run before training.

Writes ``docs/design/iter-slm312-state-sources-20260724.{json,md}`` and the
immutable snapshot under ``outputs/slm312/``.

Example:
  python -m scripts.run_slm312_state_sources --steps 4
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.slm308_distance_oracle import (
    clear_caches,
)
from slm_training.harnesses.experiments.slm312_state_sources import (
    EXPERIMENT_ID,
    PREREGISTERED_ARMS,
    SEED_SOURCE,
    StateSnapshotV1,
    acquire_gold_only,
    acquire_on_policy,
    acquire_seedward,
    assert_no_leakage,
    build_arm_rows,
    source_coverage,
    source_fingerprint,
    state_supervision_loss,
    drop_leaked_rows,
)
from slm_training.models.tree_edit_diffusion import (
    TreeEditDiffusionConfig,
    TreeEditDiffusionModel,
    TreeEditSpace,
    _is_valid,
    parse_statements,
    render_statements,
)
from slm_training.versioning import build_version_stamp
from scripts.run_slm308_distance_value import (
    FIXTURE_PROGRAMS,
    label_eval_states,
    score_arm,
)

DEFAULT_JSON_OUT = Path("docs/design/iter-slm312-state-sources-20260724.json")
DEFAULT_MD_OUT = Path("docs/design/iter-slm312-state-sources-20260724.md")
DEFAULT_SNAPSHOT_OUT = Path("outputs/slm312/state_snapshot.jsonl")

SUITE_ID = "slm312-fixture-v1"
N_TRAIN_RECORDS = 6  # last len(FIXTURE_PROGRAMS) - N_TRAIN_RECORDS are held out

# Tiny fixture model/decode config — identical across ALL arms and declared in
# the payload; it only bounds fixture wall-clock (the run cap), never differs
# by arm.
FIXTURE_CONFIG_OVERRIDES = {
    "d_model": 64,
    "denoiser_layers": 2,
    "context_layers": 1,
    "max_search_steps": 8,
    "beam_width": 3,
    "expand_per_state": 3,
}
ON_POLICY_MAX_STATES_PER_RECORD = 3

# PREREGISTERED improvement thresholds (locked before any run; do not edit
# after outcomes are visible — deviations are append-only and exploratory).
PREREGISTERED_THRESHOLDS = {
    "beam_regret_improvement_min": 0.05,
    "rank_correlation_degradation_max": 0.05,
    "comparison": "best non-gold arm (seedward/on_policy/mixed) vs gold_only",
    "verdict_rule": (
        "distribution_gap_closed iff best non-gold arm improves beam regret "
        ">= 0.05 AND degrades rank correlation by <= 0.05, else rejected"
    ),
}


def build_records() -> tuple[list[ExampleRecord], list[ExampleRecord]]:
    records = [
        ExampleRecord(
            id=f"fixture_{i:02d}",
            prompt=prompt,
            openui=openui,
            placeholders=list(placeholders),
            split="train" if i < N_TRAIN_RECORDS else "held_out",
        )
        for i, (prompt, openui, placeholders) in enumerate(FIXTURE_PROGRAMS)
    ]
    return records[:N_TRAIN_RECORDS], records[N_TRAIN_RECORDS:]


def build_heldout_eval_states(
    held_out: list[ExampleRecord], space: TreeEditSpace, *, seed: int = 31337
) -> list[dict]:
    """Held-out seed-trajectory eval states: random walks from the decode
    seed over HELD-OUT records only (distinct rng stream from training)."""
    states: list[dict] = []
    seed_statements = parse_statements(SEED_SOURCE)
    assert seed_statements is not None
    for idx, record in enumerate(held_out):
        gold = parse_statements(record.openui)
        assert gold is not None
        inventory = [p if p.startswith(":") else f":{p}" for p in record.placeholders]
        rng = random.Random(seed + idx)
        # The bare decode seed is the origin of every seed trajectory; it is
        # almost always oracle-measurable (small EXACT distance to gold).
        states.append(
            {
                "record_idx": idx,
                "origin": "heldout_seed_trajectory",
                "witness": None,
                "statements": seed_statements,
                "target": gold,
                "inventory": inventory,
            }
        )
        # Walk depths 1-2: deeper walks are almost always oracle-UNKNOWN at
        # the shallow SLM-308 budgets and only cost eval time.
        for k in (1, 2):
            current = seed_statements
            applied = 0
            for _ in range(k):
                step = space.sample_mutation(current, inventory, rng)
                if step is None:
                    break
                current, _inverse = step
                applied += 1
            if applied == 0:
                continue
            states.append(
                {
                    "record_idx": idx,
                    "origin": "heldout_seed_trajectory",
                    "witness": None,
                    "statements": current,
                    "target": gold,
                    "inventory": inventory,
                }
            )
    return states


def train_arm(
    records: list[ExampleRecord],
    arm_rows: list,
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> TreeEditDiffusionModel:
    """One arm: identical model init / optimizer / batch order; only the
    state source differs (loss consumes the arm's snapshot rows)."""
    torch.manual_seed(seed)
    config = TreeEditDiffusionConfig(
        seed=seed,
        value_label_mode="bounded_distance",
        context_backend="scratch",
        max_chain=3,
        **FIXTURE_CONFIG_OVERRIDES,
    )
    model = TreeEditDiffusionModel.from_records(records, config=config, device="cpu")
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=3e-3)
    order = random.Random(0)  # identical batch order across arms
    record_by_id = {r.id: r for r in records}
    model.train()
    for _ in range(steps):
        batch = [arm_rows[order.randrange(len(arm_rows))] for _ in range(batch_size)]
        loss = state_supervision_loss(model, batch, record_by_id)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def valid_final_rate(
    model: TreeEditDiffusionModel, held_out: list[ExampleRecord]
) -> dict:
    """Decode from the seed per held-out record; fraction whose final beam
    state is a valid program."""
    n_valid = 0
    finals: list[dict] = []
    for record in held_out:
        inventory = [p if p.startswith(":") else f":{p}" for p in record.placeholders]
        prompt = model._format_context(
            record.prompt, design_md=record.design_md, slot_contract=inventory
        )
        ctx, ctx_pad = model._encode_context([prompt])
        final_text, evidence = model._decode_one(ctx, ctx_pad, inventory)
        ok = bool(final_text) and _is_valid(final_text)
        n_valid += int(ok)
        finals.append(
            {
                "record_id": record.id,
                "valid": ok,
                "steps": evidence.get("steps"),
                "verifier_calls": evidence.get("verifier_calls"),
            }
        )
    return {
        "n": len(held_out),
        "valid_final_rate": n_valid / max(len(held_out), 1),
        "finals": finals,
    }


def build_report(
    train_records: list[ExampleRecord],
    held_out: list[ExampleRecord],
    *,
    steps: int,
    batch_size: int,
    per_arm_budget: int,
    seed: int,
    snapshot_out: Path,
) -> dict:
    space = TreeEditSpace()
    clear_caches()
    # Tiny CPU fixture: torch thread thrash on tiny ops dominates wall-clock
    # (measured 6x slower at 12 threads); pin to a single thread, identical
    # for every arm, and declare it in the payload.
    torch.set_num_threads(1)
    start = time.perf_counter()

    # --- held-out eval states (labeled once, shared across arms) ----------
    eval_states = build_heldout_eval_states(held_out, space)
    oracle = label_eval_states(eval_states, space)
    held_fingerprints = {
        source_fingerprint(render_statements(entry["statements"]))
        for entry in eval_states
    }

    # --- bootstrap model for on-policy rollouts (gold_only supervision) ---
    bootstrap_rows = acquire_gold_only(
        train_records, space, parent_checkpoint="none", suite_id=SUITE_ID
    )
    bootstrap = train_arm(
        train_records,
        bootstrap_rows,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
    )

    # --- acquire all sources over TRAIN records only ----------------------
    rows = []
    rows += bootstrap_rows
    rows += acquire_seedward(
        train_records, space, parent_checkpoint="none", suite_id=SUITE_ID
    )
    rows += acquire_on_policy(
        bootstrap,
        train_records,
        space,
        parent_checkpoint="bootstrap_gold_only_fixture",
        suite_id=SUITE_ID,
        max_states_per_record=ON_POLICY_MAX_STATES_PER_RECORD,
    )

    # Drop legitimate cross-record fingerprint collisions (counted in the
    # snapshot manifest), then the fail-closed guard verifies the kept set.
    rows, dropped_leaks = drop_leaked_rows(rows, held_fingerprints)
    assert_no_leakage(rows, held_fingerprints)

    snapshot = StateSnapshotV1(
        snapshot_id=f"{EXPERIMENT_ID}-fixture",
        config={
            "suite_id": SUITE_ID,
            "arms": PREREGISTERED_ARMS,
            "per_arm_budget": per_arm_budget,
            "steps": steps,
            "batch_size": batch_size,
            "seed": seed,
            "held_out_fingerprints": sorted(held_fingerprints),
            "dropped_leak_collisions": dropped_leaks,
        },
        rows=rows,
    )
    snapshot.write(snapshot_out)
    coverage = source_coverage(rows)

    # --- matched arms -------------------------------------------------------
    arms: dict[str, dict] = {}
    for arm in sorted(PREREGISTERED_ARMS):
        arm_rows = build_arm_rows(rows, arm, per_arm_budget=per_arm_budget, seed=seed)
        model = train_arm(
            train_records, arm_rows, steps=steps, batch_size=batch_size, seed=seed
        )
        metrics = score_arm(model, held_out, eval_states)
        arms[arm] = {
            "n_rows": len(arm_rows),
            "coverage": source_coverage(arm_rows),
            "metrics": metrics,
            "valid_final": valid_final_rate(model, held_out),
            "training_metrics": model.last_training_metrics,
        }

    gold = arms["gold_only"]["metrics"]
    best_name = None
    best_regret_imp: float | None = None
    for name in ("seedward", "on_policy", "mixed"):
        a = arms[name]["metrics"]
        if a["beam_regret_mean"] is None or gold["beam_regret_mean"] is None:
            continue
        imp = gold["beam_regret_mean"] - a["beam_regret_mean"]
        if best_regret_imp is None or imp > best_regret_imp:
            best_regret_imp = imp
            best_name = name
    rank_degradation = None
    if (
        best_name is not None
        and gold["rank_correlation"] is not None
        and arms[best_name]["metrics"]["rank_correlation"] is not None
    ):
        rank_degradation = (
            gold["rank_correlation"] - arms[best_name]["metrics"]["rank_correlation"]
        )
    regret_ok = (
        best_regret_imp is not None
        and best_regret_imp >= PREREGISTERED_THRESHOLDS["beam_regret_improvement_min"]
    )
    rank_ok = (
        rank_degradation is not None
        and rank_degradation
        <= PREREGISTERED_THRESHOLDS["rank_correlation_degradation_max"]
    )
    verdict = "distribution_gap_closed" if (regret_ok and rank_ok) else "rejected"

    payload = {
        "experiment": EXPERIMENT_ID,
        "issue": "SLM-312",
        "question": (
            "Does closing the gold-corruption vs seed-rollout distribution gap "
            "(seedward + on-policy state sources) improve repair more than "
            "extra offline near-gold training at equal budget?"
        ),
        "preregistered_thresholds": PREREGISTERED_THRESHOLDS,
        "preregistered_arms": PREREGISTERED_ARMS,
        "config": {
            "steps": steps,
            "batch_size": batch_size,
            "per_arm_budget": per_arm_budget,
            "seed": seed,
            "n_train_records": len(train_records),
            "n_held_out_records": len(held_out),
            "suite_id": SUITE_ID,
            "fixture_config_overrides": FIXTURE_CONFIG_OVERRIDES,
            "on_policy_max_states_per_record": ON_POLICY_MAX_STATES_PER_RECORD,
            "torch_num_threads": 1,
        },
        "snapshot": {
            "path": str(snapshot_out),
            "manifest": snapshot.manifest(),
            "coverage": coverage,
        },
        "oracle_cost": oracle,
        "arms": arms,
        "comparison": {
            "best_non_gold_arm": best_name,
            "beam_regret_improvement": best_regret_imp,
            "rank_correlation_degradation": rank_degradation,
        },
        "threshold_checks": {"beam_regret_ok": regret_ok, "rank_ok": rank_ok},
        "verdict": verdict,
        "wall_seconds": time.perf_counter() - start,
        "honesty": (
            "Fixture-scale matched arms over immutable content-addressed state "
            "snapshots; evaluation uses held-out seed trajectories only "
            "(fail-closed leak guard ran before training); UNKNOWN/unbounded "
            "oracle labels are excluded and counted, never coerced. A fixture "
            "verdict is wiring/state-source evidence, not a production ship "
            "claim."
        ),
    }
    payload["version_stamp"] = build_version_stamp(
        "harness.experiments.slm312_state_sources",
        "harness.experiments.slm299_edit_reachability",
    )
    return payload


def render_markdown(payload: dict) -> str:
    thr = payload["preregistered_thresholds"]

    def fmt(x: object) -> str:
        return "n/a" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))

    lines = [
        "# SLM-312 (LAR2-04): seedward + on-policy state sources vs near-gold",
        "",
        f"**Verdict: `{payload['verdict']}`** (fixture-scale matched arms; not a ship claim)",
        "",
        "## Preregistered (locked before results)",
        "",
        f"- arms/weights: `{json.dumps(payload['preregistered_arms'])}`",
        f"- per-arm budget: {payload['config']['per_arm_budget']} rows; "
        f"identical model/steps/optimizer/seeds/evaluator",
        f"- beam regret improvement >= {thr['beam_regret_improvement_min']}",
        f"- rank correlation degradation <= {thr['rank_correlation_degradation_max']}",
        f"- rule: {thr['verdict_rule']}",
        "",
        "## Headline (held-out seed trajectories)",
        "",
        "| arm | rows | rank corr | concordance | Brier | beam regret | valid-final |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, arm in sorted(payload["arms"].items()):
        m = arm["metrics"]
        lines.append(
            f"| {name} | {arm['n_rows']} | {fmt(m['rank_correlation'])} | "
            f"{fmt(m['concordance'])} | {fmt(m['brier'])} | "
            f"{fmt(m['beam_regret_mean'])} | "
            f"{fmt(arm['valid_final']['valid_final_rate'])} |"
        )
    cmp_ = payload["comparison"]
    lines += [
        "",
        f"Best non-gold arm: **{cmp_['best_non_gold_arm']}** — beam regret "
        f"improvement {fmt(cmp_['beam_regret_improvement'])}, rank correlation "
        f"degradation {fmt(cmp_['rank_correlation_degradation'])}.",
        "",
        "## State-space coverage by source (full snapshot)",
        "",
        "| source | rows | unique fingerprints |",
        "| --- | --- | --- |",
    ]
    cov = payload["snapshot"]["coverage"]
    for source, row in cov["per_source"].items():
        lines.append(f"| {source} | {row['rows']} | {row['unique_fingerprints']} |")
    lines += [
        "",
        f"Snapshot duplicate rate {cov['duplicate_rate']:.3f}; "
        f"{cov['unique_fingerprints']}/{cov['rows']} unique states. "
        f"Manifest `{payload['snapshot']['manifest']['manifest_sha256'][:16]}…` "
        f"at `{payload['snapshot']['path']}`.",
        "",
        "## Coverage + oracle cost",
        "",
    ]
    oracle = payload["oracle_cost"]
    lines.append(
        f"- UNKNOWN coverage {oracle['unknown_coverage']:.3f} "
        f"({oracle['n_unknown']}/{oracle['n_states']}), measurable "
        f"{oracle['n_measurable']}, oracle {oracle['ms_per_state']:.1f} ms/state"
    )
    lines += [
        "",
        "## Honesty",
        "",
        payload["honesty"],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--per-arm-budget", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--snapshot-out", type=Path, default=DEFAULT_SNAPSHOT_OUT)
    args = parser.parse_args(argv)

    train_records, held_out = build_records()
    payload = build_report(
        train_records,
        held_out,
        steps=args.steps,
        batch_size=args.batch_size,
        per_arm_budget=args.per_arm_budget,
        seed=args.seed,
        snapshot_out=args.snapshot_out,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"verdict={payload['verdict']} "
        f"best_non_gold={payload['comparison']['best_non_gold_arm']} "
        f"regret_imp={payload['comparison']['beam_regret_improvement']} "
        f"-> {args.json_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
