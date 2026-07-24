#!/usr/bin/env python3
"""SLM-314 (LAR2-05): winner-take-all vs single-gold vs multi-gold CE.

Question: when a prompt has multiple hard-valid canonical AST modes, does
winner-take-all (MCL) training — min-loss mode per example plus a small
preregistered coverage/floor term — retain MORE verifier-accepted modes than
single-gold CE (which drops all but one mode at the dataset level) without a
semantic regression, at equal model / steps / optimizer / seed budget? The
duplicated-example multi-gold CE arm is the averaging baseline.

Preregistered (locked before any run; written into the payload):

- frozen dataset: ``src/slm_training/resources/data/slm314_multimode/modes.jsonl``
  (8 prompts x 2 verifier-accepted canonical AST modes; mode identity =
  alpha-invariant canonical AST fingerprints + sha256 freeze manifest);
- arms {single_gold, multi_gold, wta} with identical model init, steps,
  optimizer, per-step prompt coverage, and seeds;
- WTA floor epsilon 0.1; a frozen mode counts as RETAINED when its
  deterministic eval loss IMPROVES vs the matched untrained baseline (same
  init seed, same fixed eval rng) by more than 0.0 AND keeps pace with its
  prompt's best-improved mode within 0.5 loss units;
- primary verdict ``wta_preserves_modes`` iff coverage(wta) >
  coverage(single_gold) AND v2 semantic verdict regression vs single_gold
  <= 5 points; else ``rejected`` — honestly computed, never narrated;
- synthetic two-mode fixture assertions (collapse/retain/hybrid thresholds in
  ``slm314_winner_take_all``) must hold as the mechanism proof.

Evaluation per arm: per-mode retention (deterministic eval loss), mode
coverage, collapse indicator, decode hard-valid rate, decoded-AST mode-hit
rate (canonical fingerprint match against the prompt's frozen modes), and v2
semantic verdict rate (``binding_aware_meaningful_v2``). Fixture-scale: 8
prompts is far below any ship-gate prompt count — this is mechanism evidence,
not a ship claim.

Writes ``docs/design/iter-slm314-winner-take-all-20260724.{json,md}`` and
durable WTA telemetry under ``outputs/slm314/``.

Example:
  python -m scripts.run_slm314_winner_take_all --steps 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.meaningful_program import binding_aware_meaningful_v2
from slm_training.harnesses.experiments.slm314_winner_take_all import (
    DEFAULT_RESOURCE,
    EXPERIMENT_ID,
    FLOOR_EPSILON,
    PromptGroupV1,
    _seed_int,
    dataset_manifest,
    mode_supervision_loss,
    read_frozen_dataset,
    run_synthetic_two_mode,
    single_gold_groups,
    synthetic_assertions,
    winner_take_all_loss,
)
from slm_training.models.tree_edit_diffusion import (
    TreeEditDiffusionConfig,
    TreeEditDiffusionModel,
    _is_valid,
    parse_statements,
)
from slm_training.harnesses.experiments.slm312_state_sources import ast_fingerprint
from slm_training.versioning import build_version_stamp

DEFAULT_JSON_OUT = Path("docs/design/iter-slm314-winner-take-all-20260724.json")
DEFAULT_MD_OUT = Path("docs/design/iter-slm314-winner-take-all-20260724.md")
DEFAULT_TELEMETRY_OUT = Path("outputs/slm314/wta_telemetry.jsonl")

ARMS = ("single_gold", "multi_gold", "wta")

# Tiny fixture model/decode config — identical across ALL arms and declared in
# the payload; it only bounds fixture wall-clock (the run cap), never differs
# by arm.
FIXTURE_CONFIG_OVERRIDES = {
    "d_model": 64,
    "denoiser_layers": 2,
    "context_layers": 1,
    "max_chain": 3,
    "max_search_steps": 6,
    "beam_width": 3,
    "expand_per_state": 3,
}

# PREREGISTERED thresholds (locked before any run; deviations are append-only
# and exploratory).
PREREGISTERED_THRESHOLDS = {
    "floor_epsilon": FLOOR_EPSILON,
    "retained_improvement_min": 0.0,
    "retention_gap_margin": 0.5,
    "semantic_regression_max_points": 0.05,
    "verdict_rule": (
        "wta_preserves_modes iff coverage(wta) > coverage(single_gold) AND "
        "v2 semantic verdict regression vs single_gold <= 0.05, else rejected"
    ),
}


def build_model(records: list[ExampleRecord], *, seed: int) -> TreeEditDiffusionModel:
    """Identical model init for every arm (same tokenizer texts, same seed)."""
    torch.manual_seed(seed)
    config = TreeEditDiffusionConfig(
        seed=seed,
        # mutation_count value labels: the SLM-308 bounded-distance oracle is
        # too expensive for this fixture's per-mode WTA loss (measured 5x
        # wall-clock); arms stay matched and the mode is declared.
        value_label_mode="mutation_count",
        context_backend="scratch",
        **FIXTURE_CONFIG_OVERRIDES,
    )
    return TreeEditDiffusionModel.from_records(records, config=config, device="cpu")


def train_arm(
    arm: str,
    groups: list[PromptGroupV1],
    records: list[ExampleRecord],
    *,
    steps: int,
    lr: float,
    seed: int,
    telemetry_rows: list[dict] | None = None,
) -> TreeEditDiffusionModel:
    """One arm. Every step covers every prompt group exactly once for all
    arms (identical prompt budget); arms differ ONLY in loss weighting:
    single_gold = one mode per prompt, multi_gold = plain CE over duplicated
    per-mode examples, wta = min-loss mode + floor term."""
    model = build_model(records, seed=seed)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    sg_records = [r for g in single_gold_groups(groups) for r in g.records()]
    mg_records = [r for g in groups for r in g.records()]
    model.train()
    for step in range(steps):
        if arm == "single_gold":
            loss = model.training_loss(sg_records)
        elif arm == "multi_gold":
            loss = model.training_loss(mg_records)
        else:
            total = None
            for group in groups:
                group_loss, telemetry = winner_take_all_loss(
                    model, group, floor_epsilon=FLOOR_EPSILON, seed=seed + step
                )
                total = group_loss if total is None else total + group_loss
                if telemetry_rows is not None:
                    telemetry_rows.append({"step": step, **telemetry})
            loss = total
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def per_mode_eval_losses(
    model: TreeEditDiffusionModel, groups: list[PromptGroupV1]
) -> dict[tuple[str, str], float]:
    """Deterministic per-mode eval loss (fixed eval rng, shared across arms)."""
    losses: dict[tuple[str, str], float] = {}
    for group in groups:
        for mode, record in zip(group.modes, group.records()):
            losses[(group.group_id, mode.canonical_fingerprint)] = float(
                mode_supervision_loss(
                    model,
                    record,
                    seed=_seed_int(
                        "slm314-eval", group.group_id, mode.canonical_fingerprint
                    ),
                ).detach().cpu()
            )
    return losses


def eval_per_mode(
    model: TreeEditDiffusionModel,
    groups: list[PromptGroupV1],
    baseline: dict[tuple[str, str], float],
) -> dict:
    """A mode is RETAINED when its deterministic eval loss improves on the
    matched untrained baseline (same init seed, same eval rng) by more than
    the preregistered margin AND it keeps pace with its prompt's
    best-improved mode within ``retention_gap_margin`` loss units."""
    margin = PREREGISTERED_THRESHOLDS["retained_improvement_min"]
    gap = PREREGISTERED_THRESHOLDS["retention_gap_margin"]
    losses = per_mode_eval_losses(model, groups)
    per_group: list[dict] = []
    retained = 0
    total = 0
    for group in groups:
        improvements = [
            baseline[(group.group_id, m.canonical_fingerprint)]
            - losses[(group.group_id, m.canonical_fingerprint)]
            for m in group.modes
        ]
        best = max(improvements)
        rows = []
        for mode, improvement in zip(group.modes, improvements):
            key = (group.group_id, mode.canonical_fingerprint)
            loss = losses[key]
            kept = improvement > margin and improvement >= best - gap
            retained += int(kept)
            total += 1
            rows.append(
                {
                    "mode_index": mode.mode_index,
                    "canonical_fingerprint": mode.canonical_fingerprint,
                    "eval_loss": loss,
                    "baseline_loss": baseline[key],
                    "improvement": improvement,
                    "retained": kept,
                }
            )
        per_group.append({"group_id": group.group_id, "modes": rows})
    coverage = retained / max(total, 1)
    return {
        "n_modes": total,
        "n_retained": retained,
        "coverage": coverage,
        "collapse_indicator": 1.0 - coverage,
        "per_group": per_group,
    }


@torch.no_grad()
def eval_decode(
    model: TreeEditDiffusionModel, groups: list[PromptGroupV1]
) -> dict:
    """Decode once per prompt (beam decode is deterministic): hard-valid rate,
    mode-hit rate (decoded AST fingerprint in the prompt's frozen mode set),
    and v2 semantic verdict rate."""
    n_valid = 0
    n_hit = 0
    n_v2 = 0
    rows = []
    for group in groups:
        inventory = [p if p.startswith(":") else f":{p}" for p in group.modes[0].placeholders]
        prompt = model._format_context(group.prompt, slot_contract=inventory)
        ctx, ctx_pad = model._encode_context([prompt])
        text, _evidence = model._decode_one(ctx, ctx_pad, inventory)
        valid = bool(text) and _is_valid(text)
        hit = False
        if valid:
            statements = parse_statements(text)
            if statements is not None:
                fp = ast_fingerprint(statements)
                hit = fp in {m.canonical_fingerprint for m in group.modes}
        report = binding_aware_meaningful_v2(
            text,
            record=ExampleRecord(
                id=group.group_id,
                prompt=group.prompt,
                openui=group.modes[0].source,
                placeholders=list(inventory),
            ),
        )
        n_valid += int(valid)
        n_hit += int(hit)
        n_v2 += int(report.verdict)
        rows.append(
            {
                "group_id": group.group_id,
                "valid": valid,
                "mode_hit": hit,
                "v2_verdict": bool(report.verdict),
            }
        )
    n = max(len(groups), 1)
    return {
        "n_prompts": len(groups),
        "hard_valid_rate": n_valid / n,
        "mode_hit_rate": n_hit / n,
        "v2_verdict_rate": n_v2 / n,
        "per_group": rows,
    }


def build_report(
    groups: list[PromptGroupV1],
    *,
    steps: int,
    lr: float,
    seed: int,
    telemetry_out: Path,
) -> dict:
    torch.set_num_threads(1)  # tiny CPU fixture; identical for every arm
    start = time.perf_counter()
    records = [r for g in groups for r in g.records()]

    # Matched untrained baseline (identical init seed/config): retention is
    # measured as per-mode eval-loss IMPROVEMENT against this baseline.
    baseline = per_mode_eval_losses(build_model(records, seed=seed), groups)

    telemetry_rows: list[dict] = []
    arms: dict[str, dict] = {}
    for arm in ARMS:
        model = train_arm(
            arm,
            groups,
            records,
            steps=steps,
            lr=lr,
            seed=seed,
            telemetry_rows=telemetry_rows if arm == "wta" else None,
        )
        per_mode = eval_per_mode(model, groups, baseline)
        decode = eval_decode(model, groups)
        arms[arm] = {
            "per_mode": per_mode,
            "decode": decode,
            "coverage": per_mode["coverage"],
            "collapse_indicator": per_mode["collapse_indicator"],
            "training_metrics": model.last_training_metrics,
        }

    telemetry_out.parent.mkdir(parents=True, exist_ok=True)
    with telemetry_out.open("w", encoding="utf-8") as fh:
        for row in telemetry_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    synthetic = run_synthetic_two_mode()
    synthetic_checks = synthetic_assertions(synthetic)

    coverage_gain = arms["wta"]["coverage"] - arms["single_gold"]["coverage"]
    semantic_regression = (
        arms["single_gold"]["decode"]["v2_verdict_rate"]
        - arms["wta"]["decode"]["v2_verdict_rate"]
    )
    coverage_ok = coverage_gain > 0.0
    semantic_ok = (
        semantic_regression
        <= PREREGISTERED_THRESHOLDS["semantic_regression_max_points"]
    )
    verdict = "wta_preserves_modes" if (coverage_ok and semantic_ok) else "rejected"

    payload = {
        "experiment": EXPERIMENT_ID,
        "issue": "SLM-314",
        "question": (
            "Does winner-take-all (MCL) training with a preregistered floor "
            "term retain more hard-valid canonical AST modes than single-gold "
            "CE without a semantic regression, at equal budget?"
        ),
        "preregistered_thresholds": PREREGISTERED_THRESHOLDS,
        "arms": ARMS,
        "config": {
            "steps": steps,
            "lr": lr,
            "seed": seed,
            "n_prompts": len(groups),
            "n_modes": sum(len(g.modes) for g in groups),
            "dataset": str(DEFAULT_RESOURCE),
            "dataset_manifest": dataset_manifest(groups),
            "fixture_config_overrides": FIXTURE_CONFIG_OVERRIDES,
            "torch_num_threads": 1,
        },
        "synthetic_two_mode": {
            "probabilities": synthetic,
            "assertions": synthetic_checks,
            "all_assertions_pass": all(synthetic_checks.values()),
        },
        "arm_results": arms,
        "comparison": {
            "coverage_gain_wta_vs_single_gold": coverage_gain,
            "semantic_regression_wta_vs_single_gold": semantic_regression,
            "coverage_multi_gold": arms["multi_gold"]["coverage"],
        },
        "threshold_checks": {
            "coverage_ok": coverage_ok,
            "semantic_ok": semantic_ok,
            "synthetic_ok": all(synthetic_checks.values()),
        },
        "telemetry_path": str(telemetry_out),
        "verdict": verdict,
        "wall_seconds": time.perf_counter() - start,
        "honesty": (
            "Fixture-scale: 8 prompts x 2 modes is far below any ship-gate "
            "prompt count — mechanism evidence only, not a production ship "
            "claim. Mode identity is alpha-invariant canonical AST "
            "fingerprints (never serialization strings); the frozen dataset "
            "is hash-verified before training. All arms share model init, "
            "steps, optimizer, per-step prompt coverage, and seeds. Decode is "
            "deterministic beam search (unique valid ASTs per prompt == 1 by "
            "construction); coverage is measured by deterministic per-mode "
            "eval loss instead. The SLM-130 ambiguity sets are wiring reports "
            "without a committed multi-mode corpus, so this dataset is "
            "synthesized and frozen in-repo."
        ),
    }
    payload["version_stamp"] = build_version_stamp(
        "harness.experiments.slm314_winner_take_all",
        "harness.experiments.slm299_edit_reachability",
    )
    return payload


def render_markdown(payload: dict) -> str:
    def f(x: float) -> str:
        return f"{x:.3f}"

    lines = [
        "# SLM-314 (LAR2-05): winner-take-all over multi-mode gold",
        "",
        f"**Verdict: `{payload['verdict']}`** (fixture-scale matched arms; not a ship claim)",
        "",
        "## Preregistered (locked before results)",
        "",
        f"- arms: `{json.dumps(payload['arms'])}` — identical model init, steps, "
        "optimizer, per-step prompt coverage, seeds",
        f"- floor epsilon: {payload['preregistered_thresholds']['floor_epsilon']}; "
        f"retained iff eval-loss improvement > "
        f"{payload['preregistered_thresholds']['retained_improvement_min']} vs matched "
        f"baseline and within {payload['preregistered_thresholds']['retention_gap_margin']} "
        "of the prompt's best-improved mode",
        f"- rule: {payload['preregistered_thresholds']['verdict_rule']}",
        f"- frozen dataset: `{payload['config']['dataset']}` "
        f"({payload['config']['n_prompts']} prompts x "
        f"{payload['config']['n_modes'] // payload['config']['n_prompts']} modes, "
        f"manifest `{payload['config']['dataset_manifest']['manifest_sha256'][:16]}…`)",
        "",
        "## Synthetic two-mode proof (deterministic)",
        "",
        "| arm | p(mode A) | p(mode B) | p(invalid hybrid) |",
        "| --- | --- | --- | --- |",
    ]
    for arm, probs in payload["synthetic_two_mode"]["probabilities"].items():
        lines.append(
            f"| {arm} | {f(probs['p_mode_a'])} | {f(probs['p_mode_b'])} | "
            f"{f(probs['p_invalid_hybrid'])} |"
        )
    checks = payload["synthetic_two_mode"]["assertions"]
    lines += [
        "",
        "Assertions: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in checks.items()),
        "",
        "## Matched arms (real tree-edit model, frozen multi-mode dataset)",
        "",
        "| arm | mode coverage | collapse | hard-valid | mode-hit | v2 verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in payload["arms"]:
        r = payload["arm_results"][arm]
        lines.append(
            f"| {arm} | {f(r['coverage'])} | {f(r['collapse_indicator'])} | "
            f"{f(r['decode']['hard_valid_rate'])} | {f(r['decode']['mode_hit_rate'])} | "
            f"{f(r['decode']['v2_verdict_rate'])} |"
        )
    cmp_ = payload["comparison"]
    lines += [
        "",
        f"Coverage gain (wta − single_gold): **{f(cmp_['coverage_gain_wta_vs_single_gold'])}**; "
        f"v2 semantic regression: **{f(cmp_['semantic_regression_wta_vs_single_gold'])}** "
        f"(budget 0.05). Multi-gold coverage: {f(cmp_['coverage_multi_gold'])}.",
        "",
        f"WTA per-step selected-mode telemetry: `{payload['telemetry_path']}`.",
        "",
        "## Honesty",
        "",
        payload["honesty"],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_RESOURCE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--telemetry-out", type=Path, default=DEFAULT_TELEMETRY_OUT)
    args = parser.parse_args(argv)

    groups = read_frozen_dataset(args.dataset)
    payload = build_report(
        groups,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        telemetry_out=args.telemetry_out,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"verdict={payload['verdict']} "
        f"coverage_gain={payload['comparison']['coverage_gain_wta_vs_single_gold']:.3f} "
        f"semantic_regression={payload['comparison']['semantic_regression_wta_vs_single_gold']:.3f} "
        f"-> {args.json_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
