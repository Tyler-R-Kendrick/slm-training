#!/usr/bin/env python3
"""SLM-310 (LAR2-03): matched fixture experiment — action alignment in X22.

Question: does aligning the corruption sampler with a declared inverse-action
distribution (ADD-balanced) close the training-target vs decode-demand gap,
and does corrected STOP-slot accounting (STOP consumes an expand_per_state
slot only when its frozen candidate is retained) change search efficiency —
each lever isolated, at equal model / data / optimizer budget?

Cells (2×2, levers isolated; A is the baseline):

- A: default corruption sampler × legacy STOP accounting
- B: ADD-balanced corruption sampler × legacy STOP accounting
- C: default corruption sampler × corrected STOP accounting
- D: ADD-balanced corruption sampler × corrected STOP accounting

Preregistered (written into the output payload BEFORE any result; locked —
do not edit after outcomes are visible):

- T1 (corruption lever): within each STOP arm, the ADD-balanced cell must
  raise the ADD training-target share by >= 0.10 absolute vs the default
  cell (sampler hits its declared distribution);
- T2 (corruption lever): within each STOP arm, the ADD-balanced cell must
  not reduce the valid-final rate below the default cell's (no semantic
  regression);
- T3 (STOP lever): within each corruption arm, the corrected cell must
  consume strictly fewer or equal STOP budget slots than the legacy cell
  (duplicate STOPs no longer burn budget) and must not reduce the
  valid-final rate;
- T4 (isolation): B−A differs only in the sampler, C−A only in STOP
  accounting — verified structurally by config diff, not by narration.
- Verdict per lever: ``adopted`` iff its thresholds hold in BOTH cells of
  the pair, else ``rejected``. A negative result is retained, never
  combined.

The distribution audit (training-target vs decode-demand) runs on the
baseline cell and its preregistered prediction decides whether a
class-balanced/focal action loss is warranted; the prediction and the
decision are recorded in the payload (no loss arm was added unless the audit
predicted it — see ``loss_decision``).

Writes ``docs/design/iter-slm310-action-alignment-20260724.{json,md}``.

Example:
  python -m scripts.run_slm310_action_alignment --steps 8
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from slm_training.dsl.parser import validate
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.slm310_action_alignment import (
    action_calibration,
    decode_demand_distribution,
    run_distribution_audit,
    training_target_distribution,
)
from slm_training.models.tree_edit_diffusion import (
    ACTION_STOP,
    TreeEditDiffusionConfig,
    TreeEditDiffusionModel,
)
from slm_training.versioning import build_version_stamp

EXPERIMENT_ID = "slm310-action-alignment"
DEFAULT_JSON_OUT = Path("docs/design/iter-slm310-action-alignment-20260724.json")
DEFAULT_MD_OUT = Path("docs/design/iter-slm310-action-alignment-20260724.md")

# PREREGISTERED thresholds (locked before any run; deviations are append-only
# and exploratory).
PREREGISTERED_THRESHOLDS = {
    "t1_add_target_share_gain_min": 0.10,
    "t2_valid_final_no_regression": True,
    "t3_stop_budget_not_worse_and_valid_final_no_regression": True,
    "t4_lever_isolation_structural": True,
    "verdict_rule": (
        "per lever: adopted iff its thresholds hold in both cells of the "
        "matched pair, else rejected; levers are never combined"
    ),
}

# Declared inverse-action distribution for the ADD-balanced corruption arm:
# ADD lifted well above its default ~10% share (only REMOVE mutations yield
# ADD inverses), all other reachable inverse actions uniform.
ADD_BALANCED_DISTRIBUTION = {
    "REPLACE": 1.0,
    "ADD": 4.0,
    "REMOVE": 1.0,
    "ADD_CONTAINER": 1.0,
    "REMOVE_CONTAINER": 1.0,
    "INSERT_SUBTREE": 1.0,
    "REPLACE_SUBTREE": 1.0,
    "REPLACE_STATEMENT": 1.0,
    "BIND_PLACEHOLDER": 1.0,
}

CELLS = {
    "cell_a": {"corruption": None, "stop": "legacy"},
    "cell_b": {"corruption": ADD_BALANCED_DISTRIBUTION, "stop": "legacy"},
    "cell_c": {"corruption": None, "stop": "corrected"},
    "cell_d": {"corruption": ADD_BALANCED_DISTRIBUTION, "stop": "corrected"},
}

# Tiny fixture corpus (same shape as the SLM-308 matched experiment).
FIXTURE_PROGRAMS: list[tuple[str, str, list[str]]] = [
    ("hero with cta", 'root = Stack([t, b], "column")\nt = TextContent(":hero.title")\nb = Button(":cta.label")', [":hero.title", ":cta.label"]),
    ("simple text", 'root = Stack([t], "column")\nt = TextContent(":body")', [":body"]),
    ("form with input", 'root = Form([i, s], "column")\ni = TextInput(":form.email")\ns = Button(":form.submit")', [":form.email", ":form.submit"]),
    ("card with image", 'root = Stack([c], "column")\nc = Card([im, t], "column")\nim = Image(":img.src")\nt = TextContent(":img.caption")', [":img.src", ":img.caption"]),
    ("two texts", 'root = Stack([a, b], "column")\na = TextContent(":title")\nb = TextContent(":subtitle")', [":title", ":subtitle"]),
    ("button only", 'root = Stack([b], "column")\nb = Button(":cta")', [":cta"]),
    ("nested card", 'root = Stack([c], "column")\nc = Card([t], "column")\nt = TextContent(":card.body")', [":card.body"]),
    ("form pair", 'root = Form([i], "column")\ni = TextInput(":q")', [":q"]),
]


def build_fixture_records() -> list[ExampleRecord]:
    return [
        ExampleRecord(
            id=f"fixture_{i:02d}",
            prompt=prompt,
            openui=openui,
            placeholders=list(placeholders),
            split="train",
        )
        for i, (prompt, openui, placeholders) in enumerate(FIXTURE_PROGRAMS)
    ]


def train_cell(
    records: list[ExampleRecord],
    corruption: dict[str, float] | None,
    stop: str,
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> TreeEditDiffusionModel:
    """Train one cell; cells share corpus, init seed, batch order, optimizer
    budget, and value_label_mode — only the cell's lever(s) differ."""
    torch.manual_seed(seed)
    config = TreeEditDiffusionConfig(
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        seed=seed,
        max_chain=3,
        value_label_mode="mutation_count",
        context_backend="scratch",
        corruption_action_distribution=corruption,
        stop_slot_accounting=stop,
    )
    model = TreeEditDiffusionModel.from_records(records, config=config, device="cpu")
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=3e-3)
    order = random.Random(0)  # identical batch order across cells
    model.train()
    for _ in range(steps):
        batch = [records[order.randrange(len(records))] for _ in range(batch_size)]
        loss = model.training_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def score_cell(
    model: TreeEditDiffusionModel,
    records: list[ExampleRecord],
    *,
    audit_samples: int,
    seed: int,
) -> dict:
    """Per-cell metrics: dead-candidate rate, applicable-ADD recall, action
    calibration, verifier calls, semantic outcomes (valid-final rate)."""
    weights = model._inverse_action_weights()
    training = training_target_distribution(
        records,
        model.space,
        seed=seed,
        samples=audit_samples,
        max_chain=model.config.max_chain,
        inverse_action_weights=weights,
    )
    demand = decode_demand_distribution(model, records, include_outputs=True)
    outputs = demand.pop("outputs")
    valid_final = 0
    for text in outputs:
        try:
            validate(text)
            valid_final += 1
        except Exception:  # noqa: BLE001
            pass
    calib = action_calibration(training, demand["overall"])
    overall = demand["overall"]
    proposals = [
        p for ev in demand["evidence"] for p in ev.get("proposals", [])
    ]
    stop_budget = sum(
        1 for p in proposals if p["action"] == ACTION_STOP and p["consumed_budget"]
    )
    stop_visited = sum(1 for p in proposals if p["action"] == ACTION_STOP)
    return {
        "training_targets": training,
        "decode_demand": {
            "overall": overall,
            "by_suite": demand["by_suite"],
            "by_source": demand["by_source"],
        },
        "calibration": calib,
        "dead_candidate_rate": overall["dead_candidate_rate"],
        "applicable_add_recall": overall["applicable_add_recall"],
        "verifier_calls": overall["verifier_calls"],
        "stop_budget_consumed": stop_budget,
        "stop_proposals_visited": stop_visited,
        "valid_final_rate": valid_final / max(len(records), 1),
        "n_records": len(records),
        "training_metrics": model.last_training_metrics,
    }


def build_report(
    records: list[ExampleRecord],
    *,
    steps: int,
    batch_size: int,
    seed: int,
    audit_samples: int,
) -> dict:
    cells: dict[str, dict] = {}
    for cell, spec in CELLS.items():
        model = train_cell(
            records,
            spec["corruption"],
            spec["stop"],
            steps=steps,
            batch_size=batch_size,
            seed=seed,
        )
        metrics = score_cell(model, records, audit_samples=audit_samples, seed=seed)
        cells[cell] = {"levers": spec, "metrics": metrics}

    # T4: lever isolation is structural — pairwise config diffs must name
    # exactly one lever.
    def lever_diff(x: str, y: str) -> list[str]:
        return [
            key
            for key in ("corruption", "stop")
            if CELLS[x][key] != CELLS[y][key]
        ]

    isolation = {
        "b_minus_a": lever_diff("cell_b", "cell_a"),
        "c_minus_a": lever_diff("cell_c", "cell_a"),
        "d_minus_b": lever_diff("cell_d", "cell_b"),
        "d_minus_c": lever_diff("cell_d", "cell_c"),
    }
    t4_ok = (
        isolation["b_minus_a"] == ["corruption"]
        and isolation["c_minus_a"] == ["stop"]
        and isolation["d_minus_b"] == ["stop"]
        and isolation["d_minus_c"] == ["corruption"]
    )

    thr = PREREGISTERED_THRESHOLDS
    checks: dict[str, dict] = {}
    # T1/T2: corruption lever, compared within each STOP arm (A→B, C→D).
    t1 = {}
    t2 = {}
    for base, balanced in (("cell_a", "cell_b"), ("cell_c", "cell_d")):
        a_add = cells[base]["metrics"]["training_targets"]["shares"]["ADD"]
        b_add = cells[balanced]["metrics"]["training_targets"]["shares"]["ADD"]
        t1[f"{balanced}_minus_{base}"] = b_add - a_add
        a_vf = cells[base]["metrics"]["valid_final_rate"]
        b_vf = cells[balanced]["metrics"]["valid_final_rate"]
        t2[f"{balanced}_vs_{base}"] = b_vf >= a_vf
    t1_ok = all(g >= thr["t1_add_target_share_gain_min"] for g in t1.values())
    t2_ok = all(t2.values())
    # T3: STOP lever, compared within each corruption arm (A→C, B→D).
    t3 = {}
    for legacy, corrected in (("cell_a", "cell_c"), ("cell_b", "cell_d")):
        leg = cells[legacy]["metrics"]
        cor = cells[corrected]["metrics"]
        t3[f"{corrected}_vs_{legacy}"] = {
            "stop_budget_legacy": leg["stop_budget_consumed"],
            "stop_budget_corrected": cor["stop_budget_consumed"],
            "budget_ok": cor["stop_budget_consumed"] <= leg["stop_budget_consumed"],
            "valid_final_ok": cor["valid_final_rate"] >= leg["valid_final_rate"],
        }
    t3_ok = all(v["budget_ok"] and v["valid_final_ok"] for v in t3.values())

    checks = {
        "t1_add_target_share_gains": t1,
        "t1_ok": t1_ok,
        "t2_valid_final": t2,
        "t2_ok": t2_ok,
        "t3_stop_accounting": t3,
        "t3_ok": t3_ok,
        "t4_lever_isolation": isolation,
        "t4_ok": t4_ok,
    }
    verdicts = {
        "corruption_add_balanced": "adopted" if (t1_ok and t2_ok) else "rejected",
        "stop_slot_corrected": "adopted" if t3_ok else "rejected",
    }

    # Distribution audit on the baseline cell decides whether a
    # class-balanced/focal action loss is warranted (preregistered rule).
    audit_model = train_cell(
        records, None, "legacy", steps=steps, batch_size=batch_size, seed=seed
    )
    audit = run_distribution_audit(
        audit_model, records, seed=seed, samples=audit_samples, max_chain=3
    )
    predicted = audit["loss_reweighting_prediction"]["predicted"]
    loss_decision = {
        "audit_prediction": audit["loss_reweighting_prediction"],
        "decision": (
            "not_added"
            if not predicted
            else "not_added_preregistered_cells_unchanged"
        ),
        "rationale": (
            "The audit's preregistered rule predicted "
            f"reweighting={predicted}. The matched 2x2 was preregistered "
            "without a loss arm; adding one post-hoc would break lever "
            "isolation, so it is deferred to a follow-up preregistered cell "
            "rather than combined here."
        ),
    }

    payload = {
        "experiment": EXPERIMENT_ID,
        "issue": "SLM-310",
        "question": (
            "Does ADD-balanced corruption sampling close the train/demand "
            "action gap, and does corrected STOP-slot accounting improve "
            "search budget use — each lever isolated at fixture scale?"
        ),
        "preregistered_thresholds": PREREGISTERED_THRESHOLDS,
        "config": {
            "steps": steps,
            "batch_size": batch_size,
            "seed": seed,
            "audit_samples": audit_samples,
            "n_records": len(records),
            "value_label_mode": "mutation_count",
            "add_balanced_distribution": ADD_BALANCED_DISTRIBUTION,
            "cells": CELLS,
        },
        "cells": cells,
        "threshold_checks": checks,
        "verdicts": verdicts,
        "baseline_audit": audit,
        "loss_decision": loss_decision,
        "honesty": (
            "Fixture-scale matched cells; decode-demand metrics come from "
            "SLM-310 per-proposal telemetry (visited = enumerated candidates "
            "the decode loop actually considered). A fixture verdict is "
            "wiring/distribution evidence, not a production ship claim. "
            "Negative results are retained per lever, never combined."
        ),
    }
    payload["version_stamp"] = build_version_stamp(
        "harness.experiments.slm310_action_alignment",
        "harness.experiments.slm299_edit_reachability",
    )
    return payload


def render_markdown(payload: dict) -> str:
    cells = payload["cells"]
    thr = payload["preregistered_thresholds"]
    checks = payload["threshold_checks"]

    def fmt(x: object) -> str:
        return "n/a" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))

    lines = [
        "# SLM-310 (LAR2-03): action alignment — corruption sampler × STOP-slot accounting",
        "",
        f"**Verdicts: corruption=ADD-balanced `{payload['verdicts']['corruption_add_balanced']}`, "
        f"STOP-slot corrected `{payload['verdicts']['stop_slot_corrected']}`** "
        "(fixture-scale matched cells; not a ship claim)",
        "",
        "## Preregistered thresholds (locked before results)",
        "",
        f"- T1: ADD training-target share gain >= {thr['t1_add_target_share_gain_min']} (both STOP arms)",
        "- T2: ADD-balanced never reduces valid-final rate",
        "- T3: corrected STOP accounting consumes <= legacy STOP budget and never reduces valid-final rate",
        "- T4: lever isolation is structural (single-lever config diffs)",
        f"- rule: {thr['verdict_rule']}",
        "",
        "## Cells (levers isolated)",
        "",
        "| cell | corruption | STOP accounting | dead-candidate rate | applicable-ADD recall | ADD target share | verifier calls | STOP budget | valid-final |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell, entry in cells.items():
        m = entry["metrics"]
        lev = entry["levers"]
        lines.append(
            f"| {cell} | {'ADD-balanced' if lev['corruption'] else 'default'} "
            f"| {lev['stop']} | {fmt(m['dead_candidate_rate'])} "
            f"| {fmt(m['applicable_add_recall'])} "
            f"| {fmt(m['training_targets']['shares']['ADD'])} "
            f"| {m['verifier_calls']} | {m['stop_budget_consumed']} "
            f"| {fmt(m['valid_final_rate'])} |"
        )
    lines += [
        "",
        "## Threshold checks",
        "",
        f"- T1 gains: {checks['t1_add_target_share_gains']} → ok={checks['t1_ok']}",
        f"- T2 valid-final: {checks['t2_valid_final']} → ok={checks['t2_ok']}",
        f"- T3 STOP accounting: {checks['t3_stop_accounting']} → ok={checks['t3_ok']}",
        f"- T4 isolation: {checks['t4_lever_isolation']} → ok={checks['t4_ok']}",
        "",
        "## Baseline distribution audit (cell A)",
        "",
    ]
    audit = payload["baseline_audit"]
    lines.append(
        f"- dead-candidate rate {fmt(audit['decode_demand']['overall']['dead_candidate_rate'])}, "
        f"applicable-ADD recall {fmt(audit['decode_demand']['overall']['applicable_add_recall'])}, "
        f"calibration MAD {fmt(audit['calibration']['mean_abs_deviation'])}"
    )
    pred = payload["loss_decision"]["audit_prediction"]
    lines += [
        f"- top rejection reasons: {audit['decode_demand']['overall']['rejection_reasons']}",
        f"- loss-reweighting prediction: {pred['predicted']} ({pred['rule']})",
        f"- decision: {payload['loss_decision']['decision']} — {payload['loss_decision']['rationale']}",
        "",
        "## Honesty",
        "",
        payload["honesty"],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--audit-samples", type=int, default=400)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    records = build_fixture_records()
    payload = build_report(
        records,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        audit_samples=args.audit_samples,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"verdicts={payload['verdicts']} "
        f"t1={payload['threshold_checks']['t1_ok']} "
        f"t3={payload['threshold_checks']['t3_ok']} "
        f"-> {args.json_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
