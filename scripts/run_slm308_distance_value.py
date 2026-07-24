#!/usr/bin/env python3
"""SLM-308 (LAR2-02): matched fixture experiment — mutation-count vs bounded-distance value labels.

Question: does replacing the X22 tree-edit model's mutation-count value
labels (``1 - applied/(max_chain+1)``) with SLM-308 oracle labels (normalized
cost-to-go from the bounded reverse-BFS distance oracle + pairwise
parent/improving-child ranking) improve value quality and value-guided beam
selection, at equal model / data / optimizer budget?

Preregistered (written into the output payload BEFORE any result):

- primary: beam selection regret vs oracle-best must improve by **>= 0.05**
  absolute (arm A regret − arm B regret, distance units);
- secondary: Spearman rank correlation between value and oracle distance
  must improve by **>= 0.10** (arm B − arm A);
- verdict ``adopted`` iff both thresholds are met, else ``rejected`` —
  honestly computed from the measured arms, never narrated.

Arms share the tiny fixture corpus, model init seed, mutation chains, batch
order, and optimizer budget; they differ ONLY in ``value_label_mode``.
Evaluation covers near-gold states (gold mutated 1..3 edits) AND
seed-trajectory states (random walks from the decode seed): Spearman rank
correlation, pairwise concordance, Brier + per-bin calibration over distance
bins d=0..8+, beam selection regret vs oracle-best, UNKNOWN coverage, and
oracle cost (nodes expanded / ms). States the oracle cannot measure (UNKNOWN
or unbounded) are excluded from distance-referenced metrics and counted,
never coerced.

Writes ``docs/design/iter-slm308-distance-value-20260724.{json,md}``.

Example:
  python -m scripts.run_slm308_distance_value --steps 8
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
    DistanceKind,
    cache_stats,
    clear_caches,
    distance_to_target,
    effective_distance,
)
from slm_training.models.tree_edit_diffusion import (
    ACTION_STOP,
    TreeEditDiffusionConfig,
    TreeEditDiffusionModel,
    TreeEditSpace,
    parse_statements,
    render_statements,
)
from slm_training.versioning import build_version_stamp

EXPERIMENT_ID = "slm308-distance-value"
DEFAULT_JSON_OUT = Path("docs/design/iter-slm308-distance-value-20260724.json")
DEFAULT_MD_OUT = Path("docs/design/iter-slm308-distance-value-20260724.md")

# PREREGISTERED improvement thresholds (locked before any run; do not edit
# after outcomes are visible — deviations are append-only and exploratory).
PREREGISTERED_THRESHOLDS = {
    "beam_regret_improvement_min": 0.05,
    "rank_correlation_improvement_min": 0.10,
    "direction": "arm_b(bounded_distance) minus arm_a(mutation_count)",
    "verdict_rule": "adopted iff both thresholds are met, else rejected",
}

EVAL_MAX_DEPTH = 8
EVAL_NODE_BUDGET = 8

# Tiny fixture corpus: small valid programs with 1-2 placeholders.
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

SEED_SOURCE = 'root = Stack([], "column")'


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


# --- metrics -----------------------------------------------------------------


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation; None when undefined (ties everywhere)."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            mean_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = mean_rank
            i = j + 1
        return out

    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def concordance(preds: list[float], refs: list[float]) -> float | None:
    """Fraction of comparable pairs the prediction orders correctly (ties in
    reference skipped; prediction ties count as half)."""
    total = 0
    correct = 0.0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            if refs[i] == refs[j]:
                continue
            total += 1
            want = refs[i] < refs[j]
            if preds[i] == preds[j]:
                correct += 0.5
            elif (preds[i] < preds[j]) == want:
                correct += 1.0
    return (correct / total) if total else None


def calibration_bins(
    preds: list[float], refs: list[float], max_depth: int = EVAL_MAX_DEPTH
) -> list[dict[str, float | int]]:
    """Per distance bin d=0..8+: mean predicted value vs mean normalized
    cost-to-go target (1 - d/max_depth). Bins are by rounded reference
    distance; d>=8 folds into the final bin."""
    bins: dict[int, list[tuple[float, float]]] = {}
    for pred, ref in zip(preds, refs):
        d = min(int(round(ref)), max_depth)
        bins.setdefault(d, []).append((pred, 1.0 - min(ref, max_depth) / max_depth))
    return [
        {
            "distance_bin": d,
            "n": len(rows),
            "mean_predicted_value": sum(r[0] for r in rows) / len(rows),
            "mean_target": sum(r[1] for r in rows) / len(rows),
        }
        for d, rows in sorted(bins.items())
    ]


# --- eval states ---------------------------------------------------------------


def build_eval_states(
    records: list[ExampleRecord], space: TreeEditSpace, *, seed: int = 20260724
) -> list[dict]:
    """Deterministic near-gold and seed-trajectory states per record."""
    states: list[dict] = []
    for idx, record in enumerate(records):
        gold = parse_statements(record.openui)
        assert gold is not None
        inventory = [
            p if p.startswith(":") else f":{p}" for p in record.placeholders
        ]
        # Near-gold: mutate gold 1..3 edits.
        rng = random.Random(seed + idx)
        for k in (1, 2, 3):
            current = gold
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
                    "origin": "near_gold",
                    "witness": applied,
                    "statements": current,
                    "target": gold,
                    "inventory": inventory,
                }
            )
        # Seed-trajectory: random walks from the decode seed (1, 3 edits).
        seed_statements = parse_statements(SEED_SOURCE)
        assert seed_statements is not None
        for k in (1, 3):
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
                    "origin": "seed_trajectory",
                    "witness": None,  # not a gold mutation chain: no witness
                    "statements": current,
                    "target": gold,
                    "inventory": inventory,
                }
            )
    return states


def label_eval_states(states: list[dict], space: TreeEditSpace) -> dict:
    """Oracle labels for every eval state (shared across arms) + oracle cost."""
    clear_caches()
    start = time.perf_counter()
    n_unknown = 0
    for entry in states:
        label = distance_to_target(
            entry["statements"],
            entry["target"],
            space=space,
            inventory=entry["inventory"],
            max_depth=EVAL_MAX_DEPTH,
            node_budget=EVAL_NODE_BUDGET,
            upper_bound_witness=entry["witness"],
        )
        entry["label"] = label
        entry["ref_distance"] = effective_distance(label)
        if label.kind is DistanceKind.UNKNOWN:
            n_unknown += 1
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    stats = cache_stats()
    return {
        "n_states": len(states),
        "n_unknown": n_unknown,
        "unknown_coverage": n_unknown / max(len(states), 1),
        "n_measurable": sum(1 for e in states if e["ref_distance"] is not None),
        "wall_ms_total": elapsed_ms,
        "ms_per_state": elapsed_ms / max(len(states), 1),
        "cache": stats,
    }


# --- training ------------------------------------------------------------------


def train_arm(
    records: list[ExampleRecord],
    mode: str,
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> TreeEditDiffusionModel:
    torch.manual_seed(seed)
    config = TreeEditDiffusionConfig(
        seed=seed,
        value_label_mode=mode,
        context_backend="scratch",
        max_chain=3,
    )
    model = TreeEditDiffusionModel.from_records(records, config=config, device="cpu")
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=3e-3)
    order = random.Random(0)  # identical batch order across arms
    model.train()
    for _ in range(steps):
        batch = [records[order.randrange(len(records))] for _ in range(batch_size)]
        loss = model.training_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


# --- arm scoring -----------------------------------------------------------------


@torch.no_grad()
def _model_value(model: TreeEditDiffusionModel, prompt: str, source: str) -> float:
    ctx, ctx_pad = model._encode_context([prompt])
    out = model.policy(
        model._state_batch([source]), model.tokenizer.pad_id, ctx, ctx_pad
    )
    return float(out["value"][0])


@torch.no_grad()
def beam_regret(
    model: TreeEditDiffusionModel,
    record: ExampleRecord,
    entry: dict,
) -> float | None:
    """Value-guided single-step selection regret vs oracle-best, in distance
    units. Candidates mirror decode: STOP plus the policy's top applicable
    edits; the value head picks; the oracle scores every candidate. None when
    any candidate is oracle-unmeasurable (excluded, never coerced)."""
    inventory = entry["inventory"]
    statements = entry["statements"]
    prompt = model._format_context(
        record.prompt, design_md=record.design_md, slot_contract=inventory
    )
    ctx, ctx_pad = model._encode_context([prompt])
    out = model.policy(
        model._state_batch([render_statements(statements)]),
        model.tokenizer.pad_id,
        ctx,
        ctx_pad,
    )
    candidates = model._enumerate_edits(out, 0, len(statements), len(inventory))
    # (child statements or None for STOP)
    options: list[list | None] = []
    expanded = 0
    for _score, edit in candidates:
        if expanded >= model.config.expand_per_state:
            break
        if edit.action == ACTION_STOP:
            options.append(None)
            expanded += 1
            continue
        child = model.space.apply(statements, edit, inventory)
        if child is None:
            continue
        options.append(child)
        expanded += 1
    if not options:
        return None
    child_sources = [
        render_statements(child) if child is not None else render_statements(statements)
        for child in options
    ]
    values = model.policy(
        model._state_batch(child_sources),
        model.tokenizer.pad_id,
        ctx.expand(len(options), -1, -1),
        ctx_pad.expand(len(options), -1),
    )["value"]
    distances: list[float] = []
    for child in options:
        # A candidate child of a witness-backed state is one edit further from
        # gold along the same witness path: hi = witness + 1 is proven.
        witness = (
            entry["witness"] + 1 if entry["witness"] is not None else None
        )
        label = distance_to_target(
            child if child is not None else statements,
            entry["target"],
            space=model.space,
            inventory=inventory,
            max_depth=EVAL_MAX_DEPTH,
            node_budget=EVAL_NODE_BUDGET,
            upper_bound_witness=witness,
        )
        ref = effective_distance(label)
        if ref is None:
            return None
        distances.append(ref)
    pick = max(range(len(options)), key=lambda i: float(values[i]))
    return distances[pick] - min(distances)


def score_arm(
    model: TreeEditDiffusionModel,
    records: list[ExampleRecord],
    states: list[dict],
) -> dict:
    measurable = [e for e in states if e["ref_distance"] is not None]
    preds: list[float] = []
    refs: list[float] = []
    regrets: list[float] = []
    n_regret_excluded = 0
    for entry in measurable:
        record = records[entry["record_idx"]]
        prompt = model._format_context(
            record.prompt,
            design_md=record.design_md,
            slot_contract=entry["inventory"],
        )
        value = _model_value(model, prompt, render_statements(entry["statements"]))
        preds.append(value)
        refs.append(entry["ref_distance"])
        regret = beam_regret(model, record, entry)
        if regret is None:
            n_regret_excluded += 1
        else:
            regrets.append(regret)
    # Higher value must mean closer => correlation against NEGATIVE distance.
    rank = spearman(preds, [-r for r in refs])
    conc = concordance(preds, [-r for r in refs])
    targets = [1.0 - min(r, EVAL_MAX_DEPTH) / EVAL_MAX_DEPTH for r in refs]
    brier = (
        sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(preds)
        if preds
        else None
    )
    by_origin: dict[str, dict] = {}
    for origin in ("near_gold", "seed_trajectory"):
        sel = [
            (p, r)
            for p, r, e in zip(preds, refs, measurable)
            if e["origin"] == origin
        ]
        if sel:
            by_origin[origin] = {
                "n": len(sel),
                "rank_correlation": spearman(
                    [p for p, _ in sel], [-r for _, r in sel]
                ),
            }
    return {
        "n_measurable": len(measurable),
        "rank_correlation": rank,
        "concordance": conc,
        "brier": brier,
        "calibration_bins": calibration_bins(preds, refs),
        "beam_regret_mean": (sum(regrets) / len(regrets)) if regrets else None,
        "beam_regret_n": len(regrets),
        "beam_regret_excluded": n_regret_excluded,
        "by_origin": by_origin,
    }


# --- report ---------------------------------------------------------------------


def build_report(
    records: list[ExampleRecord],
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> dict:
    space = TreeEditSpace()
    states = build_eval_states(records, space)
    # Oracle labels are arm-independent: label once, share across arms (the
    # oracle cache is keyed by grammar/action versions + AST hashes).
    oracle = label_eval_states(states, space)
    arms: dict[str, dict] = {}
    for arm, mode in (("arm_a", "mutation_count"), ("arm_b", "bounded_distance")):
        model = train_arm(records, mode, steps=steps, batch_size=batch_size, seed=seed)
        metrics = score_arm(model, records, states)
        arms[arm] = {
            "value_label_mode": mode,
            "metrics": metrics,
            "training_metrics": model.last_training_metrics,
        }
    a, b = arms["arm_a"]["metrics"], arms["arm_b"]["metrics"]

    def _delta(key: str, reverse: bool = False) -> float | None:
        if a.get(key) is None or b.get(key) is None:
            return None
        return (a[key] - b[key]) if reverse else (b[key] - a[key])

    improvements = {
        "rank_correlation": _delta("rank_correlation"),
        "concordance": _delta("concordance"),
        "beam_regret": _delta("beam_regret_mean", reverse=True),
        "brier": _delta("brier", reverse=True),
    }
    rank_ok = (
        improvements["rank_correlation"] is not None
        and improvements["rank_correlation"]
        >= PREREGISTERED_THRESHOLDS["rank_correlation_improvement_min"]
    )
    regret_ok = (
        improvements["beam_regret"] is not None
        and improvements["beam_regret"]
        >= PREREGISTERED_THRESHOLDS["beam_regret_improvement_min"]
    )
    verdict = "adopted" if (rank_ok and regret_ok) else "rejected"
    payload = {
        "experiment": EXPERIMENT_ID,
        "issue": "SLM-308",
        "question": (
            "Do bounded-distance oracle value labels beat mutation-count value "
            "labels at equal fixture budget (rank correlation + beam regret)?"
        ),
        "preregistered_thresholds": PREREGISTERED_THRESHOLDS,
        "config": {
            "steps": steps,
            "batch_size": batch_size,
            "seed": seed,
            "n_records": len(records),
            "eval_max_depth": EVAL_MAX_DEPTH,
            "eval_node_budget": EVAL_NODE_BUDGET,
        },
        "oracle_cost": oracle,
        "arms": arms,
        "improvements": improvements,
        "threshold_checks": {
            "rank_correlation_ok": rank_ok,
            "beam_regret_ok": regret_ok,
        },
        "verdict": verdict,
        "honesty": (
            "Fixture-scale matched arms; UNKNOWN/unbounded states are excluded "
            "from distance-referenced metrics and counted, never coerced. A "
            "fixture verdict is wiring/label evidence, not a production ship "
            "claim."
        ),
    }
    payload["version_stamp"] = build_version_stamp(
        "harness.experiments.slm308_distance_value",
        "harness.experiments.slm299_edit_reachability",
    )
    return payload


def render_markdown(payload: dict) -> str:
    a = payload["arms"]["arm_a"]["metrics"]
    b = payload["arms"]["arm_b"]["metrics"]
    imp = payload["improvements"]
    thr = payload["preregistered_thresholds"]

    def fmt(x: object) -> str:
        return "n/a" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))

    lines = [
        "# SLM-308 (LAR2-02): bounded-distance value labels vs mutation-count",
        "",
        f"**Verdict: `{payload['verdict']}`** (fixture-scale matched arms; not a ship claim)",
        "",
        "## Preregistered thresholds (locked before results)",
        "",
        f"- beam regret improvement (A−B) >= {thr['beam_regret_improvement_min']}",
        f"- rank correlation improvement (B−A) >= {thr['rank_correlation_improvement_min']}",
        f"- rule: {thr['verdict_rule']}",
        "",
        "## Headline",
        "",
        "| metric | arm A (mutation_count) | arm B (bounded_distance) | improvement |",
        "| --- | --- | --- | --- |",
        f"| rank correlation | {fmt(a['rank_correlation'])} | {fmt(b['rank_correlation'])} | {fmt(imp['rank_correlation'])} |",
        f"| concordance | {fmt(a['concordance'])} | {fmt(b['concordance'])} | {fmt(imp['concordance'])} |",
        f"| Brier | {fmt(a['brier'])} | {fmt(b['brier'])} | {fmt(imp['brier'])} |",
        f"| beam regret (mean) | {fmt(a['beam_regret_mean'])} | {fmt(b['beam_regret_mean'])} | {fmt(imp['beam_regret'])} |",
        "",
        "## Coverage + oracle cost",
        "",
    ]
    oracle = payload["oracle_cost"]
    lines.append(
        f"- UNKNOWN coverage {oracle['unknown_coverage']:.3f} "
        f"({oracle['n_unknown']}/{oracle['n_states']}), measurable "
        f"{oracle['n_measurable']}, oracle {oracle['ms_per_state']:.1f} ms/state, "
        f"{oracle['cache']['map_nodes']} nodes expanded"
    )
    lines += [
        "",
        "## Calibration (arm B, distance bins d=0..8+)",
        "",
        "| d | n | mean predicted value | mean target |",
        "| --- | --- | --- | --- |",
    ]
    for row in b["calibration_bins"]:
        lines.append(
            f"| {row['distance_bin']} | {row['n']} | "
            f"{row['mean_predicted_value']:.3f} | {row['mean_target']:.3f} |"
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
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    records = build_fixture_records()
    payload = build_report(
        records, steps=args.steps, batch_size=args.batch_size, seed=args.seed
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"verdict={payload['verdict']} "
        f"rank_imp={payload['improvements']['rank_correlation']} "
        f"regret_imp={payload['improvements']['beam_regret']} "
        f"-> {args.json_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
