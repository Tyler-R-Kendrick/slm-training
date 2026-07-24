"""SLM-310 (LAR2-03): action-alignment audit for the X22 tree-edit model.

Question: does the inverse-edit supervision distribution (what training
targets ask the policy to repair) match the decode-demand distribution (what
the value-guided beam actually proposes, finds applicable, and selects)?

Two measurement sides, both fixture-scale and deterministic:

- ``training_target_distribution`` replays the training-time sampling loop
  (a ``stop_fraction`` share of clean-state STOP targets + 1..max_chain
  forward-noise chains whose last inverse edit is the supervised target) and
  counts inverse actions.
- ``decode_demand_distribution`` decodes every record and aggregates the
  SLM-310 per-proposal telemetry (``evidence["proposals"]`` /
  ``evidence["states"]``): per action, how many enumerated candidates were
  visited, applicable, and selected — overall, by source record, by suite.

Derived metrics:

- dead-candidate rate: visited proposals with a rejection reason
  (inapplicable or duplicate) over all visited proposals;
- applicable-ADD recall: over the exact decode states where at least one
  ADD edit is applicable in the full edit space, the fraction where the
  decode loop visited at least one applicable ADD proposal;
- action calibration: per-action decode-visited share vs training-target
  share (mean absolute deviation over actions).
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.plugin import GenerationRequest
from slm_training.models.tree_edit_diffusion import (
    ACTION_ADD,
    ACTION_NAMES,
    ACTION_STOP,
    Edit,
    LEAF_COMPONENTS,
    MAX_SLOTS,
    TreeEditDiffusionModel,
    TreeEditSpace,
    parse_statements,
)

#: Preregistered audit rule (locked before any experiment run): recommend a
#: class-balanced/focal action loss only when the measured misalignment is
#: large enough — applicable-ADD recall below this floor OR an ADD
#: train/demand share gap above this bound.
ADD_RECALL_FLOOR = 0.5
ADD_SHARE_GAP = 0.20


def _inventory(record: ExampleRecord) -> list[str]:
    return [
        p if p.startswith(":") else f":{p}"
        for p in (record.placeholders or [])
    ][:MAX_SLOTS]


def training_target_distribution(
    records: list[ExampleRecord],
    space: TreeEditSpace,
    *,
    seed: int = 0,
    samples: int = 400,
    max_chain: int = 4,
    inverse_action_weights: dict[int, float] | None = None,
    stop_fraction: float = 0.2,
) -> dict[str, Any]:
    """Replay the training-time target sampler and count inverse actions.

    Mirrors ``TreeEditDiffusionModel.training_loss``: with probability
    ``stop_fraction`` the target is STOP (clean state); otherwise a
    1..max_chain forward-noise chain is applied and the last inverse edit is
    the target. Records whose program fails to parse are skipped (as in
    training). Deterministic under ``seed``.
    """
    rng = random.Random(seed)
    parsable = [r for r in records if parse_statements(r.openui or "")]
    counts: Counter[int] = Counter()
    attempted = 0
    for _ in range(samples):
        if not parsable:
            break
        record = parsable[rng.randrange(len(parsable))]
        statements = parse_statements(record.openui)
        assert statements is not None
        inventory = _inventory(record)
        attempted += 1
        if rng.random() < stop_fraction:
            counts[ACTION_STOP] += 1
            continue
        k = rng.randint(1, max_chain)
        current = statements
        inverse: Edit | None = None
        for _ in range(k):
            step = space.sample_mutation(
                current,
                inventory,
                rng,
                inverse_action_weights=inverse_action_weights,
            )
            if step is None:
                break
            current, inverse = step
        if inverse is not None:
            counts[inverse.action] += 1
    total = sum(counts.values())
    return {
        "samples": attempted,
        "n_targets": total,
        "counts": {
            ACTION_NAMES[a]: counts.get(a, 0) for a in range(len(ACTION_NAMES))
        },
        "shares": {
            ACTION_NAMES[a]: (counts.get(a, 0) / total if total else 0.0)
            for a in range(len(ACTION_NAMES))
        },
        "inverse_action_weights": (
            None
            if inverse_action_weights is None
            else {
                ACTION_NAMES[a]: w for a, w in sorted(inverse_action_weights.items())
            }
        ),
    }


def applicable_add_exists(
    statements: list, inventory: list[str], space: TreeEditSpace
) -> bool:
    """True when at least one ADD edit is applicable to the state."""
    leaf_comps = [
        i for i, c in enumerate(space.components) if c in LEAF_COMPONENTS
    ]
    for stmt in range(len(statements)):
        for comp in leaf_comps:
            for slot in range(min(len(inventory), MAX_SLOTS)):
                if space.apply(
                    statements, Edit(ACTION_ADD, stmt, comp, slot), inventory
                ):
                    return True
    return False


def _new_bucket() -> dict[str, Any]:
    return {
        "visited": Counter(),
        "applicable": Counter(),
        "selected": Counter(),
        "dead": Counter(),
        "rejection_reasons": Counter(),
        "verifier_calls": 0,
        "states_with_applicable_add": 0,
        "states_recalled": 0,
    }


def _add_proposals(bucket: dict[str, Any], proposals: list[dict[str, Any]]) -> None:
    for p in proposals:
        name = p["action_name"]
        bucket["visited"][name] += 1
        if p["applicable"]:
            bucket["applicable"][name] += 1
        if p["selected"]:
            bucket["selected"][name] += 1
        if p["rejection_reason"] is not None:
            bucket["dead"][name] += 1
            bucket["rejection_reasons"][p["rejection_reason"]] += 1


def _add_recall(
    bucket: dict[str, Any],
    ev: dict[str, Any],
    inventory: list[str],
    space: TreeEditSpace,
) -> None:
    """Exact applicable-ADD recall over the states this decode expanded."""
    proposals = ev.get("proposals", [])
    visited_add_states = {
        (p["step"], p["beam_row"])
        for p in proposals
        if p["action"] == ACTION_ADD and p["applicable"]
    }
    for entry in ev.get("states", []):
        statements = parse_statements(entry["source"])
        if statements is None:
            continue
        if not applicable_add_exists(statements, inventory, space):
            continue
        bucket["states_with_applicable_add"] += 1
        if (entry["step"], entry["beam_row"]) in visited_add_states:
            bucket["states_recalled"] += 1


def _finalize(bucket: dict[str, Any]) -> dict[str, Any]:
    visited = sum(bucket["visited"].values())
    dead = sum(bucket["dead"].values())
    n_add_states = bucket["states_with_applicable_add"]
    return {
        "visited": dict(sorted(bucket["visited"].items())),
        "applicable": dict(sorted(bucket["applicable"].items())),
        "selected": dict(sorted(bucket["selected"].items())),
        "rejection_reasons": dict(sorted(bucket["rejection_reasons"].items())),
        "visited_total": visited,
        "dead_total": dead,
        "dead_candidate_rate": (dead / visited) if visited else None,
        "visited_shares": {
            name: (count / visited if visited else 0.0)
            for name, count in sorted(bucket["visited"].items())
        },
        "verifier_calls": bucket["verifier_calls"],
        "states_with_applicable_add": n_add_states,
        "states_recalled": bucket["states_recalled"],
        "applicable_add_recall": (
            bucket["states_recalled"] / n_add_states if n_add_states else None
        ),
    }


def decode_demand_distribution(
    model: TreeEditDiffusionModel,
    records: list[ExampleRecord],
    *,
    include_outputs: bool = False,
) -> dict[str, Any]:
    """Decode every record and aggregate proposal telemetry by action —
    overall, by source record, and by suite (split)."""
    requests = [
        GenerationRequest(
            prompt=r.prompt,
            slot_contract=tuple(_inventory(r)),
            design_md=r.design_md,
        )
        for r in records
    ]
    outputs = model.generate_batch_requests(requests)
    evidence = model.consume_generation_evidence()
    overall = _new_bucket()
    by_source: dict[str, Any] = {}
    by_suite: dict[str, Any] = {}
    for record, ev in zip(records, evidence):
        inventory = _inventory(record)
        proposals = ev.get("proposals", [])
        overall["verifier_calls"] += ev.get("verifier_calls", 0)
        _add_proposals(overall, proposals)
        _add_recall(overall, ev, inventory, model.space)
        src = by_source.setdefault(record.id, _new_bucket())
        _add_proposals(src, proposals)
        _add_recall(src, ev, inventory, model.space)
        suite = by_suite.setdefault(record.split or "unspecified", _new_bucket())
        _add_proposals(suite, proposals)
        _add_recall(suite, ev, inventory, model.space)
    result = {
        "overall": _finalize(overall),
        "by_source": {k: _finalize(v) for k, v in sorted(by_source.items())},
        "by_suite": {k: _finalize(v) for k, v in sorted(by_suite.items())},
    }
    if include_outputs:
        result["outputs"] = outputs
        result["evidence"] = evidence
    return result


def action_calibration(
    training: dict[str, Any], demand_overall: dict[str, Any]
) -> dict[str, Any]:
    """Per-action training-target share vs decode visited share + MAD."""
    target = training["shares"]
    visited = demand_overall["visited_shares"]
    rows = {}
    mad_terms = []
    for name in ACTION_NAMES:
        t = target.get(name, 0.0)
        d = visited.get(name, 0.0)
        rows[name] = {"target_share": t, "visited_share": d, "gap": d - t}
        mad_terms.append(abs(d - t))
    return {
        "per_action": rows,
        "mean_abs_deviation": sum(mad_terms) / len(mad_terms),
    }


def run_distribution_audit(
    model: TreeEditDiffusionModel,
    records: list[ExampleRecord],
    *,
    seed: int = 0,
    samples: int = 400,
    max_chain: int = 4,
    inverse_action_weights: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Full audit: training-target vs decode-demand distributions, dead
    candidates, applicable-ADD recall, calibration, and the preregistered
    loss-reweighting prediction."""
    training = training_target_distribution(
        records,
        model.space,
        seed=seed,
        samples=samples,
        max_chain=max_chain,
        inverse_action_weights=inverse_action_weights,
    )
    demand = decode_demand_distribution(model, records)
    calib = action_calibration(training, demand["overall"])
    recall = demand["overall"]["applicable_add_recall"]
    add_gap = calib["per_action"]["ADD"]["gap"]
    predicted = bool(
        (recall is not None and recall < ADD_RECALL_FLOOR)
        or abs(add_gap) > ADD_SHARE_GAP
    )
    return {
        "training_targets": training,
        "decode_demand": demand,
        "calibration": calib,
        "loss_reweighting_prediction": {
            "predicted": predicted,
            "rule": (
                f"predict reweighting iff applicable-ADD recall < "
                f"{ADD_RECALL_FLOOR} or |ADD visited share - ADD target "
                f"share| > {ADD_SHARE_GAP}"
            ),
            "add_share_gap": add_gap,
            "applicable_add_recall": recall,
        },
    }
