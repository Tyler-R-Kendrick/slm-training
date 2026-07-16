"""Judge grammar-legal continuations replayed from one exact decode state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from slm_training.data.quality import independent_judge
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.eval_runner import (
    _is_meaningful_program,
    _placeholder_fidelity,
    _reward_for_prediction,
    component_type_recall,
    structural_similarity,
)


_METRICS = (
    "placeholder_fidelity",
    "component_recall",
    "structural_similarity",
    "reward",
)
SEMANTIC_VERIFIER_V1 = "independent_judge+meaningful_program+pareto_v1"


def _sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def semantic_outcome(record: ExampleRecord, text: str) -> dict[str, Any]:
    """Run independent admission and end-to-end semantic checks on one output."""
    candidate = replace(record, openui=text, accepted_outputs=[])
    judge = independent_judge(candidate)
    meaningful, meaningful_error, serialized = _is_meaningful_program(
        text, gold=record
    )
    scored = serialized or text
    return {
        "judge": judge,
        "meaningful": meaningful,
        "meaningful_error": meaningful_error,
        "verified": bool(judge["ok"] and meaningful),
        "metrics": {
            "placeholder_fidelity": round(_placeholder_fidelity(scored, record), 6),
            "component_recall": round(
                component_type_recall(scored, record.openui), 6
            ),
            "structural_similarity": round(
                structural_similarity(scored, record.openui), 6
            ),
            "reward": round(_reward_for_prediction(scored, record), 6),
        },
    }


def label_pareto_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    """Return verified nondominated tokens and failed/dominated legal tokens."""
    verified = [row for row in candidates if row.get("verified") is True]

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_metrics = left["metrics"]
        right_metrics = right["metrics"]
        no_worse = all(left_metrics[name] >= right_metrics[name] for name in _METRICS)
        better = any(left_metrics[name] > right_metrics[name] for name in _METRICS)
        return no_worse and better

    good = [
        int(row["token_id"])
        for row in verified
        if not any(
            other is not row and dominates(other, row) for other in verified
        )
    ]
    good_rows = [row for row in verified if int(row["token_id"]) in good]
    bad = [
        int(row["token_id"])
        for row in candidates
        if row.get("verified") is not True
        or any(dominates(frontier, row) for frontier in good_rows)
    ]
    return sorted(set(good)), sorted(set(bad) - set(good))


def mine_semantic_counterfactuals(
    model: Any,
    recorder: Any,
    record: ExampleRecord,
    context_text: str,
    *,
    max_states: int = 4,
    max_candidates: int = 4,
    seed: int = 0,
) -> dict[str, int]:
    """Replay bounded grammar-legal alternatives and append qualified evidence."""
    import torch

    if max_states < 1 or max_candidates < 2:
        raise ValueError("counterfactual mining requires states >= 1 and candidates >= 2")
    if bool(getattr(model.config, "grammar_sample_decode", False)):
        raise ValueError("same-state counterfactual replay requires deterministic decode")
    commits = [
        commit
        for step in recorder.steps
        for commit in step.get("commits", ())
        if commit.get("phase") == "compiler_tree"
        and len(commit.get("allowed_id_set") or ()) > 1
        and commit.get("pre_canvas") is not None
    ]
    unique: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    for commit in commits:
        key = (tuple(map(int, commit["pre_canvas"])), int(commit["t"]))
        if key not in seen:
            seen.add(key)
            unique.append(commit)
        if len(unique) >= max_states:
            break

    stats = {
        "states": len(unique),
        "candidates": 0,
        "judge_passed": 0,
        "verified": 0,
        "events": 0,
    }
    if not unique:
        return stats
    ctx, ctx_pad = model._encode_context([context_text])
    contract = (
        list(record.placeholders)
        if getattr(model.config, "slot_contract_constrained_decode", False)
        else None
    )
    previous_recorder = getattr(model, "trace_recorder", None)
    model.trace_recorder = None
    try:
        with torch.inference_mode():
            for commit in unique:
                canvas = list(map(int, commit["pre_canvas"]))
                position = int(commit["t"])
                legal = sorted({int(value) for value in commit["allowed_id_set"]})
                selected = int(commit["id"])
                if selected not in legal:
                    raise ValueError("selected compiler token is absent from legal support")
                canvas_tensor = torch.tensor(
                    [canvas], dtype=torch.long, device=model.device_name
                )
                hidden = model._denoiser_hidden(canvas_tensor, ctx, ctx_pad)
                scores = model._project_candidates(hidden[0, position], tuple(legal))
                ranked = [
                    legal[index]
                    for index in torch.argsort(scores, descending=True).tolist()
                ]
                candidate_ids = [selected]
                candidate_ids.extend(token for token in ranked if token != selected)
                candidate_ids = candidate_ids[:max_candidates]
                outcomes: list[dict[str, Any]] = []
                for token_id in candidate_ids:
                    prefix = tuple(canvas[:position]) + (int(token_id),)
                    decoded = model._compiler_ltr_decode_one(
                        ctx,
                        ctx_pad,
                        len(canvas),
                        mode="tree",
                        slot_contract=contract,
                        _initial_prefix=prefix,
                        _disable_trajectory_fork=True,
                    )
                    raw_text = model._decode_openui(
                        decoded, placeholders=list(record.placeholders)
                    )
                    text = model._ensure_valid_openui(
                        raw_text,
                        ctx,
                        ctx_pad,
                        len(canvas),
                        attempts=0,
                        slot_contract=contract,
                    )
                    outcome = semantic_outcome(record, text)
                    outcomes.append(
                        {
                            "token_id": int(token_id),
                            "token": model.tokenizer.id_to_token.get(
                                int(token_id), ""
                            ),
                            "selected": int(token_id) == selected,
                            "raw_text": raw_text,
                            "text": text,
                            "finalization_changed": raw_text.strip() != text.strip(),
                            **outcome,
                        }
                    )
                stats["candidates"] += len(outcomes)
                stats["judge_passed"] += sum(
                    row["judge"]["ok"] is True for row in outcomes
                )
                stats["verified"] += sum(
                    row["verified"] is True for row in outcomes
                )
                good, bad = label_pareto_candidates(outcomes)
                state_identity = {
                    "context_text": context_text,
                    "pre_canvas": canvas,
                    "position": position,
                    "seed": int(seed),
                }
                state_hash = _sha(state_identity)
                recorder.event(
                    "counterfactual_probe",
                    same_state_verified=True,
                    state_hash=state_hash,
                    pre_canvas=canvas,
                    position=position,
                    selected_token_id=selected,
                    legal_token_ids=legal,
                    good_token_ids=good,
                    bad_token_ids=bad,
                    qualified=bool(good and bad),
                    rejection_reason=(
                        None
                        if good and bad
                        else (
                            "no_verified_frontier"
                            if not good
                            else "no_semantic_separation"
                        )
                    ),
                    verifier={
                        "name": SEMANTIC_VERIFIER_V1,
                        "metrics": list(_METRICS),
                    },
                    candidates=outcomes,
                )
                if not good or not bad:
                    continue
                recorder.event(
                    "counterfactual_decision",
                    same_state_verified=True,
                    state_hash=state_hash,
                    pre_canvas=canvas,
                    position=position,
                    selected_token_id=selected,
                    good_token_ids=good,
                    bad_token_ids=bad,
                    legal_token_ids=legal,
                    evidence_confidence=1.0,
                    decision_kind=str(
                        commit.get("decision_kind") or "compiler_tree"
                    ),
                    verifier={
                        "name": SEMANTIC_VERIFIER_V1,
                        "metrics": list(_METRICS),
                    },
                )
                stats["events"] += 1
    finally:
        model.trace_recorder = previous_recorder
    return stats
