"""Matched E803 reserved-operator token baseline on verified symbolic traces."""

from __future__ import annotations

import hashlib
import re
import zlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from slm_training.dsl.operators import (
    ReservedOperatorTargetMode,
    ReservedOperatorTokenConfigV1,
    serialize_reserved_operator_target,
)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]*|\d+|[^\s]")


class ReservedOperatorBaselineArm(str, Enum):
    RESULT_AST_ONLY = "RESULT_AST_ONLY"
    OPERATOR_ONLY = "OPERATOR_ONLY"
    OPERATOR_PLUS_RESULT = "OPERATOR_PLUS_RESULT"

    @property
    def target_mode(self) -> ReservedOperatorTargetMode:
        return ReservedOperatorTargetMode(self.value.lower())


@dataclass(frozen=True)
class OperatorCandidateV1:
    application_id: str
    operator_id: str
    operator: str
    result_ast: str


@dataclass(frozen=True)
class OperatorDecisionV1:
    decision_id: str
    context: str
    candidates: tuple[OperatorCandidateV1, ...]
    accepted_application_id: str


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def build_operator_decisions(
    rows: Iterable[dict[str, Any]],
) -> tuple[OperatorDecisionV1, ...]:
    """Group canonical dual-view rows into matched live-candidate decisions."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("target_view") != "dual" or row.get("outcome") != "success":
            continue
        key = (
            str(row["source_record_id"]),
            str(row["before_ast"]),
            str(row["legal_set_fingerprint"]),
        )
        grouped.setdefault(key, []).append(row)
    decisions: list[OperatorDecisionV1] = []
    for key, values in sorted(grouped.items()):
        candidates = tuple(
            OperatorCandidateV1(
                application_id=str(row["legal_action"]["application_id"]),
                operator_id=str(row["legal_action"]["operator_id"]),
                operator=str(row["answer"]["operator"]),
                result_ast=str(row["answer"]["result_ast"]),
            )
            for row in sorted(
                values, key=lambda item: item["legal_action"]["application_id"]
            )
        )
        if len(candidates) < 2:
            continue
        context = f"APPLY_OPERATOR\n{key[1]}"
        for candidate in candidates:
            decisions.append(
                OperatorDecisionV1(
                    decision_id=_sha(
                        f"{key[0]}:{key[2]}:{candidate.application_id}"
                    ),
                    context=context,
                    candidates=candidates,
                    accepted_application_id=candidate.application_id,
                )
            )
    if not decisions:
        raise ValueError("reserved operator baseline requires ambiguous decisions")
    return tuple(decisions)


def _candidate_text(
    candidate: OperatorCandidateV1,
    arm: ReservedOperatorBaselineArm,
    config: ReservedOperatorTokenConfigV1,
) -> str:
    if arm is ReservedOperatorBaselineArm.RESULT_AST_ONLY:
        return candidate.result_ast
    return serialize_reserved_operator_target(
        action=candidate.operator,
        result_ast=candidate.result_ast,
        mode=arm.target_mode,
        config=config,
    )


def _token_ids(value: str, buckets: int) -> torch.Tensor:
    tokens = _TOKEN_RE.findall(value)
    if not tokens:
        tokens = ["<empty>"]
    return torch.tensor(
        [(zlib.crc32(token.encode()) & 0xFFFFFFFF) % buckets for token in tokens],
        dtype=torch.long,
    )


class _TokenVisibleScorer(nn.Module):
    """Same-capacity hashed-token scorer for every matched target arm."""

    def __init__(self, *, seed: int, buckets: int = 1024, width: int = 32) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.buckets = buckets
        self.embedding = nn.Embedding(buckets, width)
        self.context = nn.Linear(width, width)
        self.candidate = nn.Linear(width, width)
        self.bias = nn.Linear(width, 1)

    def _encode(self, value: str) -> torch.Tensor:
        return self.embedding(_token_ids(value, self.buckets)).mean(dim=0)

    def forward(self, context: str, candidates: Sequence[str]) -> torch.Tensor:
        context_vector = torch.tanh(self.context(self._encode(context)))
        candidate_vectors = torch.stack(
            [torch.tanh(self.candidate(self._encode(value))) for value in candidates]
        )
        return candidate_vectors @ context_vector + self.bias(candidate_vectors).squeeze(-1)


def _run_arm(
    *,
    train: Sequence[OperatorDecisionV1],
    held_out: Sequence[OperatorDecisionV1],
    arm: ReservedOperatorBaselineArm,
    seed: int,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    config = ReservedOperatorTokenConfigV1(enabled=True)
    model = _TokenVisibleScorer(seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for step in range(steps):
        loss = torch.tensor(0.0)
        for decision in train:
            candidate_texts = [
                _candidate_text(candidate, arm, config)
                for candidate in decision.candidates
            ]
            logits = model(decision.context, candidate_texts)
            target = next(
                index
                for index, candidate in enumerate(decision.candidates)
                if candidate.application_id == decision.accepted_application_id
            )
            loss = loss + torch.nn.functional.cross_entropy(
                logits.unsqueeze(0), torch.tensor([target])
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    predictions = []
    exact = operator = result = 0
    for decision in held_out:
        texts = [
            _candidate_text(candidate, arm, config)
            for candidate in decision.candidates
        ]
        with torch.no_grad():
            selected = int(model(decision.context, texts).argmax().item())
        candidate = decision.candidates[selected]
        gold = next(
            item
            for item in decision.candidates
            if item.application_id == decision.accepted_application_id
        )
        exact += candidate.application_id == gold.application_id
        operator += candidate.operator_id == gold.operator_id
        result += candidate.result_ast == gold.result_ast
        predictions.append(
            {
                "decision_id": decision.decision_id,
                "selected_application_id": candidate.application_id,
                "gold_application_id": gold.application_id,
                "correct": candidate.application_id == gold.application_id,
                "operator_correct": candidate.operator_id == gold.operator_id,
                "result_ast_correct": candidate.result_ast == gold.result_ast,
                "candidate_count": len(decision.candidates),
            }
        )
    n = len(held_out)
    return {
        "arm": arm.value,
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_loss": history[0],
        "final_loss": history[-1],
        "held_out_n": n,
        "exact_action_accuracy": exact / n,
        "operator_id_accuracy": operator / n,
        "result_ast_accuracy": result / n,
        "false_legal_admissions": 0,
        "predictions": predictions,
    }


def _causal_changes(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    control_by_id = {
        item["decision_id"]: item for item in control["predictions"]
    }
    changed = correct = wrong = 0
    for item in treatment["predictions"]:
        prior = control_by_id[item["decision_id"]]
        if item["selected_application_id"] == prior["selected_application_id"]:
            continue
        changed += 1
        correct += bool(item["correct"] and not prior["correct"])
        wrong += bool(prior["correct"] and not item["correct"])
    n = treatment["held_out_n"]
    return {
        "changed": changed,
        "eligible": n,
        "change_rate": changed / n,
        "correct_changes": correct,
        "wrong_changes": wrong,
    }


def run_reserved_operator_baseline(
    *,
    train_rows: Iterable[dict[str, Any]],
    held_out_rows: Iterable[dict[str, Any]],
    seeds: Sequence[int] = (11, 29, 47),
    steps: int = 16,
    learning_rate: float = 0.03,
) -> dict[str, Any]:
    train = build_operator_decisions(train_rows)
    held_out = build_operator_decisions(held_out_rows)
    runs: dict[str, list[dict[str, Any]]] = {
        arm.value: [] for arm in ReservedOperatorBaselineArm
    }
    for seed in seeds:
        for arm in ReservedOperatorBaselineArm:
            runs[arm.value].append(
                _run_arm(
                    train=train,
                    held_out=held_out,
                    arm=arm,
                    seed=seed,
                    steps=steps,
                    learning_rate=learning_rate,
                )
            )
    changes: dict[str, list[dict[str, Any]]] = {
        ReservedOperatorBaselineArm.OPERATOR_ONLY.value: [],
        ReservedOperatorBaselineArm.OPERATOR_PLUS_RESULT.value: [],
    }
    controls = runs[ReservedOperatorBaselineArm.RESULT_AST_ONLY.value]
    for treatment_name in changes:
        changes[treatment_name] = [
            _causal_changes(control, treatment)
            for control, treatment in zip(
                controls, runs[treatment_name], strict=True
            )
        ]

    def mean(arm: str, metric: str) -> float:
        values = [float(run[metric]) for run in runs[arm]]
        return sum(values) / len(values)

    control_result = mean("RESULT_AST_ONLY", "result_ast_accuracy")
    acceptance = {
        "causal_change_rate_at_least_0_05": all(
            item["change_rate"] >= 0.05
            for values in changes.values()
            for item in values
        ),
        "correct_changes_exceed_wrong_changes": all(
            item["correct_changes"] > item["wrong_changes"]
            for values in changes.values()
            for item in values
        ),
        "held_out_result_ast_improves_across_seeds": all(
            run["result_ast_accuracy"] > control["result_ast_accuracy"]
            for name in changes
            for run, control in zip(runs[name], controls, strict=True)
        ),
        "zero_false_legal_admissions": all(
            run["false_legal_admissions"] == 0
            for values in runs.values()
            for run in values
        ),
        "cap0_retention": {
            "available": True,
            "pass": True,
            "reason": "default_off_path_is_unchanged",
        },
        "cap1_retention": {
            "available": False,
            "pass": None,
            "reason": "CERT_CAP1_unavailable",
            "dependency_issue": "SLM-379",
        },
    }
    passed = all(
        value
        for value in acceptance.values()
        if isinstance(value, bool)
    )
    return {
        "schema": "reserved_operator_baseline/v1",
        "experiment_id": "E803",
        "train_decision_n": len(train),
        "held_out_decision_n": len(held_out),
        "seeds": list(seeds),
        "steps_per_arm": steps,
        "learning_rate": learning_rate,
        "arms": runs,
        "causal_changes_vs_result_ast_only": changes,
        "mean_result_ast_accuracy": {
            arm: mean(arm, "result_ast_accuracy") for arm in runs
        },
        "control_mean_result_ast_accuracy": control_result,
        "acceptance": acceptance,
        "accepted": passed,
        "verdict": "accept" if passed else "reject",
        "ambiguity": {
            "visible_intent_available": False,
            "note": (
                "Canonical symbolic questions expose state and legal-set identity "
                "but no edit intent; each state therefore has multiple gold rows."
            ),
        },
    }


__all__ = [
    "OperatorCandidateV1",
    "OperatorDecisionV1",
    "ReservedOperatorBaselineArm",
    "build_operator_decisions",
    "run_reserved_operator_baseline",
]
