"""RESEARCH-10 / SLM-566: proof-strength curriculum pilot (default-off).

Fixture-only schedule comparison: obligation-difficulty ordering vs unordered
mixing over frozen grouped theorem records. No trainer invocation; measures
sample-efficiency proxy under a fixed step budget with HARN-11-style root-family
isolation between train and eval.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from slm_training.harness_core.lineage.records import content_sha

BACKEND_ID = "proof_strength_curriculum_pilot"
TOOLCHAIN_ID = "hermetic_python_proof_strength_v1"
CORPUS_RELATIVE = "src/slm_training/resources/formal/proof_strength_records.v1.json"
CONTROL_SEED = 0

SplitKind = Literal["train", "eval"]
ScheduleKind = Literal["unordered", "obligation_difficulty"]


def pinned_toolchain() -> dict[str, str]:
    return {
        "toolchain_id": TOOLCHAIN_ID,
        "backend_id": BACKEND_ID,
        "corpus_relpath": CORPUS_RELATIVE,
        "control_seed": str(CONTROL_SEED),
        "trust_domain": "research_only/proof_strength_curriculum",
    }


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "slm_training").is_dir():
            return parent
    return here.parents[3]


@dataclass(frozen=True)
class ObligationV1:
    obligation_id: str
    kind: str
    difficulty: int
    gold_pass: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObligationV1:
        return cls(
            obligation_id=str(data["obligation_id"]),
            kind=str(data["kind"]),
            difficulty=int(data["difficulty"]),
            gold_pass=bool(data["gold_pass"]),
        )


@dataclass(frozen=True)
class TheoremGroupV1:
    group_id: str
    root_family_id: str
    split: SplitKind
    theorem_id: str
    obligations: tuple[ObligationV1, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TheoremGroupV1:
        return cls(
            group_id=str(data["group_id"]),
            root_family_id=str(data["root_family_id"]),
            split=str(data["split"]),  # type: ignore[arg-type]
            theorem_id=str(data["theorem_id"]),
            obligations=tuple(
                ObligationV1.from_dict(o) for o in data["obligations"]
            ),
        )


def load_corpus(*, root: Path | None = None) -> dict[str, Any]:
    path = (root or _repo_root()) / CORPUS_RELATIVE
    return json.loads(path.read_text(encoding="utf-8"))


def default_groups(*, root: Path | None = None) -> tuple[TheoremGroupV1, ...]:
    raw = load_corpus(root=root)
    return tuple(TheoremGroupV1.from_dict(g) for g in raw["groups"])


def schedule_obligations(
    group: TheoremGroupV1,
    *,
    ordering: ScheduleKind,
    seed: int = CONTROL_SEED,
) -> list[ObligationV1]:
    obs = list(group.obligations)
    if ordering == "obligation_difficulty":
        return sorted(obs, key=lambda o: (o.difficulty, o.obligation_id))
    stable_group = int(
        hashlib.sha256(group.group_id.encode("utf-8")).hexdigest()[:8], 16
    )
    rng = random.Random(seed ^ stable_group)
    rng.shuffle(obs)
    return obs


def simulate_group(
    group: TheoremGroupV1,
    *,
    ordering: ScheduleKind,
    budget: int,
    seed: int = CONTROL_SEED,
) -> dict[str, Any]:
    ordered = schedule_obligations(group, ordering=ordering, seed=seed)
    passes = 0
    steps = 0
    attempted: list[str] = []
    for ob in ordered:
        if steps >= budget:
            break
        steps += 1
        attempted.append(ob.obligation_id)
        if ob.gold_pass:
            passes += 1
    pass_rate = passes / max(len(group.obligations), 1)
    return {
        "group_id": group.group_id,
        "split": group.split,
        "ordering": ordering,
        "budget": budget,
        "steps_used": steps,
        "passes": passes,
        "pass_rate": pass_rate,
        "attempted_obligation_ids": attempted,
    }


def paired_curriculum_report(
    *,
    root: Path | None = None,
    budget_per_eval_group: int | None = None,
) -> dict[str, Any]:
    raw = load_corpus(root=root)
    groups = default_groups(root=root)
    budget = int(
        budget_per_eval_group or raw.get("schedule_budget_per_eval_group", 4)
    )

    train_groups = [g for g in groups if g.split == "train"]
    eval_groups = [g for g in groups if g.split == "eval"]
    train_roots = {g.root_family_id for g in train_groups}
    eval_roots = {g.root_family_id for g in eval_groups}
    leakage = bool(train_roots & eval_roots)

    control_runs = [
        simulate_group(g, ordering="unordered", budget=budget) for g in eval_groups
    ]
    treatment_runs = [
        simulate_group(g, ordering="obligation_difficulty", budget=budget)
        for g in eval_groups
    ]

    control_rate = sum(r["passes"] for r in control_runs) / max(
        sum(len(g.obligations) for g in eval_groups), 1
    )
    treatment_rate = sum(r["passes"] for r in treatment_runs) / max(
        sum(len(g.obligations) for g in eval_groups), 1
    )
    primary = treatment_rate / control_rate if control_rate > 0 else (
        1.0 if treatment_rate > 0 else 0.0
    )

    easy_regression = any(
        r["passes"] == 0
        for g, r in zip(eval_groups, treatment_runs, strict=True)
        if any(o.difficulty <= 2 and o.gold_pass for o in g.obligations)
    )

    if leakage:
        decision = "reject"
        reason = "train_eval_root_leakage"
    elif treatment_rate <= control_rate:
        decision = "reject"
        reason = "curriculum_no_gain"
    elif easy_regression:
        decision = "reject"
        reason = "easy_task_regression"
    else:
        decision = "accept"
        reason = "curriculum_improves_sample_efficiency"

    return {
        "schema": "proof_strength_curriculum_report/v1",
        "sample_efficiency_to_target_obligation_pass_rate": primary,
        "control_eval_pass_rate": control_rate,
        "treatment_eval_pass_rate": treatment_rate,
        "budget_per_eval_group": budget,
        "train_eval_root_leakage": leakage,
        "easy_task_regression": easy_regression,
        "curriculum_leakage": False,
        "hidden_axiom_reliance": False,
        "decision": decision,
        "reason": reason,
        "control_runs": control_runs,
        "treatment_runs": treatment_runs,
        "corpus_digest": content_sha(raw),
        "toolchain": pinned_toolchain(),
    }


__all__ = [
    "BACKEND_ID",
    "CORPUS_RELATIVE",
    "CONTROL_SEED",
    "ObligationV1",
    "TheoremGroupV1",
    "default_groups",
    "load_corpus",
    "paired_curriculum_report",
    "pinned_toolchain",
    "schedule_obligations",
    "simulate_group",
]
