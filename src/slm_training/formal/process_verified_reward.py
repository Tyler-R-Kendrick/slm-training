"""RESEARCH-11 / SLM-574: process-verified reward shaping pilot (default-off).

Fixture-only comparison of terminal-only proof rewards vs dense process signals
derived only from verified tactic prefixes and checked failure localization.
No trainer invocation; measures reward correlation and leakage guards on frozen
proof-attempt records.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.harness_core.lineage.records import content_sha

BACKEND_ID = "process_verified_reward_pilot"
TOOLCHAIN_ID = "hermetic_python_process_reward_v1"
CORPUS_RELATIVE = "src/slm_training/resources/formal/process_verified_reward_corpus.v1.json"

PREFIX_STEP_REWARD = 0.15
FAILURE_LOCALIZATION_REWARD = 0.1
TERMINAL_SUCCESS_REWARD = 1.0
MAX_PROCESS_REWARD = 0.85


def pinned_toolchain() -> dict[str, str]:
    return {
        "toolchain_id": TOOLCHAIN_ID,
        "backend_id": BACKEND_ID,
        "corpus_relpath": CORPUS_RELATIVE,
        "trust_domain": "research_only/process_verified_reward",
    }


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "slm_training").is_dir():
            return parent
    return here.parents[3]


@dataclass(frozen=True)
class ProofAttemptV1:
    attempt_id: str
    final_success: bool
    verified_prefix_steps: int
    total_steps: int
    verified_failure_class: str | None
    leakage_fields_present: bool
    notes: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProofAttemptV1:
        fc = data.get("verified_failure_class")
        return cls(
            attempt_id=str(data["attempt_id"]),
            final_success=bool(data["final_success"]),
            verified_prefix_steps=int(data["verified_prefix_steps"]),
            total_steps=int(data["total_steps"]),
            verified_failure_class=str(fc) if fc else None,
            leakage_fields_present=bool(data.get("leakage_fields_present", False)),
            notes=str(data.get("notes", "")),
        )


def load_corpus(*, root: Path | None = None) -> dict[str, Any]:
    path = (root or _repo_root()) / CORPUS_RELATIVE
    return json.loads(path.read_text(encoding="utf-8"))


def default_attempts(*, root: Path | None = None) -> tuple[ProofAttemptV1, ...]:
    raw = load_corpus(root=root)
    return tuple(ProofAttemptV1.from_dict(a) for a in raw["attempts"])


def terminal_reward(attempt: ProofAttemptV1) -> float:
    return TERMINAL_SUCCESS_REWARD if attempt.final_success else 0.0


def process_verified_reward(attempt: ProofAttemptV1) -> float:
    """Dense reward from verified evidence only; ignores unverified step counts."""

    if attempt.leakage_fields_present:
        return terminal_reward(attempt)

    reward = 0.0
    if attempt.verified_prefix_steps > 0:
        reward += min(
            attempt.verified_prefix_steps * PREFIX_STEP_REWARD,
            MAX_PROCESS_REWARD,
        )
    if attempt.verified_failure_class and attempt.verified_failure_class != "unlocalized":
        reward += FAILURE_LOCALIZATION_REWARD
    if attempt.final_success:
        reward = min(reward + TERMINAL_SUCCESS_REWARD, 1.0)
    return min(reward, 1.0)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def evaluate_control(attempt: ProofAttemptV1) -> dict[str, Any]:
    return {
        "arm": "terminal_only",
        "attempt_id": attempt.attempt_id,
        "reward": terminal_reward(attempt),
        "final_success": attempt.final_success,
    }


def evaluate_treatment(attempt: ProofAttemptV1) -> dict[str, Any]:
    proc = process_verified_reward(attempt)
    unverifiable_inflation = (
        attempt.verified_prefix_steps == 0
        and attempt.total_steps >= 4
        and proc > terminal_reward(attempt)
    )
    return {
        "arm": "process_verified",
        "attempt_id": attempt.attempt_id,
        "reward": proc,
        "final_success": attempt.final_success,
        "verified_prefix_steps": attempt.verified_prefix_steps,
        "unverifiable_intermediate_reward": unverifiable_inflation,
    }


def paired_reward_report(*, root: Path | None = None) -> dict[str, Any]:
    base = root or _repo_root()
    raw = load_corpus(root=base)
    attempts = default_attempts(root=base)

    control_rows = [evaluate_control(a) for a in attempts]
    treatment_rows = [evaluate_treatment(a) for a in attempts]

    success_flags = [1.0 if a.final_success else 0.0 for a in attempts]
    terminal_rewards = [row["reward"] for row in control_rows]
    process_rewards = [row["reward"] for row in treatment_rows]

    terminal_corr = _pearson(terminal_rewards, success_flags)
    process_corr = _pearson(process_rewards, success_flags)

    primary = process_corr

    unverifiable_rate = sum(
        1 for row in treatment_rows if row["unverifiable_intermediate_reward"]
    ) / max(len(attempts), 1)
    leakage_violations = sum(
        1
        for att, treat in zip(attempts, treatment_rows, strict=True)
        if att.leakage_fields_present and treat["reward"] > terminal_reward(att)
    )
    hacking = sum(
        1
        for att, treat in zip(attempts, treatment_rows, strict=True)
        if not att.final_success and treat["reward"] >= TERMINAL_SUCCESS_REWARD
    )

    if leakage_violations > 0:
        decision = "reject"
        reason = "leakage_label_influenced_reward"
    elif hacking > 0:
        decision = "reject"
        reason = "reward_hacking_semantically_invalid"
    elif unverifiable_rate > 0:
        decision = "reject"
        reason = "unverifiable_intermediate_reward_inflation"
    elif process_corr <= terminal_corr:
        decision = "reject"
        reason = "process_reward_no_correlation_gain"
    else:
        decision = "accept"
        reason = "process_verified_shaping_improves_correlation"

    return {
        "schema": "process_verified_reward_report/v1",
        "verified_process_reward_correlation_with_final_proof_success": primary,
        "terminal_reward_correlation": terminal_corr,
        "process_reward_correlation": process_corr,
        "correlation_gain": process_corr - terminal_corr,
        "unverifiable_intermediate_reward_rate": unverifiable_rate,
        "reward_hacking_incidents": hacking,
        "leakage_violations": leakage_violations,
        "decision": decision,
        "reason": reason,
        "control_rows": control_rows,
        "treatment_rows": treatment_rows,
        "corpus_digest": content_sha(raw),
        "toolchain": pinned_toolchain(),
    }


__all__ = [
    "BACKEND_ID",
    "CORPUS_RELATIVE",
    "ProofAttemptV1",
    "default_attempts",
    "evaluate_control",
    "evaluate_treatment",
    "load_corpus",
    "paired_reward_report",
    "pinned_toolchain",
    "process_verified_reward",
    "terminal_reward",
]
