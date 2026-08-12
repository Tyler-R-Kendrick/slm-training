"""RESEARCH-15 / SLM-569: anytime-valid promotion pilot (default-off).

Fixture-only simulation comparing naive fixed-n Holm peeking (control) vs a
mixture e-process confidence sequence (treatment) on matched adaptive Bernoulli
promotion streams. Does not replace deterministic ship gates or production
promotion authority.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from slm_training.evals.power_protocol import _binom_tail_prob, holm_bonferroni
from slm_training.harness_core.lineage.records import content_sha

BACKEND_ID = "anytime_valid_promotion_pilot"
TOOLCHAIN_ID = "hermetic_python_anytime_valid_v1"
CORPUS_RELATIVE = (
    "src/slm_training/resources/formal/anytime_valid_promotion_corpus.v1.json"
)

DecisionArm = Literal["holm_adaptive_peek", "mixture_e_process"]


def pinned_toolchain() -> dict[str, str]:
    return {
        "toolchain_id": TOOLCHAIN_ID,
        "backend_id": BACKEND_ID,
        "corpus_relpath": CORPUS_RELATIVE,
        "trust_domain": "research_only/anytime_valid_promotion",
    }


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "slm_training").is_dir():
            return parent
    return here.parents[3]


@dataclass(frozen=True)
class ScenarioV1:
    scenario_id: str
    kind: Literal["null", "planted"]
    seed: int
    success_p: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "kind": self.kind,
            "seed": self.seed,
            "success_p": self.success_p,
        }


def load_corpus(*, root: Path | None = None) -> tuple[dict[str, Any], tuple[ScenarioV1, ...]]:
    path = (root or _repo_root()) / CORPUS_RELATIVE
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenarios: list[ScenarioV1] = []
    for index in range(int(raw["null_scenarios"]["count"])):
        seed = int(raw["null_scenarios"]["seed_base"]) + index
        scenarios.append(
            ScenarioV1(
                scenario_id=f"null_{index + 1:03d}",
                kind="null",
                seed=seed,
                success_p=float(raw["null_scenarios"]["success_p"]),
            )
        )
    for index in range(int(raw["planted_scenarios"]["count"])):
        seed = int(raw["planted_scenarios"]["seed_base"]) + index
        scenarios.append(
            ScenarioV1(
                scenario_id=f"planted_{index + 1:03d}",
                kind="planted",
                seed=seed,
                success_p=float(raw["planted_scenarios"]["success_p"]),
            )
        )
    return raw, tuple(scenarios)


def default_scenarios(*, root: Path | None = None) -> tuple[ScenarioV1, ...]:
    _, scenarios = load_corpus(root=root)
    return scenarios


def bernoulli_stream(
    *,
    seed: int,
    length: int,
    success_p: float,
) -> tuple[int, ...]:
    rng = random.Random(seed)
    return tuple(1 if rng.random() < success_p else 0 for _ in range(length))


def _one_sided_binomial_p(successes: int, n: int, *, null_p: float) -> float:
    p = _binom_tail_prob(n, successes, null_p, upper=True)
    return min(1.0, max(0.0, p))


def _mixture_e_step(
    *,
    e_states: list[float],
    observation: int,
    lambdas: Sequence[float],
) -> float:
    for index, lam in enumerate(lambdas):
        factor = 1.0 + lam * (2 * observation - 1)
        if factor <= 0.0:
            raise ValueError("non-positive e-process multiplier")
        e_states[index] *= factor
    return sum(e_states) / len(e_states)


def evaluate_arm(
    stream: Sequence[int],
    *,
    arm: DecisionArm,
    alpha: float,
    null_p: float,
    holm_family_size: int,
    lambdas: Sequence[float],
) -> dict[str, Any]:
    if arm == "mixture_e_process":
        threshold = 1.0 / alpha
        e_states = [1.0 for _ in lambdas]
        stop_t = len(stream)
        promoted = False
        mixture = 1.0
        for index, observation in enumerate(stream, start=1):
            mixture = _mixture_e_step(
                e_states=e_states, observation=observation, lambdas=lambdas
            )
            if mixture >= threshold:
                promoted = True
                stop_t = index
                break
        return {
            "arm": arm,
            "promoted": promoted,
            "stop_t": stop_t,
            "final_e_value": mixture,
            "threshold": threshold,
        }

    stop_t = len(stream)
    promoted = False
    final_p = 1.0
    for index in range(1, len(stream) + 1):
        prefix = stream[:index]
        successes = sum(prefix)
        p_primary = _one_sided_binomial_p(successes, index, null_p=null_p)
        hypotheses = [("primary", p_primary)]
        for aux in range(holm_family_size - 1):
            hypotheses.append((f"aux_{aux}", 1.0))
        holm = holm_bonferroni(hypotheses, alpha=alpha)
        primary_row = next(row for row in holm if row["hypothesis_id"] == "primary")
        final_p = float(primary_row["p_value"])
        if primary_row["rejected"]:
            promoted = True
            stop_t = index
            break
    return {
        "arm": arm,
        "promoted": promoted,
        "stop_t": stop_t,
        "final_p_value": final_p,
        "holm_family_size": holm_family_size,
    }


def _fixed_n_oracle(
    stream: Sequence[int],
    *,
    alpha: float,
    null_p: float,
    holm_family_size: int,
) -> bool:
    successes = sum(stream)
    p_primary = _one_sided_binomial_p(successes, len(stream), null_p=null_p)
    holm = holm_bonferroni(
        [("primary", p_primary)]
        + [(f"aux_{index}", 1.0) for index in range(holm_family_size - 1)],
        alpha=alpha,
    )
    return bool(next(row for row in holm if row["hypothesis_id"] == "primary")["rejected"])


def paired_calibration_report(*, root: Path | None = None) -> dict[str, Any]:
    raw, scenarios = load_corpus(root=root)
    alpha = float(raw["alpha"])
    null_p = float(raw["null_p"])
    stream_length = int(raw["stream_length"])
    holm_family_size = int(raw["holm_family_size"])
    lambdas = tuple(float(item) for item in raw["lambda_grid"])
    type_i_tolerance = float(raw["type_i_tolerance"])

    null_rows: list[dict[str, Any]] = []
    planted_rows: list[dict[str, Any]] = []
    ledger_integrity_failures = 0

    for scenario in scenarios:
        stream = bernoulli_stream(
            seed=scenario.seed,
            length=stream_length,
            success_p=scenario.success_p,
        )
        if len(stream) != stream_length or any(x not in (0, 1) for x in stream):
            ledger_integrity_failures += 1
            continue
        control = evaluate_arm(
            stream,
            arm="holm_adaptive_peek",
            alpha=alpha,
            null_p=null_p,
            holm_family_size=holm_family_size,
            lambdas=lambdas,
        )
        treatment = evaluate_arm(
            stream,
            arm="mixture_e_process",
            alpha=alpha,
            null_p=null_p,
            holm_family_size=holm_family_size,
            lambdas=lambdas,
        )
        row = {
            "scenario_id": scenario.scenario_id,
            "kind": scenario.kind,
            "seed": scenario.seed,
            "control": control,
            "treatment": treatment,
        }
        if scenario.kind == "null":
            null_rows.append(row)
        else:
            planted_rows.append(row)

    null_n = len(null_rows)
    planted_n = len(planted_rows)
    control_null_rate = (
        sum(1 for row in null_rows if row["control"]["promoted"]) / max(null_n, 1)
    )
    treatment_null_rate = (
        sum(1 for row in null_rows if row["treatment"]["promoted"]) / max(null_n, 1)
    )
    treatment_power = (
        sum(1 for row in planted_rows if row["treatment"]["promoted"]) / max(planted_n, 1)
    )
    control_power = (
        sum(1 for row in planted_rows if row["control"]["promoted"]) / max(planted_n, 1)
    )
    oracle_power = 0.0
    for scenario in scenarios:
        if scenario.kind != "planted":
            continue
        stream = bernoulli_stream(
            seed=scenario.seed,
            length=stream_length,
            success_p=scenario.success_p,
        )
        if _fixed_n_oracle(
            stream,
            alpha=alpha,
            null_p=null_p,
            holm_family_size=holm_family_size,
        ):
            oracle_power += 1
    oracle_power /= max(planted_n, 1)
    power_loss = max(0.0, 1.0 - (treatment_power / max(oracle_power, 1e-9)))
    premature_promotion_rate = treatment_null_rate
    type_i_ok = treatment_null_rate <= alpha + type_i_tolerance
    control_inflated = control_null_rate > alpha + type_i_tolerance
    usable_power = treatment_power >= max(0.3, oracle_power * 0.85)
    primary = 1.0 if type_i_ok else max(0.0, 1.0 - (treatment_null_rate - alpha))

    if not type_i_ok:
        decision = "reject"
        reason = "anytime_type_i_control_failed"
    elif not usable_power:
        decision = "reject"
        reason = "power_collapsed_unusably"
    elif power_loss > 0.35:
        decision = "reject"
        reason = "excessive_power_loss_vs_fixed_n_oracle"
    elif ledger_integrity_failures:
        decision = "reject"
        reason = "ledger_integrity_failure"
    else:
        decision = "accept"
        reason = "anytime_valid_controls_optional_stopping_with_usable_power"

    return {
        "schema": "anytime_valid_promotion_report/v1",
        "anytime_valid_type_i_error_control_under_optional_stopping": primary,
        "alpha": alpha,
        "null_scenario_count": null_n,
        "planted_scenario_count": planted_n,
        "control_null_promotion_rate": control_null_rate,
        "treatment_null_promotion_rate": treatment_null_rate,
        "control_planted_power": control_power,
        "treatment_planted_power": treatment_power,
        "fixed_n_oracle_power": oracle_power,
        "power_loss": power_loss,
        "premature_promotion_rate": premature_promotion_rate,
        "control_inflated_vs_alpha": control_inflated,
        "ledger_integrity_failures": ledger_integrity_failures,
        "type_i_tolerance": type_i_tolerance,
        "decision": decision,
        "reason": reason,
        "sample_null_rows": null_rows[:3],
        "sample_planted_rows": planted_rows[:3],
        "corpus_digest": content_sha(raw),
        "toolchain": pinned_toolchain(),
    }


__all__ = [
    "BACKEND_ID",
    "CORPUS_RELATIVE",
    "DecisionArm",
    "ScenarioV1",
    "bernoulli_stream",
    "default_scenarios",
    "evaluate_arm",
    "load_corpus",
    "paired_calibration_report",
    "pinned_toolchain",
]
