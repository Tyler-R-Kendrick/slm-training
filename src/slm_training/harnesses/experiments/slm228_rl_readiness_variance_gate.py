"""SLM-228 (RLRG0-01): RL-readiness reward-variance gate stress test.

``assess_rl_readiness`` (SLM's fail-closed RL competence gate,
:mod:`slm_training.autoresearch.rl_gate`) requires, among other things, that
the evaluation bundle's ``reward_samples`` have nonzero variance before an
``RLReadinessReport`` can be ``approved``. Reading the implementation shows
that check is purely mechanical: ``len(rewards) >= 2 and
statistics.pvariance(rewards) > 0.0``. There is no minimum sample count beyond
2 and no minimum spread/magnitude — the same bar a healthy 100-sample reward
distribution has to clear is also cleared by two samples one femto-unit
apart.

This harness asks a narrow, falsifiable, CPU-only question: **holding every
other RL-readiness requirement fixed at a passing state (frozen snapshot,
full five-suite ship gates, AgentV pass, human-feedback holdout, rico_held
n>=1500), does the real ``assess_rl_readiness`` function approve reward-sample
arms whose diversity is too degenerate to carry a useful GRPO
group-relative-advantage signal, purely because ``variance > 0`` is
technically true?**

No new gate is implemented and no existing gate is changed. This only
exercises the canonical ``assess_rl_readiness`` function from
:mod:`slm_training.autoresearch.rl_gate` (never a reimplementation) against a
battery of reward-sample arms, records its real ``approved``/``failures``
output, and compares that against an *illustrative* (not implemented, not
gating) stronger candidate check — minimum sample count and minimum
spread — to show the shape of the gap.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.autoresearch.rl_gate import assess_rl_readiness
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "CANDIDATE_MIN_SAMPLES",
    "CANDIDATE_MIN_SPREAD",
    "RewardVarianceArm",
    "RewardVarianceArmResult",
    "RlReadinessVarianceGateReport",
    "render_markdown",
    "run_variance_gate_stress_fixture",
]

MATRIX_VERSION = "rlrg0-01-v1"
MATRIX_SET = "slm228_rl_readiness_variance_gate"
EXPERIMENT_ID = "slm228-rl-readiness-variance-gate"

# Illustrative-only candidate thresholds for a stronger reward-diversity check.
# These are NOT implemented in `assess_rl_readiness` and this harness makes no
# claim that these exact values are correct; they exist only to show, on a
# concrete deterministic scale, that a minimally stronger discriminator would
# separate the gameable arms from the healthy one.
CANDIDATE_MIN_SAMPLES = 8
CANDIDATE_MIN_SPREAD = 0.05

_HYPOTHESIS = (
    "The real assess_rl_readiness reward-variance requirement "
    "(len(reward_samples) >= 2 and statistics.pvariance(reward_samples) > 0) "
    "approves reward-sample arms with degenerate diversity -- vanishing "
    "spread and/or only two samples -- whenever every other RL-readiness "
    "requirement is independently satisfied, because the check has no "
    "minimum sample count beyond 2 and no minimum spread/magnitude floor."
)

_FALSIFIER = (
    "Every degenerate arm (near-zero spread with n>=2, or n==2 with a wide "
    "spread) is rejected by assess_rl_readiness while the healthy diverse "
    "arm and the two negative controls (zero variance, single sample) behave "
    "as expected -- i.e., some safeguard beyond the read source already "
    "closes this gap, or the mechanism cannot be exercised at all."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no checkpoint, GPU run, RL train step, or "
    "ship-gate claim is made or implied.",
    "This exercises the real, unmodified assess_rl_readiness function against "
    "constructed evaluation payloads; the suite/AgentV/frozen-snapshot fields "
    "are fixture data built to satisfy every non-reward requirement, not a "
    "real production evaluation.",
    "The CANDIDATE_MIN_SAMPLES / CANDIDATE_MIN_SPREAD thresholds are "
    "illustrative diagnostics only. They are not implemented in rl_gate.py, "
    "not proposed as the correct values, and passing/failing them makes no "
    "gate or promotion claim.",
    "Whether tiny reward variance actually harms GRPO-lite training (as "
    "opposed to only being a weak proxy in the readiness report) is not "
    "measured here; this harness is about the readiness *gate*, not RL "
    "training dynamics.",
    "Arms use hand-picked reward_samples values, not rewards sampled from any "
    "real policy or environment.",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: Any) -> str:
    return _sha256(_canonical_json(value))


def _base_passing_payload() -> dict[str, Any]:
    """A frozen production evaluation bundle satisfying every non-reward gate.

    Suite metrics clear the DEFAULT_SHIP_GATES floors with margin, every
    suite carries fallback_count=0 (certified, non-fallback), rico_held is
    exactly at the n>=1500 floor, and AgentV/frozen-snapshot metadata are
    marked passing. Only ``reward_samples`` varies across arms.
    """
    suite_metrics = {
        "n": 32,
        "parse_rate": 1.0,
        "structural_similarity": 1.0,
        "component_type_recall": 1.0,
        "fallback_count": 0,
        "placeholder_fidelity": 1.0,
        "reward_score": 1.0,
    }
    return {
        "evaluation_snapshot": {
            "metadata": {
                "kind": "frozen_production_evaluation",
                "suite_sizes": {"rico_held": 1500},
                "human_feedback_holdout_n": 10,
            }
        },
        "suites": {
            "smoke": dict(suite_metrics),
            "held_out": dict(suite_metrics),
            "adversarial": dict(suite_metrics),
            "ood": dict(suite_metrics),
            "rico_held": {**suite_metrics, "n": 1500},
        },
        "agentv": {"passed": True},
        "reward_samples": [],
    }


@dataclass(frozen=True)
class RewardVarianceArm:
    """One reward_samples arm to stress the readiness gate with."""

    name: str
    description: str
    reward_samples: tuple[float, ...]
    is_negative_control: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "reward_samples": list(self.reward_samples),
            "is_negative_control": self.is_negative_control,
        }


def build_default_arms() -> list[RewardVarianceArm]:
    """Six deterministic arms: one healthy positive, four gameable-candidate
    arms, and two negative controls that must still be rejected."""
    return [
        RewardVarianceArm(
            name="healthy_diverse_n8",
            description=(
                "8 samples evenly spread across [0.05, 0.95]: the shape a "
                "real GRPO-lite reward distribution with a working signal "
                "should have."
            ),
            reward_samples=(0.05, 0.15, 0.30, 0.45, 0.55, 0.70, 0.85, 0.95),
        ),
        RewardVarianceArm(
            name="two_sample_wide",
            description=(
                "The minimum n=2 the check accepts, but with a wide "
                "[0.1, 0.9] spread -- tests whether sample count alone is "
                "gameable even with genuinely different rewards."
            ),
            reward_samples=(0.1, 0.9),
        ),
        RewardVarianceArm(
            name="two_sample_epsilon",
            description=(
                "n=2 with a ~1e-9 spread: technically nonzero variance, no "
                "real reward diversity."
            ),
            reward_samples=(0.5, 0.500000001),
        ),
        RewardVarianceArm(
            name="large_n_epsilon_outlier",
            description=(
                "100 samples, 99 identical and 1 perturbed by 1e-7: mimics "
                "floating-point jitter from an otherwise near-deterministic "
                "reward function at a sample count no reasonable minimum-n "
                "check would flag."
            ),
            reward_samples=tuple([0.5] * 99 + [0.5000001]),
        ),
        RewardVarianceArm(
            name="all_identical_control",
            description=(
                "10 identical reward samples: zero variance. Negative "
                "control -- the current gate must still reject this."
            ),
            reward_samples=tuple([0.5] * 10),
            is_negative_control=True,
        ),
        RewardVarianceArm(
            name="single_sample_control",
            description=(
                "A single reward sample: below the len>=2 floor. Negative "
                "control -- the current gate must still reject this."
            ),
            reward_samples=(0.5,),
            is_negative_control=True,
        ),
    ]


@dataclass(frozen=True)
class RewardVarianceArmResult:
    """The real assess_rl_readiness outcome for one arm, plus the
    illustrative candidate-check comparison."""

    arm: RewardVarianceArm
    n_samples: int
    spread: float | None
    reward_variance: float
    approved: bool
    failures: tuple[str, ...]
    candidate_would_pass: bool
    gameable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.to_dict(),
            "n_samples": self.n_samples,
            "spread": self.spread,
            "reward_variance": self.reward_variance,
            "approved": self.approved,
            "failures": list(self.failures),
            "candidate_would_pass": self.candidate_would_pass,
            "gameable": self.gameable,
        }


def _evaluate_arm(arm: RewardVarianceArm) -> RewardVarianceArmResult:
    payload = _base_passing_payload()
    payload["reward_samples"] = list(arm.reward_samples)
    report = assess_rl_readiness(payload)

    n = len(arm.reward_samples)
    spread = (max(arm.reward_samples) - min(arm.reward_samples)) if n else None
    candidate_would_pass = bool(
        n >= CANDIDATE_MIN_SAMPLES and spread is not None and spread >= CANDIDATE_MIN_SPREAD
    )
    # "Gameable" = the real gate approved this arm, but the illustrative
    # stronger candidate check would not have.
    gameable = bool(report.approved and not candidate_would_pass)

    return RewardVarianceArmResult(
        arm=arm,
        n_samples=n,
        spread=spread,
        reward_variance=report.reward_variance,
        approved=report.approved,
        failures=report.failures,
        candidate_would_pass=candidate_would_pass,
        gameable=gameable,
    )


def _resolve_disposition(
    results: list[RewardVarianceArmResult],
) -> tuple[str, str]:
    controls = [r for r in results if r.arm.is_negative_control]
    controls_ok = all(not r.approved for r in controls)
    healthy = next((r for r in results if r.arm.name == "healthy_diverse_n8"), None)
    healthy_ok = bool(healthy and healthy.approved and healthy.candidate_would_pass)

    if not controls_ok:
        return (
            "inconclusive",
            "At least one negative-control arm (zero variance or a single "
            "sample) was still approved; the reward-variance check is not "
            "behaving as its source implies, so the gameable-arm comparison "
            "below is not a clean read.",
        )
    if not healthy_ok:
        return (
            "inconclusive",
            "The healthy diverse arm was not approved (or did not clear the "
            "illustrative candidate check itself); the fixture payload does "
            "not isolate the reward-variance requirement cleanly.",
        )

    gameable = [r for r in results if r.gameable]
    if gameable:
        names = ", ".join(r.arm.name for r in gameable)
        return (
            "gap_confirmed",
            f"{len(gameable)}/{len(results) - len(controls)} non-control arms "
            f"({names}) were approved by the real assess_rl_readiness reward-"
            "variance check while failing the illustrative stronger "
            f"candidate (n>={CANDIDATE_MIN_SAMPLES}, spread>={CANDIDATE_MIN_SPREAD}). "
            "The mechanical variance>0 bar is confirmed gameable by "
            "degenerate-but-technically-diverse reward samples.",
        )
    return (
        "no_gap_found",
        "No non-control arm was approved while failing the illustrative "
        "candidate check; the reward-variance requirement was not shown to "
        "be gameable by these constructions.",
    )


@dataclass(frozen=True)
class RlReadinessVarianceGateReport:
    """Full fixture report for SLM-228."""

    schema: str = "RlReadinessVarianceGateReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm228-rl-readiness-variance-gate"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    candidate_min_samples: int = CANDIDATE_MIN_SAMPLES
    candidate_min_spread: float = CANDIDATE_MIN_SPREAD
    results: tuple[RewardVarianceArmResult, ...] = field(default_factory=tuple)
    gate_hash: str = ""
    disposition: str = "inconclusive"
    disposition_rationale: str = ""
    honest_caveats: tuple[str, ...] = _HONEST_CAVEATS
    version_stamp: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "candidate_min_samples": self.candidate_min_samples,
            "candidate_min_spread": self.candidate_min_spread,
            "results": [r.to_dict() for r in self.results],
            "gate_hash": self.gate_hash,
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "honest_caveats": list(self.honest_caveats),
            "version_stamp": dict(self.version_stamp),
            "timestamp": self.timestamp,
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RlReadinessVarianceGateReport":
        results = tuple(
            RewardVarianceArmResult(
                arm=RewardVarianceArm(
                    name=str(r["arm"]["name"]),
                    description=str(r["arm"]["description"]),
                    reward_samples=tuple(r["arm"]["reward_samples"]),
                    is_negative_control=bool(r["arm"].get("is_negative_control", False)),
                ),
                n_samples=int(r["n_samples"]),
                spread=r.get("spread"),
                reward_variance=float(r["reward_variance"]),
                approved=bool(r["approved"]),
                failures=tuple(r.get("failures", ())),
                candidate_would_pass=bool(r["candidate_would_pass"]),
                gameable=bool(r["gameable"]),
            )
            for r in data.get("results", ())
        )
        return cls(
            schema=str(data.get("schema", "RlReadinessVarianceGateReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", EXPERIMENT_ID)),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            candidate_min_samples=int(data.get("candidate_min_samples", CANDIDATE_MIN_SAMPLES)),
            candidate_min_spread=float(data.get("candidate_min_spread", CANDIDATE_MIN_SPREAD)),
            results=results,
            gate_hash=str(data.get("gate_hash", "")),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def run_variance_gate_stress_fixture(
    *,
    arms: list[RewardVarianceArm] | None = None,
    run_id: str | None = None,
) -> RlReadinessVarianceGateReport:
    """Run every arm through the real assess_rl_readiness gate and compare
    against the illustrative stronger candidate check."""
    arms = arms if arms is not None else build_default_arms()
    results = [_evaluate_arm(arm) for arm in arms]
    disposition, rationale = _resolve_disposition(results)

    payload = {"row_digests": sorted(_digest(r.to_dict()) for r in results)}
    gate_hash = _sha256(_canonical_json(payload))

    return RlReadinessVarianceGateReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        results=tuple(results),
        gate_hash=gate_hash,
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm228_rl_readiness_variance_gate",
        ),
    )


def render_markdown(report: RlReadinessVarianceGateReport) -> str:
    lines = [
        f"# SLM-228 (RLRG0-01): RL-readiness reward-variance gate stress test ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
        f"**Illustrative candidate:** n>={report.candidate_min_samples}, spread>={report.candidate_min_spread}",
        f"**Gate hash:** `{report.gate_hash[:16]}...`",
        f"**Disposition:** {report.disposition} — {report.disposition_rationale}",
        "",
        "## Hypothesis",
        "",
        report.hypothesis,
        "",
        "## Falsifier",
        "",
        report.falsifier,
        "",
        "## Honest caveats",
        "",
        *(f"- {c}" for c in report.honest_caveats),
        "",
        "## Per-arm results",
        "",
        "| arm | n | spread | reward_variance | approved (real gate) | candidate would pass | gameable | control |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.results:
        spread = f"{r.spread:.2e}" if r.spread is not None else "—"
        lines.append(
            f"| {r.arm.name} | {r.n_samples} | {spread} | {r.reward_variance:.2e} | "
            f"{r.approved} | {r.candidate_would_pass} | {r.gameable} | {r.arm.is_negative_control} |"
        )
    lines += [
        "",
        "## Arm descriptions",
        "",
    ]
    for r in report.results:
        lines.append(f"- **{r.arm.name}**: {r.arm.description}")
    lines += [
        "",
        "## No-go for promotion",
        "",
        "This report is wiring/fixture evidence only. It does not change "
        "`assess_rl_readiness`, does not train or run RL, and makes no ship "
        "or gate claim. It documents a concrete gap in the existing "
        "mechanical reward-variance check as a candidate for a future, "
        "separately reviewed hardening change (never implemented here).",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python -m scripts.run_slm228_rl_readiness_variance_gate --mode plan-only",
        "python -m scripts.run_slm228_rl_readiness_variance_gate --mode fixture",
        "```",
        "",
    ]
    return "\n".join(lines)
