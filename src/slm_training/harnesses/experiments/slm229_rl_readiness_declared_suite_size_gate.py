"""SLM-229 (RLRG0-02): RL-readiness declared-vs-actual suite-size gate stress test.

SLM-228 showed that ``assess_rl_readiness``'s reward-variance requirement is
purely mechanical (``variance > 0``) and gameable by degenerate-but-technically
-diverse reward samples. This harness continues that lineage against a
*different* mechanical requirement in the same function, still exercising the
real, unmodified :func:`slm_training.autoresearch.rl_gate.assess_rl_readiness`
(and, further, the fail-closed :func:`slm_training.autoresearch.rl_gate.assert_rl_ready`
used to actually unlock RL).

Reading the source shows the production-scale ``rico_held`` floor is computed
as::

    suite_sizes[name] = max(actual_suite_n, declared_metadata_suite_sizes[name])

i.e. ``evaluation_snapshot.metadata.suite_sizes`` -- a self-reported,
unverified field on the payload -- can only ever *raise* the reported size,
never be checked against the actually-evaluated ``suites["rico_held"]["n"]``.
The canonical honest ship gates (``evaluate_ship_gates``) do independently
floor the *actual* suite ``n`` at ``DEFAULT_MIN_SUITE_N`` (20, no per-suite
override for ``rico_held``) -- but 20 is far below the 1500-record
production-scale bar ``assess_rl_readiness`` is supposed to enforce
specifically for ``rico_held``. Any actual suite between 20 and 1499 records,
paired with a declared-metadata claim of >=1500, therefore clears both
checks.

This harness asks a narrow, falsifiable, CPU-only question: **holding every
other RL-readiness requirement fixed at a passing state (frozen snapshot,
full five-suite ship gates at floor-clearing metrics, AgentV pass, human-
feedback holdout, healthy reward variance), does the real
``assess_rl_readiness`` -- and the downstream fail-closed ``assert_rl_ready``
-- approve a ``rico_held`` suite whose *actual* evaluated record count is far
below 1500, purely because ``evaluation_snapshot.metadata.suite_sizes``
declares a value of 1500 or more?**

No new gate is implemented and no existing gate is changed. This only
exercises the canonical ``assess_rl_readiness`` / ``assert_rl_ready``
functions from :mod:`slm_training.autoresearch.rl_gate` (never a
reimplementation) against a battery of declared-vs-actual suite-size arms,
records their real ``approved``/``failures``/raise-or-not output, and
compares that against an *illustrative* (not implemented, not gating)
stronger candidate check -- requiring the actual suite ``n`` alone (ignoring
declared metadata) to clear the 1500 floor -- to show the shape of the gap.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.autoresearch.rl_gate import assess_rl_readiness, assert_rl_ready
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "REQUIRED_RICO_HELD_N",
    "SuiteSizeArm",
    "SuiteSizeArmResult",
    "RlReadinessSuiteSizeGateReport",
    "render_markdown",
    "run_suite_size_gate_stress_fixture",
]

MATRIX_VERSION = "rlrg0-02-v1"
MATRIX_SET = "slm229_rl_readiness_declared_suite_size_gate"
EXPERIMENT_ID = "slm229-rl-readiness-declared-suite-size-gate"

# The real floor assess_rl_readiness enforces for rico_held. Mirrored here
# (not reimplemented) purely to compute the illustrative candidate check.
REQUIRED_RICO_HELD_N = 1500

_HYPOTHESIS = (
    "The real assess_rl_readiness rico_held>=1500 requirement -- computed as "
    "max(actual suites['rico_held']['n'], declared "
    "evaluation_snapshot.metadata.suite_sizes['rico_held']) -- is satisfied "
    "by a self-reported declared suite_sizes claim alone, decoupled from the "
    "actually-evaluated rico_held record count, whenever the actual count "
    "independently clears the unrelated, much lower honest-ship-gate n floor "
    "(DEFAULT_MIN_SUITE_N=20, no rico_held override)."
)

_FALSIFIER = (
    "Every arm whose actual rico_held n is below 1500 is rejected by "
    "assess_rl_readiness / assert_rl_ready regardless of what the declared "
    "metadata suite_sizes field claims -- i.e., the check already ties its "
    "size floor to the actually-evaluated suite, or the mechanism cannot be "
    "exercised at all."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no checkpoint, GPU run, RL train step, or "
    "ship-gate claim is made or implied.",
    "This exercises the real, unmodified assess_rl_readiness and "
    "assert_rl_ready functions against constructed evaluation payloads; the "
    "reward/AgentV/frozen-snapshot fields are fixture data built to satisfy "
    "every non-suite-size requirement, not a real production evaluation.",
    "The candidate check (actual suite n alone, ignoring declared metadata, "
    "must clear 1500) is an illustrative diagnostic only. It is not "
    "implemented in rl_gate.py, not proposed as the correct fix, and "
    "passing/failing it makes no gate or promotion claim.",
    "Whether a smaller-than-1500 rico_held suite actually harms RL training "
    "(as opposed to only being a weak proxy in the readiness report) is not "
    "measured here; this harness is about the readiness *gate*, not RL "
    "training dynamics.",
    "Arms use hand-picked suite metrics and metadata, not a real evaluation "
    "run against a real checkpoint.",
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
    """A frozen production evaluation bundle satisfying every non-suite-size
    gate: healthy reward variance, AgentV pass, frozen-snapshot metadata, and
    every suite's metrics comfortably clearing the DEFAULT_SHIP_GATES floors.
    Only ``suites.rico_held.n`` and ``metadata.suite_sizes`` vary across arms.
    """
    suite_metrics = {
        "n": 32,
        "meaningful_program_rate": 1.0,
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
                "human_feedback_holdout_n": 10,
            }
        },
        "suites": {
            "smoke": dict(suite_metrics),
            "held_out": dict(suite_metrics),
            "adversarial": dict(suite_metrics),
            "ood": dict(suite_metrics),
            "rico_held": dict(suite_metrics),
        },
        "agentv": {"passed": True},
        "reward_samples": [0.05, 0.15, 0.30, 0.45, 0.55, 0.70, 0.85, 0.95],
    }


@dataclass(frozen=True)
class SuiteSizeArm:
    """One (actual rico_held n, declared metadata rico_held n) arm."""

    name: str
    description: str
    actual_n: int
    declared_n: int | None
    is_negative_control: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "actual_n": self.actual_n,
            "declared_n": self.declared_n,
            "is_negative_control": self.is_negative_control,
        }


def build_default_arms() -> list[SuiteSizeArm]:
    """Six deterministic arms: one healthy positive, three gameable-candidate
    arms, and two negative controls that must still be rejected."""
    return [
        SuiteSizeArm(
            name="matched_actual_1500",
            description=(
                "Actual rico_held n=1500, declared metadata matches. The "
                "shape a genuine production-scale evaluation should have."
            ),
            actual_n=1500,
            declared_n=1500,
        ),
        SuiteSizeArm(
            name="declared_only_ship_gate_floor_n20",
            description=(
                "Actual rico_held n=20 -- exactly the honest-ship-gate "
                "DEFAULT_MIN_SUITE_N floor and nothing more -- with declared "
                "metadata claiming 1500. Tests whether the 1500 floor is "
                "satisfiable at the smallest actual size that independently "
                "clears the (unrelated, much lower) ship-gate n check."
            ),
            actual_n=20,
            declared_n=1500,
        ),
        SuiteSizeArm(
            name="declared_only_smoke_scale_n25",
            description=(
                "Actual rico_held n=25 -- a smoke/dev-scale suite -- with "
                "declared metadata claiming 1500."
            ),
            actual_n=25,
            declared_n=1500,
        ),
        SuiteSizeArm(
            name="declared_far_exceeds_actual_n100",
            description=(
                "Actual rico_held n=100 with declared metadata claiming "
                "5000 -- an order-of-magnitude over-claim -- to check "
                "whether the gap widens without limit."
            ),
            actual_n=100,
            declared_n=5000,
        ),
        SuiteSizeArm(
            name="no_declared_field_small_actual_control",
            description=(
                "Actual rico_held n=20 (still clears the ship-gate floor) "
                "with no declared suite_sizes metadata at all. Negative "
                "control -- without a declared-size claim to inflate the "
                "reported size, the current gate must still reject this."
            ),
            actual_n=20,
            declared_n=None,
            is_negative_control=True,
        ),
        SuiteSizeArm(
            name="declared_below_floor_control",
            description=(
                "Actual rico_held n=20 with declared metadata honestly "
                "claiming 50 (also below 1500, no inflation attempted). "
                "Negative control -- the current gate must still reject "
                "this."
            ),
            actual_n=20,
            declared_n=50,
            is_negative_control=True,
        ),
    ]


@dataclass(frozen=True)
class SuiteSizeArmResult:
    """The real assess_rl_readiness / assert_rl_ready outcome for one arm,
    plus the illustrative candidate-check comparison."""

    arm: SuiteSizeArm
    reported_rico_held_n: int
    approved: bool
    failures: tuple[str, ...]
    assert_rl_ready_raised: bool
    assert_rl_ready_error: str | None
    candidate_would_pass: bool
    gameable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.to_dict(),
            "reported_rico_held_n": self.reported_rico_held_n,
            "approved": self.approved,
            "failures": list(self.failures),
            "assert_rl_ready_raised": self.assert_rl_ready_raised,
            "assert_rl_ready_error": self.assert_rl_ready_error,
            "candidate_would_pass": self.candidate_would_pass,
            "gameable": self.gameable,
        }


def _evaluate_arm(arm: SuiteSizeArm) -> SuiteSizeArmResult:
    payload = _base_passing_payload()
    payload["suites"]["rico_held"]["n"] = arm.actual_n
    if arm.declared_n is not None:
        payload["evaluation_snapshot"]["metadata"]["suite_sizes"] = {
            "rico_held": arm.declared_n
        }

    report = assess_rl_readiness(payload)

    assert_rl_ready_raised = False
    assert_rl_ready_error: str | None = None
    if report.approved:
        try:
            assert_rl_ready(report)
        except ValueError as exc:  # pragma: no cover - defensive; gate is fail-closed
            assert_rl_ready_raised = True
            assert_rl_ready_error = str(exc)
    else:
        try:
            assert_rl_ready(report)
        except ValueError as exc:
            assert_rl_ready_raised = True
            assert_rl_ready_error = str(exc)

    candidate_would_pass = arm.actual_n >= REQUIRED_RICO_HELD_N
    # "Gameable" = the real gate approved this arm (and the downstream
    # fail-closed assert_rl_ready did not reject it either), but the
    # illustrative stronger candidate check -- actual n alone -- would not.
    gameable = bool(
        report.approved and not assert_rl_ready_raised and not candidate_would_pass
    )

    return SuiteSizeArmResult(
        arm=arm,
        reported_rico_held_n=report.suite_sizes.get("rico_held", 0),
        approved=report.approved,
        failures=report.failures,
        assert_rl_ready_raised=assert_rl_ready_raised,
        assert_rl_ready_error=assert_rl_ready_error,
        candidate_would_pass=candidate_would_pass,
        gameable=gameable,
    )


def _resolve_disposition(
    results: list[SuiteSizeArmResult],
) -> tuple[str, str]:
    controls = [r for r in results if r.arm.is_negative_control]
    controls_ok = all((not r.approved) or r.assert_rl_ready_raised for r in controls)
    healthy = next((r for r in results if r.arm.name == "matched_actual_1500"), None)
    healthy_ok = bool(
        healthy
        and healthy.approved
        and not healthy.assert_rl_ready_raised
        and healthy.candidate_would_pass
    )

    if not controls_ok:
        return (
            "inconclusive",
            "At least one negative-control arm (no declared-size inflation "
            "attempted, actual n below the floor) was still approved by "
            "both assess_rl_readiness and assert_rl_ready; the suite-size "
            "check is not behaving as its source implies, so the gameable-"
            "arm comparison below is not a clean read.",
        )
    if not healthy_ok:
        return (
            "inconclusive",
            "The healthy matched-size arm was not approved (or did not "
            "clear the illustrative candidate check itself); the fixture "
            "payload does not isolate the suite-size requirement cleanly.",
        )

    gameable = [r for r in results if r.gameable]
    if gameable:
        names = ", ".join(r.arm.name for r in gameable)
        return (
            "gap_confirmed",
            f"{len(gameable)}/{len(results) - len(controls)} non-control arms "
            f"({names}) were approved by the real assess_rl_readiness and "
            "assert_rl_ready while their actual rico_held record count fell "
            f"below the illustrative candidate floor (actual n>={REQUIRED_RICO_HELD_N}, "
            "ignoring declared metadata). The declared-vs-actual suite-size "
            "requirement is confirmed gameable by a self-reported "
            "suite_sizes claim decoupled from the evaluated data.",
        )
    return (
        "no_gap_found",
        "No non-control arm was approved while failing the illustrative "
        "candidate check; the declared-vs-actual suite-size requirement was "
        "not shown to be gameable by these constructions.",
    )


@dataclass(frozen=True)
class RlReadinessSuiteSizeGateReport:
    """Full fixture report for SLM-229."""

    schema: str = "RlReadinessSuiteSizeGateReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm229-rl-readiness-declared-suite-size-gate"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    required_rico_held_n: int = REQUIRED_RICO_HELD_N
    results: tuple[SuiteSizeArmResult, ...] = field(default_factory=tuple)
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
            "required_rico_held_n": self.required_rico_held_n,
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
    def from_dict(cls, data: dict[str, Any]) -> "RlReadinessSuiteSizeGateReport":
        results = tuple(
            SuiteSizeArmResult(
                arm=SuiteSizeArm(
                    name=str(r["arm"]["name"]),
                    description=str(r["arm"]["description"]),
                    actual_n=int(r["arm"]["actual_n"]),
                    declared_n=(
                        None
                        if r["arm"].get("declared_n") is None
                        else int(r["arm"]["declared_n"])
                    ),
                    is_negative_control=bool(r["arm"].get("is_negative_control", False)),
                ),
                reported_rico_held_n=int(r["reported_rico_held_n"]),
                approved=bool(r["approved"]),
                failures=tuple(r.get("failures", ())),
                assert_rl_ready_raised=bool(r.get("assert_rl_ready_raised", False)),
                assert_rl_ready_error=r.get("assert_rl_ready_error"),
                candidate_would_pass=bool(r["candidate_would_pass"]),
                gameable=bool(r["gameable"]),
            )
            for r in data.get("results", ())
        )
        return cls(
            schema=str(data.get("schema", "RlReadinessSuiteSizeGateReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", EXPERIMENT_ID)),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            required_rico_held_n=int(data.get("required_rico_held_n", REQUIRED_RICO_HELD_N)),
            results=results,
            gate_hash=str(data.get("gate_hash", "")),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def run_suite_size_gate_stress_fixture(
    *,
    arms: list[SuiteSizeArm] | None = None,
    run_id: str | None = None,
) -> RlReadinessSuiteSizeGateReport:
    """Run every arm through the real assess_rl_readiness / assert_rl_ready
    gates and compare against the illustrative stronger candidate check."""
    arms = arms if arms is not None else build_default_arms()
    results = [_evaluate_arm(arm) for arm in arms]
    disposition, rationale = _resolve_disposition(results)

    payload = {"row_digests": sorted(_digest(r.to_dict()) for r in results)}
    gate_hash = _sha256(_canonical_json(payload))

    return RlReadinessSuiteSizeGateReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        results=tuple(results),
        gate_hash=gate_hash,
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm229_rl_readiness_declared_suite_size_gate",
        ),
    )


def render_markdown(report: RlReadinessSuiteSizeGateReport) -> str:
    lines = [
        f"# SLM-229 (RLRG0-02): RL-readiness declared-vs-actual suite-size gate stress test ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
        f"**Illustrative candidate:** actual rico_held n>={report.required_rico_held_n} (declared metadata ignored)",
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
        "| arm | actual n | declared n | reported n (max) | approved (real gate) | assert_rl_ready raised | candidate would pass | gameable | control |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.results:
        declared = "—" if r.arm.declared_n is None else str(r.arm.declared_n)
        lines.append(
            f"| {r.arm.name} | {r.arm.actual_n} | {declared} | {r.reported_rico_held_n} | "
            f"{r.approved} | {r.assert_rl_ready_raised} | {r.candidate_would_pass} | "
            f"{r.gameable} | {r.arm.is_negative_control} |"
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
        "`assess_rl_readiness` or `assert_rl_ready`, does not train or run "
        "RL, and makes no ship or gate claim. It documents a concrete gap in "
        "the existing mechanical declared-vs-actual suite-size check as a "
        "candidate for a future, separately reviewed hardening change (never "
        "implemented here).",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python -m scripts.run_slm229_rl_readiness_declared_suite_size_gate --mode plan-only",
        "python -m scripts.run_slm229_rl_readiness_declared_suite_size_gate --mode fixture",
        "```",
        "",
    ]
    return "\n".join(lines)
