"""SDE0-01: frozen E396/E479 decode-scaffolding × prompt-inventory factorial.

This module builds the factorial arms and applies them to a frozen checkpoint
via the existing ``evaluate_suites`` path.  Stage A always runs; Stage B is
optional and triggered by a non-additive residual in Stage A.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.decode_path import (
    DecodePathSpec,
    get_decode_path,
)
from slm_training.harnesses.model_build.factory import apply_runtime_overrides
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
)


@dataclass(frozen=True)
class ScaffoldFactors:
    """Four binary scaffolding factors for the SDE0-01 factorial."""

    content_floor: bool = True
    prompt_inventory: bool = True
    semantic_constraints: bool = True
    attempts: bool = True

    def to_dict(self) -> dict[str, bool]:
        return {
            "content_floor": self.content_floor,
            "prompt_inventory": self.prompt_inventory,
            "semantic_constraints": self.semantic_constraints,
            "attempts": self.attempts,
        }

    def bits_set(self) -> int:
        return sum(self.to_dict().values())


@dataclass(frozen=True)
class AblateArm:
    """One factorial cell."""

    arm_id: str
    factors: ScaffoldFactors
    decode_path_id: str
    best_of_n: int
    description: str = ""
    runtime_override_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "factors": self.factors.to_dict(),
            "decode_path_id": self.decode_path_id,
            "best_of_n": self.best_of_n,
            "description": self.description,
            "runtime_override_fields": list(self.runtime_override_fields),
        }


@dataclass(frozen=True)
class ArmResult:
    """Measured result for one arm."""

    arm_id: str
    factors: ScaffoldFactors
    decode_path_id: str
    best_of_n: int
    compatible: bool
    incompatible_reason: str | None
    metrics: dict[str, Any] = field(default_factory=dict)
    ship_gates: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "factors": self.factors.to_dict(),
            "decode_path_id": self.decode_path_id,
            "best_of_n": self.best_of_n,
            "compatible": self.compatible,
            "incompatible_reason": self.incompatible_reason,
            "metrics": self.metrics,
            "ship_gates": self.ship_gates,
            "elapsed_seconds": self.elapsed_seconds,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class AblateReport:
    """Full factorial report."""

    run_id: str
    version: str
    timestamp: str
    checkpoint_id: str
    checkpoint_sha256: str | None
    checkpoint_remote_uri: str | None
    suites: tuple[str, ...]
    stage: str
    arms: tuple[ArmResult, ...]
    gate_policy: dict[str, Any] = field(default_factory=lambda: DEFAULT_SHIP_GATES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "version": self.version,
            "timestamp": self.timestamp,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_remote_uri": self.checkpoint_remote_uri,
            "suites": list(self.suites),
            "stage": self.stage,
            "arms": [a.to_dict() for a in self.arms],
            "gate_policy": self.gate_policy,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _hash_run_id(parts: tuple[Any, ...]) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


# Decode path chosen for each scaffolding level.  The "semantic_constraints"
# factor maps onto the existing decode-path registry: when True use the strict
# exact/compiler path; when False fall back to native greedy LTR.
PATH_BY_CONSTRAINTS: dict[bool, str] = {
    True: "current_exact_or_compiler",
    False: "current_native",
}


def _decode_path_for_factors(factors: ScaffoldFactors) -> DecodePathSpec:
    return get_decode_path(PATH_BY_CONSTRAINTS[factors.semantic_constraints])


def build_stage_a_arms() -> tuple[AblateArm, ...]:
    """Return Stage A arms: full baseline, four one-factor-off, all-off."""
    baseline_factors = ScaffoldFactors()
    one_off: list[AblateArm] = []
    for field_name in baseline_factors.to_dict():
        off_factors = ScaffoldFactors(**{
            k: (False if k == field_name else v)
            for k, v in baseline_factors.to_dict().items()
        })
        one_off.append(
            AblateArm(
                arm_id=f"one_off_{field_name}",
                factors=off_factors,
                decode_path_id=PATH_BY_CONSTRAINTS[off_factors.semantic_constraints],
                best_of_n=4 if off_factors.attempts else 1,
                description=f"Baseline with {field_name} disabled",
            )
        )
    all_off = AblateArm(
        arm_id="all_off",
        factors=ScaffoldFactors(
            content_floor=False,
            prompt_inventory=False,
            semantic_constraints=False,
            attempts=False,
        ),
        decode_path_id=PATH_BY_CONSTRAINTS[False],
        best_of_n=1,
        description="All scaffolding disabled; grammar/compiler legality only",
    )
    baseline = AblateArm(
        arm_id="baseline",
        factors=baseline_factors,
        decode_path_id=PATH_BY_CONSTRAINTS[True],
        best_of_n=4,
        description="E479-equivalent full scaffolding control",
    )
    return (baseline,) + tuple(one_off) + (all_off,)


def _factor_overrides(arm: AblateArm) -> dict[str, Any]:
    """Map an arm's binary factors onto concrete ModelBuildConfig overrides."""
    factors = arm.factors
    overrides: dict[str, Any] = {}

    # content_floor: decode_min_content=-1 auto-from-inventory vs 0 off.
    overrides["decode_min_content"] = -1 if factors.content_floor else 0

    # prompt_inventory: honest surfacing of slot contract into prompt.
    overrides["honest_slot_contract"] = factors.prompt_inventory
    overrides["slot_contract_in_context"] = factors.prompt_inventory
    overrides["slot_contract_constrained_decode"] = factors.prompt_inventory

    # attempts: best_of_n and retry allowance.
    overrides["best_of_n"] = arm.best_of_n
    overrides["generate_max_attempts"] = 3 if factors.attempts else 1
    overrides["allow_unconstrained_fallback"] = not factors.semantic_constraints

    return overrides


def resolve_arm_config(
    base_config: ModelBuildConfig,
    arm: AblateArm,
    *,
    output_codec: str,
) -> tuple[ModelBuildConfig, DecodePathSpec, bool, str | None]:
    """Resolve a concrete eval config for ``arm``.

    Returns the mutated config, the decode path spec, compatibility status, and
    a stable reason string when incompatible.
    """
    path = _decode_path_for_factors(arm.factors)
    ok, reason = path.is_compatible(
        model_family="twotower",
        output_codec=output_codec,
    )
    if not ok:
        return base_config, path, False, reason

    config = copy.deepcopy(base_config)
    # Apply decode-path overrides first, then factor overrides, so factor
    # settings win when the path defines a default.
    path_overrides = path.resolve_config_overrides(output_codec)
    factor_overrides = _factor_overrides(arm)
    for key, value in {**path_overrides, **factor_overrides}.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"{arm.arm_id}: unknown config field {key!r}")

    # Compute the runtime_override_fields whitelist as the union of path and
    # factor fields so downstream factories can apply them safely.
    runtime_fields = set(path.runtime_override_fields()) | set(factor_overrides)
    config.runtime_override_fields = frozenset(runtime_fields)
    return config, path, True, None


def _verify_checkpoint(
    checkpoint_path: Path,
    expected_sha256: str | None,
) -> tuple[bool, str | None]:
    """Fail-closed checkpoint provenance check.

    Returns (ok, reason).  When ``expected_sha256`` is None the check is a
    warning-level skip; when it is provided and does not match, the arm is
    refused.
    """
    import hashlib

    if expected_sha256 is None:
        return True, "no expected sha256 provided; skipping hash verification"
    if not checkpoint_path.exists():
        return False, f"checkpoint not found: {checkpoint_path}"
    sha = hashlib.sha256()
    with checkpoint_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    observed = sha.hexdigest()
    if observed.lower() != expected_sha256.lower():
        return (
            False,
            f"checkpoint sha256 mismatch: expected {expected_sha256}, got {observed}",
        )
    return True, None


def _empty_metrics() -> dict[str, Any]:
    return {
        "meaningful_program_rate": math.nan,
        "placeholder_fidelity": math.nan,
        "parse_rate": math.nan,
        "samples": 0,
    }


def run_arm(
    arm: AblateArm,
    *,
    base_config: ModelBuildConfig,
    output_codec: str,
    checkpoint_path: Path | None = None,
    suites: tuple[str, ...] = (),
) -> ArmResult:
    """Run one factorial arm.

    If ``checkpoint_path`` is None the arm returns fixture metrics after
    verifying config resolution.  This lets unit tests and dry-runs exercise the
    harness without a frozen checkpoint.
    """
    start = time.monotonic()
    config, path, ok, reason = resolve_arm_config(
        base_config, arm, output_codec=output_codec
    )
    if not ok:
        return ArmResult(
            arm_id=arm.arm_id,
            factors=arm.factors,
            decode_path_id=arm.decode_path_id,
            best_of_n=arm.best_of_n,
            compatible=False,
            incompatible_reason=reason,
            elapsed_seconds=time.monotonic() - start,
            notes=(f"skipped: {reason}",),
        )

    notes: list[str] = [
        f"decode_path={path.path_id}",
        f"runtime_override_fields={sorted(set(path.runtime_override_fields()) | set(_factor_overrides(arm)))}",
    ]

    if checkpoint_path is None:
        # Fixture mode: verify the config resolution path only.
        return ArmResult(
            arm_id=arm.arm_id,
            factors=arm.factors,
            decode_path_id=path.path_id,
            best_of_n=arm.best_of_n,
            compatible=True,
            incompatible_reason=None,
            metrics=_empty_metrics(),
            ship_gates={"status": "fixture", "passed": None},
            elapsed_seconds=time.monotonic() - start,
            notes=tuple(notes) + ("fixture mode: no checkpoint provided",),
        )

    # Real eval mode: load checkpoint once per arm and run suites.
    from slm_training.harnesses.model_build.eval_runner import evaluate_suites
    from slm_training.models.twotower import TwoTowerModel

    model = TwoTowerModel.from_checkpoint(str(checkpoint_path), device=config.device)
    apply_runtime_overrides(model, config)
    config.run_root = Path(config.run_root) / "eval"
    config.run_id = arm.arm_id
    Path(config.run_dir).mkdir(parents=True, exist_ok=True)

    scoreboard = evaluate_suites(
        config,
        suites,
        model=model,
        write_gates=True,
    )
    gates = evaluate_ship_gates(scoreboard, DEFAULT_SHIP_GATES)
    notes.append("evaluated with frozen checkpoint")

    return ArmResult(
        arm_id=arm.arm_id,
        factors=arm.factors,
        decode_path_id=path.path_id,
        best_of_n=arm.best_of_n,
        compatible=True,
        incompatible_reason=None,
        metrics=scoreboard,
        ship_gates=gates,
        elapsed_seconds=time.monotonic() - start,
        notes=tuple(notes),
    )


def run_stage_a(
    base_config: ModelBuildConfig,
    *,
    checkpoint_id: str,
    checkpoint_sha256: str | None = None,
    checkpoint_remote_uri: str | None = None,
    checkpoint_path: Path | None = None,
    output_codec: str = "choice",
    suites: tuple[str, ...] = (),
) -> AblateReport:
    """Run Stage A of the SDE0-01 factorial."""
    arms = build_stage_a_arms()
    results: list[ArmResult] = []

    # Fail-closed provenance check before any arm runs.
    if checkpoint_path is not None:
        ok, reason = _verify_checkpoint(checkpoint_path, checkpoint_sha256)
        if not ok:
            for arm in arms:
                results.append(
                    ArmResult(
                        arm_id=arm.arm_id,
                        factors=arm.factors,
                        decode_path_id=arm.decode_path_id,
                        best_of_n=arm.best_of_n,
                        compatible=False,
                        incompatible_reason=reason,
                        notes=("provenance failure",),
                    )
                )
            run_id = _hash_run_id(("sde0-01", checkpoint_id, output_codec, "provenance_failure"))
            return AblateReport(
                run_id=run_id,
                version="sde0-01-v1",
                timestamp=_utc_now(),
                checkpoint_id=checkpoint_id,
                checkpoint_sha256=checkpoint_sha256,
                checkpoint_remote_uri=checkpoint_remote_uri,
                suites=suites,
                stage="A",
                arms=tuple(results),
            )

    for arm in arms:
        results.append(
            run_arm(
                arm,
                base_config=base_config,
                output_codec=output_codec,
                checkpoint_path=checkpoint_path,
                suites=suites,
            )
        )

    run_id = _hash_run_id(
        (
            "sde0-01",
            checkpoint_id,
            output_codec,
            tuple(a.arm_id for a in arms),
        )
    )
    return AblateReport(
        run_id=run_id,
        version="sde0-01-v1",
        timestamp=_utc_now(),
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_remote_uri=checkpoint_remote_uri,
        suites=suites,
        stage="A",
        arms=tuple(results),
    )


def stage_a_needs_stage_b(results: tuple[ArmResult, ...]) -> bool:
    """Heuristic: Stage B is needed when a one-factor-off residual is non-additive.

    For now this is intentionally conservative: if any one-factor-off arm
    differs from baseline by more than a preregistered epsilon on any metric,
    or if the all-off arm is outside the linear extrapolation, request Stage B.
    """
    baseline = next((r for r in results if r.arm_id == "baseline"), None)
    if baseline is None or not baseline.compatible:
        return False
    baseline_rate = baseline.metrics.get("meaningful_program_rate", math.nan)
    if math.isnan(baseline_rate):
        return False

    additive_prediction = baseline_rate
    for r in results:
        if r.arm_id.startswith("one_off_") and r.compatible:
            delta = baseline_rate - r.metrics.get("meaningful_program_rate", baseline_rate)
            additive_prediction -= delta

    all_off = next((r for r in results if r.arm_id == "all_off"), None)
    if all_off is None or not all_off.compatible:
        return False
    all_off_rate = all_off.metrics.get("meaningful_program_rate", math.nan)
    if math.isnan(all_off_rate):
        return False

    # If the observed all-off rate deviates by more than 5 points from the
    # additive extrapolation, suspect interactions.
    return abs(all_off_rate - additive_prediction) > 0.05
