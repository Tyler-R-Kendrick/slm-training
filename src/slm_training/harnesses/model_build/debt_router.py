"""SLM-212 (SDE5-05): deterministic constraint-debt routing over decode paths.

Routing is a thin deterministic policy: it selects among existing MaskGIT,
constrained left-to-right (LTR), and ASAp decode paths.  It never changes
grammar legality, verifier behavior, or model weights.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from slm_training.harnesses.model_build.config import ModelBuildConfig

ROUTE = Literal["maskgit", "ltr", "asap"]
ROUTES: tuple[ROUTE, ...] = ("maskgit", "ltr", "asap")

ROUTE_MODES: dict[str, ROUTE | None] = {
    "off": None,
    "fixed_maskgit": "maskgit",
    "fixed_ltr": "ltr",
    "fixed_asap": "asap",
    "debt_router": "debt_router",
}

# Map public signal names to ConstraintDebtV1 attribute names.
SIGNAL_FIELDS: dict[str, str] = {
    "D_legal": "legal_debt",
    "D_good_proxy": "good_debt",
    "legal_mass_deficit": "legal_mass_deficit",
    "pre_post_mask_kl": "pre_post_mask_kl",
}

CHEAP_ROUTE: ROUTE = "maskgit"
STRICT_ROUTE: ROUTE = "ltr"


@dataclass(frozen=True)
class DebtRoutingPolicy:
    """Resolved, immutable routing policy.

    ``mode == "off"`` means routing does not override the model's existing decode
    settings (``grammar_ltr_primary``, ``asap_decode``, etc.).  Fixed modes force
    a single decode path.  ``debt_router`` uses the signal thresholds below.

    ``calibrator_hash`` captures the artifact used to produce the policy; an
    empty string means no calibrator was loaded.
    """

    mode: str = "off"
    signal: str = "D_legal"
    threshold_high: float = 2.0
    threshold_low: float | None = None
    hysteresis: int = 1
    fallback_policy: str = "fixed_maskgit"
    budget_mode: str = "equal_verifier_budget"
    calibrator_path: Path | None = None
    calibrator_hash: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ROUTE_MODES:
            raise ValueError(f"unknown routing mode: {self.mode!r}")
        if self.signal not in SIGNAL_FIELDS:
            raise ValueError(f"unknown routing signal: {self.signal!r}")
        if self.fallback_policy not in ("fixed_maskgit", "fixed_ltr", "fixed_asap"):
            raise ValueError(f"unknown fallback policy: {self.fallback_policy!r}")
        if self.budget_mode not in (
            "equal_verifier_budget",
            "equal_forward_budget",
            "equal_wall_budget",
        ):
            raise ValueError(f"unknown budget mode: {self.budget_mode!r}")
        if self.hysteresis < 1:
            raise ValueError("hysteresis must be at least 1")

    @property
    def effective_threshold_low(self) -> float:
        return self.threshold_low if self.threshold_low is not None else self.threshold_high

    @property
    def fixed_route(self) -> ROUTE | None:
        """Return the route for fixed modes; None for off/debt_router."""
        route = ROUTE_MODES[self.mode]
        if route == "debt_router":
            return None
        return route

    @classmethod
    def from_config(cls, config: ModelBuildConfig) -> "DebtRoutingPolicy":
        """Build a policy from ModelBuildConfig.

        This does **not** load a calibrator; use :class:`CalibratedDebtRouter` for
        artifact-aware routing.
        """
        return cls(
            mode=config.constraint_debt_routing_mode,
            signal=config.constraint_debt_routing_signal,
            threshold_high=config.constraint_debt_routing_threshold_high,
            threshold_low=config.constraint_debt_routing_threshold_low,
            hysteresis=config.constraint_debt_routing_hysteresis,
            fallback_policy=config.constraint_debt_routing_fallback_policy,
            budget_mode=config.constraint_debt_routing_budget_mode,
            calibrator_path=config.constraint_debt_routing_calibrator_path,
        )

    def identity_hash(self) -> str:
        """Stable hash of the resolved policy (for decode identity / traces)."""
        payload = {
            "mode": self.mode,
            "signal": self.signal,
            "threshold_high": self.threshold_high,
            "threshold_low": self.threshold_low,
            "hysteresis": self.hysteresis,
            "fallback_policy": self.fallback_policy,
            "budget_mode": self.budget_mode,
            "calibrator_hash": self.calibrator_hash,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_signal_value(signal_name: str, debt_row: object) -> float:
    """Read a signal from a ConstraintDebtV1-like row, returning 0.0 when missing."""
    field_name = SIGNAL_FIELDS.get(signal_name)
    if field_name is None:
        raise ValueError(f"unknown signal: {signal_name!r}")
    value = getattr(debt_row, field_name, None)
    if value is None:
        return 0.0
    return float(value)


def decide_route(
    signal_value: float,
    policy: DebtRoutingPolicy,
    previous_route: ROUTE | None = None,
    step: int = 0,
    state: dict[str, Any] | None = None,
) -> tuple[ROUTE, dict[str, Any]]:
    """Return the deterministic route for one decode step.

    The returned ``state`` dict should be passed back on the next call to enforce
    hysteresis.  ``step`` is recorded in the state for trace replay but does not
    affect routing logic.

    Fixed modes return their route immediately and ignore the signal.
    """
    if policy.mode == "off":
        return CHEAP_ROUTE, {"step": step, "route": CHEAP_ROUTE, "reason": "routing_off"}

    fixed = policy.fixed_route
    if fixed is not None:
        return fixed, {"step": step, "route": fixed, "reason": f"fixed_{policy.mode}"}

    state = state if state is not None else {}
    high_count: int = state.get("high_count", 0)
    low_count: int = state.get("low_count", 0)
    current_route: ROUTE = state.get("current_route") or _route_from_fallback(policy.fallback_policy)

    low_threshold = policy.effective_threshold_low
    reason = "hold"

    if signal_value >= policy.threshold_high:
        high_count += 1
        low_count = 0
        if high_count >= policy.hysteresis:
            current_route = STRICT_ROUTE
            reason = "high_debt"
    elif signal_value <= low_threshold:
        low_count += 1
        high_count = 0
        if low_count >= policy.hysteresis:
            current_route = CHEAP_ROUTE
            reason = "low_debt"
    else:
        high_count = 0
        low_count = 0

    new_state = {
        "step": step,
        "route": current_route,
        "current_route": current_route,
        "signal_value": signal_value,
        "high_count": high_count,
        "low_count": low_count,
        "reason": reason,
    }
    return current_route, new_state


def _route_from_fallback(fallback_policy: str) -> ROUTE:
    route = ROUTE_MODES.get(fallback_policy)
    if route in ROUTES:
        return route  # type: ignore[return-value]
    return CHEAP_ROUTE


@dataclass
class CalibratedDebtRouter:
    """Load a JSON calibrator artifact and fall back to static on mismatch."""

    policy: DebtRoutingPolicy = field(default_factory=DebtRoutingPolicy)
    calibration_error: str | None = None

    @classmethod
    def from_config(cls, config: ModelBuildConfig) -> "CalibratedDebtRouter":
        """Build a router, loading the calibrator if a path is configured."""
        base = DebtRoutingPolicy.from_config(config)
        path = config.constraint_debt_routing_calibrator_path
        if path is None:
            return cls(policy=base)
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            warnings.warn(f"calibrator not found at {path}; using static fallback")
            return cls(policy=_fallback_policy(base), calibration_error="missing_calibrator")
        except json.JSONDecodeError as exc:
            warnings.warn(f"calibrator JSON invalid at {path}: {exc}; using static fallback")
            return cls(policy=_fallback_policy(base), calibration_error="invalid_json")

        schema = artifact.get("schema", "")
        if schema != "debt_router_calibrator/v1":
            warnings.warn(f"calibrator schema mismatch {schema!r}; using static fallback")
            return cls(policy=_fallback_policy(base), calibration_error="schema_mismatch")

        stored_hash = artifact.get("artifact_hash", "")
        computed_hash = _calibrator_hash(artifact)
        if stored_hash and stored_hash != computed_hash:
            warnings.warn(
                f"calibrator hash mismatch at {path}; expected {computed_hash[:16]}, "
                f"got {stored_hash[:16]}; using static fallback"
            )
            return cls(policy=_fallback_policy(base), calibration_error="hash_mismatch")

        # Artifact is trusted: override base policy with calibrated values.
        policy = DebtRoutingPolicy(
            mode=artifact.get("mode", base.mode),
            signal=artifact.get("signal", base.signal),
            threshold_high=float(artifact.get("threshold_high", base.threshold_high)),
            threshold_low=(float(v) if (v := artifact.get("threshold_low")) is not None else base.threshold_low),
            hysteresis=int(artifact.get("hysteresis", base.hysteresis)),
            fallback_policy=artifact.get("fallback_policy", base.fallback_policy),
            budget_mode=artifact.get("budget_mode", base.budget_mode),
            calibrator_path=path,
            calibrator_hash=computed_hash,
        )
        return cls(policy=policy)

    def decide(
        self,
        signal_value: float,
        previous_route: ROUTE | None = None,
        step: int = 0,
        state: dict[str, Any] | None = None,
    ) -> tuple[ROUTE, dict[str, Any]]:
        return decide_route(signal_value, self.policy, previous_route, step, state)


def _fallback_policy(base: DebtRoutingPolicy) -> DebtRoutingPolicy:
    """Return a static policy that uses the configured fallback route."""
    return DebtRoutingPolicy(
        mode=base.fallback_policy,
        signal=base.signal,
        threshold_high=base.threshold_high,
        threshold_low=base.threshold_low,
        hysteresis=base.hysteresis,
        fallback_policy=base.fallback_policy,
        budget_mode=base.budget_mode,
        calibrator_path=base.calibrator_path,
        calibrator_hash="",
    )


def _calibrator_hash(artifact: dict[str, Any]) -> str:
    """Stable hash of the calibrator fields (excluding artifact_hash)."""
    payload = {k: v for k, v in artifact.items() if k != "artifact_hash"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_calibrator_artifact(
    *,
    signal: str = "D_legal",
    threshold_high: float = 2.0,
    threshold_low: float | None = None,
    hysteresis: int = 1,
    fallback_policy: str = "fixed_maskgit",
    budget_mode: str = "equal_verifier_budget",
    calibration_split_digest: str = "",
) -> dict[str, Any]:
    """Build a versioned calibrator artifact with its own hash."""
    artifact = {
        "schema": "debt_router_calibrator/v1",
        "signal": signal,
        "threshold_high": threshold_high,
        "threshold_low": threshold_low,
        "hysteresis": hysteresis,
        "fallback_policy": fallback_policy,
        "budget_mode": budget_mode,
        "calibration_split_digest": calibration_split_digest,
    }
    artifact["artifact_hash"] = _calibrator_hash(artifact)
    return artifact


class OracleRouter:
    """Diagnostic-only router that chooses the best fixed arm per example.

    ``outcomes`` maps an example/state id to a dict of ``{route: score}``.  The
    oracle may only use precomputed diagnostic outcomes; it must never see
    verifier-good labels at serving time.
    """

    def __init__(self, outcomes: dict[str, dict[str, float]]) -> None:
        self.outcomes = outcomes

    def decide(self, example_id: str) -> tuple[ROUTE, dict[str, Any]]:
        scores = self.outcomes.get(example_id, {})
        if not scores:
            return CHEAP_ROUTE, {"reason": "oracle_missing", "example_id": example_id}
        best_route = max(scores, key=lambda r: (scores[r], r))
        if best_route not in ROUTES:
            return CHEAP_ROUTE, {"reason": "oracle_invalid", "example_id": example_id}
        return best_route, {"reason": "oracle_best", "example_id": example_id, "scores": dict(scores)}
