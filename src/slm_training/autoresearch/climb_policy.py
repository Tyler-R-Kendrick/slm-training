"""Load versioned autotrain climb policy and pure climb-loop decisions.

High-volatility continuous-loop knobs (primaries, cadence, identity fields,
recipe-null caps, EG_params floors, rung gates) live under
``resources/experiments/autotrain_climb/``. Driver/harness code loads this
artifact and stays static when numbers or inventory change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from slm_training.autoresearch.hillclimb import (
    ExhaustedKnobLedger,
    HillClimbError,
    MetricDirection,
    assert_capacity_growth_allowed,
    assert_matrix_knobs_not_exhausted,
    improvement_signed,
    infer_metric_direction,
    is_measured_null_feedback,
    knob_signature_sha256,
)
from slm_training.lineage.records import canonical_json
from slm_training.levers import MAX_RUN_MINUTES

__all__ = [
    "CLIMB_POLICY_SCHEMA",
    "CLIMB_RESOURCE_DIR",
    "PROMOTE_AUTHORITY_SCHEMA",
    "ClimbPolicy",
    "ClimbPolicyError",
    "load_climb_policy",
    "climb_policy_content_digest",
    "promote_authority_sha256",
    "loop_data_eval_identity",
    "loop_ledger_path",
    "load_loop_exhausted_ledger",
    "save_loop_exhausted_ledger",
    "cycle_role_for_index",
    "assert_cycle_cadence",
    "primary_for_role",
    "stage_wall_minutes_for_role",
    "max_consecutive_frozen_replays",
    "decode_timeout_seconds_for_role",
    "eval_suites_for_role",
    "classify_positive_metrics",
    "promotion_primary_effect_met",
    "assert_recipe_null_regime_pressure",
    "is_recipe_tweak_knobs",
    "count_recipe_nulls_in_ledger",
    "assert_rung_allows_promotion",
    "synthesis_policy_allows_sft",
]

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CLIMB_RESOURCE_DIR = _PACKAGE_ROOT / "resources" / "experiments" / "autotrain_climb"
_REPO_ROOT = _PACKAGE_ROOT.parents[1]
CLIMB_POLICY_SCHEMA = "autotrain_climb_policy/v1"
PROMOTE_AUTHORITY_SCHEMA = "autotrain_promote_authority/v1"
_DEFAULT_POLICY_PATH = CLIMB_RESOURCE_DIR / "policy.v1.json"
_PROMOTE_AUTHORITY_HARNESS_COMPONENT = "harness.autoresearch.experiment_campaign"


class ClimbPolicyError(ValueError):
    """External climb-policy resource is invalid or unsupported."""


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ClimbPolicyError(f"missing climb policy resource: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ClimbPolicyError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ClimbPolicyError(f"{path} root must be an object")
    return payload


def _require_mapping(
    payload: Mapping[str, Any], key: str, path: Path
) -> Mapping[str, Any]:
    raw = payload.get(key)
    if not isinstance(raw, Mapping):
        raise ClimbPolicyError(f"{path}: missing object field {key!r}")
    return raw


def _require_list(payload: Mapping[str, Any], key: str, path: Path) -> list[Any]:
    raw = payload.get(key)
    if not isinstance(raw, list) or not raw:
        raise ClimbPolicyError(f"{path}: missing non-empty list field {key!r}")
    return list(raw)


@dataclass(frozen=True)
class ClimbPolicy:
    """Immutable loaded climb policy with resource identity."""

    path: Path
    schema: str
    version: str
    sha256: str
    payload: Mapping[str, Any]

    @property
    def screening_primary(self) -> Mapping[str, Any]:
        return dict(self.payload["screening_primary"])

    @property
    def promotion_primary(self) -> Mapping[str, Any]:
        return dict(self.payload["promotion_primary"])

    @property
    def defaults(self) -> Mapping[str, Any]:
        return dict(self.payload["defaults"])

    @property
    def measurement(self) -> Mapping[str, Any]:
        """Optional continuous measurement budgets (defaults if absent)."""
        raw = self.payload.get("measurement")
        return dict(raw) if isinstance(raw, Mapping) else {}

    @property
    def cadence(self) -> Mapping[str, Any]:
        return dict(self.payload["cadence"])

    @property
    def exhausted_identity_fields(self) -> tuple[str, ...]:
        return tuple(str(x) for x in self.payload["exhausted_identity_fields"])

    @property
    def recipe_tweak_knobs(self) -> frozenset[str]:
        return frozenset(str(x) for x in self.payload["recipe_tweak_knobs"])

    @property
    def recipe_null_cap(self) -> Mapping[str, Any]:
        return dict(self.payload["recipe_null_cap"])

    @property
    def positive_classification(self) -> Mapping[str, Any]:
        return dict(self.payload["positive_classification"])

    @property
    def promotion_dispose(self) -> Mapping[str, Any]:
        """Promote authority knobs (effect gate + cert); defaults fail closed."""
        raw = self.payload.get("promotion_dispose")
        base = {
            "require_primary_win": True,
            "require_cert_continue": True,
            "require_dual_arm_metrics": True,
            "require_parse_non_regression": True,
            "ignore_ship_insufficient_n_for_climb": True,
            "recertify_on_authority_change": True,
        }
        if isinstance(raw, Mapping):
            base.update(dict(raw))
        return base

    @property
    def synthesis_loop(self) -> Mapping[str, Any]:
        return dict(self.payload["synthesis_loop"])

    @property
    def rung_gates(self) -> Mapping[str, Any]:
        return dict(self.payload["rung_gates"])

    @property
    def loop_ledger(self) -> Mapping[str, Any]:
        return dict(self.payload["loop_ledger"])

    def identity_dict(self) -> dict[str, str]:
        return {
            "path": _repo_relative(self.path),
            "schema": self.schema,
            "version": self.version,
            "sha256": self.sha256,
        }


def climb_policy_content_digest(payload: Mapping[str, Any]) -> str:
    """Stable digest of policy content (excludes path-only identity)."""

    body = {k: v for k, v in payload.items() if k not in {"path", "sha256"}}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def promote_authority_sha256(
    *,
    climb_policy_sha256: str,
    locked_expectations_sha256: str,
    harness_component_version: str,
    harness_component: str = _PROMOTE_AUTHORITY_HARNESS_COMPONENT,
) -> str:
    """Content hash for climb-promote authority under current harness/policy.

    Climb ``promoted`` / ``climb_accepted`` rows must carry this digest. When
    climb policy, locked promote expectations, or the continuous/promote
    harness component version changes, the digest changes and historical
    promotions must re-certify under current dispose rules before remaining
    authoritative.
    """

    body = {
        "schema": PROMOTE_AUTHORITY_SCHEMA,
        "climb_policy_sha256": str(climb_policy_sha256 or ""),
        "locked_expectations_sha256": str(locked_expectations_sha256 or ""),
        "harness_component": str(harness_component or _PROMOTE_AUTHORITY_HARNESS_COMPONENT),
        "harness_component_version": str(harness_component_version or ""),
    }
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


@lru_cache(maxsize=4)
def load_climb_policy(path: str | None = None) -> ClimbPolicy:
    """Load and validate the committed climb-policy resource."""

    policy_path = Path(path) if path else _DEFAULT_POLICY_PATH
    payload = _read_json(policy_path)
    schema = str(payload.get("schema") or "")
    version = str(payload.get("version") or "")
    if schema != CLIMB_POLICY_SCHEMA:
        raise ClimbPolicyError(
            f"{policy_path}: schema {schema!r} != required {CLIMB_POLICY_SCHEMA!r}"
        )
    if not version:
        raise ClimbPolicyError(f"{policy_path}: missing version")
    # Required high-volatility blocks
    for key in (
        "screening_primary",
        "promotion_primary",
        "defaults",
        "cadence",
        "exhausted_identity_fields",
        "recipe_tweak_knobs",
        "recipe_null_cap",
        "positive_classification",
        "synthesis_loop",
        "rung_gates",
        "loop_ledger",
    ):
        if key not in payload:
            raise ClimbPolicyError(f"{policy_path}: missing required field {key!r}")
    screening = _require_mapping(payload, "screening_primary", policy_path)
    promotion = _require_mapping(payload, "promotion_primary", policy_path)
    for block_name, block in (
        ("screening_primary", screening),
        ("promotion_primary", promotion),
    ):
        if not block.get("metric"):
            raise ClimbPolicyError(f"{policy_path}: {block_name}.metric required")
        direction = str(block.get("direction") or "")
        if direction not in {"increase", "decrease"}:
            raise ClimbPolicyError(
                f"{policy_path}: {block_name}.direction must be increase|decrease"
            )
    _require_list(payload, "exhausted_identity_fields", policy_path)
    _require_list(payload, "recipe_tweak_knobs", policy_path)
    cadence = _require_mapping(payload, "cadence", policy_path)
    if int(cadence.get("screening_cycles_per_promotion") or 0) < 1:
        raise ClimbPolicyError(
            f"{policy_path}: cadence.screening_cycles_per_promotion must be >= 1"
        )
    max_replays = int(
        _require_mapping(payload, "measurement", policy_path).get(
            "max_consecutive_frozen_replays", 2
        )
    )
    if max_replays < 1:
        raise ClimbPolicyError(
            f"{policy_path}: measurement.max_consecutive_frozen_replays must be >= 1"
        )
    positive = _require_mapping(payload, "positive_classification", policy_path)
    minimum_efficiency_gain_fraction = float(
        positive.get("minimum_efficiency_gain_fraction", -1.0)
    )
    if not 0.0 < minimum_efficiency_gain_fraction < 1.0:
        raise ClimbPolicyError(
            f"{policy_path}: positive_classification."
            "minimum_efficiency_gain_fraction must be in (0, 1)"
        )
    return ClimbPolicy(
        path=policy_path.resolve(),
        schema=schema,
        version=version,
        sha256=_file_sha256(policy_path),
        payload=payload,
    )


def primary_for_role(policy: ClimbPolicy, role: str) -> Mapping[str, Any]:
    if role == "promotion":
        return policy.promotion_primary
    if role == "screening":
        return policy.screening_primary
    raise ClimbPolicyError(f"unknown cycle role: {role!r}")


def stage_wall_minutes_for_role(policy: ClimbPolicy, role: str) -> int:
    """Return a role wall that obeys the repository-wide command cap."""

    measurement = policy.measurement
    if role == "promotion":
        key = "promotion_stage_wall_minutes"
        default = 12
    else:
        key = "screening_stage_wall_minutes"
        default = 10
    value = measurement.get(key, default)
    minutes = int(value)
    if not 1 <= minutes <= MAX_RUN_MINUTES:
        raise ClimbPolicyError(
            f"measurement.{key} must be in [1, {MAX_RUN_MINUTES}], got {minutes}"
        )
    return minutes


def max_consecutive_frozen_replays(policy: ClimbPolicy) -> int:
    """Bound identical incomplete replays before canonical harness repair."""

    value = int(policy.measurement.get("max_consecutive_frozen_replays", 2))
    if value < 1:
        raise ClimbPolicyError(
            "measurement.max_consecutive_frozen_replays must be >= 1"
        )
    return value


def decode_timeout_seconds_for_role(policy: ClimbPolicy, role: str) -> float:
    """Per-record decode timeout for continuous eval arms.

    Screening defaults lean toward thrash arm-wall fit (see continuous
    ``_fit_screening_decode_timeout_seconds`` which may clamp further).
    """

    measurement = policy.measurement
    if role == "promotion":
        key = "promotion_decode_timeout_seconds"
        default = 24.0
    else:
        key = "screening_decode_timeout_seconds"
        # Default 8s: with smoke n=3 and ~70s arm share under MAX_RUN=3m,
        # 24s×3 already exceeds the arm wall (empty scoreboards).
        default = 8.0
    value = float(measurement.get(key, default))
    if value <= 0:
        raise ClimbPolicyError(f"measurement.{key} must be > 0, got {value}")
    return value


def eval_suites_for_role(policy: ClimbPolicy, role: str) -> tuple[str, ...]:
    """Suites for continuous evaluate_model; screening defaults to smoke only."""

    measurement = policy.measurement
    if role == "promotion":
        raw = measurement.get("promotion_suites")
        default = ("smoke", "held_out")
    else:
        raw = measurement.get("screening_suites")
        default = ("smoke",)
    if raw is None:
        return default
    if not isinstance(raw, list) or not raw:
        raise ClimbPolicyError(
            f"measurement suites for role {role!r} must be a non-empty list"
        )
    return tuple(str(x) for x in raw)


def cycle_role_for_index(policy: ClimbPolicy, cycle_index: int) -> str:
    """Return screening|promotion for 1-based cycle_index under policy cadence."""

    if cycle_index < 1:
        raise ClimbPolicyError("cycle_index must be >= 1")
    n = int(policy.cadence["screening_cycles_per_promotion"])
    # Every (n+1)th cycle is promotion after n screening cycles.
    period = n + 1
    offset = (cycle_index - 1) % period
    first = str(policy.cadence.get("first_cycle_role") or "screening")
    if first == "promotion":
        return "promotion" if offset == 0 else "screening"
    return "promotion" if offset == n else "screening"


def assert_cycle_cadence(
    policy: ClimbPolicy,
    *,
    cycle_index: int,
    claimed_role: str,
    claim_class: str | None = None,
    certified_rungs: Sequence[str] | None = None,
    promotion_target_available: bool | None = None,
    confirmation_pending: bool = False,
) -> str:
    """Fail closed when a cycle claims a role outside the configured cadence.

    A promotion cadence slot is an opportunity to measure a prior screening
    winner, not authority to expose a new arm to promotion suites.  When the
    policy requires a prior screening win, an explicitly empty promotion slot
    may therefore fall back to screening.
    """

    expected = cycle_role_for_index(policy, cycle_index)
    if claimed_role not in {"screening", "promotion"}:
        raise HillClimbError(f"invalid cycle role: {claimed_role!r}")
    empty_promotion_fallback = (
        expected == "promotion"
        and claimed_role == "screening"
        and promotion_target_available is False
        and bool(policy.cadence.get("promotion_requires_prior_screening_win", True))
    )
    confirmation_fallback = (
        expected == "promotion"
        and claimed_role == "screening"
        and confirmation_pending
        and bool(policy.cadence.get("promotion_requires_prior_screening_win", True))
    )
    if claimed_role != expected and not (
        empty_promotion_fallback or confirmation_fallback
    ):
        raise HillClimbError(
            f"cycle cadence violation: cycle_index={cycle_index} expects "
            f"{expected}, claimed {claimed_role} "
            f"(screening_cycles_per_promotion="
            f"{policy.cadence['screening_cycles_per_promotion']})"
        )
    if claim_class is not None:
        defaults = policy.defaults
        if claimed_role == "promotion":
            allowed = {
                str(defaults.get("claim_class_promotion") or "promotion_candidate"),
                "ship_gate",
            }
            if claim_class not in allowed:
                raise HillClimbError(
                    f"promotion-role cycle requires claim_class in {sorted(allowed)}, "
                    f"got {claim_class!r}"
                )
        else:
            banned = {"promotion_candidate", "ship_gate"}
            if claim_class in banned:
                raise HillClimbError(
                    f"screening-role cycle cannot use promotion claim_class {claim_class!r}"
                )
    # Optional I10 rung gate — fail closed when enabled and prior rung unmet.
    assert_rung_allows_promotion(
        policy,
        claimed_role=claimed_role,
        certified_rungs=certified_rungs,
    )
    return (
        claimed_role
        if empty_promotion_fallback or confirmation_fallback
        else expected
    )


def loop_data_eval_identity(
    policy: ClimbPolicy,
    *,
    claim_class: str,
    train_version: str | None = None,
    eval_version: str | None = None,
    primary_metric: str,
    direction: str,
    data_manifest_sha256: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Hash identity fields listed in the external policy for loop-scoped exhaust."""

    values: dict[str, Any] = {
        "claim_class": claim_class,
        "train_version": train_version or "",
        "eval_version": eval_version or "",
        "primary_metric": primary_metric,
        "direction": direction,
        "data_manifest_sha256": data_manifest_sha256 or "",
    }
    if extra:
        values.update(dict(extra))
    ordered = {
        field: values.get(field, "") for field in policy.exhausted_identity_fields
    }
    return hashlib.sha256(canonical_json(ordered).encode("utf-8")).hexdigest()


def loop_ledger_path(
    autoresearch_root: Path, loop_id: str, policy: ClimbPolicy | None = None
) -> Path:
    pol = policy or load_climb_policy()
    template = str(
        pol.loop_ledger.get("relative_path_template")
        or "loops/{loop_id}/exhausted_knob_ledger.json"
    )
    rel = template.format(loop_id=loop_id)
    return Path(autoresearch_root) / rel


def load_loop_exhausted_ledger(
    autoresearch_root: Path, loop_id: str, policy: ClimbPolicy | None = None
) -> ExhaustedKnobLedger:
    return ExhaustedKnobLedger.load(
        loop_ledger_path(autoresearch_root, loop_id, policy)
    )


def save_loop_exhausted_ledger(
    ledger: ExhaustedKnobLedger,
    autoresearch_root: Path,
    loop_id: str,
    policy: ClimbPolicy | None = None,
) -> Path:
    path = loop_ledger_path(autoresearch_root, loop_id, policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger.save(path)
    return path


def is_recipe_tweak_knobs(policy: ClimbPolicy, knobs: Mapping[str, Any]) -> bool:
    """True when every set knob is in the recipe-tweak allowlist (pure recipe)."""

    keys = {k for k, v in knobs.items() if v is not None}
    if not keys:
        return False
    return keys <= set(policy.recipe_tweak_knobs)


def count_recipe_nulls_in_ledger(
    policy: ClimbPolicy,
    ledger: ExhaustedKnobLedger,
    *,
    data_eval_identity: str,
    claim_class: str,
    family: str = "recipe",
) -> int:
    """Count exhausted entries for this identity that note recipe-family nulls."""

    del family  # reserved for multi-family ledgers
    count = 0
    for entry in ledger.entries:
        if entry.data_eval_identity != data_eval_identity:
            continue
        if entry.claim_class != claim_class:
            continue
        # Only count explicit recipe-family nulls (not architecture nulls).
        if entry.reason.startswith("recipe_null") or "recipe_family" in entry.note:
            count += 1
    return count


def assert_recipe_null_regime_pressure(
    policy: ClimbPolicy,
    *,
    ledger: ExhaustedKnobLedger,
    data_eval_identity: str,
    claim_class: str,
    matrix_has_regime_transition: bool,
    proposed_knobs: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """After recipe-null cap, successor matrices must include a regime transition.

    Also rejects matrices that are purely recipe tweaks under the same identity
    once the cap is reached.
    """

    cap = int(policy.recipe_null_cap.get("max_nulls_per_family") or 0)
    if cap < 1:
        return
    nulls = count_recipe_nulls_in_ledger(
        policy,
        ledger,
        data_eval_identity=data_eval_identity,
        claim_class=claim_class,
    )
    require = bool(
        policy.recipe_null_cap.get("require_regime_transition_after_cap", True)
    )
    if nulls < cap:
        return
    if require and not matrix_has_regime_transition:
        raise HillClimbError(
            f"recipe-null cap reached ({nulls}>={cap}) for identity; "
            "successor matrix must include a regime_transition_candidate"
        )
    if proposed_knobs is not None:
        pure_recipe = all(
            is_recipe_tweak_knobs(policy, knobs) for knobs in proposed_knobs
        )
        if pure_recipe:
            raise HillClimbError(
                f"recipe-null cap reached ({nulls}>={cap}); replaying only "
                "recipe-tweak knobs under the same identity is rejected"
            )


def assert_rung_allows_promotion(
    policy: ClimbPolicy,
    *,
    claimed_role: str,
    certified_rungs: Sequence[str] | None = None,
) -> None:
    """Optional I10-style rung gate from external policy."""

    gates = policy.rung_gates
    if not gates.get("enabled"):
        return
    if claimed_role != "promotion":
        return
    if not gates.get("block_promotion_if_prior_rung_unmet", True):
        return
    order = [str(x) for x in (gates.get("order") or [])]
    current = str(gates.get("current_rung") or "")
    if not order or current not in order:
        raise HillClimbError(
            f"rung_gates enabled but current_rung {current!r} not in order {order}"
        )
    idx = order.index(current)
    if idx == 0:
        return
    prior = order[idx - 1]
    configured = gates.get("certified_rungs") or ()
    certified = set(certified_rungs if certified_rungs is not None else configured)
    if prior not in certified:
        raise HillClimbError(
            f"rung gate: cannot open promotion while prior rung {prior!r} is unmet "
            f"(current={current!r})"
        )
    evidence = gates.get("certification_evidence") or {}
    if not isinstance(evidence, Mapping) or not evidence.get(prior):
        raise HillClimbError(
            f"rung gate: certified prior rung {prior!r} lacks durable evidence"
        )


def _leaf(metric: str) -> str:
    return metric.split(".")[-1]


def _metric_from_map(metrics: Mapping[str, Any], metric_id: str) -> float | None:
    if metric_id in metrics and isinstance(metrics[metric_id], (int, float)):
        return float(metrics[metric_id])
    leaf = _leaf(metric_id)
    if leaf in metrics and isinstance(metrics[leaf], (int, float)):
        return float(metrics[leaf])
    for key, val in metrics.items():
        if str(key).endswith(f".{leaf}") and isinstance(val, (int, float)):
            return float(val)
    return None


def promotion_primary_effect_met(
    *,
    control_metrics: Mapping[str, Any] | None,
    candidate_metrics: Mapping[str, Any] | None,
    promotion_primary: Mapping[str, Any],
    require_dual_arm_metrics: bool = True,
    require_parse_non_regression: bool = True,
) -> tuple[bool, list[str], float | None]:
    """Whether dual-arm held-out (or policy) primary meets minimum_effect.

    Used by continuous promote dispose: cert ``continue`` is necessary but not
    sufficient. Returns ``(ok, reasons, improvement)``. Missing dual-arm primary
    values fail closed when ``require_dual_arm_metrics`` is true.
    """
    reasons: list[str] = []
    metric = str(promotion_primary.get("metric") or "")
    direction = str(promotion_primary.get("direction") or "increase")
    minimum_effect = float(promotion_primary.get("minimum_effect") or 0.0)
    if not metric:
        reasons.append("promote_primary_metric_unconfigured")
        return False, reasons, None

    control = control_metrics or {}
    candidate = candidate_metrics or {}
    c_val = _metric_from_map(control, metric)
    t_val = _metric_from_map(candidate, metric)
    if c_val is None or t_val is None:
        if require_dual_arm_metrics:
            reasons.append(
                f"promote_primary_metrics_missing:metric={metric}:"
                f"control={c_val} candidate={t_val}"
            )
            return False, reasons, None
        reasons.append(f"promote_primary_metrics_optional_missing:metric={metric}")
        return True, reasons, None

    raw = t_val - c_val
    improvement = float(improvement_signed(raw, direction))  # type: ignore[arg-type]

    if require_parse_non_regression:
        c_pr = _metric_from_map(control, "parse_rate")
        t_pr = _metric_from_map(candidate, "parse_rate")
        if c_pr is not None and t_pr is not None and t_pr + 1e-12 < c_pr:
            reasons.append(f"promote_parse_regression:control={c_pr} candidate={t_pr}")
            return False, reasons, improvement

    # Match classify_positive_metrics: strict greater-than min_effect.
    if not (improvement > minimum_effect):
        reasons.append(
            "promote_primary_null_or_insufficient:"
            f"metric={metric}:control={c_val}:candidate={t_val}:"
            f"min_effect={minimum_effect}:delta={improvement}"
        )
        return False, reasons, improvement

    reasons.append(
        f"promote_primary_win:{metric}:{c_val}->{t_val}:"
        f"improvement={improvement}:min_effect={minimum_effect}"
    )
    return True, reasons, improvement


def classify_positive_metrics(
    policy: ClimbPolicy,
    *,
    role: str,
    control_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    baseline_trainable_params: int | None = None,
    candidate_trainable_params: int | None = None,
    eg_params_by_seed: Sequence[float] | None = None,
    executable_unblock: bool = False,
    fixture_insufficient_n: bool = False,
) -> dict[str, Any]:
    """Direction-signed primary win + optional EG_params for Phase A positive."""

    primary = primary_for_role(policy, role)
    metric = str(primary["metric"])
    direction = str(primary["direction"])
    minimum_effect = float(primary.get("minimum_effect") or 0.0)
    reasons: list[str] = []
    positive = False

    c_val = _metric_from_map(control_metrics, metric)
    t_val = _metric_from_map(candidate_metrics, metric)
    if c_val is None or t_val is None:
        reasons.append("primary_metric_unavailable")
    else:
        raw = t_val - c_val
        improvement = improvement_signed(raw, direction)  # type: ignore[arg-type]
        non_reg_ok = True
        for nr in primary.get("require_non_regression_metrics") or []:
            if not isinstance(nr, Mapping):
                continue
            nr_metric = str(nr.get("metric_leaf") or nr.get("metric") or "")
            nr_dir = str(nr.get("direction") or "increase")
            nr_min = float(nr.get("minimum_effect") or 0.0)
            c_nr = _metric_from_map(control_metrics, nr_metric)
            t_nr = _metric_from_map(candidate_metrics, nr_metric)
            if c_nr is None or t_nr is None:
                continue
            nr_imp = improvement_signed(t_nr - c_nr, nr_dir)  # type: ignore[arg-type]
            if nr_imp < nr_min:
                non_reg_ok = False
                reasons.append(f"non_regression_fail:{nr_metric}:{c_nr}->{t_nr}")
        if improvement > minimum_effect and non_reg_ok:
            positive = True
            reasons.append(
                f"primary_metric_win:{metric}:{c_val}->{t_val}"
                f":improvement={improvement}"
            )
        else:
            reasons.append(
                f"primary_metric_null_or_worse:{metric}:"
                f"control={c_val} candidate={t_val} improvement={improvement}"
            )

    pos_cfg = policy.positive_classification
    if executable_unblock and pos_cfg.get("allow_executable_unblock", True):
        positive = True
        reasons.append("executable_unblock:candidate_completed_after_control_error")

    if (
        fixture_insufficient_n
        and pos_cfg.get("fixture_insufficient_n_alone_not_positive", True)
        and not any(r.startswith("primary_metric_win") for r in reasons)
        and not any(r.startswith("executable_unblock") for r in reasons)
    ):
        positive = False
        reasons.append("fixture_insufficient_n_alone")

    # Capacity charge
    if pos_cfg.get("require_size_match_or_eg_params", True):
        if (
            baseline_trainable_params is not None
            and candidate_trainable_params is not None
            and int(candidate_trainable_params) > int(baseline_trainable_params)
        ):
            try:
                assert_capacity_growth_allowed(
                    baseline_params=int(baseline_trainable_params),
                    candidate_params=int(candidate_trainable_params),
                    eg_params_by_seed=eg_params_by_seed,
                    eg_params_lcb_min=float(pos_cfg.get("eg_params_lcb_min") or 1.0),
                )
            except HillClimbError as exc:
                positive = False
                reasons.append(f"eg_params_block:{exc}")

    if not any(
        r.startswith("primary_metric_win") or r.startswith("executable_unblock")
        for r in reasons
    ):
        positive = False
        if not reasons:
            reasons.append("no_positive_signal")

    return {
        "positive": positive,
        "stack_layer": positive,
        "reasons": reasons,
        "primary_metric": metric,
        "direction": direction,
        "role": role,
        "control_metrics": dict(control_metrics),
        "candidate_metrics": dict(candidate_metrics),
        "policy_sha256": policy.sha256,
        "policy_version": policy.version,
    }


def synthesis_policy_allows_sft(
    policy: ClimbPolicy,
    *,
    train_dir: Path,
) -> bool:
    """True when synthesis gate policy permits SFT (delegates to hillclimb gate)."""

    if not policy.synthesis_loop.get("block_sft_on_open_recommendations", True):
        return True
    from slm_training.autoresearch.hillclimb import (
        assert_synthesis_feedback_cleared_for_sft,
    )

    names = (
        tuple(str(x) for x in (policy.synthesis_loop.get("action_filenames") or ()))
        or None
    )
    waiver = policy.synthesis_loop.get("allow_global_waiver_code", "*")
    assert_synthesis_feedback_cleared_for_sft(
        Path(train_dir),
        allow_missing_feedback=True,
        action_filenames=names,
        allow_global_waiver_code=str(waiver) if waiver is not None else "*",
    )
    return True


def train_eval_from_knobs(
    knobs_list: Sequence[Mapping[str, Any]],
) -> tuple[str | None, str | None]:
    """Pick train_version / eval_version from proposed experiment knobs."""

    trains: list[str] = []
    evals: list[str] = []
    for knobs in knobs_list:
        if not isinstance(knobs, Mapping):
            continue
        if knobs.get("train_version"):
            trains.append(str(knobs["train_version"]))
        if knobs.get("eval_version"):
            evals.append(str(knobs["eval_version"]))
    train = sorted(set(trains))[0] if trains else None
    eval_v = sorted(set(evals))[0] if evals else None
    return train, eval_v


def promotion_seed_floor(policy: ClimbPolicy) -> tuple[int, bool]:
    """Return (min_seeds, require_multi_seed) from promotion_primary policy."""

    primary = policy.promotion_primary
    require = bool(primary.get("require_multi_seed", True))
    min_seeds = int(primary.get("min_seeds") or 2)
    if min_seeds < 1:
        min_seeds = 2
    return min_seeds, require


# Re-export helpers tests may want without hunting hillclimb.
__all__ += [
    "assert_matrix_knobs_not_exhausted",
    "is_measured_null_feedback",
    "knob_signature_sha256",
    "infer_metric_direction",
    "HillClimbError",
    "MetricDirection",
]
