"""Evidence-governance gates for consistent autotrain hill-climbing.

Pure predicates first (testable without I/O), then small ledger/file helpers.
These gates separate fixture busywork from promotion-class climb steps.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.harness_core.promotion_engine import check_parameter_efficiency
from slm_training.lineage.records import canonical_json

MetricDirection = Literal["increase", "decrease"]

__all__ = [
    "CLIMB_CLAIM_CLASSES",
    "NON_CLIMB_CLAIM_CLASSES",
    "MIN_CLIMB_SEEDS",
    "EXHAUSTED_LEDGER_FILENAME",
    "MetricDirection",
    "ClimbStepEligibility",
    "ExhaustedKnobEntry",
    "ExhaustedKnobLedger",
    "SynthesisGateResult",
    "climb_step_eligibility",
    "assert_climb_label_allowed",
    "validate_causal_campaign_shape",
    "knob_signature_sha256",
    "data_eval_identity",
    "record_null_outcome",
    "assert_matrix_knobs_not_exhausted",
    "open_synthesis_recommendations",
    "assert_synthesis_feedback_cleared_for_sft",
    "assert_capacity_growth_allowed",
    "exhausted_ledger_path",
    "load_exhausted_ledger",
    "save_exhausted_ledger",
    "matrix_data_eval_identity",
    "data_generation_sha256",
    "is_measured_null_feedback",
    "resolve_primary_effect",
    "resolve_baseline_primary",
    "infer_metric_direction",
    "improvement_signed",
    "record_null_from_knob_signature_json",
    "HillClimbError",
    "PARSE_RATE_PERFECT",
    "invalid_grammar_reasons",
    "parse_rate_illegal",
    "HILLCLIMB_STAGNATION_CADENCE",
    "hillclimb_iteration_report",
    "assess_hillclimb_speculation",
    "stagnation_review",
    "CLIMB_CHAMPION_SCHEMA",
    "CHAMPION_EPOCHS_EXHAUSTED",
    "DEFAULT_MAX_CUMULATIVE_EPOCHS",
    "ClimbChampionSidecar",
    "climb_champion_dir",
    "climb_champion_checkpoint_path",
    "climb_champion_sidecar_path",
    "dump_climb_champion",
    "load_climb_champion",
    "attach_initialize_from",
    "assert_warm_start_launch",
    "assert_champion_eval_disjoint",
    "champion_epoch_park_reason",
    "write_climb_champion",
    "maybe_advance_climb_champion",
    "seed_climb_champion",
]

CLIMB_CLAIM_CLASSES: frozenset[str] = frozenset(
    {"promotion_candidate", "ship_gate"}
)
NON_CLIMB_CLAIM_CLASSES: frozenset[str] = frozenset(
    {"wiring", "fixture", "diagnostic", "screening"}
)
MIN_CLIMB_SEEDS = 2
# I6: completed documents must parse. Not a lever; never < 1 in autotrain.
PARSE_RATE_PERFECT = 1.0


def parse_rate_illegal(value: Any) -> bool:
    """True when a measured parse_rate is below perfect (unmeasured is not illegal)."""
    if value is None:
        return False
    try:
        return float(value) + 1e-12 < PARSE_RATE_PERFECT
    except (TypeError, ValueError):
        return True


def invalid_grammar_reasons(
    metrics: Mapping[str, Any] | None, *, arm: str
) -> list[str]:
    """I6: any measured parse_rate < 1 on completed docs is invalid grammar."""
    if not isinstance(metrics, Mapping):
        return []
    reasons: list[str] = []
    for key, raw in metrics.items():
        if str(key).split(".")[-1] != "parse_rate":
            continue
        if parse_rate_illegal(raw):
            reasons.append(
                f"invalid_grammar:{arm}:{key}={raw}<{PARSE_RATE_PERFECT:g}"
            )
    return reasons

_SYNTHESIS_ACTION_NAMES = (
    "synthesis_feedback_actions.json",
    "synthesis_feedback_action.json",
)


CLIMB_CHAMPION_SCHEMA = "climb_champion/v1"
CHAMPION_EPOCHS_EXHAUSTED = "champion_epochs_exhausted"
DEFAULT_MAX_CUMULATIVE_EPOCHS = 50


class HillClimbError(ValueError):
    """A hill-climb governance gate refused the requested action."""


@dataclass
class ClimbChampionSidecar:
    """Lineage for ``loops/<id>/champion/last.pt``."""

    source_campaign: str
    cumulative_steps: int
    train_data_manifest_sha: str
    cumulative_epochs: float
    trainable_params: int | None = None
    knobs: dict[str, Any] = field(default_factory=dict)
    schema: str = CLIMB_CHAMPION_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = self.schema
        return payload


def climb_champion_dir(loop_dir: Path) -> Path:
    return Path(loop_dir) / "champion"


def climb_champion_checkpoint_path(loop_dir: Path) -> Path:
    return climb_champion_dir(loop_dir) / "last.pt"


def climb_champion_sidecar_path(loop_dir: Path) -> Path:
    return climb_champion_dir(loop_dir) / "last.json"


def dump_climb_champion(sidecar: ClimbChampionSidecar, loop_dir: Path) -> Path:
    path = climb_champion_sidecar_path(loop_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sidecar.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_climb_champion(loop_dir: Path) -> ClimbChampionSidecar | None:
    path = climb_champion_sidecar_path(loop_dir)
    ckpt = climb_champion_checkpoint_path(loop_dir)
    if not path.is_file() or not ckpt.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise HillClimbError("champion_sidecar:invalid")
    return ClimbChampionSidecar(
        source_campaign=str(raw.get("source_campaign") or ""),
        cumulative_steps=int(raw.get("cumulative_steps") or 0),
        train_data_manifest_sha=str(raw.get("train_data_manifest_sha") or ""),
        cumulative_epochs=float(raw.get("cumulative_epochs") or 0.0),
        trainable_params=(
            int(raw["trainable_params"])
            if raw.get("trainable_params") is not None
            else None
        ),
        knobs=dict(raw.get("knobs") or {}),
        schema=str(raw.get("schema") or CLIMB_CHAMPION_SCHEMA),
    )


def assert_warm_start_launch(
    control_knobs: Mapping[str, Any],
    candidate_knobs: Mapping[str, Any],
    *,
    trainable_params: tuple[int | None, int | None] | None = None,
) -> None:
    """Identical K, data, checkpoint, and params for paired warm-start arms."""

    if control_knobs.get("steps") != candidate_knobs.get("steps"):
        raise HillClimbError("warm_start:unequal_extra_steps")
    if control_knobs.get("initialize_from") != candidate_knobs.get("initialize_from"):
        raise HillClimbError("warm_start:unequal_checkpoint")
    if control_knobs.get("train_version") != candidate_knobs.get("train_version"):
        raise HillClimbError("warm_start:unequal_train_data")
    if control_knobs.get("seed") != candidate_knobs.get("seed"):
        raise HillClimbError("warm_start:unequal_seed")
    from slm_training.levers import require_size_matched_arms

    require_size_matched_arms(
        control_knobs, candidate_knobs, context="warm_start"
    )
    if trainable_params is not None:
        left, right = trainable_params
        if left is not None and right is not None and int(left) != int(right):
            raise HillClimbError("warm_start:unequal_params")


def attach_initialize_from(
    control_knobs: Mapping[str, Any],
    candidate_knobs: Mapping[str, Any],
    checkpoint: Path | str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    control = dict(control_knobs)
    candidate = dict(candidate_knobs)
    if checkpoint is not None:
        path = str(checkpoint)
        control["initialize_from"] = path
        candidate["initialize_from"] = path
    assert_warm_start_launch(control, candidate)
    return control, candidate


def assert_champion_eval_disjoint(
    *,
    train_data_manifest_sha: str | None,
    eval_data_manifest_sha: str | None,
) -> None:
    """Fail closed: missing or colliding train/eval identities are leakage."""

    train = str(train_data_manifest_sha or "").strip()
    eval_sha = str(eval_data_manifest_sha or "").strip()
    if not train:
        raise HillClimbError("champion_leakage:missing_train_manifest")
    if not eval_sha:
        raise HillClimbError("champion_leakage:missing_eval_manifest")
    if train == eval_sha:
        raise HillClimbError("champion_leakage:train_eval_manifest_collision")


def champion_epoch_park_reason(
    sidecar: ClimbChampionSidecar | None,
    *,
    max_cumulative_epochs: float = DEFAULT_MAX_CUMULATIVE_EPOCHS,
) -> str | None:
    if sidecar is None:
        return None
    if float(sidecar.cumulative_epochs) > float(max_cumulative_epochs):
        return CHAMPION_EPOCHS_EXHAUSTED
    return None


def write_climb_champion(
    loop_dir: Path,
    *,
    checkpoint: Path,
    sidecar: ClimbChampionSidecar,
) -> ClimbChampionSidecar:
    dest = climb_champion_checkpoint_path(loop_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = Path(checkpoint)
    if not src.is_file():
        raise HillClimbError(f"champion_checkpoint_missing:{src}")
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    dump_climb_champion(sidecar, loop_dir)
    return sidecar


def maybe_advance_climb_champion(
    loop_dir: Path,
    *,
    confirmed: bool,
    checkpoint: Path | None,
    source_campaign: str,
    extra_steps: int,
    train_data_manifest_sha: str,
    record_count: int,
    knobs: Mapping[str, Any] | None = None,
    trainable_params: int | None = None,
) -> ClimbChampionSidecar | None:
    """Replace persistent champion only after a confirmed win."""

    current = load_climb_champion(loop_dir)
    if not confirmed or checkpoint is None:
        return current
    n = max(int(record_count) or 1, 1)
    prior_steps = current.cumulative_steps if current else 0
    prior_epochs = current.cumulative_epochs if current else 0.0
    sidecar = ClimbChampionSidecar(
        source_campaign=str(source_campaign),
        cumulative_steps=prior_steps + int(extra_steps),
        train_data_manifest_sha=str(train_data_manifest_sha),
        cumulative_epochs=prior_epochs + (float(extra_steps) / n),
        trainable_params=trainable_params,
        knobs=dict(knobs or {}),
    )
    return write_climb_champion(loop_dir, checkpoint=checkpoint, sidecar=sidecar)


def seed_climb_champion(
    loop_dir: Path,
    *,
    confirmed_artifacts: Sequence[Mapping[str, Any]] | None = None,
    baseline_checkpoint: Path | None = None,
    source_campaign: str = "",
    extra_steps: int = 0,
    train_data_manifest_sha: str = "",
    record_count: int = 1,
    knobs: Mapping[str, Any] | None = None,
    trainable_params: int | None = None,
) -> ClimbChampionSidecar | None:
    """Seed from best confirmed artifact, else one baseline train checkpoint."""

    existing = load_climb_champion(loop_dir)
    if existing is not None:
        return existing
    chosen: Path | None = None
    campaign = source_campaign
    artifact_knobs = dict(knobs or {})
    for row in reversed(list(confirmed_artifacts or ())):
        if str(row.get("status") or "") not in {
            "confirmed",
            "climb_accepted",
            "promoted",
        }:
            continue
        raw = row.get("checkpoint") or row.get("checkpoint_path")
        if not raw:
            continue
        path = Path(str(raw))
        if path.is_file():
            chosen = path
            campaign = str(row.get("campaign_id") or campaign)
            if isinstance(row.get("knobs"), dict):
                artifact_knobs = dict(row["knobs"])
            break
    if chosen is None and baseline_checkpoint is not None:
        if Path(baseline_checkpoint).is_file():
            chosen = Path(baseline_checkpoint)
    if chosen is None:
        return None
    sidecar = ClimbChampionSidecar(
        source_campaign=campaign,
        cumulative_steps=int(extra_steps),
        train_data_manifest_sha=str(train_data_manifest_sha),
        cumulative_epochs=float(extra_steps) / max(int(record_count) or 1, 1),
        trainable_params=trainable_params,
        knobs=artifact_knobs,
    )
    return write_climb_champion(loop_dir, checkpoint=chosen, sidecar=sidecar)


@dataclass(frozen=True)
class ClimbStepEligibility:
    eligible: bool
    failures: tuple[str, ...]
    claim_class: str

    @property
    def is_climb_step(self) -> bool:
        return self.eligible


def climb_step_eligibility(
    *,
    claim_class: str,
    locked_eval_manifest_sha256: str | None,
    seeds: Sequence[int],
    primary_endpoint_seed_values: Sequence[float] | None = None,
    primary_lcb: float | None = None,
    baseline_primary: float | None = None,
    minimum_effect: float = 0.0,
    label_as_climb: bool = True,
    direction: MetricDirection = "increase",
    min_seeds: int | None = None,
    require_multi_seed: bool = True,
) -> ClimbStepEligibility:
    """Return whether an outcome may be recorded as a hill-climb step.

    Fixture/wiring/diagnostic/screening can run, but never as climb progress.
    Promotion-class requires locked held-out identity and multi-seed primary.
    ``direction`` signs the primary so both higher-is-better and lower-is-better
    metrics use the same improvement ≥ minimum_effect rule.
    ``min_seeds`` / ``require_multi_seed`` come from external climb policy when set.
    """

    failures: list[str] = []
    seed_floor = int(min_seeds) if min_seeds is not None else MIN_CLIMB_SEEDS
    if seed_floor < 1:
        seed_floor = MIN_CLIMB_SEEDS
    if not label_as_climb:
        return ClimbStepEligibility(
            eligible=False,
            failures=("label_as_climb_false",),
            claim_class=claim_class,
        )
    if claim_class in NON_CLIMB_CLAIM_CLASSES:
        failures.append(f"claim_class_not_climb:{claim_class}")
    elif claim_class not in CLIMB_CLAIM_CLASSES:
        failures.append(f"claim_class_unknown:{claim_class}")
    if claim_class in CLIMB_CLAIM_CLASSES:
        if not locked_eval_manifest_sha256:
            failures.append("locked_eval_manifest_sha256_missing")
        if require_multi_seed and len(tuple(seeds)) < seed_floor:
            failures.append(f"insufficient_seeds:{len(tuple(seeds))}<{seed_floor}")
        # Multi-seed primary values are mandatory for climb eligibility — seeds
        # alone never prove an LCB (fail closed when omitted).
        if require_multi_seed and primary_endpoint_seed_values is None:
            failures.append("primary_seed_values_missing")
        elif primary_endpoint_seed_values is not None:
            vals = [float(v) for v in primary_endpoint_seed_values]
            if require_multi_seed and len(vals) < seed_floor:
                failures.append("primary_seed_values_incomplete")
            elif any(not math.isfinite(v) for v in vals):
                failures.append("primary_seed_values_non_finite")
            else:
                bound = _seed_conservative_bound(vals, direction=direction)
                if primary_lcb is not None:
                    # Caller-supplied bound overrides (must already be the
                    # conservative side for the metric direction).
                    bound = float(primary_lcb)
                if baseline_primary is not None:
                    raw = bound - float(baseline_primary)
                    improvement = improvement_signed(raw, direction)
                    if improvement < float(minimum_effect):
                        failures.append("primary_lcb_below_baseline_plus_effect")
    return ClimbStepEligibility(
        eligible=not failures, failures=tuple(failures), claim_class=claim_class
    )


def assert_climb_label_allowed(
    *,
    claim_class: str,
    locked_eval_manifest_sha256: str | None,
    seeds: Sequence[int],
    label_as_climb: bool = True,
) -> ClimbStepEligibility:
    """Fail closed when callers try to brand non-climb work as a climb step."""

    result = climb_step_eligibility(
        claim_class=claim_class,
        locked_eval_manifest_sha256=locked_eval_manifest_sha256,
        seeds=seeds,
        label_as_climb=label_as_climb,
    )
    if label_as_climb and not result.eligible:
        raise HillClimbError(
            "cannot label outcome as hill-climb step: "
            + ", ".join(result.failures)
        )
    return result


def validate_causal_campaign_shape(
    campaign: ExperimentCampaignV1,
    *,
    min_seeds: int | None = None,
    require_multi_seed: bool | None = None,
) -> tuple[str, ...]:
    """Promotion-class campaigns must declare causal arm shape + kill criteria.

    Seed floor comes from external climb policy when ``min_seeds`` /
    ``require_multi_seed`` are omitted (``promotion_primary.min_seeds`` /
    ``require_multi_seed``). Explicit kwargs override for tests.
    """

    if campaign.claim_class not in CLIMB_CLAIM_CLASSES:
        return ()
    seed_floor = MIN_CLIMB_SEEDS
    require_ms = True
    if min_seeds is not None or require_multi_seed is not None:
        seed_floor = int(min_seeds) if min_seeds is not None else MIN_CLIMB_SEEDS
        require_ms = True if require_multi_seed is None else bool(require_multi_seed)
    else:
        try:
            from slm_training.autoresearch.climb_policy import (
                load_climb_policy,
                promotion_seed_floor,
            )

            seed_floor, require_ms = promotion_seed_floor(load_climb_policy())
        except Exception:  # noqa: BLE001 — policy optional for pure unit fixtures
            seed_floor, require_ms = MIN_CLIMB_SEEDS, True
    if not require_ms:
        seed_floor = max(1, int(seed_floor) if seed_floor else 1)
    elif seed_floor < 1:
        seed_floor = MIN_CLIMB_SEEDS

    failures: list[str] = []
    if len(campaign.seeds) < seed_floor:
        failures.append(f"insufficient_seeds:{len(campaign.seeds)}<{seed_floor}")
    if not campaign.locked_eval_manifest_sha256:
        failures.append("locked_eval_manifest_sha256_missing")
    arm_ids = {arm.arm_id for arm in campaign.arms}
    if not any(arm.role == "control" for arm in campaign.arms):
        failures.append("missing_control_arm")
    kinds = {c.kind for c in campaign.controls}
    if "positive" not in kinds:
        failures.append("missing_positive_matched_control")
    if "negative" not in kinds:
        failures.append("missing_destructive_negative_control")
    mechanism_off = tuple(getattr(campaign, "mechanism_off_arm_ids", ()) or ())
    if not mechanism_off:
        failures.append("missing_mechanism_off_arm_ids")
    else:
        unknown = set(mechanism_off) - arm_ids
        if unknown:
            failures.append(
                "mechanism_off_arm_unknown:" + ",".join(sorted(unknown))
            )
    kills = tuple(getattr(campaign, "executable_kill_criteria", ()) or ())
    if not kills:
        # Fall back: non-empty stopping_rules alone are documentary; require
        # explicit kill criteria field for climb campaigns.
        failures.append("missing_executable_kill_criteria")
    return tuple(failures)


def assert_causal_campaign_shape(
    campaign: ExperimentCampaignV1,
    *,
    min_seeds: int | None = None,
    require_multi_seed: bool | None = None,
) -> None:
    failures = validate_causal_campaign_shape(
        campaign, min_seeds=min_seeds, require_multi_seed=require_multi_seed
    )
    if failures:
        raise HillClimbError(
            "promotion-class campaign missing causal shape: "
            + ", ".join(failures)
        )


def knob_signature_sha256(knobs: Mapping[str, Any]) -> str:
    """Stable hash of a knob dict (exclude_none semantics)."""

    cleaned = {k: v for k, v in knobs.items() if v is not None}
    return hashlib.sha256(
        canonical_json(cleaned).encode("utf-8")
    ).hexdigest()


def data_eval_identity(
    *,
    data_manifest_sha256: str | None = None,
    locked_eval_manifest_sha256: str | None = None,
    claim_class: str,
) -> str:
    payload = {
        "data_manifest_sha256": data_manifest_sha256 or "",
        "locked_eval_manifest_sha256": locked_eval_manifest_sha256 or "",
        "claim_class": claim_class,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def data_generation_sha256(
    data_generation: Mapping[str, Any] | None,
) -> str | None:
    """Stable digest of corpus-generation knobs; None when the block is absent."""

    if not data_generation:
        return None
    if hasattr(data_generation, "model_dump"):
        cleaned = data_generation.model_dump(exclude_none=True, mode="json")
    else:
        cleaned = {key: value for key, value in data_generation.items() if value is not None}
    if not cleaned:
        return None
    return hashlib.sha256(canonical_json(cleaned).encode("utf-8")).hexdigest()


@dataclass
class ExhaustedKnobEntry:
    knob_signature_sha256: str
    data_eval_identity: str
    claim_class: str
    reason: str = "primary_lcb_within_noise"
    note: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "knob_signature_sha256": self.knob_signature_sha256,
            "data_eval_identity": self.data_eval_identity,
            "claim_class": self.claim_class,
            "reason": self.reason,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExhaustedKnobEntry:
        return cls(
            knob_signature_sha256=str(raw["knob_signature_sha256"]),
            data_eval_identity=str(raw["data_eval_identity"]),
            claim_class=str(raw["claim_class"]),
            reason=str(raw.get("reason") or "primary_lcb_within_noise"),
            note=str(raw.get("note") or ""),
        )


@dataclass
class ExhaustedKnobLedger:
    """Append-only exhausted-knob records for a campaign artifact tree."""

    entries: list[ExhaustedKnobEntry] = field(default_factory=list)

    def is_exhausted(
        self,
        *,
        knob_signature_sha256: str,
        data_eval_identity: str,
        claim_class: str,
    ) -> bool:
        for entry in self.entries:
            if (
                entry.knob_signature_sha256 == knob_signature_sha256
                and entry.data_eval_identity == data_eval_identity
                and entry.claim_class == claim_class
            ):
                return True
        return False

    def record_null(
        self,
        *,
        knob_signature_sha256: str,
        data_eval_identity: str,
        claim_class: str,
        reason: str = "primary_lcb_within_noise",
        note: str = "",
    ) -> ExhaustedKnobEntry:
        entry = ExhaustedKnobEntry(
            knob_signature_sha256=knob_signature_sha256,
            data_eval_identity=data_eval_identity,
            claim_class=claim_class,
            reason=reason,
            note=note,
        )
        if not self.is_exhausted(
            knob_signature_sha256=knob_signature_sha256,
            data_eval_identity=data_eval_identity,
            claim_class=claim_class,
        ):
            self.entries.append(entry)
        return entry

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "exhausted_knob_ledger/v1",
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExhaustedKnobLedger:
        entries = [
            ExhaustedKnobEntry.from_dict(item)
            for item in (raw.get("entries") or [])
        ]
        return cls(entries=entries)

    def save(self, path: Path) -> None:
        """Atomically write the ledger (temp file + replace)."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> ExhaustedKnobLedger:
        if not path.is_file():
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def record_null_outcome(
    ledger: ExhaustedKnobLedger,
    *,
    knobs: Mapping[str, Any],
    data_eval_identity: str,
    claim_class: str,
    reason: str = "primary_lcb_within_noise",
    note: str = "",
) -> ExhaustedKnobEntry:
    return ledger.record_null(
        knob_signature_sha256=knob_signature_sha256(knobs),
        data_eval_identity=data_eval_identity,
        claim_class=claim_class,
        reason=reason,
        note=note,
    )


def assert_matrix_knobs_not_exhausted(
    *,
    knob_signatures: Sequence[str],
    ledger: ExhaustedKnobLedger,
    data_eval_identity: str,
    claim_class: str,
) -> None:
    """Reject matrix proposals that replay exhausted knob signatures."""

    blocked = [
        sig
        for sig in knob_signatures
        if ledger.is_exhausted(
            knob_signature_sha256=sig,
            data_eval_identity=data_eval_identity,
            claim_class=claim_class,
        )
    ]
    if blocked:
        raise HillClimbError(
            "hypothesis matrix reuses exhausted knob signature(s): "
            + ", ".join(blocked[:5])
        )


@dataclass(frozen=True)
class SynthesisGateResult:
    allowed: bool
    open_recommendations: tuple[dict[str, Any], ...]
    reason: str


def open_synthesis_recommendations(
    feedback: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    """Return still-open synthesis recommendations that require action."""

    if not feedback:
        return ()
    recs = feedback.get("recommendations") or feedback.get("actions_required") or []
    if not isinstance(recs, list):
        return ()
    open_recs: list[dict[str, Any]] = []
    for item in recs:
        if not isinstance(item, Mapping):
            continue
        # Explicitly closed rows are ignored.
        if item.get("status") in {"done", "closed", "waived", "acted"}:
            continue
        # Require actionable codes (matches train_data.feedback emitters).
        code = str(item.get("code") or item.get("recommendation_code") or "")
        if not code:
            continue
        open_recs.append(dict(item))
    return tuple(open_recs)


def _load_json(path: Path) -> dict[str, Any] | None:
    """Return a dict from JSON, or None if the path is missing.

    Corrupt / non-object JSON raises :class:`HillClimbError` (fail closed)
    rather than looking like a missing artifact.
    """

    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HillClimbError(f"corrupt JSON at {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise HillClimbError(f"expected JSON object at {path}, got {type(raw).__name__}")
    return raw


def assert_synthesis_feedback_cleared_for_sft(
    train_dir: Path,
    *,
    action_path: Path | None = None,
    allow_missing_feedback: bool = True,
    action_filenames: Sequence[str] | None = None,
    allow_global_waiver_code: str | None = "*",
) -> SynthesisGateResult:
    """Refuse SFT when synthesis_feedback has open recommendations and no action.

    Action record schema (next to train data)::

        {
          "actions": [{"code": "...", "target": "...", "note": "..."}],
          "waivers": [{"code": "...", "reason": "..."}]
        }

    A **waiver** entry with matching ``code`` (or the configured global waiver
    code, default ``"*"``) clears the gate without claiming the synthesizer was
    fixed. Action rows must name concrete codes — the global waiver code is only
    valid on waivers. ``action_filenames`` / ``allow_global_waiver_code`` come
    from external climb policy when set.
    """

    train_dir = Path(train_dir)
    feedback_path = train_dir / "synthesis_feedback.json"
    feedback = _load_json(feedback_path)
    if feedback is None:
        if allow_missing_feedback:
            return SynthesisGateResult(
                allowed=True, open_recommendations=(), reason="no_feedback_artifact"
            )
        raise HillClimbError(
            f"missing synthesis_feedback.json under {train_dir} "
            "(required for SFT after data build)"
        )
    open_recs = open_synthesis_recommendations(feedback)
    if not open_recs:
        return SynthesisGateResult(
            allowed=True, open_recommendations=(), reason="no_open_recommendations"
        )
    names = tuple(action_filenames) if action_filenames else _SYNTHESIS_ACTION_NAMES
    waiver_star = (
        str(allow_global_waiver_code)
        if allow_global_waiver_code is not None
        else "*"
    )
    action_file = action_path
    if action_file is None:
        for name in names:
            candidate = train_dir / name
            if candidate.is_file():
                action_file = candidate
                break
    actions_doc = _load_json(action_file) if action_file else None
    acted_codes: set[str] = set()
    global_waiver = False
    if actions_doc:
        for row in actions_doc.get("actions") or []:
            if not isinstance(row, Mapping) or not row.get("code"):
                continue
            code = str(row["code"])
            if code == waiver_star:
                raise HillClimbError(
                    "synthesis_feedback_actions: actions may not use global waiver "
                    f"code {waiver_star!r}; put global clearances under waivers only"
                )
            acted_codes.add(code)
        for row in actions_doc.get("waivers") or []:
            if not isinstance(row, Mapping) or not row.get("code"):
                continue
            code = str(row["code"])
            acted_codes.add(code)
            if code == waiver_star:
                global_waiver = True
    if global_waiver:
        return SynthesisGateResult(
            allowed=True,
            open_recommendations=open_recs,
            reason="global_waiver",
        )
    still_open = [
        rec
        for rec in open_recs
        if str(rec.get("code") or rec.get("recommendation_code") or "")
        not in acted_codes
    ]
    if still_open:
        codes = sorted(
            {
                str(r.get("code") or r.get("recommendation_code") or "?")
                for r in still_open
            }
        )
        raise HillClimbError(
            "SFT blocked: open synthesis_feedback recommendations without "
            f"action/waiver record: {codes}. Write "
            f"{train_dir / 'synthesis_feedback_actions.json'} after fixing "
            "the synthesizer (or an explicit waiver)."
        )
    return SynthesisGateResult(
        allowed=True,
        open_recommendations=open_recs,
        reason="actions_recorded",
    )


def assert_capacity_growth_allowed(
    *,
    baseline_params: int | None,
    candidate_params: int | None,
    eg_params_by_seed: Sequence[float] | None,
    eg_params_lcb_min: float = 1.0,
) -> dict[str, Any]:
    """Fail closed when capacity grows without EG_params LCB evidence."""

    check = check_parameter_efficiency(
        baseline_params=baseline_params,
        candidate_params=candidate_params,
        eg_params_by_seed=eg_params_by_seed,
        eg_params_lcb_min=eg_params_lcb_min,
    )
    if not check.get("pass"):
        raise HillClimbError(
            "capacity growth blocked for climb/promotion: "
            + str(check.get("reason") or check)
        )
    return check


EXHAUSTED_LEDGER_FILENAME = "exhausted_knob_ledger.json"


def exhausted_ledger_path(campaign_root: Path) -> Path:
    """Canonical ledger path under a campaign store root."""

    return Path(campaign_root) / EXHAUSTED_LEDGER_FILENAME


def load_exhausted_ledger(campaign_root: Path) -> ExhaustedKnobLedger:
    return ExhaustedKnobLedger.load(exhausted_ledger_path(campaign_root))


def save_exhausted_ledger(ledger: ExhaustedKnobLedger, campaign_root: Path) -> Path:
    path = exhausted_ledger_path(campaign_root)
    ledger.save(path)
    return path


def matrix_data_eval_identity(
    *,
    evidence_snapshot_id: str,
    primary_metric: str,
    claim_class: str = "fixture",
    locked_eval_manifest_sha256: str | None = None,
    data_generation: Mapping[str, Any] | None = None,
) -> str:
    """Stable identity for exhausted-knob scope in the exploratory matrix loop."""

    material = f"{evidence_snapshot_id}:{primary_metric}"
    digest = data_generation_sha256(data_generation)
    if digest:
        material = f"{material}:{digest}"
    return data_eval_identity(
        data_manifest_sha256=hashlib.sha256(material.encode("utf-8")).hexdigest(),
        locked_eval_manifest_sha256=locked_eval_manifest_sha256,
        claim_class=claim_class,
    )


def _as_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _metric_lookup(
    metrics: Mapping[str, Any],
    *names: str,
    allow_suffix: bool = True,
    exclude_key_prefixes: Sequence[str] = (),
) -> float | None:
    """Look up a metric value by exact key, then optional safe suffix match.

    Suffix matches skip keys whose dotted path starts with an excluded prefix
    (e.g. ``baseline.`` / ``control.``) so baseline rows never shadow the
    candidate absolute when resolving the primary metric itself.
    """

    for name in names:
        if name in metrics:
            parsed = _as_finite_float(metrics[name])
            if parsed is not None:
                return parsed
    if not allow_suffix:
        return None
    excluded = tuple(exclude_key_prefixes)
    for name in names:
        suffix = f".{name}"
        for key, raw in metrics.items():
            key_s = str(key)
            if key_s != name and not key_s.endswith(suffix):
                continue
            if any(key_s == p.rstrip(".") or key_s.startswith(p) for p in excluded):
                continue
            # Also skip baseline/control when the match is only via a longer name
            # that embeds those roles (baseline.smoke.latency_ms_p50).
            head = key_s.split(".", 1)[0]
            if head in {"baseline", "control", "matched_control"}:
                continue
            parsed = _as_finite_float(raw)
            if parsed is not None:
                return parsed
    return None


def resolve_baseline_primary(
    metrics: Mapping[str, Any],
    *,
    primary_metric: str,
    baseline_primary: float | None = None,
) -> float | None:
    """Resolve a matched-control / baseline primary from kwargs or metrics."""

    if baseline_primary is not None:
        parsed = _as_finite_float(baseline_primary)
        if parsed is not None:
            return parsed
    return _metric_lookup(
        metrics,
        f"{primary_metric}_baseline",
        f"baseline.{primary_metric}",
        f"control.{primary_metric}",
        f"matched_control.{primary_metric}",
        "baseline_primary",
        "control_primary",
        "matched_control_primary",
        "baseline",
    )


def infer_metric_direction(primary_metric: str) -> MetricDirection:
    """Map a metric id to increase/decrease when no endpoint direction is set.

    Aligns with continuous autotrain defaults: latency / loss / nll / error /
    wall-time leaves are lower-is-better; everything else defaults to increase.
    """

    name = primary_metric.lower()
    leaf = name.split(".")[-1]
    lower_tokens = (
        "latency",
        "loss",
        "nll",
        "error",
        "wall_ms",
        "time_ms",
        "cost",
        "duration",
    )
    if any(token in leaf or token in name for token in lower_tokens):
        return "decrease"
    return "increase"


def improvement_signed(raw: float, direction: MetricDirection) -> float:
    """Sign a raw (candidate − baseline) delta so positive always means better."""

    return float(raw) if direction == "increase" else -float(raw)


def _seed_conservative_bound(
    numbers: Sequence[float], *, direction: MetricDirection
) -> float:
    """Conservative multi-seed bound: LCB for increase, UCB for decrease."""

    vals = [float(v) for v in numbers]
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    se = math.sqrt(var / len(vals))
    if direction == "increase":
        return mean - 1.96 * se
    return mean + 1.96 * se


def resolve_primary_effect(
    metrics: Mapping[str, Any],
    *,
    primary_metric: str,
    direction: MetricDirection,
    baseline_primary: float | None = None,
    primary_seed_values: Sequence[float] | None = None,
) -> tuple[float | None, str | None]:
    """Return ``(improvement, source)`` where positive always means better.

    Raw deltas are computed as candidate − baseline (or explicit raw delta),
    then signed by ``direction`` (increase → +raw, decrease → −raw). Absolute
    primary metrics alone never count as an effect.
    """

    # 1) Explicit effect / delta keys (preferred). Raw candidate−baseline units.
    raw = _metric_lookup(
        metrics,
        f"{primary_metric}_delta",
        "primary_delta",
        "delta",
        "effect",
    )
    if raw is not None:
        return improvement_signed(raw, direction), "explicit_delta"
    # "improvement" is already improvement-signed (positive = better).
    signed_improvement = _metric_lookup(metrics, "improvement")
    if signed_improvement is not None:
        return signed_improvement, "explicit_improvement"

    baseline = resolve_baseline_primary(
        metrics, primary_metric=primary_metric, baseline_primary=baseline_primary
    )

    # 2) Multi-seed conservative bound vs baseline.
    if primary_seed_values is not None:
        vals = [_as_finite_float(v) for v in primary_seed_values]
        if all(v is not None for v in vals) and len(vals) >= MIN_CLIMB_SEEDS:
            numbers = [float(v) for v in vals]  # type: ignore[arg-type]
            bound = _seed_conservative_bound(numbers, direction=direction)
            if baseline is not None:
                return (
                    improvement_signed(bound - baseline, direction),
                    "seed_bound_vs_baseline",
                )

    # 3) Explicit primary bound vs baseline — pick the conservative side for
    # the metric direction (LCB for increase, UCB for decrease).
    if direction == "increase":
        bound = _metric_lookup(
            metrics,
            f"{primary_metric}_lcb",
            "primary_lcb",
            "lcb",
            allow_suffix=False,
        )
    else:
        bound = _metric_lookup(
            metrics,
            f"{primary_metric}_ucb",
            "primary_ucb",
            "ucb",
            allow_suffix=False,
        )
    if bound is not None and baseline is not None:
        return (
            improvement_signed(bound - baseline, direction),
            "primary_bound_vs_baseline",
        )

    # 4) Absolute primary vs baseline (never absolute alone). Exclude
    # baseline/control-prefixed keys so they cannot double as the candidate.
    absolute = _metric_lookup(
        metrics,
        primary_metric,
        exclude_key_prefixes=("baseline.", "control.", "matched_control."),
    )
    if absolute is not None and baseline is not None:
        return (
            improvement_signed(absolute - baseline, direction),
            "absolute_vs_baseline",
        )

    return None, None


def is_measured_null_feedback(
    *,
    outcome_status: str,
    metrics: Mapping[str, Any],
    primary_metric: str,
    direction: MetricDirection | None = None,
    minimum_effect: float = 0.0,
    baseline_primary: float | None = None,
    primary_seed_values: Sequence[float] | None = None,
) -> bool:
    """True when a completed run's primary *improvement* is within noise (null).

    Absolute eval scores alone are never null. ``direction`` must reflect the
    primary's polarity (or is inferred from the metric id). Null iff
    improvement ≤ minimum_effect after direction signing.
    """

    if outcome_status != "completed":
        return False
    resolved_direction = (
        direction if direction is not None else infer_metric_direction(primary_metric)
    )
    improvement, _source = resolve_primary_effect(
        metrics,
        primary_metric=primary_metric,
        direction=resolved_direction,
        baseline_primary=baseline_primary,
        primary_seed_values=primary_seed_values,
    )
    if improvement is None:
        return False
    return improvement <= float(minimum_effect)


def record_null_from_knob_signature_json(
    ledger: ExhaustedKnobLedger,
    *,
    knob_signature_json: str,
    data_eval_identity: str,
    claim_class: str,
    reason: str = "primary_lcb_within_noise",
    note: str = "",
) -> ExhaustedKnobEntry | None:
    """Record exhaustion from the stored knob_signature JSON blob.

    Returns the entry when the signature parses; None when the signature is
    not a JSON object (still leave the ledger unchanged).
    """

    try:
        knobs = json.loads(knob_signature_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(knobs, dict):
        return None
    return record_null_outcome(
        ledger,
        knobs=knobs,
        data_eval_identity=data_eval_identity,
        claim_class=claim_class,
        reason=reason,
        note=note,
    )


# --- per-cycle hill-climb reports + stagnation review (continuous loop) ---

HILLCLIMB_STAGNATION_CADENCE = 10

# Speculations that would weaken ship gates, Lean floors, or decode invariants.
_UNVIABLE_SPECULATION_IDS = frozenset(
    {
        "weaken_ship_gates",
        "screen_below_lean_floor",
        "allow_unconstrained_fallback",
        "grow_capacity_to_green_gate",
        "allow_parse_rate_below_perfect",
    }
)

_VIABLE_AUTO_APPLY = {
    "retire_knob_do_not_confirm": "skip_identical_quality_confirms",
    "keep_ship_gates_do_not_enqueue_confirm": "skip_volume_gate_confirms",
    "fit_decode_timeout_to_n_times_p50": "refit_screening_decode",
    "execute_i10_heal_then_process_arm_not_rematch": "prefer_process_arm",
    "must_generate_screening_n": "generate_lean_floor_eval",
    "replay_frozen_or_fit_decode_to_wall": "refit_screening_decode",
}


def hillclimb_iteration_report(
    *,
    campaign_id: str,
    cycle_index: int | None,
    positive: bool,
    measurement_complete: bool | None,
    reasons: Sequence[str],
    control_metrics: Mapping[str, Any] | None = None,
    candidate_metrics: Mapping[str, Any] | None = None,
    primary_metric: str = "",
) -> dict[str, Any]:
    """Compact well / wrong / speculate payload for one cycle."""
    well: list[str] = []
    wrong: list[str] = []
    speculate: list[str] = []
    if measurement_complete:
        well.append("measurement_complete")
    elif measurement_complete is False:
        wrong.append("measurement_incomplete")
        speculate.append("replay_frozen_or_fit_decode_to_wall")
    if positive:
        well.append("quality_aware_positive")
    else:
        wrong.append("non_positive")
        speculate.append("keep_rotating_size_matched_arms")
    joined = " ".join(str(r) for r in reasons)
    for reason in reasons:
        text = str(reason)
        if text.startswith("mechanism_no_effect"):
            wrong.append(text)
            speculate.append("retire_knob_do_not_confirm")
        if "fixture_insufficient_n_alone" in text or "fixture_volume_gate_ship_only" in text:
            wrong.append(text)
            speculate.append("keep_ship_gates_do_not_enqueue_confirm")
        if "must_generate" in text or "screening_n_infeasible" in text:
            wrong.append(text)
            speculate.append("must_generate_screening_n")
        if "decode" in text.lower() or "timeout" in text.lower() or "wall_" in text:
            speculate.append("fit_decode_timeout_to_n_times_p50")
        if "rebuild_data" in text or "heal_resume" in text or "park" in text.lower():
            speculate.append("execute_i10_heal_then_process_arm_not_rematch")
    if "weaken" in joined.lower() and "gate" in joined.lower():
        speculate.append("weaken_ship_gates")
    if any(str(r).startswith("invalid_grammar:") for r in reasons):
        wrong.append("invalid_grammar")
        speculate.append("reject_illegal_completions")
        speculate.append("allow_parse_rate_below_perfect")
    # Dedupe while preserving order.
    def _uniq(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    deltas: dict[str, Any] = {}
    if isinstance(control_metrics, Mapping) and isinstance(candidate_metrics, Mapping):
        keys = set(control_metrics) | set(candidate_metrics)
        for key in sorted(keys):
            c0, c1 = control_metrics.get(key), candidate_metrics.get(key)
            if isinstance(c0, (int, float)) and isinstance(c1, (int, float)):
                deltas[str(key)] = float(c1) - float(c0)
    return {
        "schema": "hillclimb_iteration/v1",
        "campaign_id": campaign_id,
        "cycle_index": cycle_index,
        "positive": bool(positive),
        "progress": bool(positive),
        "measurement_complete": measurement_complete,
        "primary_metric": primary_metric,
        "deltas": deltas,
        "went_well": _uniq(well),
        "went_wrong": _uniq(wrong),
        "speculate": _uniq(speculate),
        "reasons": [str(r) for r in reasons],
    }


def assess_hillclimb_speculation(speculation_id: str) -> dict[str, Any]:
    """Viability of a hill-climb speculation. Weakening gates is never viable."""
    sid = str(speculation_id)
    if sid in _UNVIABLE_SPECULATION_IDS:
        return {
            "id": sid,
            "viable": False,
            "auto_apply": None,
            "reason": "rejects_invariant_or_ship_gate_weakening",
        }
    apply_kind = _VIABLE_AUTO_APPLY.get(sid)
    return {
        "id": sid,
        "viable": True,
        "auto_apply": apply_kind,
        "reason": "matches_recorded_fail_closed_repair",
    }


def stagnation_review(
    iterations: Sequence[Mapping[str, Any]],
    *,
    cadence: int = HILLCLIMB_STAGNATION_CADENCE,
) -> dict[str, Any] | None:
    """If the last ``cadence`` iterations made no progress, emit a review plan."""
    rows = [dict(item) for item in iterations if isinstance(item, Mapping)]
    if len(rows) < cadence:
        return None
    tail = rows[-cadence:]
    if any(bool(item.get("progress") or item.get("positive")) for item in tail):
        return None
    counts: dict[str, int] = {}
    for item in tail:
        for spec in item.get("speculate") or []:
            counts[str(spec)] = counts.get(str(spec), 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    assessments = [assess_hillclimb_speculation(sid) for sid, _n in ranked]
    viable = [a for a in assessments if a["viable"] and a.get("auto_apply")]
    rejected = [a for a in assessments if not a["viable"]]
    return {
        "schema": "hillclimb_stagnation_review/v1",
        "cadence": cadence,
        "no_progress_streak": cadence,
        "campaign_ids": [str(item.get("campaign_id") or "") for item in tail],
        "speculation_counts": dict(ranked),
        "viable_applies": [a["auto_apply"] for a in viable],
        "rejected": rejected,
        "assessments": assessments,
        "plan": (
            "Apply viable fail-closed repairs from the last "
            f"{cadence} no-progress cycles; never weaken ship gates."
        ),
    }
