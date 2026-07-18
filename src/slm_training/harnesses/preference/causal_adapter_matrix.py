"""LDI1-03 causal adapter & objective campaign matrix (SLM-122).

Orchestration-only harness that plans the matched causal exact-state intervention
campaign over the merged LDI1-02 trainer (:mod:`causal_trainer`). It builds one
canonical arm matrix — Stage 0 objective controls (C0–C6), Stage 1 rank/placement,
Stage 2 actuator method — where every arm differs from its stage baseline only in
declared levers, then classifies each arm with the campaign's eligibility and
falsification gates.

This layer is Torch-free and runs **no** training on its own. An arm is executed
only when the caller supplies an adapter-enabled causal policy and an admitted
DecisionEventV2 corpus; otherwise the arm resolves to ``expired`` — never a
fabricated metric. The ship-grade run requires GPU, a pinned causal checkpoint,
and a corpus that passes the LDI0-03 objective-support gate. No quality claim is
made here; this is wiring only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal

from slm_training.lineage.tracks import CAUSAL_LORA_RECIPE

__all__ = [
    "ArmStatus",
    "ArmLevers",
    "CampaignArm",
    "ArmResult",
    "CorpusSupport",
    "CampaignConfig",
    "SUPPORTED_OBJECTIVES",
    "SUPPORTED_METHODS",
    "SET_VALUED_OBJECTIVES",
    "build_stage0",
    "build_stage1",
    "build_stage2",
    "classify_arm",
    "only_declared_levers_differ",
    "needs_replication",
    "run_arm",
    "describe_campaign",
]

# ``expired`` is the honest CPU/no-corpus outcome: the arm was well-formed and
# admissible but no executable policy + admitted corpus was available, so it did
# not run. It is never conflated with ``completed``.
ArmStatus = Literal["admitted", "blocked", "not_supported", "completed", "expired"]
_VALID_STATUS: frozenset[str] = frozenset(
    {"admitted", "blocked", "not_supported", "completed", "expired"}
)

# C1–C4 objective bindings; C0 is the untrained parent baseline (objective None).
SUPPORTED_OBJECTIVES: tuple[str, ...] = (
    "unlikelihood",
    "ftpo_single",
    "ftpo_set",
    "legal_set_mass",
)
# Objectives whose supervision is set-valued and therefore fail closed when the
# corpus lacks multiple verifier-backed alternatives (never silently narrowed).
SET_VALUED_OBJECTIVES: frozenset[str] = frozenset({"ftpo_set", "legal_set_mass"})
SUPPORTED_METHODS: tuple[str, ...] = ("lora", "dora", "pissa", "adalora")
# Methods that must be explicitly enabled; absent a verified implementation they
# yield ``not_supported`` rather than an implicit LoRA fallback.
_EXPERIMENTAL_METHODS: frozenset[str] = frozenset({"adalora"})

_DEFAULT_RANK = int(CAUSAL_LORA_RECIPE["rank"])
_DEFAULT_ALPHA = int(CAUSAL_LORA_RECIPE["alpha"])
_DEFAULT_TARGETS: tuple[str, ...] = tuple(CAUSAL_LORA_RECIPE["target_modules"])


@dataclass(frozen=True)
class ArmLevers:
    """The matched levers for one arm. Fields not listed in an arm's
    ``declared_levers`` must equal the stage baseline."""

    objective: str | None  # None == untrained parent baseline (C0)
    method: str = "lora"
    rank: int = _DEFAULT_RANK
    alpha: int = _DEFAULT_ALPHA
    target_modules: tuple[str, ...] = _DEFAULT_TARGETS
    include_lm_head: bool = False
    layer_pattern: str | None = None  # e.g. "last_k:4"; None == all target blocks
    reference_tether: bool = False
    balanced_sampler: bool = False
    dropout: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "method": self.method,
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": list(self.target_modules),
            "include_lm_head": self.include_lm_head,
            "layer_pattern": self.layer_pattern,
            "reference_tether": self.reference_tether,
            "balanced_sampler": self.balanced_sampler,
            "dropout": self.dropout,
        }


@dataclass(frozen=True)
class CampaignArm:
    """One matched arm. ``declared_levers`` names the fields this arm is allowed
    to vary versus its stage baseline; the matrix invariant checks the rest are
    identical."""

    arm_id: str
    stage: int
    label: str
    levers: ArmLevers
    declared_levers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "stage": self.stage,
            "label": self.label,
            "levers": self.levers.as_dict(),
            "declared_levers": list(self.declared_levers),
        }


@dataclass(frozen=True)
class CorpusSupport:
    """What the admitted DecisionEventV2 corpus can support, from the LDI0-03
    objective-support gate. Drives fail-closed classification."""

    admitted: bool = False
    has_pairs: bool = False
    has_set_valued: bool = False
    trainable_events: int = 0


@dataclass(frozen=True)
class CampaignConfig:
    """Canonical manifest inputs. Every arm is generated from this; only declared
    levers differ per arm."""

    campaign: str = "LDI-causal-adapter"
    base_model_id: str = ""
    base_model_revision: str = ""
    ranks: tuple[int, ...] = (16, 32, 64)
    methods: tuple[str, ...] = ("lora", "dora", "pissa")
    last_k_blocks: int = 4
    replication_seeds: int = 3
    allow_experimental_methods: bool = False


@dataclass(frozen=True)
class ArmResult:
    """Outcome for one arm. ``metrics`` is populated only for ``completed`` arms;
    ``expired``/``blocked``/``not_supported`` carry a reason and no metrics."""

    arm_id: str
    status: ArmStatus
    reason: str
    trainable_parameters: int | None = None
    seeds_run: tuple[int, ...] = ()
    metrics: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUS:
            raise ValueError(f"invalid arm status: {self.status!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "status": self.status,
            "reason": self.reason,
            "trainable_parameters": self.trainable_parameters,
            "seeds_run": list(self.seeds_run),
            "metrics": dict(self.metrics) if self.metrics is not None else None,
        }


# --- Stage builders -------------------------------------------------------

# C0–C6 objective controls at one pinned adapter configuration (standard LoRA
# r32 over the repo's causal targets); only the declared lever varies.
_STAGE0: tuple[tuple[str, str | None, dict[str, Any], tuple[str, ...]], ...] = (
    ("C0", None, {}, ()),  # parent, no update — the matched baseline
    ("C1", "unlikelihood", {}, ("objective",)),
    ("C2", "ftpo_single", {}, ("objective",)),
    ("C3", "ftpo_set", {}, ("objective",)),
    ("C4", "legal_set_mass", {}, ("objective",)),
    ("C5", None, {"reference_tether": True}, ("objective", "reference_tether")),
    (
        "C6",
        None,
        {"reference_tether": True, "balanced_sampler": True},
        ("objective", "reference_tether", "balanced_sampler"),
    ),
)


def build_stage0(config: CampaignConfig) -> list[CampaignArm]:
    """Stage 0: parent + objective controls at pinned LoRA r32.

    C5/C6 bind the best Stage-0 objective at run time; in the manifest they carry
    ``objective=None`` as a placeholder resolved by :func:`build_stage1`-style
    selection, and declare ``objective`` as a varied lever.
    """
    arms: list[CampaignArm] = []
    for label, objective, overrides, declared in _STAGE0:
        levers = replace(
            ArmLevers(objective=objective, rank=_DEFAULT_RANK, alpha=_DEFAULT_ALPHA),
            **overrides,
        )
        arms.append(
            CampaignArm(
                arm_id=f"{config.campaign}/s0/{label.lower()}",
                stage=0,
                label=label,
                levers=levers,
                declared_levers=declared,
            )
        )
    return arms


def build_stage1(config: CampaignConfig, *, best_objective: str) -> list[CampaignArm]:
    """Stage 1: rank × placement sweep on the best eligible Stage-0 objective.

    ``alpha`` is matched to rank (alpha == rank) — the predeclared scaling — so a
    rank change is not confounded by an unmatched alpha.
    """
    if best_objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"best_objective must be a supported objective: {best_objective!r}")
    placements: tuple[tuple[str, str | None], ...] = (
        ("all", None),
        (f"last{config.last_k_blocks}", f"last_k:{config.last_k_blocks}"),
    )
    arms: list[CampaignArm] = []
    for rank in sorted(set(config.ranks)):
        for place_label, pattern in placements:
            declared = ("rank", "alpha") + (("layer_pattern",) if pattern else ())
            arms.append(
                CampaignArm(
                    arm_id=f"{config.campaign}/s1/r{rank}/{place_label}",
                    stage=1,
                    label=f"rank{rank}-{place_label}",
                    levers=ArmLevers(
                        objective=best_objective,
                        rank=rank,
                        alpha=rank,  # matched scaling
                        layer_pattern=pattern,
                    ),
                    declared_levers=declared,
                )
            )
    return arms


def build_stage2(
    config: CampaignConfig, *, best_objective: str, best_rank: int
) -> list[CampaignArm]:
    """Stage 2: actuator method sweep at a fixed target map and matched budget."""
    if best_objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"best_objective must be a supported objective: {best_objective!r}")
    arms: list[CampaignArm] = []
    for method in config.methods:
        arms.append(
            CampaignArm(
                arm_id=f"{config.campaign}/s2/{method}",
                stage=2,
                label=f"method-{method}",
                levers=ArmLevers(
                    objective=best_objective, method=method, rank=best_rank, alpha=best_rank
                ),
                declared_levers=("method",),
            )
        )
    return arms


# --- Eligibility & falsification -----------------------------------------


def classify_arm(
    arm: CampaignArm,
    *,
    corpus: CorpusSupport,
    allow_experimental_methods: bool = False,
) -> ArmResult:
    """Admit, block, or reject one arm before any compute.

    Fail-closed rules (never silently narrowed): an unsupported/unverified method
    is ``not_supported`` (no LoRA fallback); a set-valued objective without
    set-valued corpus support is ``blocked`` (``blocked_by_corpus``); an
    unadmitted corpus blocks every trainable arm. C0 (parent) needs no corpus.
    """
    levers = arm.levers
    method = levers.method
    if method not in SUPPORTED_METHODS:
        return ArmResult(arm.arm_id, "not_supported", f"unknown actuator method {method!r}")
    if method in _EXPERIMENTAL_METHODS and not allow_experimental_methods:
        return ArmResult(
            arm.arm_id,
            "not_supported",
            f"{method} requires an explicit verified opt-in; not falling back to lora",
        )

    # C0 parent baseline is always admissible — it performs no update.
    if levers.objective is None and not levers.reference_tether and not levers.balanced_sampler:
        return ArmResult(arm.arm_id, "admitted", "parent baseline (no update)")

    if not corpus.admitted:
        return ArmResult(arm.arm_id, "blocked", "blocked_by_corpus: corpus not admitted")
    if levers.objective in SET_VALUED_OBJECTIVES and not corpus.has_set_valued:
        return ArmResult(
            arm.arm_id,
            "blocked",
            "blocked_by_corpus: set-valued objective lacks multi-alternative support",
        )
    if levers.objective is not None and not corpus.has_pairs and not corpus.has_set_valued:
        return ArmResult(
            arm.arm_id, "blocked", "blocked_by_corpus: no verifier-backed action evidence"
        )
    return ArmResult(arm.arm_id, "admitted", "corpus supports objective")


def only_declared_levers_differ(baseline: CampaignArm, arm: CampaignArm) -> bool:
    """True iff ``arm`` differs from ``baseline`` only in its ``declared_levers``.

    The matrix invariant: an arm may not silently vary an undeclared lever.
    """
    allowed = set(arm.declared_levers)
    base = baseline.levers.as_dict()
    cur = arm.levers.as_dict()
    for key in base:
        if base[key] != cur[key] and key not in allowed:
            return False
    return True


def needs_replication(result: ArmResult) -> bool:
    """Positive completed arms must be repeated across the declared seeds; blocked,
    not-supported, expired, and non-improving arms do not consume replication."""
    if result.status != "completed" or result.metrics is None:
        return False
    pre = result.metrics.get("pre_loss")
    post = result.metrics.get("post_loss")
    if pre is None or post is None:
        return False
    return post < pre - 1e-9  # improved held-out loss -> must replicate


# --- Execution (guarded; expires without an executable policy + corpus) ---


def run_arm(
    arm: CampaignArm,
    *,
    corpus: CorpusSupport,
    seed: int,
    policy_factory: Callable[[ArmLevers], Any] | None = None,
    train_items: Sequence[Any] = (),
    held_out: Sequence[Any] = (),
    strata: Sequence[Any] = (),
    allow_experimental_methods: bool = False,
) -> ArmResult:
    """Classify then optionally execute one arm.

    Execution happens only when ``policy_factory`` and ``train_items`` are both
    supplied (i.e. a real adapter-enabled policy and an admitted corpus exist).
    In every other case an otherwise-admissible arm resolves to ``expired`` —
    the honest CPU/no-corpus outcome — and never fabricates metrics.
    """
    verdict = classify_arm(
        arm, corpus=corpus, allow_experimental_methods=allow_experimental_methods
    )
    if verdict.status != "admitted":
        return verdict

    if policy_factory is None or not train_items:
        return ArmResult(
            arm.arm_id,
            "expired",
            "no executable policy + admitted corpus in this environment "
            "(ship-grade run requires GPU + pinned checkpoint + admitted corpus)",
        )

    # Lazy import: keep the planning/gating layer Torch-free.
    from slm_training.harnesses.preference.causal_trainer import train_causal_local

    if arm.levers.objective is None:
        # Parent baseline: evaluate only, no update.
        from slm_training.harnesses.preference.causal_trainer import evaluate_items

        policy = policy_factory(arm.levers)
        metrics = evaluate_items(policy, held_out or train_items, objective="unlikelihood")
        return ArmResult(
            arm.arm_id,
            "completed",
            "parent baseline evaluated",
            trainable_parameters=0,
            seeds_run=(seed,),
            metrics={"pre_loss": metrics["loss"], "post_loss": metrics["loss"]},
        )

    policy = policy_factory(arm.levers)
    summary = train_causal_local(
        train_items,
        policy,
        objective=arm.levers.objective,
        strata=strata,
        seed=seed,
        held_out=held_out,
        non_target_tether=1.0 if arm.levers.reference_tether else 0.0,
    )
    return ArmResult(
        arm.arm_id,
        "completed",
        summary.get("claim", "wiring only; no quality claim"),
        trainable_parameters=int(summary["trainable_parameters"]),
        seeds_run=(seed,),
        metrics={"pre_loss": summary["pre"]["loss"], "post_loss": summary["post"]["loss"]},
    )


def describe_campaign(arms: Sequence[CampaignArm]) -> dict[str, Any]:
    """A deterministic, JSON-safe dry-run description of the arm matrix."""
    by_stage: dict[int, int] = {}
    for arm in arms:
        by_stage[arm.stage] = by_stage.get(arm.stage, 0) + 1
    return {
        "arm_count": len(arms),
        "arms_by_stage": {str(k): by_stage[k] for k in sorted(by_stage)},
        "arms": [arm.as_dict() for arm in arms],
        "claim": "wiring only; no quality claim; ship-grade run requires GPU + checkpoint",
    }
