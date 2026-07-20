"""SLM-156 (SPV3-03): shared recursive refinement for SemanticPlanV1.

Fixture/wiring-only harness. Defines a small shared refinement cell over a
fixed-shape plan state, trains it to recover synthetic one-factor corruptions,
and compares one-pass, deeper non-shared, fixed-depth shared, adaptive, and
diagnostic arms under a common manifest. No production TwoTower wiring is
touched; no ship-gate claim is made.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "PLAN_REFINEMENT_CAMPAIGN_ID",
    "RefinementArm",
    "RefinementArmKind",
    "CommonConfig",
    "PlanRefinementState",
    "RefinementTrace",
    "RefinementRecord",
    "RefinementRow",
    "RefinementManifest",
    "RefinementReport",
    "PlanRefinementCell",
    "PlanRefinementModel",
    "build_manifest",
    "validate_manifest",
    "run_fixture_campaign",
    "render_markdown",
]

MATRIX_VERSION = "spv3-03-v1"
MATRIX_SET = "slm156_plan_refinement"
PLAN_REFINEMENT_CAMPAIGN_ID = "slm156-plan-refinement"


class RefinementArmKind(str, Enum):
    """Arm category for the recursion comparison."""

    ONE_PASS = "one_pass"
    DEEPER = "deeper"
    SHARED_FIXED = "shared_fixed"
    SHARED_ADAPTIVE = "shared_adaptive"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class CommonConfig:
    """Frozen orthogonal controls shared by every arm."""

    num_roles: int = 8
    num_archetypes: int = 4
    max_depth: int = 4
    n_train: int = 64
    n_eval: int = 16
    seeds: tuple[int, ...] = (0, 1, 2)
    lr: float = 1e-2
    epochs: int = 10
    adaptive_halt_threshold: float = 0.7
    metric_versions: dict[str, str] = field(default_factory=lambda: {"meaningful": "2.0.0"})

    @property
    def state_dim(self) -> int:
        """Plan-state vector dimension: archetype + roles + confidence."""
        return self.num_archetypes + self.num_roles + 1

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["seeds"] = list(self.seeds)
        data["state_dim"] = self.state_dim
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommonConfig":
        return cls(
            num_roles=data.get("num_roles", 8),
            num_archetypes=data.get("num_archetypes", 4),
            max_depth=data.get("max_depth", 4),
            n_train=data.get("n_train", 256),
            n_eval=data.get("n_eval", 64),
            seeds=tuple(data.get("seeds", [0, 1, 2])),
            lr=data.get("lr", 1e-2),
            epochs=data.get("epochs", 30),
            adaptive_halt_threshold=data.get("adaptive_halt_threshold", 0.7),
            metric_versions=data.get("metric_versions", {"meaningful": "2.0.0"}),
        )


@dataclass(frozen=True)
class RefinementArm:
    """One arm in the plan-refinement comparison."""

    arm_id: str
    kind: RefinementArmKind
    name: str
    description: str
    depth: int = 1
    adaptive: bool = False
    stochastic: bool = False
    uses_diagnostics: bool = False
    uses_gold: bool = False
    promotable: bool = True
    diagnostic: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["kind"] = self.kind.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RefinementArm":
        return cls(
            arm_id=data["arm_id"],
            kind=RefinementArmKind(data.get("kind", "one_pass")),
            name=data.get("name", ""),
            description=data.get("description", ""),
            depth=data.get("depth", 1),
            adaptive=data.get("adaptive", False),
            stochastic=data.get("stochastic", False),
            uses_diagnostics=data.get("uses_diagnostics", False),
            uses_gold=data.get("uses_gold", False),
            promotable=data.get("promotable", True),
            diagnostic=data.get("diagnostic", False),
        )


@dataclass(frozen=True)
class PlanRefinementState:
    """Simplified fixed-shape plan state used by the fixture cell."""

    archetype_logit: torch.Tensor
    role_logits: torch.Tensor
    confidence: torch.Tensor
    step: int = 0
    halted: bool = False

    def vector(self) -> torch.Tensor:
        return torch.cat(
            [
                self.archetype_logit.flatten(),
                self.role_logits.flatten(),
                self.confidence.flatten(),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "archetype_logit": self.archetype_logit.tolist(),
            "role_logits": self.role_logits.tolist(),
            "confidence": self.confidence.tolist(),
            "step": self.step,
            "halted": self.halted,
        }


@dataclass(frozen=True)
class RefinementTrace:
    """Per-example trace envelope for one refinement arm."""

    trace_id: str
    arm_id: str
    initial_fingerprint: str
    depth_reached: int
    halted: bool
    per_step_values: tuple[float, ...]
    final_state: dict[str, Any]
    cost_counters: dict[str, int]
    metrics: dict[str, float]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["per_step_values"] = list(self.per_step_values)
        return data


@dataclass(frozen=True)
class RefinementRecord:
    """Per-example result for one arm/seed."""

    record_id: str
    arm_id: str
    seed: int
    accepted: bool
    plan_score: float
    depth: int
    forwards: int
    trace: RefinementTrace

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["trace"] = self.trace.to_dict()
        return data


@dataclass(frozen=True)
class RefinementRow:
    """Aggregated row for one arm/seed."""

    arm_id: str
    kind: RefinementArmKind
    seed: int
    promotable: bool
    n_records: int
    mean_plan_score: float
    mean_depth: float
    mean_forwards: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["kind"] = self.kind.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RefinementRow":
        return cls(
            arm_id=data["arm_id"],
            kind=RefinementArmKind(data.get("kind", "one_pass")),
            seed=data["seed"],
            promotable=data.get("promotable", True),
            n_records=data["n_records"],
            mean_plan_score=data["mean_plan_score"],
            mean_depth=data["mean_depth"],
            mean_forwards=data["mean_forwards"],
            notes=list(data.get("notes", [])),
        )


@dataclass(frozen=True)
class RefinementManifest:
    """Preregistered manifest for the SLM-156 campaign."""

    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = PLAN_REFINEMENT_CAMPAIGN_ID
    hypothesis: str = (
        "A small shared refinement cell applied recursively to SemanticPlanV1 "
        "improves plan-factor recovery over a parameter-matched one-pass predictor "
        "on coupled corruptions, and adaptive halting preserves quality at lower "
        "average depth."
    )
    falsifier: str = (
        "Recursion changes plans but not final correctness, deeper non-shared "
        "matches it at equal FLOPs, adaptive halting collapses to min/max depth, "
        "or diagnostics leak gold information."
    )
    common_config: CommonConfig = field(default_factory=CommonConfig)
    arms: tuple[RefinementArm, ...] = ()
    claim_class: str = "wiring"
    status: str = "not_run"

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["common_config"] = self.common_config.to_dict()
        data["arms"] = [arm.to_dict() for arm in self.arms]
        return data

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class RefinementReport:
    """Full fixture report for SLM-156."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    manifest: RefinementManifest
    rows: list[RefinementRow]
    version_stamp: dict[str, Any] = field(default_factory=dict)
    claim_class: str = "wiring"

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "manifest": self.manifest.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


class PlanRefinementCell(nn.Module):
    """Shared residual refinement cell over a flat plan-state vector."""

    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.halt_head = nn.Linear(state_dim, 1)
        self.value_head = nn.Linear(state_dim, 1)

    def forward(
        self, state_vec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = self.net(state_vec)
        halt_logit = self.halt_head(state_vec)
        value = self.value_head(state_vec)
        return state_vec + delta, halt_logit.squeeze(-1), value.squeeze(-1)


class PlanRefinementModel(nn.Module):
    """Apply the shared cell recursively; support fixed, adaptive, stochastic."""

    def __init__(
        self,
        state_dim: int,
        cell: PlanRefinementCell | None = None,
        max_depth: int = 4,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.max_depth = max_depth
        self.cell = cell if cell is not None else PlanRefinementCell(state_dim)

    def refine(
        self,
        state: PlanRefinementState,
        *,
        adaptive: bool = False,
        stochastic: bool = False,
        halt_threshold: float = 0.7,
        rng: random.Random | None = None,
    ) -> tuple[PlanRefinementState, list[float], list[float]]:
        """Run refinement and return final state, per-step values, halt probs."""
        current = state
        values: list[float] = []
        halt_probs: list[float] = []
        for step in range(self.max_depth):
            vec = current.vector().unsqueeze(0)
            new_vec, halt_logit, value = self.cell(vec)
            prob = float(torch.sigmoid(halt_logit).item())
            values.append(float(value.item()))
            halt_probs.append(prob)
            new_state = PlanRefinementState(
                archetype_logit=new_vec[0, : current.archetype_logit.numel()],
                role_logits=new_vec[
                    0,
                    current.archetype_logit.numel() : current.archetype_logit.numel()
                    + current.role_logits.numel(),
                ],
                confidence=new_vec[0, -1:],
                step=step + 1,
                halted=False,
            )
            if adaptive and prob >= halt_threshold:
                new_state = PlanRefinementState(
                    archetype_logit=new_state.archetype_logit,
                    role_logits=new_state.role_logits,
                    confidence=new_state.confidence,
                    step=new_state.step,
                    halted=True,
                )
                current = new_state
                break
            if stochastic and rng is not None and rng.random() < prob:
                current = new_state
                break
            current = new_state
        return current, values, halt_probs


class OnePassPredictor(nn.Module):
    """Parameter-matched one-pass baseline (same capacity as one cell step)."""

    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        return state_vec + self.net(state_vec)


class DeeperPredictor(nn.Module):
    """Deeper non-shared predictor with roughly matched parameter count."""

    def __init__(self, state_dim: int, hidden_dim: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers.append(nn.Linear(state_dim if not layers else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        return state_vec + self.net(state_vec)


def _state_fingerprint(state: PlanRefinementState) -> str:
    from slm_training.lineage.records import content_sha

    return content_sha(state.to_dict())


def _make_state(
    cfg: CommonConfig,
    archetype: int,
    roles: set[int],
    confidence: float,
    device: str = "cpu",
) -> PlanRefinementState:
    arch = torch.zeros(cfg.num_archetypes, device=device)
    arch[archetype] = 1.0
    role = torch.zeros(cfg.num_roles, device=device)
    for r in roles:
        role[r] = 1.0
    return PlanRefinementState(
        archetype_logit=arch,
        role_logits=role,
        confidence=torch.tensor([confidence], device=device),
    )


def _corrupt_state(
    state: PlanRefinementState,
    rng: random.Random,
) -> PlanRefinementState:
    """Apply one-factor corruption: archetype, role, or confidence."""
    arch = state.archetype_logit.clone()
    role = state.role_logits.clone()
    conf = state.confidence.clone()
    kind = rng.choice(("archetype", "role", "confidence"))
    if kind == "archetype":
        idx = rng.randrange(len(arch))
        arch = torch.zeros_like(arch)
        arch[idx] = 1.0
    elif kind == "role":
        idx = rng.randrange(len(role))
        role[idx] = 1.0 - role[idx]
    else:
        conf = torch.clamp(conf + rng.uniform(-0.3, 0.3), 0.0, 1.0)
    return PlanRefinementState(
        archetype_logit=arch,
        role_logits=role,
        confidence=conf,
    )


def _state_score(state: PlanRefinementState, gold: PlanRefinementState) -> float:
    with torch.no_grad():
        arch_match = int(state.archetype_logit.argmax() == gold.archetype_logit.argmax())
        role_iou = float(
            (
                (state.role_logits > 0.5) & (gold.role_logits > 0.5)
            ).float().sum()
            / (
                (state.role_logits > 0.5) | (gold.role_logits > 0.5)
            ).float().clamp_min(1.0).sum()
        )
        conf_err = float((state.confidence - gold.confidence).abs().item())
    return (arch_match + role_iou + (1.0 - conf_err)) / 3.0


def _train_models(
    cfg: CommonConfig,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[PlanRefinementModel, OnePassPredictor, DeeperPredictor]:
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # Build synthetic gold/corrupted pairs.
    train_pairs: list[tuple[PlanRefinementState, PlanRefinementState]] = []
    for _ in range(cfg.n_train):
        arch = rng.randrange(cfg.num_archetypes)
        roles = {rng.randrange(cfg.num_roles) for _ in range(rng.randint(2, 4))}
        gold = _make_state(cfg, arch, roles, rng.random(), device)
        corrupted = _corrupt_state(gold, rng)
        train_pairs.append((corrupted, gold))

    shared = PlanRefinementModel(cfg.state_dim, max_depth=cfg.max_depth).to(device)
    one_pass = OnePassPredictor(cfg.state_dim).to(device)
    deeper = DeeperPredictor(cfg.state_dim, n_layers=2).to(device)

    opt_shared = torch.optim.Adam(shared.parameters(), lr=cfg.lr)
    opt_one = torch.optim.Adam(one_pass.parameters(), lr=cfg.lr)
    opt_deep = torch.optim.Adam(deeper.parameters(), lr=cfg.lr)

    for _ in range(cfg.epochs):
        rng.shuffle(train_pairs)
        for corrupted, gold in train_pairs:
            for opt, model, depth in (
                (opt_one, one_pass, 1),
                (opt_deep, deeper, 1),
                (opt_shared, shared, cfg.max_depth),
            ):
                opt.zero_grad()
                if isinstance(model, PlanRefinementModel):
                    final, _, _ = model.refine(corrupted)
                    pred = final.vector()
                else:
                    pred = model(corrupted.vector().unsqueeze(0))[0]
                loss = F.mse_loss(pred, gold.vector())
                loss.backward()
                opt.step()

    return shared, one_pass, deeper


def _run_arm_records(
    arm: RefinementArm,
    models: tuple[PlanRefinementModel, OnePassPredictor, DeeperPredictor],
    eval_pairs: list[tuple[PlanRefinementState, PlanRefinementState]],
    cfg: CommonConfig,
    seed: int,
) -> list[RefinementRecord]:
    shared, one_pass, deeper = models
    rng = random.Random(seed)
    records: list[RefinementRecord] = []

    for idx, (corrupted, gold) in enumerate(eval_pairs):
        if arm.uses_gold:
            final = gold
            values: list[float] = [1.0]
            halt_probs: list[float] = []
            forwards = 0
        elif arm.kind is RefinementArmKind.ONE_PASS:
            with torch.no_grad():
                pred = one_pass(corrupted.vector().unsqueeze(0))[0]
            final = PlanRefinementState(
                archetype_logit=pred[: cfg.num_archetypes],
                role_logits=pred[cfg.num_archetypes : cfg.num_archetypes + cfg.num_roles],
                confidence=pred[-1:],
            )
            values = [0.5]
            halt_probs = []
            forwards = 1
        elif arm.kind is RefinementArmKind.DEEPER:
            with torch.no_grad():
                pred = deeper(corrupted.vector().unsqueeze(0))[0]
            final = PlanRefinementState(
                archetype_logit=pred[: cfg.num_archetypes],
                role_logits=pred[cfg.num_archetypes : cfg.num_archetypes + cfg.num_roles],
                confidence=pred[-1:],
            )
            values = [0.5]
            halt_probs = []
            forwards = 2
        elif arm.kind is RefinementArmKind.SHARED_FIXED:
            with torch.no_grad():
                depth_model = PlanRefinementModel(
                    cfg.state_dim, cell=shared.cell, max_depth=arm.depth
                )
                final, values, halt_probs = depth_model.refine(corrupted, adaptive=False)
            forwards = arm.depth
        elif arm.kind is RefinementArmKind.SHARED_ADAPTIVE:
            with torch.no_grad():
                final, values, halt_probs = shared.refine(
                    corrupted,
                    adaptive=True,
                    halt_threshold=cfg.adaptive_halt_threshold,
                )
            forwards = final.step
        else:  # stochastic diagnostic
            with torch.no_grad():
                final, values, halt_probs = shared.refine(
                    corrupted, stochastic=True, rng=rng
                )
            forwards = final.step

        score = _state_score(final, gold)
        accepted = score >= 0.8
        trace = RefinementTrace(
            trace_id=f"{arm.arm_id}-s{seed}-{idx}",
            arm_id=arm.arm_id,
            initial_fingerprint=_state_fingerprint(corrupted),
            depth_reached=final.step,
            halted=final.halted,
            per_step_values=tuple(values),
            final_state=final.to_dict(),
            cost_counters={"forwards": forwards, "cell_calls": forwards},
            metrics={"plan_score": score, "accepted": float(accepted)},
            notes=[
                f"kind={arm.kind.value}",
                f"depth={arm.depth}",
                "fixture-only: synthetic plan-state recovery",
            ],
        )
        records.append(
            RefinementRecord(
                record_id=f"rec{idx}",
                arm_id=arm.arm_id,
                seed=seed,
                accepted=accepted,
                plan_score=score,
                depth=final.step,
                forwards=forwards,
                trace=trace,
            )
        )
    return records


def _aggregate_records(
    arm: RefinementArm,
    seed: int,
    records: list[RefinementRecord],
) -> RefinementRow:
    n = len(records)
    if not n:
        return RefinementRow(
            arm_id=arm.arm_id,
            kind=arm.kind,
            seed=seed,
            promotable=arm.promotable and not arm.diagnostic,
            n_records=0,
            mean_plan_score=0.0,
            mean_depth=0.0,
            mean_forwards=0.0,
            notes=["empty"],
        )
    notes = [
        f"kind={arm.kind.value}",
        f"depth={arm.depth}",
        "fixture-only: synthetic plan-state recovery",
    ]
    if arm.diagnostic:
        notes.append("diagnostic arm")
    if not arm.promotable:
        notes.append("non-promotable")
    return RefinementRow(
        arm_id=arm.arm_id,
        kind=arm.kind,
        seed=seed,
        promotable=arm.promotable and not arm.diagnostic,
        n_records=n,
        mean_plan_score=sum(r.plan_score for r in records) / n,
        mean_depth=sum(r.depth for r in records) / n,
        mean_forwards=sum(r.forwards for r in records) / n,
        notes=notes,
    )


def build_manifest() -> RefinementManifest:
    """Return the default SLM-156 fixture manifest."""
    arms = (
        RefinementArm(
            arm_id="A_one_pass",
            kind=RefinementArmKind.ONE_PASS,
            name="one_pass_predictor",
            description="Single forward through a parameter-matched one-pass predictor.",
            depth=1,
        ),
        RefinementArm(
            arm_id="B_deeper_non_shared",
            kind=RefinementArmKind.DEEPER,
            name="deeper_non_shared",
            description="Deeper non-shared predictor with matched parameter budget.",
            depth=1,
        ),
        RefinementArm(
            arm_id="C_shared_fixed_2",
            kind=RefinementArmKind.SHARED_FIXED,
            name="shared_fixed_2",
            description="Shared cell applied for exactly 2 recursions.",
            depth=2,
        ),
        RefinementArm(
            arm_id="D_shared_fixed_4",
            kind=RefinementArmKind.SHARED_FIXED,
            name="shared_fixed_4",
            description="Shared cell applied for exactly 4 recursions.",
            depth=4,
        ),
        RefinementArm(
            arm_id="E_shared_adaptive",
            kind=RefinementArmKind.SHARED_ADAPTIVE,
            name="shared_adaptive",
            description="Shared cell with calibrated halt/value head.",
            depth=4,
            adaptive=True,
        ),
        RefinementArm(
            arm_id="F_shared_diagnostics",
            kind=RefinementArmKind.SHARED_ADAPTIVE,
            name="shared_with_diagnostics",
            description="Adaptive shared cell plus inference-available diagnostics placeholder.",
            depth=4,
            adaptive=True,
            uses_diagnostics=True,
        ),
        RefinementArm(
            arm_id="G_stochastic_value",
            kind=RefinementArmKind.SHARED_FIXED,
            name="stochastic_trajectories",
            description="Stochastic trajectory sampling with value selection (fixture only).",
            depth=4,
            stochastic=True,
        ),
        RefinementArm(
            arm_id="H_gold_oracle",
            kind=RefinementArmKind.DIAGNOSTIC,
            name="gold_plan_oracle",
            description="Gold plan oracle ceiling.",
            depth=0,
            uses_gold=True,
            promotable=False,
            diagnostic=True,
        ),
    )
    return RefinementManifest(arms=arms)


def validate_manifest(manifest: RefinementManifest) -> list[str]:
    """Validate manifest shape and honest constraints."""
    errors: list[str] = []
    if not manifest.arms:
        errors.append("arms must not be empty")
    seen: set[str] = set()
    for arm in manifest.arms:
        if arm.arm_id in seen:
            errors.append(f"duplicate arm_id: {arm.arm_id}")
        seen.add(arm.arm_id)
        if arm.uses_gold and arm.promotable:
            errors.append(f"{arm.arm_id}: gold-oracle arm must be non-promotable")
        if arm.diagnostic and arm.promotable:
            errors.append(f"{arm.arm_id}: diagnostic arm must be non-promotable")
        if arm.adaptive and arm.kind is not RefinementArmKind.SHARED_ADAPTIVE:
            errors.append(f"{arm.arm_id}: adaptive flag requires shared_adaptive kind")
        if arm.stochastic and arm.kind is RefinementArmKind.DIAGNOSTIC:
            errors.append(f"{arm.arm_id}: stochastic flag invalid for diagnostic arm")
    cfg = manifest.common_config
    if cfg.state_dim <= 0:
        errors.append("common_config.state_dim must be positive")
    if cfg.max_depth <= 0:
        errors.append("common_config.max_depth must be positive")
    return errors


def run_fixture_campaign(
    manifest: RefinementManifest | None = None,
    *,
    run_id: str = "slm156_fixture",
    output_dir: Path | None = None,
) -> RefinementReport:
    """Run the SLM-156 plan-refinement fixture campaign."""
    manifest = manifest or build_manifest()
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))

    cfg = manifest.common_config
    rng = random.Random(0)
    eval_pairs: list[tuple[PlanRefinementState, PlanRefinementState]] = []
    for _ in range(cfg.n_eval):
        arch = rng.randrange(cfg.num_archetypes)
        roles = {rng.randrange(cfg.num_roles) for _ in range(rng.randint(2, 4))}
        gold = _make_state(cfg, arch, roles, rng.random())
        corrupted = _corrupt_state(gold, rng)
        eval_pairs.append((corrupted, gold))

    rows: list[RefinementRow] = []
    for seed in cfg.seeds:
        models = _train_models(cfg, seed=seed)
        for arm in manifest.arms:
            records = _run_arm_records(arm, models, eval_pairs, cfg, seed)
            rows.append(_aggregate_records(arm, seed, records))

    report = RefinementReport(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=PLAN_REFINEMENT_CAMPAIGN_ID,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        rows=rows,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm156_plan_refinement",
        ),
        claim_class="wiring",
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm156_plan_refinement_report.json")
    return report


def render_markdown(report: RefinementReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-156 (SPV3-03): Shared recursive SemanticPlanV1 refinement ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`",
        "",
        f"Version: `{report.matrix_version}`",
        "",
        f"Status: **{report.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no production "
        "TwoTower wiring was touched, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        report.manifest.hypothesis,
        "",
        "## Falsifier",
        "",
        report.manifest.falsifier,
        "",
        "## Arms",
        "",
        "| Arm | Kind | Depth | Adaptive | Diagnostic | Description |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in report.manifest.arms:
        lines.append(
            f"| {arm.arm_id} | {arm.kind.value} | {arm.depth} | {arm.adaptive} | "
            f"{arm.diagnostic} | {arm.description} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Arm | Seed | Records | Mean plan score | Mean depth | Mean forwards |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        lines.append(
            f"| {row.arm_id} | {row.seed} | {row.n_records} | "
            f"{row.mean_plan_score:.3f} | {row.mean_depth:.1f} | {row.mean_forwards:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "This is a fixture wiring run. It validates that a shared refinement cell, "
            "fixed-depth and adaptive recursion, a deeper non-shared control, and an "
            "oracle ceiling can be registered under a common manifest with deterministic "
            "cost accounting. Real claims require a trained SemanticPlanV1 predictor, "
            "held-out causal downstream evaluation, and wall-clock measurement.",
            "",
        ]
    )
    return "\n".join(lines)
