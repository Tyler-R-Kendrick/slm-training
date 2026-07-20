"""SLM-180 SDE4-02 minimum controller capacity fixture harness.

A wiring-only capacity ladder that trains tiny CPU MLPs on deterministic synthetic
decisions.  No GPU training, no ship-gate claim, no coupling to production
TwoTower wiring.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slm_training.versioning import build_version_stamp

__all__ = [
    "COMPETENCE_TARGET",
    "ControllerCapacityArm",
    "ControllerCapacityManifest",
    "ControllerCapacityReport",
    "ControllerCapacityRow",
    "ControllerCapacityRung",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "build_manifest",
    "fixture_decisions",
    "render_markdown",
    "run_fixture_ladder",
]

MATRIX_VERSION = "sde4-02-v1"
MATRIX_SET = "sde4-02-min-controller"
COMPETENCE_TARGET = 0.66

DEFAULT_SEMANTIC_DIM = 8
DEFAULT_ACTION_COUNT = 5
DEFAULT_STATE_COUNT = 8
DEFAULT_HIDDEN_DIMS = (8, 16, 32, 64, 128)


@dataclass(frozen=True)
class ControllerCapacityArm:
    """One controller-capacity arm with a monotonic width ladder."""

    arm_id: str
    hidden_dims: tuple[int, ...]
    seeds: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "hidden_dims": list(self.hidden_dims),
            "seeds": list(self.seeds),
        }


@dataclass(frozen=True)
class ControllerCapacityRung:
    """A single rung on the capacity ladder."""

    rung_id: str
    rank: int
    hidden_dim: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rung_id": self.rung_id,
            "rank": self.rank,
            "hidden_dim": self.hidden_dim,
        }


@dataclass(frozen=True)
class ControllerCapacityManifest:
    """Preregistered manifest for the SDE4-02 minimum-controller-capacity ladder."""

    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    hypothesis: str = (
        "A minimum-width CPU controller can be identified for a deterministic "
        "synthetic decision task: as MLP hidden dimension increases, train "
        "accuracy crosses a fixed competence threshold and the smallest "
        "qualifying rung marks the minimum viable controller capacity."
    )
    falsifier: str = (
        "No rung in the ladder reaches the competence target on the training "
        "split, or a smaller-width rung consistently outperforms every larger "
        "rung, indicating that the ladder does not monotonically characterize "
        "controller capacity for this recipe."
    )
    competence_target: float = COMPETENCE_TARGET
    semantic_dim: int = DEFAULT_SEMANTIC_DIM
    action_count: int = DEFAULT_ACTION_COUNT
    state_count: int = DEFAULT_STATE_COUNT
    rungs: tuple[ControllerCapacityRung, ...] = ()
    seeds: tuple[int, ...] = (0, 1, 2)
    base_recipe: dict[str, Any] = field(
        default_factory=lambda: {
            "device": "cpu",
            "optimizer": "Adam",
            "learning_rate": 0.01,
            "train_steps": 200,
            "batch_mode": "cyclic",
            "loss": "cross_entropy",
            "eval_split": "holdout_25_percent_states",
            "max_wall_minutes": 3.0,
        }
    )
    claim_class: str = "wiring"
    status: str = "not_run"

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["base_recipe_hash"] = self.recipe_hash()
        data["rungs"] = [r.to_dict() for r in self.rungs]
        return data

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def recipe_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.base_recipe, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class ControllerCapacityRow:
    """Measured result for one (rung, seed) fixture train."""

    rung_id: str
    hidden_dim: int
    seed: int
    train_accuracy: float
    val_accuracy: float
    trainable_parameters: int
    active_parameters: int
    meets_competence_target: bool
    status: str = "fixture_trained"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class ControllerCapacityReport:
    """Full fixture ladder report."""

    matrix_set: str
    matrix_version: str
    run_id: str
    status: str
    manifest: ControllerCapacityManifest
    rows: list[ControllerCapacityRow]
    selected_rung_id: str | None
    capacity_threshold_not_identifiable: bool
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "run_id": self.run_id,
            "status": self.status,
            "manifest": self.manifest.to_dict(),
            "rows": [r.to_dict() for r in self.rows],
            "selected_rung_id": self.selected_rung_id,
            "capacity_threshold_not_identifiable": self.capacity_threshold_not_identifiable,
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class FixtureDecision:
    """One synthetic controller decision point."""

    decision_id: str
    semantic_input: tuple[float, ...]
    history: tuple[float, ...]
    state_id: int
    state_family_id: str
    legal_actions: tuple[str, ...]
    correct_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "semantic_input": list(self.semantic_input),
            "history": list(self.history),
            "state_id": self.state_id,
            "state_family_id": self.state_family_id,
            "legal_actions": list(self.legal_actions),
            "correct_action": self.correct_action,
        }


def _make_action_vocabulary(action_count: int) -> tuple[str, ...]:
    return tuple(f"action:{i:02d}" for i in range(action_count))


def fixture_decisions(
    state_count: int = DEFAULT_STATE_COUNT,
    action_count: int = DEFAULT_ACTION_COUNT,
    semantic_dim: int = DEFAULT_SEMANTIC_DIM,
    seed: int = 0,
) -> tuple[FixtureDecision, ...]:
    """Deterministic fixture decision points with known legal action sets.

    Mirrors the CAP2-04 fixture shape without importing it. Each state gets a
    deterministic one-hot semantic input and a random history vector. Legal
    actions are a fixed-size subset of the vocabulary and the correct action is
    deterministic per state.
    """
    rng = random.Random(seed)
    actions = _make_action_vocabulary(action_count)
    decisions: list[FixtureDecision] = []
    for state_id in range(state_count):
        semantic = [0.0] * semantic_dim
        semantic[state_id % semantic_dim] = 1.0
        history = [rng.random() for _ in range(semantic_dim)]
        start = state_id % action_count
        legal = tuple(actions[(start + i) % action_count] for i in range(action_count))
        correct = legal[state_id % action_count]
        family = f"family_{state_id % 3}"
        decisions.append(
            FixtureDecision(
                decision_id=f"d{state_id:03d}",
                semantic_input=tuple(semantic),
                history=tuple(history),
                state_id=state_id,
                state_family_id=family,
                legal_actions=legal,
                correct_action=correct,
            )
        )
    return tuple(decisions)


def _split_val_states(state_count: int, seed: int) -> set[int]:
    rng = random.Random(seed)
    indices = list(range(state_count))
    rng.shuffle(indices)
    holdout = max(1, state_count // 4)
    return set(indices[:holdout])


def build_manifest(
    *,
    rungs: int = 5,
    seeds: tuple[int, ...] = (0, 1, 2),
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
    semantic_dim: int = DEFAULT_SEMANTIC_DIM,
    action_count: int = DEFAULT_ACTION_COUNT,
    state_count: int = DEFAULT_STATE_COUNT,
) -> ControllerCapacityManifest:
    """Return the SDE4-02 minimum-controller-capacity manifest."""
    if rungs < 1:
        raise ValueError("rungs must be >= 1")
    dims = tuple(hidden_dims[:rungs])
    if len(dims) < rungs:
        raise ValueError(f"need at least {rungs} hidden_dims, got {len(hidden_dims)}")
    rung_objects = tuple(
        ControllerCapacityRung(
            rung_id=f"rung_{rank:03d}_h{dim}",
            rank=rank,
            hidden_dim=dim,
        )
        for rank, dim in enumerate(dims, start=1)
    )
    return ControllerCapacityManifest(
        rungs=rung_objects,
        seeds=seeds,
        semantic_dim=semantic_dim,
        action_count=action_count,
        state_count=state_count,
        status="not_run",
        claim_class="wiring",
    )


class FixtureController:
    """Tiny CPU MLP: semantic -> hidden -> action logits.

    Instantiated inside fixture functions so the module stays importable without
    torch at the top level.
    """

    def __init__(self, semantic_dim: int, hidden_dim: int, action_count: int) -> None:
        from torch import nn

        super().__init__()
        self.semantic_dim = semantic_dim
        self.hidden_dim = hidden_dim
        self.action_count = action_count
        self.net = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_count),
        )

    def forward(self, x: Any) -> Any:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def count_active_parameters(self) -> int:
        return self.count_parameters()


def _train_fixture_controller(
    model: FixtureController,
    train_decisions: tuple[FixtureDecision, ...],
    *,
    train_steps: int = 200,
    seed: int = 0,
) -> None:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=1e-2)
    for step in range(train_steps):
        decision = train_decisions[step % len(train_decisions)]
        x = torch.tensor(
            [list(decision.semantic_input)], dtype=torch.float32
        )
        target_pos = decision.legal_actions.index(decision.correct_action)
        target = torch.tensor([target_pos], dtype=torch.long)
        logits = model.forward(x)
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _evaluate_controller(
    model: FixtureController,
    decisions: tuple[FixtureDecision, ...],
) -> float:
    import torch

    correct = 0
    with torch.no_grad():
        for decision in decisions:
            x = torch.tensor(
                [list(decision.semantic_input)], dtype=torch.float32
            )
            logits = model.forward(x)
            pred_pos = int(logits.argmax(dim=-1).item())
            pred_action = decision.legal_actions[pred_pos]
            if pred_action == decision.correct_action:
                correct += 1
    return correct / len(decisions)


def _run_rung(
    rung: ControllerCapacityRung,
    manifest: ControllerCapacityManifest,
    seed: int,
    train_steps: int,
) -> ControllerCapacityRow:
    decisions = fixture_decisions(
        state_count=manifest.state_count,
        action_count=manifest.action_count,
        semantic_dim=manifest.semantic_dim,
        seed=seed,
    )
    val_state_ids = _split_val_states(manifest.state_count, seed=seed + 7)
    train_decisions = tuple(d for d in decisions if d.state_id not in val_state_ids)
    val_decisions = tuple(d for d in decisions if d.state_id in val_state_ids)

    model = FixtureController(
        semantic_dim=manifest.semantic_dim,
        hidden_dim=rung.hidden_dim,
        action_count=manifest.action_count,
    )
    _train_fixture_controller(model, train_decisions, train_steps=train_steps, seed=seed)

    train_acc = _evaluate_controller(model, train_decisions)
    val_acc = _evaluate_controller(model, val_decisions)
    meets = train_acc >= manifest.competence_target

    notes = [
        f"train_steps={train_steps}",
        f"train_states={len(train_decisions)} val_states={len(val_decisions)}",
        "fixture-only: tiny CPU MLP on synthetic decisions",
    ]
    return ControllerCapacityRow(
        rung_id=rung.rung_id,
        hidden_dim=rung.hidden_dim,
        seed=seed,
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        trainable_parameters=model.count_parameters(),
        active_parameters=model.count_active_parameters(),
        meets_competence_target=meets,
        status="fixture_trained",
        notes=notes,
    )


def run_fixture_ladder(
    manifest: ControllerCapacityManifest,
    *,
    output_dir: Path | None = None,
    train_steps: int = 200,
    run_id: str = "sde4_02_fixture",
) -> ControllerCapacityReport:
    """Train one tiny controller per (rung, seed) and pick the smallest qualifying rung."""
    rows: list[ControllerCapacityRow] = []
    for rung in manifest.rungs:
        for seed in manifest.seeds:
            rows.append(_run_rung(rung, manifest, seed, train_steps))

    selected_rung_id: str | None = None
    for rung in manifest.rungs:
        rung_rows = [r for r in rows if r.rung_id == rung.rung_id]
        if all(r.meets_competence_target for r in rung_rows):
            selected_rung_id = rung.rung_id
            break

    report = ControllerCapacityReport(
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        rows=rows,
        selected_rung_id=selected_rung_id,
        capacity_threshold_not_identifiable=selected_rung_id is None,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.sde4_02_min_controller_capacity",
        ),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "sde4_02_min_controller_capacity_report.json")
    return report


def render_markdown(report: ControllerCapacityReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-180 / SDE4-02: Minimum controller capacity fixture ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`  ",
        f"Version: `{report.matrix_version}`  ",
        f"Status: **{report.status}**  ",
        "",
        f"Competence target (train accuracy): `{report.manifest.competence_target}`",
        "",
        "## Hypothesis",
        "",
        report.manifest.hypothesis,
        "",
        "## Falsifier",
        "",
        report.manifest.falsifier,
        "",
        "## Frozen base recipe (SHA-256)",
        "",
        f"```\n{report.manifest.recipe_hash()}\n```",
        "",
        "## Ladder",
        "",
        "| Rung | Hidden dim |",
        "| --- | --- |",
    ]
    for rung in report.manifest.rungs:
        marker = ""
        if report.selected_rung_id == rung.rung_id:
            marker = " ★ selected"
        lines.append(f"| {rung.rung_id} | {rung.hidden_dim}{marker} |")

    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Rung | Hidden dim | Seed | Train acc | Val acc | Params | Active params | Meets target |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        lines.append(
            f"| {row.rung_id} | {row.hidden_dim} | {row.seed} | "
            f"{row.train_accuracy:.4f} | {row.val_accuracy:.4f} | "
            f"{row.trainable_parameters} | {row.active_parameters} | "
            f"{row.meets_competence_target} |"
        )

    verdict = (
        "No rung met the competence target on all seeds; "
        "capacity_threshold_not_identifiable = True."
        if report.capacity_threshold_not_identifiable
        else f"Smallest qualifying rung: `{report.selected_rung_id}`."
    )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            verdict,
            "",
            "**Fixture caveat:** This is wiring-only evidence. The controllers are "
            "tiny CPU MLPs trained on a deterministic synthetic decision set with "
            "no production model, no GPU, no held-out eval suites, and no "
            "ship-gate claim. The competence threshold is a fixture probe, not a "
            "production readiness criterion.",
            "",
        ]
    )
    return "\n".join(lines)
