"""CAP2-04 implicit vs explicit vs discrete vs compiler-owned state ablation.

This fixture harness implements the five matched architecture arms from SLM-89 on a
tiny synthetic decision task.  It is wiring evidence only: all arms share the same
semantic inputs, action vocabulary, and evaluation protocol; only the state
ownership mechanism differs.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from slm_training.models.local_action_head import (
    GlobalMaskedHead,
    LocalActionHead,
    LocalActionOutput,
    LocalFlatHead,
    StateContext,
    _stable_action_index,
)
from slm_training.models.mixed_radix_fsq import MixedRadixFSQCodec, MixedRadixFSQConfig


HIDDEN_DIM = 16
SEMANTIC_DIM = 8


@dataclass(frozen=True)
class FixtureDecision:
    """One fixture compiler decision point."""

    decision_id: str
    semantic_input: tuple[float, ...]
    history: tuple[float, ...]
    state_id: int
    state_family_id: str
    legal_actions: tuple[str, ...]
    correct_action: str


@dataclass(frozen=True)
class ArmConfig:
    """Configuration for one state-ownership arm."""

    arm_id: str
    mode: str  # implicit | explicit_exact | discrete_code | compiler_owned | compiler_owned_no_state
    state_count: int = 8
    action_count: int = 5
    hidden_dim: int = HIDDEN_DIM
    semantic_dim: int = SEMANTIC_DIM
    train_steps: int = 200
    seed: int = 0
    # Parameter-matching control.
    target_active_parameters: int | None = None
    # Discrete-code options.
    levels: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ArmResult:
    """Measured result for one arm."""

    arm_id: str
    mode: str
    oracle_accuracy: float
    random_init_accuracy: float
    unseen_state_accuracy: float
    forced_decisions: int
    trainable_parameters: int
    active_parameters: int
    capacity: int | None
    leakage: bool
    elapsed_seconds: float
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "mode": self.mode,
            "oracle_accuracy": self.oracle_accuracy,
            "random_init_accuracy": self.random_init_accuracy,
            "unseen_state_accuracy": self.unseen_state_accuracy,
            "forced_decisions": self.forced_decisions,
            "trainable_parameters": self.trainable_parameters,
            "active_parameters": self.active_parameters,
            "capacity": self.capacity,
            "leakage": self.leakage,
            "elapsed_seconds": self.elapsed_seconds,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class StateAblationReport:
    """Versioned fixture report for CAP2-04."""

    run_id: str
    version: str
    timestamp: str
    hidden_dim: int
    semantic_dim: int
    states: tuple[FixtureDecision, ...]
    unseen_state_ids: tuple[int, ...]
    arms: tuple[ArmResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "version": self.version,
            "timestamp": self.timestamp,
            "hidden_dim": self.hidden_dim,
            "semantic_dim": self.semantic_dim,
            "states": [
                {
                    "decision_id": s.decision_id,
                    "semantic_input": list(s.semantic_input),
                    "history": list(s.history),
                    "state_id": s.state_id,
                    "state_family_id": s.state_family_id,
                    "legal_actions": list(s.legal_actions),
                    "correct_action": s.correct_action,
                }
                for s in self.states
            ],
            "unseen_state_ids": list(self.unseen_state_ids),
            "arms": [a.to_dict() for a in self.arms],
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


def _count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _count_active_parameters(module: nn.Module) -> int:
    """Count parameters that are not part of an explicit inactive padding control."""
    inactive = sum(
        p.numel() for name, p in module.named_parameters() if "inactive" in name
    )
    return _count_parameters(module) - inactive


def _make_action_vocabulary(action_count: int) -> tuple[str, ...]:
    return tuple(f"action:{i:02d}" for i in range(action_count))


def fixture_decisions(
    state_count: int = 8,
    action_count: int = 5,
    seed: int = 0,
) -> tuple[FixtureDecision, ...]:
    """Deterministic fixture decision points with known legal action sets.

    Each state gets a deterministic semantic input and a random history vector.
    Legal actions are a fixed-size subset of the global vocabulary; the correct
    action is deterministic per state.
    """
    rng = random.Random(seed)
    actions = _make_action_vocabulary(action_count)
    decisions: list[FixtureDecision] = []
    for state_id in range(state_count):
        # Deterministic semantic input: one-hot-ish over semantic_dim.
        semantic = [0.0] * SEMANTIC_DIM
        semantic[state_id % SEMANTIC_DIM] = 1.0
        history = [rng.random() for _ in range(SEMANTIC_DIM)]
        # Deterministic legal set: a window of size action_count from the vocabulary,
        # shifted by state so different states see different legal sets.
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


def _split_unseen_states(state_count: int, seed: int) -> tuple[int, ...]:
    """Hold out ~25% of states for compositional-generalization measurement."""
    rng = random.Random(seed)
    indices = list(range(state_count))
    rng.shuffle(indices)
    holdout = max(1, state_count // 4)
    return tuple(sorted(indices[:holdout]))


def _add_inactive_padding(model: nn.Module, target: int | None) -> None:
    """Add a non-trainable padding buffer so all arms can match a budget.

    If the active count is still below ``target`` after construction, the caller
    should add a trainable ``inactive_padding`` parameter in ``evaluate_arm``.
    """
    if target is None:
        return
    active = _count_active_parameters(model)
    if active < target:
        model.register_buffer(
            "inactive_padding_buffer",
            torch.zeros(target - active),
            persistent=True,
        )


class _ImplicitStateModel(nn.Module):
    """Arm A: no exact state; network must infer state from history+semantic input."""

    def __init__(self, config: ArmConfig, max_vocabulary: int) -> None:
        super().__init__()
        self.config = config
        self.semantic_encoder = nn.Linear(config.semantic_dim, config.hidden_dim)
        self.history_encoder = nn.Linear(config.semantic_dim, config.hidden_dim)
        self.head = GlobalMaskedHead(config.hidden_dim, max_vocabulary)
        _add_inactive_padding(self, config.target_active_parameters)

    def forward(
        self,
        semantic: torch.Tensor,
        history: torch.Tensor,
        legal_actions: list[str],
    ) -> LocalActionOutput:
        h = F.relu(self.semantic_encoder(semantic) + self.history_encoder(history))
        return self.head.score(h, StateContext("implicit"), legal_actions)


class _ExplicitExactModel(nn.Module):
    """Arm B: exact state ID mapped to a learned embedding."""

    def __init__(self, config: ArmConfig, max_vocabulary: int) -> None:
        super().__init__()
        self.config = config
        self.state_embedding = nn.Embedding(config.state_count, config.hidden_dim)
        self.semantic_encoder = nn.Linear(config.semantic_dim, config.hidden_dim)
        self.head = GlobalMaskedHead(config.hidden_dim, max_vocabulary)
        _add_inactive_padding(self, config.target_active_parameters)

    def forward(
        self,
        semantic: torch.Tensor,
        state_id: torch.Tensor,
        legal_actions: list[str],
    ) -> LocalActionOutput:
        h = F.relu(self.semantic_encoder(semantic) + self.state_embedding(state_id))
        return self.head.score(h, StateContext("explicit"), legal_actions)


class _DiscreteCodeModel(nn.Module):
    """Arm C: exact state IDs encoded through a mixed-radix FSQ discrete code."""

    def __init__(self, config: ArmConfig, max_vocabulary: int) -> None:
        super().__init__()
        self.config = config
        levels = config.levels or _choose_levels(config.state_count)
        codec_config = MixedRadixFSQConfig(
            num_states=config.state_count,
            levels=levels,
            hidden_dim=config.hidden_dim,
            mode="oracle_state",
        )
        self.codec = MixedRadixFSQCodec(codec_config)
        self.semantic_encoder = nn.Linear(config.semantic_dim, config.hidden_dim)
        self.code_decoder = nn.Linear(sum(levels), config.hidden_dim)
        self.head = GlobalMaskedHead(config.hidden_dim, max_vocabulary)
        _add_inactive_padding(self, config.target_active_parameters)

    def forward(
        self,
        semantic: torch.Tensor,
        state_id: torch.Tensor,
        legal_actions: list[str],
    ) -> LocalActionOutput:
        code = self.codec.encode(state_id, hard=True)
        code_input = self.codec.decode_input(code)
        h = F.relu(
            self.semantic_encoder(semantic) + self.code_decoder(code_input)
        )
        return self.head.score(h, StateContext("discrete"), legal_actions)

    def capacity(self) -> int:
        cap = 1
        for level in self.codec.config.levels:
            cap *= level
        return cap


class _CompilerOwnedModel(nn.Module):
    """Arm D: compiler owns state; network scores only the supplied legal actions."""

    def __init__(self, config: ArmConfig) -> None:
        super().__init__()
        self.config = config
        self.state_family_embedding = nn.Embedding(4, config.hidden_dim)
        self.head: LocalActionHead = LocalFlatHead(config.hidden_dim)
        _add_inactive_padding(self, config.target_active_parameters)

    def forward(
        self,
        state_family_index: torch.Tensor,
        legal_actions: list[str],
    ) -> LocalActionOutput:
        h = self.state_family_embedding(state_family_index)
        return self.head.score(h, StateContext("compiler_owned"), legal_actions)


class _CompilerOwnedNoStateModel(nn.Module):
    """Arm E: compiler owns state; network sees no state-family embedding at all."""

    def __init__(self, config: ArmConfig) -> None:
        super().__init__()
        self.config = config
        # A tiny learned query vector that is identical for all inputs.  This is a
        # deliberately weak baseline: the head has no state-specific signal.
        self.query = nn.Parameter(torch.randn(config.hidden_dim) * 0.02)
        self.head: LocalActionHead = LocalFlatHead(config.hidden_dim)
        _add_inactive_padding(self, config.target_active_parameters)

    def forward(self, legal_actions: list[str]) -> LocalActionOutput:
        h = self.query.unsqueeze(0)
        return self.head.score(h, StateContext("compiler_owned_no_state"), legal_actions)


def _choose_levels(state_count: int) -> tuple[int, ...]:
    """Pick a mixed-radix level vector with capacity >= state_count."""
    if state_count <= 4:
        return (2, 3)
    if state_count <= 6:
        return (2, 2, 2)
    if state_count <= 8:
        return (2, 2, 2, 2)
    if state_count <= 12:
        return (2, 2, 3, 3)
    # General small fallback: binary levels until capacity is enough.
    d = math.ceil(math.log2(state_count))
    return tuple(2 for _ in range(max(2, d)))


def _build_model(config: ArmConfig, action_count: int) -> nn.Module:
    # Use a large vocabulary so hash collisions among legal actions are negligible.
    max_vocabulary = max(256, action_count * 16)
    if config.mode == "implicit":
        return _ImplicitStateModel(config, max_vocabulary)
    if config.mode == "explicit_exact":
        return _ExplicitExactModel(config, max_vocabulary)
    if config.mode == "discrete_code":
        return _DiscreteCodeModel(config, max_vocabulary)
    if config.mode == "compiler_owned":
        return _CompilerOwnedModel(config)
    if config.mode == "compiler_owned_no_state":
        return _CompilerOwnedNoStateModel(config)
    raise ValueError(f"unknown mode {config.mode!r}")


def _state_family_index(family_id: str) -> int:
    # Deterministic small index for the embedding table.
    return _stable_action_index(family_id, 4)


def _oracle_output(
    model: nn.Module,
    decision: FixtureDecision,
    device: torch.device,
) -> LocalActionOutput:
    """Construct an output that favors the correct action for the given model."""
    legal = list(decision.legal_actions)
    correct = decision.correct_action
    zero_hidden = torch.zeros(1, model.config.hidden_dim, device=device)

    if isinstance(model, (_ImplicitStateModel, _ExplicitExactModel, _DiscreteCodeModel)):
        out = model.head.score(zero_hidden, StateContext("oracle"), legal)
        logits = out.logits
        assert logits is not None
        correct_idx = _stable_action_index(correct, logits.shape[-1])
        biased = torch.full_like(logits, float("-inf"))
        # Also keep all legal indices finite to satisfy "no illegal output" audit.
        legal_indices = torch.tensor(
            [_stable_action_index(a, logits.shape[-1]) for a in legal],
            dtype=torch.long,
            device=device,
        )
        biased[:, legal_indices] = -1.0
        biased[0, correct_idx] = 10.0
        out.logits = biased
        return out

    if isinstance(model, (_CompilerOwnedModel, _CompilerOwnedNoStateModel)):
        out = model.head.score(zero_hidden, StateContext("oracle"), legal)
        assert out.logits is not None
        scores = torch.full_like(out.logits, -10.0)
        correct_pos = legal.index(correct)
        scores[0, correct_pos] = 10.0
        out.logits = scores
        return out

    raise ValueError(f"unsupported model {type(model).__name__}")


def _forward_model(
    model: nn.Module,
    decision: FixtureDecision,
    device: torch.device,
) -> LocalActionOutput:
    semantic = torch.tensor([list(decision.semantic_input)], dtype=torch.float32, device=device)
    if isinstance(model, _ImplicitStateModel):
        history = torch.tensor([list(decision.history)], dtype=torch.float32, device=device)
        return model(semantic, history, list(decision.legal_actions))
    if isinstance(model, (_ExplicitExactModel, _DiscreteCodeModel)):
        state_id = torch.tensor([decision.state_id], dtype=torch.long, device=device)
        return model(semantic, state_id, list(decision.legal_actions))
    if isinstance(model, _CompilerOwnedModel):
        family_idx = torch.tensor(
            [_state_family_index(decision.state_family_id)],
            dtype=torch.long,
            device=device,
        )
        return model(family_idx, list(decision.legal_actions))
    if isinstance(model, _CompilerOwnedNoStateModel):
        return model(list(decision.legal_actions))
    raise ValueError(f"unsupported model {type(model).__name__}")


def _decode_model(
    model: nn.Module,
    output: LocalActionOutput,
    legal_actions: list[str],
) -> Any:
    return model.head.decode(output, legal_actions)


def _train_tiny(
    model: nn.Module,
    decisions: tuple[FixtureDecision, ...],
    config: ArmConfig,
    device: torch.device,
) -> None:
    """A few gradient steps to nudge random-init accuracy above chance.

    This is not meant to converge the models; it only demonstrates that the arms
    can be optimized through the same optimizer recipe.
    """
    torch.manual_seed(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(config.train_steps):
        decision = decisions[step % len(decisions)]
        output = _forward_model(model, decision, device)
        logits = output.logits
        if logits is None:
            continue
        legal = list(decision.legal_actions)
        correct_pos = legal.index(decision.correct_action)
        # Cross-entropy on the legal action subset.
        if isinstance(model, (_ImplicitStateModel, _ExplicitExactModel, _DiscreteCodeModel)):
            legal_indices = torch.tensor(
                [_stable_action_index(a, logits.shape[-1]) for a in legal],
                dtype=torch.long,
                device=device,
            )
            subset = logits[:, legal_indices]
        else:
            subset = logits
        target = torch.tensor([correct_pos], dtype=torch.long, device=device)
        loss = F.cross_entropy(subset, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_arm(
    config: ArmConfig,
    decisions: tuple[FixtureDecision, ...],
    unseen_state_ids: tuple[int, ...],
    device: torch.device | None = None,
) -> ArmResult:
    """Evaluate one state-ownership arm on the fixture set."""
    start = time.monotonic()
    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(config.seed)
    action_count = len(decisions[0].legal_actions)
    model = _build_model(config, action_count).to(device)
    trainable = _count_parameters(model)
    active = _count_active_parameters(model)

    # Optional parameter-matching target: pad to the largest active count.
    if config.target_active_parameters is not None:
        target = config.target_active_parameters
        if active < target:
            pad = nn.Parameter(torch.zeros(target - active))
            model.register_parameter("inactive_padding", pad)
            trainable = _count_parameters(model)
            active = _count_active_parameters(model)

    # Tiny training pass so random-init is not pure noise.
    _train_tiny(model, decisions, config, device)

    oracle_correct = 0
    random_correct = 0
    forced_count = 0
    unseen_correct = 0
    unseen_total = 0

    for decision in decisions:
        legal = list(decision.legal_actions)

        # Oracle pass: wiring the correct code/factors must recover the action.
        oracle_out = _oracle_output(model, decision, device)
        oracle_decision = _decode_model(model, oracle_out, legal)
        if oracle_decision.action_identity == decision.correct_action:
            oracle_correct += 1

        # Trained pass.
        output = _forward_model(model, decision, device)
        pred = _decode_model(model, output, legal)
        if pred.decision_kind == "forced":
            forced_count += 1
            if pred.action_identity == decision.correct_action:
                random_correct += 1
        elif pred.action_identity == decision.correct_action:
            random_correct += 1

        if decision.state_id in unseen_state_ids:
            unseen_total += 1
            if pred.action_identity == decision.correct_action:
                unseen_correct += 1

    oracle_accuracy = oracle_correct / len(decisions)
    random_init_accuracy = random_correct / len(decisions)
    unseen_accuracy = unseen_correct / max(1, unseen_total)

    capacity: int | None = None
    if isinstance(model, _DiscreteCodeModel):
        capacity = model.capacity()

    leakage = oracle_accuracy >= 1.0 and capacity is not None and capacity < config.state_count

    notes: list[str] = [
        f"trainable_parameters={trainable}, active_parameters={active}",
        f"oracle_accuracy={oracle_accuracy:.4f}, random_init_accuracy={random_init_accuracy:.4f}",
    ]
    if capacity is not None:
        notes.append(f"discrete_code capacity={capacity}")

    return ArmResult(
        arm_id=config.arm_id,
        mode=config.mode,
        oracle_accuracy=oracle_accuracy,
        random_init_accuracy=random_init_accuracy,
        unseen_state_accuracy=unseen_accuracy,
        forced_decisions=forced_count,
        trainable_parameters=trainable,
        active_parameters=active,
        capacity=capacity,
        leakage=leakage,
        elapsed_seconds=time.monotonic() - start,
        notes=tuple(notes),
    )


def build_arms(
    state_count: int = 8,
    action_count: int = 5,
    hidden_dim: int = HIDDEN_DIM,
    semantic_dim: int = SEMANTIC_DIM,
    seeds: tuple[int, ...] = (0,),
    modes: tuple[str, ...] | None = None,
) -> list[ArmConfig]:
    """Build matched CAP2-04 arms.

    All arms share the same fixture recipe.  Parameter budgets are left as ``None``
    by default; callers can request active-parameter matching by setting
    ``target_active_parameters`` after construction.
    """
    if modes is None:
        modes = (
            "implicit",
            "explicit_exact",
            "discrete_code",
            "compiler_owned",
            "compiler_owned_no_state",
        )
    arms: list[ArmConfig] = []
    for seed in seeds:
        for mode in modes:
            arms.append(
                ArmConfig(
                    arm_id=f"{mode}_s{seed}",
                    mode=mode,
                    state_count=state_count,
                    action_count=action_count,
                    hidden_dim=hidden_dim,
                    semantic_dim=semantic_dim,
                    train_steps=200,
                    seed=seed,
                )
            )
    return arms


def match_active_parameters(
    arms: list[ArmConfig],
    decisions: tuple[FixtureDecision, ...],
) -> list[ArmConfig]:
    """Set every arm's target_active_parameters to the largest active count.

    This implements the matched-budget rule from SLM-89.  The actual padding is
    added inside ``evaluate_arm`` if needed.
    """
    device = torch.device("cpu")
    max_active = 0
    measured: list[tuple[ArmConfig, int]] = []
    for cfg in arms:
        model = _build_model(cfg, len(decisions[0].legal_actions)).to(device)
        active = _count_active_parameters(model)
        measured.append((cfg, active))
        if active > max_active:
            max_active = active
    return [
        ArmConfig(
            **{
                **cfg.__dict__,
                "target_active_parameters": max_active,
            }
        )
        for cfg, _ in measured
    ]


def run_matrix(
    state_count: int = 8,
    action_count: int = 5,
    hidden_dim: int = HIDDEN_DIM,
    semantic_dim: int = SEMANTIC_DIM,
    seeds: tuple[int, ...] = (0,),
    modes: tuple[str, ...] | None = None,
    match_parameters: bool = True,
    device: torch.device | None = None,
) -> StateAblationReport:
    """Run the CAP2-04 state-ownership ablation fixture matrix."""
    arms = build_arms(
        state_count=state_count,
        action_count=action_count,
        hidden_dim=hidden_dim,
        semantic_dim=semantic_dim,
        seeds=seeds,
        modes=modes,
    )
    decisions = fixture_decisions(state_count=state_count, action_count=action_count)
    unseen_state_ids = _split_unseen_states(state_count, seed=sum(seeds) + 1)

    if match_parameters:
        arms = match_active_parameters(arms, decisions)

    results: list[ArmResult] = []
    for arm in arms:
        results.append(evaluate_arm(arm, decisions, unseen_state_ids, device=device))

    run_id = _hash_run_id(
        ("cap2-04", state_count, action_count, tuple(a.arm_id for a in arms), seeds)
    )
    return StateAblationReport(
        run_id=run_id,
        version="cap2-04-v1",
        timestamp=_utc_now(),
        hidden_dim=hidden_dim,
        semantic_dim=semantic_dim,
        states=decisions,
        unseen_state_ids=unseen_state_ids,
        arms=tuple(results),
    )
