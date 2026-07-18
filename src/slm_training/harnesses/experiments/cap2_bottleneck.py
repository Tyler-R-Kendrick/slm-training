"""CAP2-01/02 strict bottleneck phase-boundary and latent-codec matrix harness.

This module implements the controlled experiment that tests whether a
deterministic model can represent ``M`` distinct states through a fixed-length
``K``-ary bottleneck of dimension ``d`` (CAP2-01) and compares multiple latent
codec families under a common interface (CAP2-02).  When ``K**d < M`` exact
reconstruction is impossible; when ``K**d >= M`` it is representationally
possible but not guaranteed.  Fixture runs are wiring/mathematical evidence only.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from slm_training.dsl.analysis.arity.coding import (
    build_mds_7_4_2_3,
    build_shortened_ternary_hamming_7_4_3,
)
from slm_training.models.binary_lfq import BinaryLFQCodec, BinaryLFQConfig
from slm_training.models.continuous_latent import ContinuousLatentCodec, ContinuousLatentConfig
from slm_training.models.kary_bottleneck import (
    KaryBottleneck,
    KaryBottleneckConfig,
    evaluate_kary_bottleneck,
    train_kary_bottleneck,
)
from slm_training.models.latent_codec_trainer import (
    LatentCodecModel,
    evaluate_latent_codec,
    train_latent_codec,
)
from slm_training.models.learned_vq import LearnedVQCodec, LearnedVQConfig
from slm_training.models.mixed_radix_fsq import MixedRadixFSQCodec, MixedRadixFSQConfig
from slm_training.models.uniform_scalar_codec import (
    UniformScalarCodec,
    UniformScalarCodecConfig,
)


@dataclass(frozen=True)
class BottleneckArm:
    """One arm of the CAP2 discrete-bottleneck matrix."""

    arm_id: str
    K: int
    d: int
    state_count: int
    mode: str = "injective"  # injective | learned | robust | direct | randomized
    corruption: str | None = None  # "one_substitution" for robust arms
    deterministic_eval: bool = True
    hidden_dim: int = 64
    train_steps: int = 1000
    seed: int = 0
    # CAP2-02 latent-codec family selection.
    codec: str = "kary"  # kary | uniform_scalar | fsq | lfq | vq | continuous
    radixes: tuple[int, ...] | None = None
    latent_dim: int | None = None
    noise_std: float = 0.0
    rate_penalty: float = 0.0
    commitment_cost: float = 0.25

    @property
    def capacity(self) -> int:
        if self.codec == "fsq":
            if self.radixes is None:
                raise ValueError(f"fsq arm {self.arm_id} requires radixes")
            cap = 1
            for level in self.radixes:
                cap *= level
            return cap
        if self.codec in ("lfq", "binary_lfq"):
            return 2 ** self.d
        if self.codec in ("vq", "learned_vq"):
            return self.K
        if self.codec == "continuous":
            # Continuous latents are not discrete; report the latent dimension.
            return self.latent_dim if self.latent_dim is not None else self.d
        return self.K ** self.d


@dataclass(frozen=True)
class BottleneckResult:
    """Measured result for one arm."""

    arm_id: str
    K: int
    d: int
    state_count: int
    capacity: int
    mode: str
    corruption: str | None
    seed: int
    exact_reconstruction_rate: float
    collision_count: int
    occupied_codewords: int
    code_utilization: float
    empirical_entropy_bits: float
    min_code_distance: int | None
    mean_code_distance: float | None
    leakage: bool
    elapsed_seconds: float
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "K": self.K,
            "d": self.d,
            "state_count": self.state_count,
            "capacity": self.capacity,
            "mode": self.mode,
            "corruption": self.corruption,
            "seed": self.seed,
            "exact_reconstruction_rate": self.exact_reconstruction_rate,
            "collision_count": self.collision_count,
            "occupied_codewords": self.occupied_codewords,
            "code_utilization": self.code_utilization,
            "empirical_entropy_bits": self.empirical_entropy_bits,
            "min_code_distance": self.min_code_distance,
            "mean_code_distance": self.mean_code_distance,
            "leakage": self.leakage,
            "elapsed_seconds": self.elapsed_seconds,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class BottleneckMatrixReport:
    """Full fixture matrix report."""

    run_id: str
    state_count: int
    state_report_path: str | None
    arms: tuple[BottleneckResult, ...]
    version: str = "cap2-02-v1"
    timestamp: str = field(default_factory=lambda: _utc_now())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "version": self.version,
            "timestamp": self.timestamp,
            "state_count": self.state_count,
            "state_report_path": self.state_report_path,
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


def load_state_report(path: Path) -> dict[str, Any]:
    """Load a CAP1-01 state-graph report or CAP0-03 arity report.

    Raises a clear error if the report is missing required state-count fields.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if "minimized_states" not in data:
        raise ValueError(
            f"state report {path} lacks 'minimized_states'; "
            "expected a CAP1-01 StateGraphReport or CAP0-03 ArityReport"
        )
    return data


def state_count_from_report(report: dict[str, Any]) -> int:
    return int(report["minimized_states"])


def fixture_states(state_count: int) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Synthetic deterministic fixture: integer state ids and action labels."""
    actions = tuple(f"a{i % 7}" for i in range(state_count))
    return tuple(range(state_count)), actions


def _make_codebook(M: int, K: int, d: int, seed: int) -> list[tuple[int, ...]]:
    """Deterministic flat codebook of length M over K^d.

    If capacity < M the sequence wraps, producing unavoidable collisions.
    """
    rng = random.Random(seed)
    codes: list[tuple[int, ...]] = []
    for i in range(M):
        if i < K ** d:
            # lexicographic index -> mixed-radix digits
            digits = []
            n = i
            for _ in range(d):
                digits.append(n % K)
                n //= K
            codes.append(tuple(reversed(digits)))
        else:
            # capacity exhausted: reuse a random existing code (collision)
            codes.append(rng.choice(codes))
    return codes


def _code_distance_stats(codes: list[tuple[int, ...]]) -> tuple[int | None, float | None]:
    if len(codes) < 2:
        return None, None
    dists = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            dists.append(sum(1 for a, b in zip(codes[i], codes[j]) if a != b))
    return min(dists), sum(dists) / len(dists)


def _empirical_entropy(codes: list[tuple[int, ...]]) -> float:
    counts: dict[tuple[int, ...], int] = {}
    for c in codes:
        counts[c] = counts.get(c, 0) + 1
    total = len(codes)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _nearest_codeword(
    word: tuple[int, ...],
    codebook: list[tuple[int, ...]],
) -> tuple[int, ...]:
    best = codebook[0]
    best_dist = sum(1 for a, b in zip(word, best) if a != b)
    for cw in codebook[1:]:
        dist = sum(1 for a, b in zip(word, cw) if a != b)
        if dist < best_dist:
            best = cw
            best_dist = dist
    return best


def evaluate_injective_arm(
    arm: BottleneckArm,
    states: tuple[int, ...],
) -> BottleneckResult:
    """Evaluate a deterministic injective / insufficient-capacity codebook."""
    start = time.monotonic()
    codes = _make_codebook(arm.state_count, arm.K, arm.d, arm.seed)
    index: dict[tuple[int, ...], int] = {}
    collisions = 0
    for state, code in zip(states, codes):
        if code in index and index[code] != state:
            collisions += 1
        index.setdefault(code, state)

    reconstructed = [index.get(c, -1) for c in codes]
    correct = sum(1 for s, r in zip(states, reconstructed) if s == r)
    exact_rate = correct / len(states)
    occupied = len(index)
    min_dist, mean_dist = _code_distance_stats(list(index.keys()))
    entropy = _empirical_entropy(codes)
    leakage = exact_rate >= 1.0 and arm.capacity < arm.state_count
    notes: list[str] = []
    if arm.capacity < arm.state_count:
        notes.append(
            f"capacity {arm.capacity} < states {arm.state_count}; "
            f"{collisions} collisions, exact_rate={exact_rate:.4f}"
        )
    elif exact_rate < 1.0:
        notes.append(f"injective assignment failed: exact_rate={exact_rate:.4f}")
    else:
        notes.append("injective assignment reconstructed all states")

    return BottleneckResult(
        arm_id=arm.arm_id,
        K=arm.K,
        d=arm.d,
        state_count=arm.state_count,
        capacity=arm.capacity,
        mode=arm.mode,
        corruption=arm.corruption,
        seed=arm.seed,
        exact_reconstruction_rate=exact_rate,
        collision_count=collisions,
        occupied_codewords=occupied,
        code_utilization=occupied / max(1, arm.capacity),
        empirical_entropy_bits=entropy,
        min_code_distance=min_dist,
        mean_code_distance=mean_dist,
        leakage=leakage,
        elapsed_seconds=time.monotonic() - start,
        notes=tuple(notes),
    )


def evaluate_robust_arm(
    arm: BottleneckArm,
    states: tuple[int, ...],
) -> BottleneckResult:
    """Evaluate a robust code arm under one arbitrary coordinate substitution."""
    start = time.monotonic()
    if arm.K == 7 and arm.d == 4:
        codebook = list(build_mds_7_4_2_3())[: arm.state_count]
    elif arm.K == 3 and arm.d == 7:
        codebook = list(build_shortened_ternary_hamming_7_4_3())[: arm.state_count]
    else:
        raise ValueError(f"no verified construction for robust arm {arm.arm_id}")

    rng = random.Random(arm.seed)
    corrupted: list[tuple[int, ...]] = []
    for code in codebook:
        coord = rng.randrange(arm.d)
        new_symbol = rng.choice([s for s in range(arm.K) if s != code[coord]])
        corrupted.append(code[:coord] + (new_symbol,) + code[coord + 1 :])

    decoded = [_nearest_codeword(c, codebook) for c in corrupted]
    correct = sum(1 for orig, dec in zip(codebook, decoded) if orig == dec)
    exact_rate = correct / len(states)
    occupied = len(set(codebook))
    min_dist, mean_dist = _code_distance_stats(codebook)
    entropy = _empirical_entropy(codebook)
    leakage = exact_rate >= 1.0 and arm.capacity < arm.state_count
    notes = [
        f"robust code {arm.K}^{arm.d}={arm.capacity} for {arm.state_count} states; "
        f"one-substitution correction rate {exact_rate:.4f}"
    ]
    return BottleneckResult(
        arm_id=arm.arm_id,
        K=arm.K,
        d=arm.d,
        state_count=arm.state_count,
        capacity=arm.capacity,
        mode=arm.mode,
        corruption=arm.corruption,
        seed=arm.seed,
        exact_reconstruction_rate=exact_rate,
        collision_count=0,
        occupied_codewords=occupied,
        code_utilization=occupied / max(1, arm.capacity),
        empirical_entropy_bits=entropy,
        min_code_distance=min_dist,
        mean_code_distance=mean_dist,
        leakage=leakage,
        elapsed_seconds=time.monotonic() - start,
        notes=tuple(notes),
    )


def evaluate_learned_arm(
    arm: BottleneckArm,
    states: tuple[int, ...],
) -> BottleneckResult:
    """Train and evaluate a tiny learned K-ary bottleneck."""
    start = time.monotonic()
    cfg = KaryBottleneckConfig(
        num_states=arm.state_count,
        K=arm.K,
        d=arm.d,
        hidden_dim=arm.hidden_dim,
        mode="oracle_state",
        train_steps=arm.train_steps,
    )
    model = KaryBottleneck(cfg)
    state_tensor = torch.tensor(states, dtype=torch.long)
    target_tensor = torch.tensor(states, dtype=torch.long)
    train_kary_bottleneck(model, state_tensor, target_tensor, steps=arm.train_steps)
    eval_metrics = evaluate_kary_bottleneck(model, state_tensor, target_tensor)
    exact_rate = eval_metrics["exact_reconstruction_rate"]
    occupied = eval_metrics["occupied_codewords"]
    leakage = exact_rate >= 1.0 and arm.capacity < arm.state_count
    notes = [
        f"learned bottleneck hidden_dim={arm.hidden_dim} steps={arm.train_steps}; "
        f"final_loss wired via soft training"
    ]
    return BottleneckResult(
        arm_id=arm.arm_id,
        K=arm.K,
        d=arm.d,
        state_count=arm.state_count,
        capacity=arm.capacity,
        mode=arm.mode,
        corruption=arm.corruption,
        seed=arm.seed,
        exact_reconstruction_rate=exact_rate,
        collision_count=arm.state_count - occupied,
        occupied_codewords=occupied,
        code_utilization=occupied / max(1, arm.capacity),
        empirical_entropy_bits=_empirical_entropy(
            [tuple(c.tolist()) for c in model(state_tensor, hard=True)[1]]
        ),
        min_code_distance=None,
        mean_code_distance=None,
        leakage=leakage,
        elapsed_seconds=time.monotonic() - start,
        notes=tuple(notes),
    )


def evaluate_direct_control(
    arm: BottleneckArm,
    states: tuple[int, ...],
) -> BottleneckResult:
    """No-bottleneck direct classifier baseline (perfect on the fixture)."""
    start = time.monotonic()
    # Direct one-hot: each state maps to itself, no bottleneck.
    exact_rate = 1.0
    leakage = False
    notes = ["direct one-hot control: no bottleneck, perfect reconstruction by design"]
    return BottleneckResult(
        arm_id=arm.arm_id,
        K=arm.K,
        d=arm.d,
        state_count=arm.state_count,
        capacity=arm.state_count,
        mode=arm.mode,
        corruption=arm.corruption,
        seed=arm.seed,
        exact_reconstruction_rate=exact_rate,
        collision_count=0,
        occupied_codewords=arm.state_count,
        code_utilization=1.0,
        empirical_entropy_bits=math.log2(arm.state_count),
        min_code_distance=None,
        mean_code_distance=None,
        leakage=leakage,
        elapsed_seconds=time.monotonic() - start,
        notes=tuple(notes),
    )


def _build_latent_codec_model(arm: BottleneckArm) -> LatentCodecModel:
    """Construct a LatentCodecModel matching the arm's codec family."""
    if arm.codec in ("kary", "uniform_scalar"):
        cfg = UniformScalarCodecConfig(
            num_states=arm.state_count,
            K=arm.K,
            d=arm.d,
            hidden_dim=arm.hidden_dim,
            mode="oracle_state",
        )
        codec = UniformScalarCodec(cfg)
    elif arm.codec == "fsq":
        if arm.radixes is None:
            raise ValueError(f"fsq arm {arm.arm_id} requires radixes")
        cfg = MixedRadixFSQConfig(
            num_states=arm.state_count,
            levels=arm.radixes,
            hidden_dim=arm.hidden_dim,
            mode="oracle_state",
        )
        codec = MixedRadixFSQCodec(cfg)
    elif arm.codec in ("lfq", "binary_lfq"):
        cfg = BinaryLFQConfig(
            num_states=arm.state_count,
            d=arm.d,
            hidden_dim=arm.hidden_dim,
            mode="oracle_state",
        )
        codec = BinaryLFQCodec(cfg)
    elif arm.codec in ("vq", "learned_vq"):
        cfg = LearnedVQConfig(
            num_states=arm.state_count,
            codebook_size=arm.K,
            latent_dim=arm.latent_dim or arm.d,
            hidden_dim=arm.hidden_dim,
            mode="oracle_state",
            commitment_cost=arm.commitment_cost,
        )
        codec = LearnedVQCodec(cfg)
    elif arm.codec == "continuous":
        cfg = ContinuousLatentConfig(
            num_states=arm.state_count,
            latent_dim=arm.latent_dim or arm.d,
            hidden_dim=arm.hidden_dim,
            mode="oracle_state",
            noise_std=arm.noise_std,
            rate_penalty=arm.rate_penalty,
        )
        codec = ContinuousLatentCodec(cfg)
    else:
        raise ValueError(f"unknown codec {arm.codec!r}")
    return LatentCodecModel(codec, arm.state_count)


def evaluate_latent_codec_arm(
    arm: BottleneckArm,
    states: tuple[int, ...],
) -> BottleneckResult:
    """Train and evaluate a tiny learned latent-codec arm."""
    start = time.monotonic()
    model = _build_latent_codec_model(arm)
    state_tensor = torch.tensor(states, dtype=torch.long)
    target_tensor = torch.tensor(states, dtype=torch.long)
    train_latent_codec(model, state_tensor, target_tensor, steps=arm.train_steps)
    eval_metrics = evaluate_latent_codec(model, state_tensor, target_tensor)
    exact_rate = eval_metrics["exact_reconstruction_rate"]
    occupied = eval_metrics["occupied_codewords"]
    capacity = arm.capacity
    # Continuous arms cannot leak in the discrete sense; only discrete below-capacity
    # arms reaching 100% reconstruction are leakage violations.
    leakage = (
        exact_rate >= 1.0
        and capacity < arm.state_count
        and arm.codec != "continuous"
    )
    notes = [
        f"learned {arm.codec} codec hidden_dim={arm.hidden_dim} steps={arm.train_steps}; "
        f"utilization={eval_metrics['utilization']:.4f} entropy_bits={eval_metrics['empirical_entropy_bits']:.4f}"
    ]
    if arm.codec == "continuous":
        notes.append(
            f"continuous latent_dim={arm.latent_dim or arm.d} "
            f"noise_std={arm.noise_std} rate_penalty={arm.rate_penalty}"
        )
    return BottleneckResult(
        arm_id=arm.arm_id,
        K=arm.K,
        d=arm.d,
        state_count=arm.state_count,
        capacity=capacity,
        mode=arm.mode,
        corruption=arm.corruption,
        seed=arm.seed,
        exact_reconstruction_rate=exact_rate,
        collision_count=arm.state_count - occupied,
        occupied_codewords=occupied,
        code_utilization=occupied / max(1, capacity),
        empirical_entropy_bits=eval_metrics["empirical_entropy_bits"],
        min_code_distance=None,
        mean_code_distance=None,
        leakage=leakage,
        elapsed_seconds=time.monotonic() - start,
        notes=tuple(notes),
    )


def evaluate_arm(arm: BottleneckArm, states: tuple[int, ...]) -> BottleneckResult:
    if arm.mode == "injective":
        return evaluate_injective_arm(arm, states)
    if arm.mode == "robust":
        return evaluate_robust_arm(arm, states)
    if arm.mode == "learned":
        # Legacy kary learned arm; new codec families use mode="learned_codec".
        if arm.codec == "kary":
            return evaluate_learned_arm(arm, states)
        return evaluate_latent_codec_arm(arm, states)
    if arm.mode == "learned_codec":
        return evaluate_latent_codec_arm(arm, states)
    if arm.mode == "direct":
        return evaluate_direct_control(arm, states)
    raise ValueError(f"unknown arm mode {arm.mode!r}")


def build_boundary_arms(state_count: int, seeds: tuple[int, ...]) -> list[BottleneckArm]:
    arms: list[BottleneckArm] = []
    for seed in seeds:
        arms.extend(
            [
                BottleneckArm("b2d5", 2, 5, state_count, mode="injective", seed=seed),
                BottleneckArm("b2d6", 2, 6, state_count, mode="injective", seed=seed),
                BottleneckArm("t3d3", 3, 3, state_count, mode="injective", seed=seed),
                BottleneckArm("t3d4", 3, 4, state_count, mode="injective", seed=seed),
            ]
        )
    return arms


def build_equal_capacity_arms(
    state_count: int, seeds: tuple[int, ...]
) -> list[BottleneckArm]:
    arms: list[BottleneckArm] = []
    for seed in seeds:
        arms.extend(
            [
                BottleneckArm("k2d6", 2, 6, state_count, mode="injective", seed=seed),
                BottleneckArm("k4d3", 4, 3, state_count, mode="injective", seed=seed),
                BottleneckArm("k8d2", 8, 2, state_count, mode="injective", seed=seed),
            ]
        )
    return arms


def build_robust_arms(state_count: int, seeds: tuple[int, ...]) -> list[BottleneckArm]:
    arms: list[BottleneckArm] = []
    for seed in seeds:
        arms.extend(
            [
                BottleneckArm(
                    "k7d4_robust",
                    7,
                    4,
                    state_count,
                    mode="robust",
                    corruption="one_substitution",
                    seed=seed,
                ),
                BottleneckArm(
                    "k3d7_robust",
                    3,
                    7,
                    state_count,
                    mode="robust",
                    corruption="one_substitution",
                    seed=seed,
                ),
            ]
        )
    return arms


def build_control_arms(state_count: int, seeds: tuple[int, ...]) -> list[BottleneckArm]:
    arms: list[BottleneckArm] = []
    for seed in seeds:
        arms.extend(
            [
                BottleneckArm("direct_one_hot", state_count, 1, state_count, mode="direct", seed=seed),
                BottleneckArm("learned_b2d6", 2, 6, state_count, mode="learned", seed=seed),
                BottleneckArm("learned_t3d4", 3, 4, state_count, mode="learned", seed=seed),
            ]
        )
    return arms


def build_latent_codec_arms(state_count: int, seeds: tuple[int, ...]) -> list[BottleneckArm]:
    """CAP2-02 latent-codec arms at roughly matched nominal capacity.

    The arms share the same target state_count and are trained with the same
    small fixture recipe.  They demonstrate that each codec family can be
    evaluated through the common harness.
    """
    arms: list[BottleneckArm] = []
    for seed in seeds:
        arms.extend(
            [
                # Mixed-radix FSQ with capacity 2*3*3*4*5 = 360 >= 41.
                BottleneckArm(
                    "fsq_2_3_3_4_5",
                    0,
                    0,
                    state_count,
                    mode="learned_codec",
                    codec="fsq",
                    radixes=(2, 3, 3, 4, 5),
                    train_steps=1200,
                    seed=seed,
                ),
                # Binary LFQ with capacity 2^6 = 64 >= 41.
                BottleneckArm(
                    "lfq_d6",
                    0,
                    6,
                    state_count,
                    mode="learned_codec",
                    codec="lfq",
                    train_steps=1200,
                    seed=seed,
                ),
                # Learned VQ with codebook size 64 >= 41.
                BottleneckArm(
                    "vq_64_d8",
                    64,
                    0,
                    state_count,
                    mode="learned_codec",
                    codec="vq",
                    latent_dim=8,
                    train_steps=1200,
                    seed=seed,
                ),
                # Continuous latent control (6 dims, no discrete capacity claim).
                BottleneckArm(
                    "continuous_d6",
                    0,
                    0,
                    state_count,
                    mode="learned_codec",
                    codec="continuous",
                    latent_dim=6,
                    train_steps=1200,
                    seed=seed,
                ),
                # Uniform scalar baseline at matched capacity.
                BottleneckArm(
                    "uniform_b2d6",
                    2,
                    6,
                    state_count,
                    mode="learned_codec",
                    codec="uniform_scalar",
                    train_steps=800,
                    seed=seed,
                ),
            ]
        )
    return arms


def build_matrix(
    state_count: int,
    *,
    seeds: tuple[int, ...] = (0,),
    arms_filter: tuple[str, ...] | None = None,
) -> list[BottleneckArm]:
    arms: list[BottleneckArm] = []
    arms.extend(build_boundary_arms(state_count, seeds))
    arms.extend(build_equal_capacity_arms(state_count, seeds))
    arms.extend(build_robust_arms(state_count, seeds))
    arms.extend(build_control_arms(state_count, seeds))
    arms.extend(build_latent_codec_arms(state_count, seeds))
    if arms_filter:
        wanted = set(arms_filter)
        arms = [a for a in arms if a.arm_id in wanted]
    return arms


def run_matrix(
    state_count: int,
    *,
    seeds: tuple[int, ...] = (0,),
    arms_filter: tuple[str, ...] | None = None,
    state_report_path: Path | None = None,
) -> BottleneckMatrixReport:
    """Run the full CAP2 fixture matrix and return a versioned report."""
    arms = build_matrix(state_count, seeds=seeds, arms_filter=arms_filter)
    states, _ = fixture_states(state_count)
    results: list[BottleneckResult] = []
    for arm in arms:
        results.append(evaluate_arm(arm, states))
    run_id = _hash_run_id(
        ("cap2-02", state_count, tuple(a.arm_id for a in arms), seeds)
    )
    return BottleneckMatrixReport(
        run_id=run_id,
        state_count=state_count,
        state_report_path=str(state_report_path) if state_report_path else None,
        arms=tuple(results),
    )
