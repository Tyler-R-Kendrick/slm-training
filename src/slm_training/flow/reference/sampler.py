"""Gillespie and fixed-grid CTMC samplers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


from slm_training.flow.reference.adapter import ActionRef, StateRef
from slm_training.flow.reference.generator import Generator
from slm_training.flow.reference.trajectory import FlowSampleV1, FlowTrajectoryV1


@dataclass(frozen=True)
class GillespieSampler:
    """Exact Gillespie (SSA) sampler for a finite CTMC."""

    generator: Generator
    max_steps: int = 10_000
    max_time: float = 1e6

    def sample(
        self,
        source: StateRef,
        rng: Any,
        terminal_check: Any | None = None,
    ) -> FlowTrajectoryV1:
        """Simulate one path until terminal or a bound is hit."""
        terminal_check = terminal_check or (lambda s: False)
        states: list[StateRef] = [source]
        actions: list[ActionRef | None] = []
        holding_times: list[float] = []
        wall_times: list[float] = [0.0]
        certificates: list[str] = []
        current = source
        total_time = 0.0
        for _ in range(self.max_steps):
            if terminal_check(current):
                break
            successors = self.generator.legal_successors(current)
            if not successors:
                break
            total_rate = sum(rate for _, _, rate in successors)
            if total_rate <= 0.0:
                break
            hold_time = rng.expovariate(total_rate)
            if total_time + hold_time > self.max_time:
                break
            # Weighted choice.
            threshold = rng.random() * total_rate
            cumsum = 0.0
            chosen_idx = successors[0][0]
            chosen_action = successors[0][1]
            for idx, action, rate in successors:
                cumsum += rate
                if cumsum >= threshold:
                    chosen_idx = idx
                    chosen_action = action
                    break
            cert_id = f"{current.fingerprint[:12]}->{self.generator.index_state[chosen_idx].fingerprint[:12]}"
            actions.append(chosen_action)
            holding_times.append(hold_time)
            total_time += hold_time
            wall_times.append(total_time)
            certificates.append(cert_id)
            current = self.generator.index_state[chosen_idx]
            states.append(current)
            if terminal_check(current):
                break
        return FlowTrajectoryV1(
            trajectory_id=f"gillespie-{rng.randint(0, 2**31 - 1)}",
            source_fingerprint=source.fingerprint,
            states=tuple(s.fingerprint for s in states),
            actions=tuple((a.action_id if a else "") for a in actions),
            holding_times=tuple(holding_times),
            wall_times=tuple(wall_times),
            certificates=tuple(certificates),
            terminal_fingerprint=states[-1].fingerprint,
            total_time=total_time,
        )


@dataclass(frozen=True)
class FixedGridSampler:
    """Fixed-time-step approximation sampler for comparison/debugging."""

    generator: Generator
    step_size: float = 0.1
    n_steps: int = 100

    def sample(
        self,
        source: StateRef,
        rng: Any,
        terminal_check: Any | None = None,
    ) -> FlowTrajectoryV1:
        """Simulate one path by drawing a successor at each fixed grid point."""
        terminal_check = terminal_check or (lambda s: False)
        states: list[StateRef] = [source]
        actions: list[ActionRef | None] = []
        holding_times: list[float] = []
        wall_times: list[float] = [0.0]
        certificates: list[str] = []
        current = source
        total_time = 0.0
        for step in range(self.n_steps):
            if terminal_check(current):
                break
            successors = self.generator.legal_successors(current)
            if not successors:
                break
            total_rate = sum(rate for _, _, rate in successors)
            if total_rate <= 0.0:
                break
            # Approximate: jump probability = rate * dt, renormalized.
            probs = [rate / total_rate for _, _, rate in successors]
            threshold = rng.random()
            cumsum = 0.0
            chosen_idx = successors[0][0]
            chosen_action = successors[0][1]
            for (idx, action, _), p in zip(successors, probs):
                cumsum += p
                if cumsum >= threshold:
                    chosen_idx = idx
                    chosen_action = action
                    break
            cert_id = f"{current.fingerprint[:12]}->{self.generator.index_state[chosen_idx].fingerprint[:12]}"
            actions.append(chosen_action)
            holding_times.append(self.step_size)
            total_time += self.step_size
            wall_times.append(total_time)
            certificates.append(cert_id)
            current = self.generator.index_state[chosen_idx]
            states.append(current)
            if terminal_check(current):
                break
        return FlowTrajectoryV1(
            trajectory_id=f"fixedgrid-{rng.randint(0, 2**31 - 1)}",
            source_fingerprint=source.fingerprint,
            states=tuple(s.fingerprint for s in states),
            actions=tuple((a.action_id if a else "") for a in actions),
            holding_times=tuple(holding_times),
            wall_times=tuple(wall_times),
            certificates=tuple(certificates),
            terminal_fingerprint=states[-1].fingerprint,
            total_time=total_time,
        )


def sample_endpoint_distribution(
    sampler: GillespieSampler | FixedGridSampler,
    source: StateRef,
    n_samples: int,
    rng: Any,
    terminal_check: Any,
) -> FlowSampleV1:
    """Sample many trajectories and return empirical terminal distribution."""
    counts: dict[str, int] = {}
    trajectories: list[FlowTrajectoryV1] = []
    for _ in range(n_samples):
        traj = sampler.sample(source, rng, terminal_check=terminal_check)
        trajectories.append(traj)
        counts[traj.terminal_fingerprint] = counts.get(traj.terminal_fingerprint, 0) + 1
    total = sum(counts.values())
    empirical = {fp: c / total for fp, c in counts.items()} if total else {}
    return FlowSampleV1(
        source_fingerprint=source.fingerprint,
        n_samples=n_samples,
        empirical_terminal_distribution=empirical,
        trajectories=tuple(trajectories[:10]),  # keep payload small
    )


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(p || q) with smoothing floor for numerical stability."""
    total = 0.0
    for k, pv in p.items():
        qv = q.get(k, 0.0)
        if pv > 0.0:
            total += pv * math.log(pv / max(qv, 1e-12))
    return total


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    """Total variation distance between discrete distributions."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
