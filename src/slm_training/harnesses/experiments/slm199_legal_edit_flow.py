"""SLM-199 exact-rate oracle and adapted OpenUI flow fixture."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from slm_training.data.flow.bridge_corpus import (
    RequestEditContractV1,
    canonical_fingerprint,
    load_corpus,
)
from slm_training.flow.reference import (
    ExactEnumerator,
    FlowTargetRowV1,
    GeneratorBuilder,
    GillespieSampler,
    build_uniform_rate_fn,
    check_generator,
    endpoint_distribution,
)
from slm_training.flow.reference.adapters import ChoiceSequenceAdapter
from slm_training.flow.reference.generator import Generator
from slm_training.flow.samplers import ProductionLegalEditFlowSampler
from slm_training.flow.targets import from_bridge_rows, from_exact_rows
from slm_training.flow.termination import FixedKPolicy
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.models.legal_edit_batch import LegalEditBatch
from slm_training.models.legal_edit_flow import (
    ExactRateTable,
    LegalEditFlow,
    LegalEditFlowConfig,
    legal_edit_flow_losses,
)

DEFAULT_CORPUS = Path(
    "src/slm_training/resources/data/train/slm196_legal_edit_bridge_fixture"
)
DEFAULT_RECORDS = Path("tests/fixtures/slm196_legal_edit_bridge/records.jsonl")
MATRIX_SET = "slm199_legal_edit_flow"
FIXED_EDIT_BUDGET = 4


def _schedule_progress(rows: Sequence[Any]) -> torch.Tensor:
    """Return inference-visible progress against the fixed decode budget."""
    return torch.tensor(
        [
            min(1.0, float(row.step_index) / FIXED_EDIT_BUDGET)
            for row in rows
        ],
        dtype=torch.float32,
    )


def _event_count_distribution(
    generator: Generator, source_index: int, terminal_indices: set[int]
) -> dict[int, float]:
    memo: dict[int, dict[int, float]] = {}

    def visit(index: int, active: frozenset[int]) -> dict[int, float]:
        if index in terminal_indices:
            return {0: 1.0}
        if index in active:
            raise ValueError("exact event-count fixture must be acyclic")
        if index in memo:
            return memo[index]
        successors = generator.legal_successors(index)
        total = sum(rate for _, _, rate in successors)
        if total <= 0.0:
            raise ValueError("nonterminal exact state has no positive hazard")
        result: dict[int, float] = {}
        for successor, _, rate in successors:
            for count, probability in visit(
                successor, active | {index}
            ).items():
                result[count + 1] = (
                    result.get(count + 1, 0.0) + rate * probability / total
                )
        memo[index] = result
        return result

    return visit(source_index, frozenset())


def _exact_oracle(
    seed: int,
    samples: int,
    *,
    fit_steps: int = 100,
    deadline: float | None = None,
) -> dict[str, Any]:
    # Closed, acyclic, single-path grammar: the exact event count is five
    # (expand S/A/B, emit a/b) and the terminal endpoint is unique.
    adapter = ChoiceSequenceAdapter(
        productions={"S": [["A", "B"]], "A": [["a"]], "B": [["b"]]},
        max_length=2,
        max_states=20,
    )
    graph = ExactEnumerator(adapter, max_states=20).enumerate()
    generator = GeneratorBuilder(graph).build_dense(build_uniform_rate_fn(1.0))
    source = graph.initial_states[0]
    source_idx = generator.state_index[source.fingerprint]
    exact_rows: list[FlowTargetRowV1] = []
    for index, state in generator.index_state.items():
        successors = generator.legal_successors(index)
        if not successors:
            continue
        candidate_ids = tuple(
            generator.index_state[next_index].fingerprint
            for next_index, _, _ in successors
        )
        exact_rows.append(
            FlowTargetRowV1(
                row_id=state.fingerprint,
                source_fingerprint=source.fingerprint,
                target_fingerprint=graph.terminal_states[0].fingerprint,
                time=1.0,
                state_fingerprint=state.fingerprint,
                exact_live_candidates=candidate_ids,
                target_rates={
                    generator.index_state[next_index].fingerprint: rate
                    for next_index, _, rate in successors
                },
                total_hazard=generator.hazard(index),
                next_state_fingerprints=candidate_ids,
                endpoint_class=adapter.terminal_class(
                    graph.terminal_states[0]
                ),
            )
        )
    targets = from_exact_rows(tuple(exact_rows))
    rate_model = ExactRateTable(targets)
    optimizer = torch.optim.Adam(rate_model.parameters(), lr=0.1)
    target_tensors = tuple(
        torch.tensor(target.edge_rates, dtype=torch.float32)
        for target in targets
    )
    rate_loss_history: list[float] = []
    for _ in range(fit_steps):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("exact rate fit exceeded max_wall_minutes")
        optimizer.zero_grad(set_to_none=True)
        predicted = rate_model()
        loss = torch.stack(
            [
                torch.nn.functional.mse_loss(value, target)
                for value, target in zip(
                    predicted, target_tensors, strict=True
                )
            ]
        ).mean()
        loss.backward()
        optimizer.step()
        rate_loss_history.append(float(loss.detach()))
    predicted_rows = rate_model()
    predicted_q = np.zeros_like(generator.Q)
    predicted_rates: dict[tuple[int, int], float] = {}
    for target, row_rates in zip(targets, predicted_rows, strict=True):
        source_index = generator.state_index[target.row_id]
        for candidate_id, rate_tensor in zip(
            target.candidate_ids, row_rates, strict=True
        ):
            target_index = generator.state_index[candidate_id]
            rate = float(rate_tensor.detach())
            predicted_q[source_index, target_index] = rate
            predicted_rates[(source_index, target_index)] = rate
    for index in range(generator.n_states):
        predicted_q[index, index] = -predicted_q[index].sum()
    predicted_generator = Generator(
        Q=predicted_q,
        state_index=generator.state_index,
        index_state=generator.index_state,
        action_for_pair=generator.action_for_pair,
        rates=predicted_rates,
    )
    p0 = np.zeros(generator.n_states)
    p0[source_idx] = 1.0
    exact = endpoint_distribution(generator.Q, p0, 50.0)
    predicted_exact = endpoint_distribution(predicted_generator.Q, p0, 50.0)
    terminal_indices = {
        generator.state_index[state.fingerprint] for state in graph.terminal_states
    }
    exact_terminal_mass = float(exact[list(terminal_indices)].sum())
    predicted_terminal_mass = float(
        predicted_exact[list(terminal_indices)].sum()
    )
    sampler = GillespieSampler(
        predicted_generator, max_steps=10, max_time=50.0
    )
    rng = random.Random(seed)
    terminal_counts: dict[str, int] = {}
    event_counts: list[int] = []
    deterministic_replay: list[tuple[str, ...]] = []
    for _ in range(samples):
        trajectory = sampler.sample(source, rng, terminal_check=adapter.is_terminal)
        terminal_counts[trajectory.terminal_fingerprint] = (
            terminal_counts.get(trajectory.terminal_fingerprint, 0) + 1
        )
        event_counts.append(len(trajectory.actions))
        if len(deterministic_replay) < 3:
            deterministic_replay.append(trajectory.actions)
    empirical = {
        key: value / samples for key, value in terminal_counts.items()
    }
    exact_distribution = {
        generator.index_state[index].fingerprint: float(exact[index])
        for index in terminal_indices
    }
    predicted_distribution = {
        generator.index_state[index].fingerprint: float(predicted_exact[index])
        for index in terminal_indices
    }
    keys = set(empirical) | set(exact_distribution)
    empirical_tv = 0.5 * sum(
        abs(empirical.get(key, 0.0) - exact_distribution.get(key, 0.0))
        for key in keys
    )
    analytic_tv = 0.5 * sum(
        abs(
            predicted_distribution.get(key, 0.0)
            - exact_distribution.get(key, 0.0)
        )
        for key in set(predicted_distribution) | set(exact_distribution)
    )
    exact_event_distribution = _event_count_distribution(
        generator, source_idx, terminal_indices
    )
    empirical_event_distribution = {
        count: event_counts.count(count) / samples for count in set(event_counts)
    }
    event_keys = set(exact_event_distribution) | set(
        empirical_event_distribution
    )
    event_count_tv = 0.5 * sum(
        abs(
            exact_event_distribution.get(key, 0.0)
            - empirical_event_distribution.get(key, 0.0)
        )
        for key in event_keys
    )
    legal_pairs = {
        (
            generator.state_index[transition.source.fingerprint],
            generator.state_index[transition.target.fingerprint],
        )
        for transition in graph.transitions
    }
    illegal_rate_sum = sum(
        abs(float(predicted_generator.Q[i, j]))
        for i in range(generator.n_states)
        for j in range(generator.n_states)
        if i != j and (i, j) not in legal_pairs
    )
    return {
        "domain": adapter.domain_id,
        "closed": graph.n_states < 20 and bool(graph.terminal_states),
        "states": graph.n_states,
        "transitions": graph.n_transitions,
        "generator_errors": check_generator(predicted_generator.Q),
        "illegal_edge_rate_sum": illegal_rate_sum,
        "rate_fit": {
            "rows": len(targets),
            "edges": sum(len(target.candidate_ids) for target in targets),
            "initial_mse": rate_loss_history[0],
            "final_mse": rate_loss_history[-1],
            "max_abs_error": max(
                abs(float(value) - expected)
                for values, target in zip(
                    predicted_rows, targets, strict=True
                )
                for value, expected in zip(
                    values.detach(), target.edge_rates, strict=True
                )
            ),
            "source_target_hazard": targets[0].total_hazard,
            "source_predicted_hazard": float(predicted_rows[0].sum().detach()),
        },
        "horizon": 50.0,
        "samples": samples,
        "exact_terminal_mass": exact_terminal_mass,
        "predicted_terminal_mass": predicted_terminal_mass,
        "empirical_terminal_mass": sum(empirical.values()),
        "analytic_endpoint_tv": analytic_tv,
        "empirical_endpoint_tv": empirical_tv,
        "exact_event_count_distribution": {
            str(key): value for key, value in exact_event_distribution.items()
        },
        "empirical_event_count_distribution": {
            str(key): value
            for key, value in empirical_event_distribution.items()
        },
        "event_count_tv": event_count_tv,
        "event_count_min": min(event_counts),
        "event_count_max": max(event_counts),
        "seed_replay_actions": [list(item) for item in deterministic_replay],
    }


def _records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_fixture(
    *,
    corpus_dir: Path = DEFAULT_CORPUS,
    records_path: Path = DEFAULT_RECORDS,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    train_steps: int = 8,
    exact_samples: int = 256,
    max_wall_minutes: float = float(MAX_RUN_MINUTES),
) -> dict[str, Any]:
    if not 0 < max_wall_minutes <= MAX_RUN_MINUTES:
        raise ValueError("max_wall_minutes must be in (0, 3]")
    if not seeds:
        raise ValueError("at least one seed is required")
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if exact_samples <= 0:
        raise ValueError("exact_samples must be positive")
    started = time.monotonic()
    deadline = started + max_wall_minutes * 60
    exact = _exact_oracle(
        seeds[0], exact_samples, deadline=deadline
    )
    rows, candidate_sets, manifest = load_corpus(corpus_dir)
    train_rows = [row for row in rows if row.split == "train"]
    targets = from_bridge_rows(train_rows)
    batch = LegalEditBatch.pack(train_rows, candidate_sets)
    model = LegalEditFlow(LegalEditFlowConfig(enabled=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    history: list[dict[str, float]] = []
    progress = _schedule_progress(train_rows)
    for _ in range(train_steps):
        if time.monotonic() > deadline:
            raise TimeoutError("SLM-199 fixture exceeded max_wall_minutes")
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            batch,
            schedule_progress=progress,
        )
        total, components = legal_edit_flow_losses(prediction, batch, targets)
        total.backward()
        optimizer.step()
        history.append(
            {"total": float(total.detach())}
            | {key: float(value.detach()) for key, value in components.items()}
        )
    with torch.no_grad():
        final_prediction = model(
            batch,
            schedule_progress=progress,
        )
    unknown_rate_mass = float(
        final_prediction.edge_rates[batch.unknown_mask].sum()
        / final_prediction.edge_rates.sum()
    )
    sampler = ProductionLegalEditFlowSampler(model)
    production: list[dict[str, Any]] = []
    records = _records(records_path)
    if not records:
        raise ValueError("at least one production fixture record is required")
    for index, record in enumerate(records):
        if time.monotonic() > deadline:
            raise TimeoutError("SLM-199 fixture exceeded max_wall_minutes")
        seed = seeds[index % len(seeds)]
        trace = sampler.sample(
            record["source_program"],
            RequestEditContractV1.from_dict(record["request_contract"]),
            termination=FixedKPolicy(k=2, max_steps=2),
            max_steps=2,
            seed=seed,
        )
        production.append(
            {
                "record_id": record["id"],
                "verified_output": trace.verified_output,
                "stop_reason": trace.stop_reason,
                "edits": len(trace.decisions),
                "all_selected_live": all(
                    item["selected_candidate_id"] in item["candidate_ids"]
                    for item in trace.decisions
                ),
                "candidate_sets_refreshed": len(
                    {item["candidate_set_digest"] for item in trace.decisions}
                )
                == len(trace.decisions),
                "unknown_candidate_events": trace.unknown_candidate_events,
                "target_exact": trace.final_fingerprint
                == canonical_fingerprint(record["target_program"]),
            }
        )
    elapsed = time.monotonic() - started
    if elapsed > max_wall_minutes * 60:
        raise TimeoutError("SLM-199 fixture exceeded max_wall_minutes")
    return {
        "schema": "SLM199LegalEditFlowReportV1",
        "matrix_set": MATRIX_SET,
        "status": "measured_fixture",
        "claim_class": "wiring",
        "exact_oracle": exact,
        "production_adapter": {
            "fidelity": "adapted_path_approximation",
            "rows": len(rows),
            "train_rows": len(train_rows),
            "dataset_manifest": manifest.get("dataset_id", ""),
            "unknown_supervised_as_negative": any(
                set(target.supervised_candidate_ids) & set(row.unknown_candidate_ids)
                for target, row in zip(targets, train_rows, strict=True)
            ),
            "unknown_rate_mass_after_fit": unknown_rate_mass,
            "unknown_indirect_objective": (
                "UNKNOWN remains in set normalization and receives indirect "
                "ranking pressure, but no direct edge or hazard regression label"
            ),
            "loss_initial": history[0],
            "loss_final": history[-1],
            "samples": production,
        },
        "default_path": {
            "flow_enabled_by_default": LegalEditFlowConfig().enabled,
            "flow_time_encoding": model.config.scorer.time_encoding,
            "existing_direct_policy_modified": False,
        },
        "recipe": {
            "device": "cpu",
            "backend": "torch+numpy exact closed fixture",
            "train_steps": train_steps,
            "seeds": list(seeds),
            "exact_samples": exact_samples,
            "max_wall_minutes": max_wall_minutes,
        },
        "checkpoint": {
            "written": False,
            "reason": "fixture-only evidence; no promotion",
        },
        "honest_verdict": "adapted_time_conditioned_edit_policy_fixture_only",
        "confirmation": {
            "status": "blocked",
            "owner": "VFA1-02",
            "reason": "SLM-199 does not establish a held-out flow win",
        },
        "elapsed_seconds": elapsed,
    }
