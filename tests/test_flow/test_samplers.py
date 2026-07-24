from __future__ import annotations

import json
import random
from pathlib import Path

from slm_training.data.flow.bridge_corpus import RequestEditContractV1
from slm_training.flow.reference import (
    ExactEnumerator,
    GeneratorBuilder,
    GillespieSampler,
    build_uniform_rate_fn,
)
from slm_training.flow.reference.adapters import ChoiceSequenceAdapter
from slm_training.flow.samplers import (
    ProductionLegalEditFlowSampler,
    sample_exact_reference,
)
from slm_training.flow.termination import FixedKPolicy
from slm_training.models.legal_edit_flow import LegalEditFlow, LegalEditFlowConfig


def _record() -> dict:
    return json.loads(
        Path("tests/fixtures/slm196_legal_edit_bridge/records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )


def test_exact_reference_sampler_is_seed_deterministic() -> None:
    adapter = ChoiceSequenceAdapter(
        productions={"S": [["A", "B"]], "A": [["a"]], "B": [["b"]]},
        max_length=2,
        max_states=20,
    )
    graph = ExactEnumerator(adapter, max_states=20).enumerate()
    generator = GeneratorBuilder(graph).build_dense(build_uniform_rate_fn(1.0))
    sampler = GillespieSampler(generator, max_steps=10, max_time=50.0)
    left = sample_exact_reference(
        sampler,
        graph.initial_states[0],
        random.Random(7),
        terminal_check=adapter.is_terminal,
    )
    right = sample_exact_reference(
        sampler,
        graph.initial_states[0],
        random.Random(7),
        terminal_check=adapter.is_terminal,
    )
    assert left.actions == right.actions
    assert left.holding_times == right.holding_times
    assert len(left.actions) == 5
    assert adapter.is_terminal(
        generator.index_state[generator.state_index[left.terminal_fingerprint]]
    )


def test_production_sampler_refreshes_live_unknowns_and_is_bounded() -> None:
    record = _record()
    sampler = ProductionLegalEditFlowSampler(
        LegalEditFlow(LegalEditFlowConfig(enabled=True))
    )
    kwargs = {
        "termination": FixedKPolicy(k=2, max_steps=2),
        "max_steps": 2,
        "seed": 11,
    }
    left = sampler.sample(
        record["source_program"],
        RequestEditContractV1.from_dict(record["request_contract"]),
        **kwargs,
    )
    right = sampler.sample(
        record["source_program"],
        RequestEditContractV1.from_dict(record["request_contract"]),
        **kwargs,
    )
    assert left.decisions == right.decisions
    assert len(left.decisions) == 2
    assert left.stop_reason == "FIXED_K_END"
    assert left.verified_output
    assert left.unknown_candidate_events > 0
    assert all(
        item["selected_candidate_id"] in item["candidate_ids"]
        for item in left.decisions
    )
    assert (
        left.decisions[0]["candidate_set_digest"]
        != left.decisions[1]["candidate_set_digest"]
    )


def test_production_sampler_abstains_when_final_verifier_rejects() -> None:
    record = _record()
    sampler = ProductionLegalEditFlowSampler(
        LegalEditFlow(LegalEditFlowConfig(enabled=True))
    )
    trace = sampler.sample(
        record["source_program"],
        RequestEditContractV1.from_dict(record["request_contract"]),
        termination=FixedKPolicy(k=1, max_steps=1),
        max_steps=1,
        final_verifier=lambda _: False,
    )
    assert not trace.verified_output
    assert trace.stop_reason == "final_verification_unknown"
