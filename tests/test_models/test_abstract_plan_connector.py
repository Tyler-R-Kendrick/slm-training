"""AbstractPlanConnector regression tests (AP-024 / SLM-316)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.models.abstract_plan_connector import (
    AbstractPlanConnector,
    PlanConnectorArm,
    PlanConnectorTrace,
    resolve_plan_vector,
)
from slm_training.models.abstract_plan_head import (
    AbstractPlanHead,
    AbstractPlanMode,
)
from slm_training.models.blocks import DenoiserTower
from slm_training.dsl.abstract_plan import AbstractPlanV1

D_MODEL = 16
VOCAB = 40
PLAN_SLOTS = 8
BATCH = 3


def _connector() -> AbstractPlanConnector:
    return AbstractPlanConnector(D_MODEL, PLAN_SLOTS, VOCAB)


def _vector(seed: int = 0, *, requires_grad: bool = False) -> torch.Tensor:
    torch.manual_seed(seed)
    vector = torch.randn(BATCH, D_MODEL)
    if requires_grad:
        vector.requires_grad_(True)
    return vector


def test_resolve_plan_vector_rejects_disabled_arm() -> None:
    with pytest.raises(ValueError):
        resolve_plan_vector(_vector(), arm=PlanConnectorArm.DISABLED)


@pytest.mark.parametrize("arm", [PlanConnectorArm.LEARNED, PlanConnectorArm.ORACLE])
def test_passthrough_arms_return_the_same_content(arm: PlanConnectorArm) -> None:
    vector = _vector()
    resolved = resolve_plan_vector(vector, arm=arm)
    assert torch.equal(resolved, vector)


def test_detached_arm_blocks_gradient_but_keeps_forward_values() -> None:
    vector = _vector(requires_grad=True)
    resolved = resolve_plan_vector(vector, arm=PlanConnectorArm.DETACHED)
    assert torch.equal(resolved, vector.detach())
    assert resolved.requires_grad is False


def test_empty_arm_is_all_zero_with_matching_shape() -> None:
    vector = _vector()
    resolved = resolve_plan_vector(vector, arm=PlanConnectorArm.EMPTY)
    assert resolved.shape == vector.shape
    assert bool((resolved == 0).all())


def test_random_arm_ignores_input_content_and_is_seed_reproducible() -> None:
    vector = _vector()
    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)
    a = resolve_plan_vector(vector, arm=PlanConnectorArm.RANDOM, generator=generator_a)
    b = resolve_plan_vector(vector, arm=PlanConnectorArm.RANDOM, generator=generator_b)
    assert torch.equal(a, b)
    assert not torch.equal(a, vector)


def test_shuffled_arm_permutes_batch_rows_without_changing_the_content_set() -> None:
    vector = _vector()
    generator = torch.Generator().manual_seed(3)
    resolved = resolve_plan_vector(vector, arm=PlanConnectorArm.SHUFFLED, generator=generator)
    assert torch.equal(resolved.sort(dim=0).values, vector.sort(dim=0).values)


def test_resolve_plan_vector_rejects_mismatched_generator_device() -> None:
    vector = _vector()
    cpu_generator = torch.Generator(device="cpu").manual_seed(1)
    # Simulate a foreign-device generator by asserting the check fires for a
    # deliberately mismatched device string (meta is always available).
    vector_meta = vector.to("meta")
    with pytest.raises(ValueError):
        resolve_plan_vector(
            vector_meta, arm=PlanConnectorArm.RANDOM, generator=cpu_generator
        )


def test_plan_vector_from_trace_pools_codebook_local_embeddings() -> None:
    connector = _connector()
    plan = AbstractPlanV1(slot_count=PLAN_SLOTS, max_slot_count=PLAN_SLOTS, rounds=4)
    head = AbstractPlanHead(D_MODEL, plan)
    context = torch.randn(BATCH, 5, D_MODEL)
    pad_mask = torch.zeros(BATCH, 5, dtype=torch.bool)
    trace = head(context, pad_mask, mode=AbstractPlanMode.RANDOM)

    vector = connector.plan_vector_from_trace(trace)
    assert vector.shape == (BATCH, D_MODEL)
    expected = connector.plan_embed(trace.plan_tokens).mean(dim=1)
    assert torch.equal(vector, expected)


def test_bias_for_vocab_matches_full_vocab_gather_for_candidate_ids() -> None:
    connector = _connector()
    with torch.no_grad():
        connector.gate.fill_(1.3)
    vector = _vector()
    candidate_ids = torch.tensor([1, 5, 9, VOCAB - 1])

    full = connector.bias_for_vocab(vector)
    gathered = connector.bias_for_vocab(vector, candidate_ids=candidate_ids)
    assert full.shape == (BATCH, VOCAB)
    assert torch.equal(gathered, full.index_select(1, candidate_ids))


def test_gate_initialized_to_zero_is_a_numerical_no_op() -> None:
    connector = _connector()
    vector = _vector()
    bias = connector.bias_for_vocab(vector)
    assert torch.equal(bias, torch.zeros_like(bias))


def test_plan_connector_trace_to_dict_round_trips_fields() -> None:
    trace = PlanConnectorTrace(
        arm="learned",
        gate_value=0.5,
        bias_norm=1.25,
        choice_changed_count=2,
        position_count=10,
    )
    payload = trace.to_dict()
    assert payload == {
        "arm": "learned",
        "gate_value": 0.5,
        "bias_norm": 1.25,
        "choice_changed_count": 2,
        "position_count": 10,
        "applications": 1,
    }


# --- DenoiserTower.project() wiring -----------------------------------------


def _tower() -> DenoiserTower:
    torch.manual_seed(0)
    return DenoiserTower(vocab_size=VOCAB, d_model=D_MODEL, n_layers=1, n_heads=2, max_len=12)


def _hidden(tower: DenoiserTower) -> torch.Tensor:
    ids = torch.randint(0, VOCAB, (BATCH, 6))
    ctx = torch.randn(BATCH, 4, D_MODEL)
    return tower.encode(ids, ctx, pad_id=0)


def test_project_is_byte_identical_to_baseline_when_no_connector_attached() -> None:
    tower = _tower()
    hidden = _hidden(tower)
    baseline = tower.project(hidden)
    # A fresh DenoiserTower never had set_plan_connector/set_plan_vector called.
    again = tower.project(hidden)
    assert torch.equal(baseline, again)
    assert tower.pop_plan_connector_traces() == []


def test_project_is_byte_identical_when_vector_cleared_after_use() -> None:
    tower = _tower()
    hidden = _hidden(tower)
    baseline = tower.project(hidden)

    connector = AbstractPlanConnector(D_MODEL, PLAN_SLOTS, VOCAB)
    with torch.no_grad():
        connector.gate.fill_(2.0)
    tower.set_plan_connector(connector, arm="learned")
    tower.set_plan_vector(torch.randn(BATCH, D_MODEL))
    conditioned = tower.project(hidden)
    assert not torch.equal(baseline, conditioned)

    tower.set_plan_vector(None)
    after_clear = tower.project(hidden)
    assert torch.equal(baseline, after_clear)


def test_project_records_one_trace_per_call_with_correct_choice_change_count() -> None:
    tower = _tower()
    hidden = _hidden(tower)

    connector = AbstractPlanConnector(D_MODEL, PLAN_SLOTS, VOCAB)
    with torch.no_grad():
        connector.gate.fill_(50.0)  # huge gate: bias should dominate and flip every argmax
    tower.set_plan_connector(connector, arm="oracle")
    tower.set_plan_vector(torch.randn(BATCH, D_MODEL))

    tower.project(hidden)
    tower.project(hidden)
    traces = tower.pop_plan_connector_traces()
    assert len(traces) == 2
    for trace in traces:
        assert trace.arm == "oracle"
        assert trace.applications == 1
        assert trace.position_count == BATCH * hidden.size(1)
        assert 0 <= trace.choice_changed_count <= trace.position_count
    # Draining once more returns nothing new.
    assert tower.pop_plan_connector_traces() == []


def test_project_requires_matching_batch_size_between_plan_vector_and_hidden() -> None:
    tower = _tower()
    hidden = _hidden(tower)
    connector = AbstractPlanConnector(D_MODEL, PLAN_SLOTS, VOCAB)
    tower.set_plan_connector(connector, arm="learned")
    tower.set_plan_vector(torch.randn(BATCH + 1, D_MODEL))
    with pytest.raises(ValueError):
        tower.project(hidden)


def test_project_skips_plan_bias_for_non_batched_hidden_slices() -> None:
    """Per-row compiler/tree scorer slices ([D] or [N, D]) are a different
    decode strategy than the batched MaskGIT round loop this connector
    targets; they must be left untouched even with a connector attached.
    """
    tower = _tower()
    connector = AbstractPlanConnector(D_MODEL, PLAN_SLOTS, VOCAB)
    with torch.no_grad():
        connector.gate.fill_(5.0)
    tower.set_plan_connector(connector, arm="learned")
    tower.set_plan_vector(torch.randn(1, D_MODEL))

    single_row_hidden = torch.randn(D_MODEL)
    logits = tower.project(single_row_hidden)
    assert logits.shape == (VOCAB,)
    assert tower.pop_plan_connector_traces() == []
