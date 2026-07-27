"""SelectiveLatentConnector regression tests (AP-033 / SLM-333)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.models.blocks import DenoiserTower
from slm_training.models.selective_latent_connector import (
    PRECISION_CRITICAL_FACTOR_FAMILIES,
    SOFT_FACTOR_FAMILIES,
    SelectiveLatentArm,
    SelectiveLatentConnector,
    SelectiveLatentTrace,
    resolve_selective_latent_vector,
    split_soft_precision_features,
)
from slm_training.models.semantic_plan_factors import (
    PROGRAM_FACTOR_FAMILIES,
    ProgramFactorTensorizerConfig,
    factor_dims,
)

D_MODEL = 16
OUTPUT_DIM = 40
BATCH = 3


def _connector(input_dim: int = D_MODEL, output_dim: int = OUTPUT_DIM) -> SelectiveLatentConnector:
    return SelectiveLatentConnector(input_dim, output_dim)


def _vector(seed: int = 0, *, dim: int = D_MODEL, requires_grad: bool = False) -> torch.Tensor:
    torch.manual_seed(seed)
    vector = torch.randn(BATCH, dim)
    if requires_grad:
        vector.requires_grad_(True)
    return vector


# --- factor routing -----------------------------------------------------


def test_soft_and_precision_families_partition_program_factor_families() -> None:
    assert set(PRECISION_CRITICAL_FACTOR_FAMILIES) | set(SOFT_FACTOR_FAMILIES) == set(PROGRAM_FACTOR_FAMILIES)
    assert set(PRECISION_CRITICAL_FACTOR_FAMILIES).isdisjoint(SOFT_FACTOR_FAMILIES)


def test_precision_critical_families_are_the_leading_slice() -> None:
    assert tuple(PRECISION_CRITICAL_FACTOR_FAMILIES) == PROGRAM_FACTOR_FAMILIES[: len(PRECISION_CRITICAL_FACTOR_FAMILIES)]


def test_split_soft_precision_features_matches_manual_slice() -> None:
    dims = factor_dims(ProgramFactorTensorizerConfig())
    total = sum(dims.values())
    features = torch.arange(BATCH * total, dtype=torch.float32).reshape(BATCH, total)
    precision, soft = split_soft_precision_features(features, dims)
    boundary = sum(dims[name] for name in PRECISION_CRITICAL_FACTOR_FAMILIES)
    assert torch.equal(precision, features[:, :boundary])
    assert torch.equal(soft, features[:, boundary:])
    assert precision.shape[-1] + soft.shape[-1] == total


# --- resolve_selective_latent_vector ------------------------------------


def test_resolve_selective_latent_vector_rejects_discrete_only() -> None:
    with pytest.raises(ValueError):
        resolve_selective_latent_vector(_vector(), arm=SelectiveLatentArm.DISCRETE_ONLY)


@pytest.mark.parametrize(
    "arm",
    [SelectiveLatentArm.CONTINUOUS_ONLY, SelectiveLatentArm.SELECTIVE_HYBRID, SelectiveLatentArm.ALL_CONTINUOUS],
)
def test_passthrough_arms_return_the_same_content(arm: SelectiveLatentArm) -> None:
    vector = _vector()
    resolved = resolve_selective_latent_vector(vector, arm=arm)
    assert torch.equal(resolved, vector)


def test_oracle_continuous_returns_the_oracle_vector() -> None:
    vector = _vector(seed=0)
    oracle = _vector(seed=1)
    resolved = resolve_selective_latent_vector(vector, arm=SelectiveLatentArm.ORACLE_CONTINUOUS, oracle_vector=oracle)
    assert torch.equal(resolved, oracle)
    assert not torch.equal(resolved, vector)


def test_oracle_continuous_requires_oracle_vector() -> None:
    with pytest.raises(ValueError):
        resolve_selective_latent_vector(_vector(), arm=SelectiveLatentArm.ORACLE_CONTINUOUS)


def test_oracle_continuous_rejects_shape_mismatch() -> None:
    vector = _vector(dim=D_MODEL)
    oracle = _vector(dim=D_MODEL + 1)
    with pytest.raises(ValueError):
        resolve_selective_latent_vector(vector, arm=SelectiveLatentArm.ORACLE_CONTINUOUS, oracle_vector=oracle)


def test_random_continuous_ignores_input_content_and_is_seed_reproducible() -> None:
    vector = _vector()
    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)
    a = resolve_selective_latent_vector(vector, arm=SelectiveLatentArm.RANDOM_CONTINUOUS, generator=generator_a)
    b = resolve_selective_latent_vector(vector, arm=SelectiveLatentArm.RANDOM_CONTINUOUS, generator=generator_b)
    assert torch.equal(a, b)
    assert not torch.equal(a, vector)


def test_resolve_selective_latent_vector_rejects_mismatched_generator_device() -> None:
    vector = _vector()
    cpu_generator = torch.Generator(device="cpu").manual_seed(1)
    vector_meta = vector.to("meta")
    with pytest.raises(ValueError):
        resolve_selective_latent_vector(
            vector_meta, arm=SelectiveLatentArm.RANDOM_CONTINUOUS, generator=cpu_generator
        )


# --- SelectiveLatentConnector --------------------------------------------


def test_bias_for_vocab_matches_full_gather_for_candidate_ids() -> None:
    connector = _connector()
    with torch.no_grad():
        connector.gate.fill_(1.3)
    vector = _vector()
    candidate_ids = torch.tensor([1, 5, 9, OUTPUT_DIM - 1])

    full = connector.bias_for_vocab(vector)
    gathered = connector.bias_for_vocab(vector, candidate_ids=candidate_ids)
    assert full.shape == (BATCH, OUTPUT_DIM)
    assert torch.equal(gathered, full.index_select(-1, candidate_ids))


def test_gate_initialized_to_zero_is_a_numerical_no_op() -> None:
    connector = _connector()
    vector = _vector()
    bias = connector.bias_for_vocab(vector)
    assert torch.equal(bias, torch.zeros_like(bias))


def test_selective_latent_trace_to_dict_round_trips_fields() -> None:
    trace = SelectiveLatentTrace(arm="selective_hybrid", gate_value=0.5, bias_norm=1.25)
    assert trace.to_dict() == {"arm": "selective_hybrid", "gate_value": 0.5, "bias_norm": 1.25}


# --- DenoiserTower composability (structural drop-in, no twotower.py/blocks.py edits) --


def _tower() -> DenoiserTower:
    torch.manual_seed(0)
    return DenoiserTower(vocab_size=OUTPUT_DIM, d_model=D_MODEL, n_layers=1, n_heads=2, max_len=12)


def _hidden(tower: DenoiserTower) -> torch.Tensor:
    ids = torch.randint(0, OUTPUT_DIM, (BATCH, 6))
    ctx = torch.randn(BATCH, 4, D_MODEL)
    return tower.encode(ids, ctx, pad_id=0)


def test_project_is_byte_identical_when_no_selective_connector_attached() -> None:
    """discrete_only == the connector is never constructed/attached at all --
    the hard "Off/discrete-only mode remains bit-exact" acceptance gate.
    """
    tower = _tower()
    hidden = _hidden(tower)
    baseline = tower.project(hidden)
    again = tower.project(hidden)
    assert torch.equal(baseline, again)
    assert tower.pop_plan_connector_traces() == []


def test_project_is_byte_identical_when_selective_vector_cleared_after_use() -> None:
    tower = _tower()
    hidden = _hidden(tower)
    baseline = tower.project(hidden)

    connector = SelectiveLatentConnector(D_MODEL, OUTPUT_DIM)
    with torch.no_grad():
        connector.gate.fill_(2.0)
    tower.set_plan_connector(connector, arm="selective_hybrid")
    tower.set_plan_vector(torch.randn(BATCH, D_MODEL))
    conditioned = tower.project(hidden)
    assert not torch.equal(baseline, conditioned)

    tower.set_plan_vector(None)
    after_clear = tower.project(hidden)
    assert torch.equal(baseline, after_clear)


def test_selective_latent_connector_is_a_structural_drop_in_for_set_plan_connector() -> None:
    """Proves the issue's 'condition TwoTower/MaskGIT through a gated
    continuous connector' requirement: SelectiveLatentConnector attaches via
    the exact same DenoiserTower.set_plan_connector/project path AP-024's
    AbstractPlanConnector uses, with zero changes to blocks.py/twotower.py --
    duck-typed on (gate, bias_for_vocab), never isinstance-checked there.
    """
    tower = _tower()
    hidden = _hidden(tower)

    connector = SelectiveLatentConnector(D_MODEL, OUTPUT_DIM)
    with torch.no_grad():
        connector.gate.fill_(50.0)  # huge gate: bias should dominate and flip every argmax
    tower.set_plan_connector(connector, arm="oracle_continuous")
    tower.set_plan_vector(torch.randn(BATCH, D_MODEL))

    tower.project(hidden)
    tower.project(hidden)
    traces = tower.pop_plan_connector_traces()
    assert len(traces) == 2
    for trace in traces:
        assert trace.arm == "oracle_continuous"
        assert 0 <= trace.choice_changed_count <= trace.position_count
    assert tower.pop_plan_connector_traces() == []
