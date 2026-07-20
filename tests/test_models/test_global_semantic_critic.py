"""SLM-150 (SPV2-02): global semantic energy/value critic tests."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.models.global_semantic_critic import (
    CRITIC_SCHEMA_VERSION,
    GlobalSemanticCritic,
    GlobalSemanticCriticConfig,
    SemanticEnergyOutput,
    coverage_contract_heuristic,
    global_critic_factor_loss,
    global_critic_listwise_loss,
    global_critic_pairwise_loss,
    make_fixture_examples,
    rerank_candidates,
)


def _minimal_features(component_count: int = 3) -> dict[str, float]:
    return {
        "component_count": float(component_count),
        "depth": 2.0,
        "binding_count": 1.0,
        "role_count": 2.0,
    }


def _prompt_context(pack_id: str = "openui") -> dict:
    return {
        "pack_id": pack_id,
        "n_mentioned_components": 3,
    }


def _semantic_plan() -> dict:
    return {"plan_steps": 2, "coverage_ratio": 0.75}


def _contract_features(required: int = 4) -> dict:
    return {"required_component_count": required}


def _make_critic(seed: int = 0, threshold: float = 0.0) -> GlobalSemanticCritic:
    return GlobalSemanticCritic(
        config=GlobalSemanticCriticConfig(seed=seed, confidence_threshold=threshold),
        device="cpu",
    )


def test_default_config_values() -> None:
    cfg = GlobalSemanticCriticConfig()
    assert cfg.d_model == 64
    assert cfg.hidden_dim == 64
    assert cfg.num_factors == 5
    assert cfg.scorer_id == "global-semantic-critic-v1"
    assert cfg.supported_packs == ("openui",)


def test_energy_value_sign_convention() -> None:
    critic = _make_critic()
    out = critic.score(
        _prompt_context(),
        _semantic_plan(),
        _minimal_features(),
        _contract_features(),
    )
    assert math.isfinite(out.energy)
    assert out.value == pytest.approx(-out.energy)


def test_forward_returns_required_keys() -> None:
    critic = _make_critic()
    features = critic.featurize(
        _prompt_context(),
        _semantic_plan(),
        _minimal_features(),
        _contract_features(),
    )
    batch = features.unsqueeze(0)
    outputs = critic.forward(batch)
    assert set(outputs) == {"energy", "value", "factor_energies", "confidence"}
    assert outputs["energy"].shape == (1,)
    assert outputs["factor_energies"]["coverage"].shape == (1,)
    assert outputs["confidence"].shape == (1,)


def test_pairwise_loss_prefers_positive_lower_energy() -> None:
    # Positive has higher energy than negative -> should be penalized.
    energies = torch.tensor([2.0, 0.0], requires_grad=True)
    labels = torch.tensor([1, 0])
    groups = ["g", "g"]
    loss = global_critic_pairwise_loss(energies, labels, groups, margin=0.1)
    assert float(loss) > 0.0

    # Correctly ordered: positive lower energy -> no penalty.
    good = torch.tensor([0.0, 2.0])
    loss_good = global_critic_pairwise_loss(good, labels, groups, margin=0.1)
    assert float(loss_good) == pytest.approx(0.0, abs=1e-6)


def test_pairwise_loss_skips_unknown_and_ties() -> None:
    energies = torch.tensor([0.0, 0.0, 5.0])
    labels = torch.tensor([1, -1, 0])
    groups = ["g", "g", "g"]
    # Unknown is skipped; energies[0] == energies[1] but [1] is UNKNOWN, so the
    # only valid comparison is pos(0) vs neg(2), which is correctly ordered.
    loss = global_critic_pairwise_loss(energies, labels, groups, margin=0.1)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)

    # Energy tie between positive and negative should be skipped.
    tied = torch.tensor([1.0, 1.0])
    loss_tied = global_critic_pairwise_loss(
        tied, torch.tensor([1, 0]), ["g", "g"], margin=0.1
    )
    assert float(loss_tied) == pytest.approx(0.0, abs=1e-6)


def test_listwise_loss_handles_multiple_positives_and_unknown() -> None:
    energies = torch.tensor([0.0, 0.0, 5.0, 1.0], requires_grad=True)
    labels = torch.tensor([1, 1, -1, 0])
    groups = ["g"] * 4
    loss = global_critic_listwise_loss(energies, labels, groups)
    assert math.isfinite(float(loss))

    # UNKNOWN row should not influence the loss.
    energies2 = energies.detach().clone()
    energies2[2] = 100.0
    loss2 = global_critic_listwise_loss(energies2, labels, groups)
    assert torch.isclose(loss, loss2)


def test_factor_loss_masked_mse() -> None:
    pred = {
        "coverage": torch.tensor([0.0, 2.0, 4.0]),
        "roles": torch.tensor([1.0, 3.0, 5.0]),
    }
    targets = {"coverage": 2.0, "roles": 3.0}
    mask = torch.tensor([False, True, False])
    loss = global_critic_factor_loss(pred, targets, mask)
    expected = ((2.0 - 2.0) ** 2 + (3.0 - 3.0) ** 2) / 2
    assert float(loss) == pytest.approx(expected, abs=1e-6)


def test_featurize_ignores_gold_evaluator_keys() -> None:
    critic = _make_critic()
    safe = critic.featurize(
        {"safe": 1.0, "gold_label": 999, "target_ast": 999, "judge_score": 999},
        {},
        {},
        {},
    )
    without = critic.featurize({"safe": 1.0}, {}, {}, {})
    assert torch.equal(safe, without)


def test_batch_score_equivalence() -> None:
    critic = _make_critic(seed=7)
    ctx = _prompt_context()
    plan = _semantic_plan()
    ast = _minimal_features()
    contract = _contract_features()
    single = critic.score(ctx, plan, ast, contract)

    features = critic.featurize(ctx, plan, ast, contract)
    batch = critic.forward(features.unsqueeze(0))
    assert single.energy == pytest.approx(float(batch["energy"].item()), abs=1e-5)
    assert single.confidence == pytest.approx(
        float(batch["confidence"].item()), abs=1e-5
    )
    for name in critic.FACTOR_NAMES:
        assert single.factor_energies[name] == pytest.approx(
            float(batch["factor_energies"][name].item()), abs=1e-5
        )


def test_unsupported_pack_abstains() -> None:
    critic = _make_critic()
    out = critic.score(
        {"pack_id": "unsupported"},
        _semantic_plan(),
        _minimal_features(),
        _contract_features(),
    )
    assert out.abstained
    assert out.reason_code == "unsupported_pack"
    assert out.confidence == 0.0


def test_low_confidence_abstains() -> None:
    critic = _make_critic(threshold=1.0)
    out = critic.score(
        _prompt_context(),
        _semantic_plan(),
        _minimal_features(),
        _contract_features(),
    )
    assert out.abstained
    assert out.reason_code == "low_confidence"


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    critic = _make_critic(seed=42)
    path = tmp_path / "critic.pt"
    critic.save(path)

    loaded = GlobalSemanticCritic.from_checkpoint(path, device="cpu")
    assert loaded.config == critic.config
    for p1, p2 in zip(critic.parameters(), loaded.parameters()):
        torch.testing.assert_close(p1, p2)

    identity = loaded.artifact_identity()
    assert identity["schema"] == CRITIC_SCHEMA_VERSION
    assert identity["scorer_id"] == critic.config.scorer_id


def test_checkpoint_schema_fail_closed(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    torch.save({"schema": "other/v1"}, bad)
    with pytest.raises(ValueError, match="schema mismatch"):
        GlobalSemanticCritic.from_checkpoint(bad)


def test_load_config_mismatch_fail_closed(tmp_path: Path) -> None:
    critic = _make_critic(seed=1)
    path = tmp_path / "critic.pt"
    critic.save(path)

    other = GlobalSemanticCritic(
        config=GlobalSemanticCriticConfig(seed=2), device="cpu"
    )
    with pytest.raises(ValueError, match="config does not match"):
        other.load(path)


def test_factor_heads_do_not_change_energy_or_value() -> None:
    critic = _make_critic(seed=5)
    ctx = _prompt_context()
    plan = _semantic_plan()
    ast = _minimal_features()
    contract = _contract_features()

    before = critic.score(ctx, plan, ast, contract)
    with torch.no_grad():
        for param in critic.factor_heads.parameters():
            param.fill_(1e6)
    after = critic.score(ctx, plan, ast, contract)

    assert after.energy == pytest.approx(before.energy, abs=1e-6)
    assert after.value == pytest.approx(before.value, abs=1e-6)


def test_coverage_contract_heuristic() -> None:
    assert coverage_contract_heuristic(
        {"component_count": 5}, {"required_component_count": 5}
    ) == pytest.approx(0.0)
    assert coverage_contract_heuristic(
        {"component_count": 2}, {"required_component_count": 5}
    ) == pytest.approx(0.6)
    assert coverage_contract_heuristic(
        {"component_count": 7}, {"required_component_count": 5}
    ) == pytest.approx(0.0)


def test_fixture_examples_deterministic_and_varied() -> None:
    examples = make_fixture_examples(n_groups=8, candidates_per_group=4, seed=0)
    examples2 = make_fixture_examples(n_groups=8, candidates_per_group=4, seed=0)
    assert examples == examples2
    assert len(examples) == 32
    labels = [ex.label for ex in examples]
    assert labels.count(1) > 1, "expected multiple positives"
    assert labels.count(-1) > 0, "expected at least one UNKNOWN"
    families = {ex.family for ex in examples}
    assert len(families) > 1


def test_rerank_preserves_candidate_set_and_is_deterministic() -> None:
    critic = _make_critic(seed=3, threshold=0.0)
    candidates = [
        {"candidate_id": "a", "component_count": 2.0, "depth": 1.0},
        {"candidate_id": "b", "component_count": 4.0, "depth": 3.0},
        {"candidate_id": "c", "component_count": 1.0, "depth": 2.0},
    ]
    order1, best1, _ = rerank_candidates(
        candidates, critic, _prompt_context(), _semantic_plan(), _contract_features()
    )
    order2, best2, _ = rerank_candidates(
        candidates, critic, _prompt_context(), _semantic_plan(), _contract_features()
    )
    assert set(order1) == {"a", "b", "c"}
    assert order1 == order2
    assert isinstance(best1, SemanticEnergyOutput)
    assert not best1.abstained


def test_rerank_lambda_zero_uses_local_scores_only() -> None:
    critic = _make_critic(seed=3, threshold=0.0)
    candidates = [
        {"candidate_id": "high_energy", "component_count": 1.0},
        {"candidate_id": "low_energy", "component_count": 5.0},
    ]
    # Verify the critic actually assigns different energies.
    out_high = critic.score(
        _prompt_context(), _semantic_plan(), candidates[0], _contract_features()
    )
    out_low = critic.score(
        _prompt_context(), _semantic_plan(), candidates[1], _contract_features()
    )
    assert out_high.energy != out_low.energy

    local_scores = {"high_energy": 1.0, "low_energy": 0.1}
    order, _, _ = rerank_candidates(
        candidates,
        critic,
        _prompt_context(),
        _semantic_plan(),
        _contract_features(),
        local_scores=local_scores,
        lambda_global=0.0,
    )
    assert order[0] == "high_energy"


def test_rerank_all_abstained() -> None:
    critic = _make_critic(threshold=1.0)
    candidates = [
        {"candidate_id": "a", "component_count": 2.0},
        {"candidate_id": "b", "component_count": 3.0},
    ]
    order, best, trace = rerank_candidates(
        candidates, critic, _prompt_context(), _semantic_plan(), _contract_features()
    )
    assert order == []
    assert best.abstained
    assert best.reason_code == "all_abstained"
    assert trace["scored_count"] == 0
