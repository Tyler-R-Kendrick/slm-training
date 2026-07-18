"""Torch-gated tests for LDI4-01 low-rank representation interventions (SLM-134).

Pin the intervention math and lifecycle: identity at init / no_intervention,
gradient correctness, only-declared params trainable, DiffMean train-only vector,
save/load round-trip, fail-closed config, and the matched arm set. No model
training runs. Deferred to CI (torch present) via ``importorskip``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.models.reft_intervention import (  # noqa: E402
    DiffMeanIntervention,
    InterventionSpec,
    LowRankReft,
    build_intervention,
    diffmean_vector,
    load_intervention,
    matched_arm_specs,
    save_intervention,
)


def _spec(method="reft_r1", **kw) -> InterventionSpec:
    base = dict(method=method, site="denoiser.block.3", hidden_size=6, base_checkpoint_sha="sha")
    base.update(kw)
    return InterventionSpec(**base)


def test_no_intervention_and_reft_init_are_identity() -> None:
    h = torch.randn(4, 6)
    ident = build_intervention(_spec(method="no_intervention"))
    assert torch.equal(ident(h), h)  # bit-identical parent
    reft = LowRankReft(_spec(method="reft_low_rank", rank=3))
    assert torch.allclose(reft(h), h, atol=1e-6)  # W == R, b == 0 -> zero edit


def test_reft_edits_after_perturbation_and_gradcheck() -> None:
    reft = LowRankReft(_spec(method="reft_low_rank", rank=2)).double()
    with torch.no_grad():
        reft.W.add_(torch.randn_like(reft.W) * 0.3)  # move off identity
        reft.b.add_(torch.randn_like(reft.b) * 0.3)
    h = torch.randn(3, 6, dtype=torch.double)
    assert not torch.allclose(reft(h), h)  # now a real edit
    hg = torch.randn(3, 6, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(reft, (hg,), eps=1e-6, atol=1e-4)


def test_only_reft_params_are_trainable() -> None:
    reft = LowRankReft(_spec(method="reft_r1", rank=1))
    names = {id(p) for p in reft.trainable_parameters()}
    assert names == {id(reft.R), id(reft.W), id(reft.b)}
    assert build_intervention(_spec(method="no_intervention")).trainable_parameters() == []


def test_diffmean_vector_is_train_only_and_applies() -> None:
    pos = torch.ones(5, 6)
    neg = torch.zeros(7, 6)
    v = diffmean_vector(pos, neg)
    assert torch.allclose(v, torch.ones(6))  # mean(pos) - mean(neg)
    dm = DiffMeanIntervention(_spec(method="diffmean_fixed"), v)
    h = torch.randn(2, 6)
    assert torch.allclose(dm(h), h + v)
    assert dm.trainable_parameters() == []  # non-trainable control


def test_save_load_reproduces_intervention_logits(tmp_path) -> None:
    reft = LowRankReft(_spec(method="reft_low_rank", rank=2))
    with torch.no_grad():
        reft.W.add_(torch.randn_like(reft.W) * 0.5)
    h = torch.randn(3, 6)
    before = reft(h)
    manifest = save_intervention(reft, tmp_path)
    assert manifest["trainable_parameters"] > 0
    restored = load_intervention(tmp_path)
    assert torch.allclose(restored(h), before, atol=1e-6)


def test_diffmean_save_load_roundtrip(tmp_path) -> None:
    v = diffmean_vector(torch.ones(3, 6) * 2, torch.zeros(3, 6))
    dm = build_intervention(_spec(method="diffmean_fixed"), diffmean=v)
    h = torch.randn(2, 6)
    before = dm(h)
    save_intervention(dm, tmp_path)
    restored = load_intervention(tmp_path)
    assert torch.allclose(restored(h), before, atol=1e-6)


def test_spec_fails_closed() -> None:
    with pytest.raises(ValueError):
        _spec(method="bogus")
    with pytest.raises(ValueError):
        _spec(method="reft_r1", rank=2)  # r1 requires rank 1
    with pytest.raises(ValueError):
        _spec(method="reft_low_rank", rank=99)  # > hidden_size
    with pytest.raises(ValueError):
        _spec(scale=-1.0)
    # diffmean without a vector fails closed
    with pytest.raises(ValueError):
        build_intervention(_spec(method="diffmean_fixed"))


def test_matched_arm_specs_vary_only_actuator() -> None:
    arms = matched_arm_specs(site="s", hidden_size=8, base_checkpoint_sha="sha", ranks=(1, 4, 8))
    assert "R0_parent" in arms and "R2_diffmean" in arms and "R3_reft_r1" in arms
    assert "R4_reft_r4" in arms and "R4_reft_r8" in arms
    assert arms["R0_parent"].method == "no_intervention"
    assert arms["R3_reft_r1"].rank == 1
    # every arm shares the same site/hidden/base identity
    assert {a.site for a in arms.values()} == {"s"}
    assert {a.base_checkpoint_sha for a in arms.values()} == {"sha"}
