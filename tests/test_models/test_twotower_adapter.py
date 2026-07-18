"""Torch-gated tests for TwoTower low-rank adapter attachment (LDI2-01 / SLM-123)."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord  # noqa: E402
from slm_training.models.adapters import TwoTowerAdapterSpec  # noqa: E402
from slm_training.models.adapters.low_rank import LowRankAdapter  # noqa: E402
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel  # noqa: E402

_HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)


def _records() -> list[ExampleRecord]:
    return [ExampleRecord(id="a", prompt="Hero", openui=_HERO, split="train")]


def _model() -> TwoTowerModel:
    return TwoTowerModel.from_records(
        _records(),
        config=TwoTowerConfig(d_model=32, n_heads=4, context_layers=1, denoiser_layers=1),
        device="cpu",
    )


def _spec(model: TwoTowerModel, **overrides) -> TwoTowerAdapterSpec:
    base = dict(
        method="low_rank",
        rank=2,
        alpha=4.0,
        dropout=0.0,
        target_modules=("attn_q", "attn_v"),
        base_compatibility_fingerprint=model.compatibility_fingerprint(),
        base_checkpoint_sha="ckpt",
        tokenizer_sha=model.artifact_identity()["tokenizer_sha"],
    )
    base.update(overrides)
    return TwoTowerAdapterSpec(**base)


def test_attach_resolves_targets_and_freezes_base() -> None:
    model = _model()
    model.attach_adapter(_spec(model))
    assert model.has_adapter()
    # Two targets on the single denoiser layer -> two wrapped linears.
    assert len(model._adapter_modules) == 2
    assert isinstance(model.denoiser.layers[0].self_attn.q_proj, LowRankAdapter)
    # Adapter-only mode: every trainable parameter is an adapter tensor.
    trainable = list(model.trainable_parameters())
    adapter = list(model.adapter_parameters())
    assert trainable and len(trainable) == len(adapter)


def test_fresh_adapter_is_output_identical_at_every_site() -> None:
    model = _model()
    model.attach_adapter(_spec(model))
    model.enable_adapter()
    x = torch.randn(2, 5, 32)  # d_model == 32
    for wrapper in model._adapter_modules.values():
        assert torch.equal(wrapper(x), wrapper.base(x))  # B is zero-initialized


def test_disable_restores_parent_after_weights_change() -> None:
    model = _model()
    model.attach_adapter(_spec(model))
    x = torch.randn(3, 32)
    wrapper = next(iter(model._adapter_modules.values()))
    baseline = wrapper.base(x).clone()
    with torch.no_grad():
        wrapper.lora_B.fill_(0.05)
    model.enable_adapter()
    assert not torch.allclose(wrapper(x), baseline)
    model.disable_adapter()
    assert torch.equal(wrapper(x), baseline)


def test_only_adapter_parameters_receive_gradients() -> None:
    model = _model()
    model.attach_adapter(_spec(model))
    with torch.no_grad():
        for wrapper in model._adapter_modules.values():
            wrapper.lora_B.add_(0.05)
    model.training_loss(_records()).backward()
    base = model.denoiser.layers[0].self_attn.q_proj.base.weight
    assert base.grad is None
    assert any(p.grad is not None for p in model.adapter_parameters())


def test_attach_does_not_shift_training_rng() -> None:
    model = _model()
    spec = _spec(model)
    before = torch.random.get_rng_state()
    model.attach_adapter(spec)
    assert torch.equal(before, torch.random.get_rng_state())


def test_attach_reports_actionable_target_errors() -> None:
    model = _model()
    with pytest.raises(ValueError, match="unsupported adapter target"):
        model.attach_adapter(_spec(model, target_modules=("bogus",)))
    other = _model()
    with pytest.raises(ValueError, match="out of range"):
        other.attach_adapter(_spec(other, target_layer_indices=(9,)))


def test_attach_rejects_mismatched_base_fingerprint() -> None:
    model = _model()
    with pytest.raises(ValueError, match="base compatibility fingerprint"):
        model.attach_adapter(_spec(model, base_compatibility_fingerprint="not-this-model"))


def test_merge_adapter_copy_matches_enabled_and_leaves_original() -> None:
    model = _model()
    model.attach_adapter(_spec(model))
    with torch.no_grad():
        for wrapper in model._adapter_modules.values():
            wrapper.lora_B.add_(0.03)
    model.enable_adapter()

    merged = model.merge_adapter_copy()
    assert not merged.has_adapter()
    merged_linear = merged.denoiser.layers[0].self_attn.q_proj
    assert not isinstance(merged_linear, LowRankAdapter)

    x = torch.randn(2, 5, 32)
    wrapper = model.denoiser.layers[0].self_attn.q_proj
    assert torch.allclose(merged_linear(x), wrapper(x), atol=1e-6)
    # Merge is one-way on a copy: the original still carries its removable adapter.
    assert model.has_adapter()
    assert isinstance(model.denoiser.layers[0].self_attn.q_proj, LowRankAdapter)


def test_merge_requires_an_attached_adapter() -> None:
    with pytest.raises(ValueError, match="no adapter is attached"):
        _model().merge_adapter_copy()


def test_save_and_load_adapter_round_trip(tmp_path) -> None:
    model = _model()
    model.attach_adapter(_spec(model))
    with torch.no_grad():
        for wrapper in model._adapter_modules.values():
            wrapper.lora_B.add_(0.02)
    model.enable_adapter()
    x = torch.randn(3, 32)
    trained = model.denoiser.layers[0].self_attn.q_proj(x).clone()

    model.save_adapter(tmp_path / "adapter")
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    manifest = json.loads((tmp_path / "adapter" / "adapter_manifest.json").read_text())
    assert manifest["trainable_parameter_count"] > 0
    assert manifest["module_map"]

    # A fresh model with the same seed reproduces the adapted logits after loading.
    fresh = _model()
    fresh.load_adapter(tmp_path / "adapter")
    fresh.enable_adapter()
    assert torch.allclose(
        fresh.denoiser.layers[0].self_attn.q_proj(x), trained, atol=1e-6
    )


def test_load_adapter_fails_closed_on_base_mismatch(tmp_path) -> None:
    model = _model()
    model.attach_adapter(_spec(model))
    model.save_adapter(tmp_path / "adapter")
    # A different width yields a different compatibility fingerprint.
    other = TwoTowerModel.from_records(
        _records(),
        config=TwoTowerConfig(d_model=64, n_heads=4, context_layers=1, denoiser_layers=1),
        device="cpu",
    )
    with pytest.raises(ValueError, match="fingerprint"):
        other.load_adapter(tmp_path / "adapter")


def test_active_adapter_identity_tracks_adapter_weights() -> None:
    model = _model()
    assert model.active_adapter_identity() == ""
    model.attach_adapter(_spec(model))
    before = model.active_adapter_identity()
    assert before != ""
    with torch.no_grad():
        next(iter(model._adapter_modules.values())).lora_B.add_(1.0)
    assert model.active_adapter_identity() != before
