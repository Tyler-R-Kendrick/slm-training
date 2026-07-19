"""Factory / config round-trip tests for the TwoTower removable adapter (SLM-123)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord  # noqa: E402
from slm_training.harnesses.model_build.config import ModelBuildConfig  # noqa: E402
from slm_training.harnesses.model_build.factory import build_model  # noqa: E402
from slm_training.models.adapters import TwoTowerAdapterSpec  # noqa: E402
from slm_training.models.twotower import TwoTowerModel  # noqa: E402

_HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)


def _records() -> list[ExampleRecord]:
    return [ExampleRecord(id="a", prompt="Hero", openui=_HERO, split="train")]


def _tiny_config(**overrides) -> ModelBuildConfig:
    base = dict(
        train_dir=Path("outputs/data/train/v1"),
        model_name="twotower",
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        context_backend="scratch",
        device="cpu",
    )
    base.update(overrides)
    return ModelBuildConfig(**base)


def _spec(model: TwoTowerModel) -> TwoTowerAdapterSpec:
    return TwoTowerAdapterSpec(
        method="low_rank",
        rank=2,
        alpha=4.0,
        dropout=0.0,
        target_modules=("attn_q", "attn_v"),
        base_compatibility_fingerprint=model.compatibility_fingerprint(),
        base_checkpoint_sha="ckpt",
        tokenizer_sha=model.artifact_identity()["tokenizer_sha"],
    )


def test_factory_loads_adapter_from_spec(tmp_path: Path) -> None:
    records = _records()
    base = build_model(_tiny_config(), records)
    base.attach_adapter(_spec(base))
    with torch.no_grad():
        for wrapper in base._adapter_modules.values():
            wrapper.lora_B.add_(0.03)
    base.save_adapter(tmp_path / "adapter")

    loaded = build_model(
        _tiny_config(adapter_spec=tmp_path / "adapter"),
        records,
    )
    assert loaded.has_adapter()
    trainable = list(loaded.trainable_parameters())
    adapter = list(loaded.adapter_parameters())
    assert len(trainable) == len(adapter)
    assert all(p.requires_grad for p in trainable)


def test_factory_loads_adapter_frozen(tmp_path: Path) -> None:
    records = _records()
    base = build_model(_tiny_config(), records)
    base.attach_adapter(_spec(base))
    base.save_adapter(tmp_path / "adapter")

    loaded = build_model(
        _tiny_config(adapter_spec=tmp_path / "adapter", adapter_trainable=False),
        records,
    )
    assert loaded.has_adapter()
    assert all(not p.requires_grad for p in loaded.adapter_parameters())
    assert list(loaded.trainable_parameters()) == []


def test_factory_adapter_mismatch_fails_closed(tmp_path: Path) -> None:
    records = _records()
    base = build_model(_tiny_config(), records)
    base.attach_adapter(_spec(base))
    base.save_adapter(tmp_path / "adapter")

    with pytest.raises(ValueError, match="fingerprint"):
        build_model(
            _tiny_config(d_model=64, adapter_spec=tmp_path / "adapter"),
            _records(),
            checkpoint=None,
        )
