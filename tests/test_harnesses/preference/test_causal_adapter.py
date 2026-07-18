"""Tests for the LDI1-02 PEFT actuator factory (model-free surface)."""

from __future__ import annotations

import builtins

import pytest

from slm_training.harnesses.preference.causal_adapter import (
    AdapterSpec,
    build_peft_config,
)


def _spec(**overrides: object) -> AdapterSpec:
    kwargs: dict[str, object] = {
        "base_model_id": "tiny/causal",
        "base_model_revision": "abc123",
        "tokenizer_sha": "tsha",
    }
    kwargs.update(overrides)
    return AdapterSpec(**kwargs)  # type: ignore[arg-type]


def test_lora_defaults_and_fingerprint_are_deterministic() -> None:
    spec = _spec()
    assert spec.method == "lora"
    assert spec.rank == 16 and spec.alpha == 32
    assert spec.fingerprint() == _spec().fingerprint()


def test_fingerprint_changes_with_material_config() -> None:
    assert _spec(rank=8).fingerprint() != _spec(rank=16).fingerprint()
    assert _spec(include_lm_head=True).fingerprint() != _spec().fingerprint()


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown adapter method"):
        _spec(method="bogus")


def test_adalora_is_experimental_and_requires_optin() -> None:
    with pytest.raises(ValueError, match="experimental"):
        _spec(method="adalora")
    assert _spec(method="adalora", allow_experimental=True).method == "adalora"


def test_dora_and_pissa_are_accepted_as_specs() -> None:
    assert _spec(method="dora").method == "dora"
    assert _spec(method="pissa").method == "pissa"


@pytest.mark.parametrize(
    "overrides",
    [{"rank": 0}, {"alpha": 0}, {"dropout": 1.0}, {"dropout": -0.1}, {"target_modules": ()}],
)
def test_invalid_hyperparameters_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _spec(**overrides)


def test_build_peft_config_fails_visibly_without_the_hf_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force peft to be missing so the visible-fail branch runs deterministically
    # regardless of whether the [hf] extra happens to be installed.
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "peft":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RuntimeError, match=r"slm-training\[hf\]"):
        build_peft_config(_spec(method="dora"))
