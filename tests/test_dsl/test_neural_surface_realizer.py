"""Regression tests for the VSS3-05 neural surface realizer (SLM-73).

The no-model deterministic-fallback path and the authority/kind guards are
torch-free and run here. The model-backed paths (trained generation, constrained
dead-end fallback) require torch and are skipped where torch is unavailable
(``pytest.importorskip("torch")``, this repo's convention); they are CI-run.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.neural_surface_realizer import (
    NeuralSurfaceRealizer,
    NeuralSurfaceRealizerConfig,
)
from slm_training.dsl.opaque_regions import OpaqueRegionBinding
from slm_training.dsl.pack import get_pack
from slm_training.dsl.surface import (
    SurfaceAuthority,
    SurfaceConstraint,
    SurfaceRealizationRequest,
    SurfaceSlot,
    SurfaceSlotKind,
    realize_surface_and_verify,
)

HERO = 'root = Stack([title], "column")\ntitle = TextContent(":hero.title")'


def _openui_binding(region_id: str, value: str) -> OpaqueRegionBinding:
    return OpaqueRegionBinding(region_id=region_id, scalar_value=value)


def _identifier_request(
    slots: tuple[SurfaceSlot, ...],
) -> SurfaceRealizationRequest:
    return SurfaceRealizationRequest(
        pack_id="openui",
        constraint_version="v1",
        semantic_ir_fingerprint="fp",
        slots=slots,
        context={},
    )


def _identifier_slot(
    slot_id: str = "s1",
    symbol: str = "title",
    authority: SurfaceAuthority = SurfaceAuthority.SURFACE_ONLY,
    max_bytes: int = 64,
    reserved: tuple[str, ...] = (),
) -> SurfaceSlot:
    return SurfaceSlot(
        slot_id=slot_id,
        kind=SurfaceSlotKind.INTERNAL_IDENTIFIER,
        authority=authority,
        ast_path=(),
        semantic_symbol_id=symbol,
        opaque_region_id=None,
        constraints=SurfaceConstraint(max_bytes=max_bytes, reserved=reserved),
        current_value_digest=None,
    )


# ---------------------------------------------------------------------------
# Torch-free logic: no-model fallback and authority/kind guards
# ---------------------------------------------------------------------------


def test_neural_realizer_without_model_falls_back_to_deterministic() -> None:
    """With no model, every supported slot falls back to the deterministic baseline."""
    pack = get_pack("openui")
    realizer = NeuralSurfaceRealizer(NeuralSurfaceRealizerConfig(model=None))
    result = realize_surface_and_verify(
        HERO,
        pack=pack,
        realizer=realizer,
        opaque_bindings={
            "openui:content::hero.title": _openui_binding(
                "openui:content::hero.title", ":user.title"
            )
        },
        semantic_ir_fingerprint="fp",
        prior_status="solved",
    )
    assert result.status == "solved"
    assert result.source is not None
    assert "v0 = TextContent" in result.source
    assignment = next(
        a for a in result.assignments if a.slot_id == "openui:binder:title"
    )
    assert assignment.value == "v0"
    assert assignment.provenance.startswith(
        "autoregressive_fallback:no_model:deterministic:canonical_name"
    )


def test_neural_realizer_disabled_fallback_raises() -> None:
    """When fallback is disabled, a realization failure is a hard error."""
    slot = _identifier_slot()
    request = _identifier_request((slot,))
    realizer = NeuralSurfaceRealizer(
        NeuralSurfaceRealizerConfig(model=None, fallback_to_deterministic=False)
    )
    with pytest.raises(ValueError, match="fallback is disabled"):
        realizer.realize(request)


def test_neural_realizer_rejects_semantic_authority() -> None:
    """Only SURFACE_ONLY slots may be handed to the AR model."""
    slot = _identifier_slot(authority=SurfaceAuthority.SEMANTIC)
    request = _identifier_request((slot,))
    realizer = NeuralSurfaceRealizer(NeuralSurfaceRealizerConfig(model=None))
    with pytest.raises(ValueError, match="only SURFACE_ONLY"):
        realizer.realize(request)


def test_neural_realizer_rejects_unsupported_kind() -> None:
    """STRUCTURED_STRING and other non-surface kinds are rejected before generation."""
    slot = SurfaceSlot(
        slot_id="s1",
        kind=SurfaceSlotKind.STRUCTURED_STRING,
        authority=SurfaceAuthority.SURFACE_ONLY,
        ast_path=(),
        semantic_symbol_id=None,
        opaque_region_id=None,
        constraints=SurfaceConstraint(),
        current_value_digest=None,
    )
    request = _identifier_request((slot,))
    realizer = NeuralSurfaceRealizer(NeuralSurfaceRealizerConfig(model=None))
    with pytest.raises(ValueError, match="not supported by the autoregressive"):
        realizer.realize(request)


# ---------------------------------------------------------------------------
# Model-backed paths (torch-only; CI-run)
# ---------------------------------------------------------------------------


def _tiny_ar_config():
    from slm_training.models.surface_autoregressor import SurfaceAutoregressorConfig

    return SurfaceAutoregressorConfig(d_model=32, n_layers=1, n_heads=2, max_len=64)


def _trained_neural_realizer(
    prompt_target_pairs: list[tuple[str, str]],
) -> NeuralSurfaceRealizer:
    """Train a tiny fixture model and wrap it in the neural realizer."""
    from slm_training.models.surface_autoregressor import (
        SurfaceAutoregressor,
        train_surface_autoregressor,
    )

    model = SurfaceAutoregressor(_tiny_ar_config())
    train_surface_autoregressor(
        model, prompt_target_pairs, steps=500, lr=5e-3, seed=0
    )
    model.eval()
    return NeuralSurfaceRealizer(
        NeuralSurfaceRealizerConfig(model=model, fallback_to_deterministic=True)
    )


def test_neural_realizer_trained_identifier_is_verified() -> None:
    """A trained AR model can realize a binder name through realize_surface_and_verify."""
    pytest.importorskip("torch")
    pack = get_pack("openui")
    prompt = (
        "kind=internal_identifier authority=surface_only "
        "slot_id=openui:binder:title symbol=title max=64"
    )
    realizer = _trained_neural_realizer([(prompt, "title")])
    result = realize_surface_and_verify(
        HERO,
        pack=pack,
        realizer=realizer,
        opaque_bindings={
            "openui:content::hero.title": _openui_binding(
                "openui:content::hero.title", ":user.title"
            )
        },
        semantic_ir_fingerprint="fp",
        prior_status="solved",
    )
    assert result.status == "solved"
    assert result.source is not None
    # The pack canonicalizer renames binders to canonical names, so the final
    # source uses v0; the assignment itself records the model-chosen value.
    assert "v0 = TextContent" in result.source
    assignment = next(
        a for a in result.assignments if a.slot_id == "openui:binder:title"
    )
    assert assignment.value == "title"
    assert assignment.provenance == "autoregressive"


def test_neural_realizer_dead_end_falls_back_to_deterministic() -> None:
    """A constrained dead end triggers per-slot deterministic fallback."""
    pytest.importorskip("torch")
    from slm_training.models.surface_autoregressor import SurfaceAutoregressor

    slot = _identifier_slot(max_bytes=2)
    request = _identifier_request((slot,))
    model = SurfaceAutoregressor(_tiny_ar_config())
    realizer = NeuralSurfaceRealizer(
        NeuralSurfaceRealizerConfig(model=model, fallback_to_deterministic=True)
    )
    assignments = realizer.realize(request)
    assert len(assignments) == 1
    assert assignments[0].value == "v0"
    assert "dead_end" in assignments[0].provenance
