"""Regression tests for VSS3-05 surface-realization training rows (SLM-73).

The deterministic-baseline derivation is torch-free and bridge-free (main's
``realize_surface_and_verify`` needs neither), so it runs here. The neural-model
derivation requires torch and is skipped where torch is unavailable
(``pytest.importorskip("torch")``, this repo's convention); it is CI-run.
"""

from __future__ import annotations

import pytest

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.progspec.surface_rows import derive_surface_realization_records
from slm_training.dsl.pack import get_pack
from slm_training.dsl.surface import (
    SurfaceAuthority,
    SurfaceSlotKind,
    resolve_surface_slot_extractor,
)

NO_OPAQUE = 'root = Stack([title], "column")\ntitle = Stack([], "row")'


def _spec(spec_id: str, *, split: str = "train") -> ProgramSpec:
    return ProgramSpec.from_openui(
        id=spec_id,
        openui=NO_OPAQUE,
        facts={},
        program_family_id="surf_family",
        lineage_id="surf_lineage",
        split_group_id="surf_group",
        split=split,
    )


# ---------------------------------------------------------------------------
# Deterministic baseline (torch-free, bridge-free)
# ---------------------------------------------------------------------------


def test_derive_surface_realization_records_emits_surface_only_slots() -> None:
    spec = _spec("surface_det")
    records = derive_surface_realization_records(spec)
    assert len(records) == 1
    record = records[0]
    assert record.meta["task"] == "surface_realization"
    assert record.source == "surface_realization"
    assert record.split == spec.split
    assert record.meta["split_group_id"] == spec.split_group_id
    assert record.meta["parent_id"] == spec.id
    assert record.meta["determinacy"] == "deterministic"
    assert record.meta["tier"] == "Silver"
    provenance = record.meta["provenance"]
    assert provenance["surface_slot_id"] == "openui:binder:title"
    assert provenance["surface_assignment"]["value"] == "v0"
    assert "deterministic" in provenance["surface_assignment"]["provenance"]


def test_derive_surface_realization_records_honors_include_authorities() -> None:
    spec = _spec("surface_auth")
    # An empty frozenset is falsy, so the function falls back to defaults; use a
    # non-matching authority to prove the filter is honored.
    assert (
        derive_surface_realization_records(
            spec,
            include_authorities=frozenset({"semantic"}),
        )
        == ()
    )
    assert (
        len(
            derive_surface_realization_records(
                spec,
                include_authorities=frozenset({SurfaceAuthority.SURFACE_ONLY.value}),
            )
        )
        == 1
    )
    assert (
        derive_surface_realization_records(
            spec,
            include_authorities=frozenset({SurfaceAuthority.OPAQUE_USER_VALUE.value}),
        )
        == ()
    )


def test_derive_surface_realization_records_inherits_split_and_group() -> None:
    spec = _spec("surface_split", split="held_out")
    records = derive_surface_realization_records(spec)
    assert len(records) == 1
    assert records[0].split == "held_out"
    assert records[0].meta["split_group_id"] == "surf_group"


# ---------------------------------------------------------------------------
# Neural realizer derivation (torch-only; CI-run)
# ---------------------------------------------------------------------------


def _trained_neural_realizer_for_spec(spec: ProgramSpec):
    """Train a tiny fixture model to emit a custom name for the spec's binder."""
    from slm_training.dsl.neural_surface_realizer import (
        NeuralSurfaceRealizer,
        NeuralSurfaceRealizerConfig,
    )
    from slm_training.models.surface_autoregressor import (
        SurfaceAutoregressor,
        SurfaceAutoregressorConfig,
        train_surface_autoregressor,
    )

    pack = get_pack("openui")
    extractor = resolve_surface_slot_extractor(pack)
    slots = extractor(spec.canonical_openui)
    slot = next(
        s
        for s in slots
        if s.authority is SurfaceAuthority.SURFACE_ONLY
        and s.kind is SurfaceSlotKind.INTERNAL_IDENTIFIER
    )
    realizer = NeuralSurfaceRealizer(NeuralSurfaceRealizerConfig())
    prompt = realizer._build_prompt(slot, set())
    model = SurfaceAutoregressor(
        SurfaceAutoregressorConfig(d_model=32, n_layers=1, n_heads=2, max_len=64)
    )
    train_surface_autoregressor(model, [(prompt, "mytitle")], steps=300, lr=5e-3, seed=0)
    model.eval()
    return NeuralSurfaceRealizer(
        NeuralSurfaceRealizerConfig(model=model, fallback_to_deterministic=True)
    )


def test_derive_surface_realization_records_can_use_neural_realizer() -> None:
    pytest.importorskip("torch")
    spec = _spec("surface_ar")
    realizer = _trained_neural_realizer_for_spec(spec)
    records = derive_surface_realization_records(spec, realizer=realizer)
    assert len(records) == 1
    provenance = records[0].meta["provenance"]
    assert provenance["surface_assignment"]["value"] == "mytitle"
    assert provenance["surface_assignment"]["provenance"] == "autoregressive"
