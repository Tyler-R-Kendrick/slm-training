"""Regression tests for VSS3-04 surface realization (SLM-72)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from slm_training.data.contract import CallerContentBinding, GenerationRequest
from slm_training.dsl.opaque_regions import (
    OpaqueRegionBinding,
    realize_opaque_regions,
)
from slm_training.dsl.pack import get_pack
from slm_training.dsl.surface import (
    DeterministicSurfaceRealizer,
    SurfaceAssignment,
    SurfaceAuthority,
    SurfaceConstraint,
    SurfaceRealizationRequest,
    SurfaceSlot,
    SurfaceSlotKind,
    canonicalize_input,
    realize_surface_and_verify,
    resolve_surface_slot_extractor,
    resolve_verified_template_bindings,
)

HERO = 'root = Stack([title], "column")\ntitle = TextContent(":hero.title")'
BOUND_HERO = 'root = Stack([title], "column")\ntitle = TextContent(":slot_0")'
NO_OPAQUE = 'root = Stack([title], "column")\ntitle = Stack([], "row")'


def _openui_binding(region_id: str, value: str) -> OpaqueRegionBinding:
    return OpaqueRegionBinding(region_id=region_id, scalar_value=value)


# ---------------------------------------------------------------------------
# Surface slot extraction
# ---------------------------------------------------------------------------


def test_openui_pack_surface_slot_extractor() -> None:
    pack = get_pack("openui")
    extractor = resolve_surface_slot_extractor(pack)
    slots = extractor(HERO)
    assert len(slots) == 2
    binder_slot = next(s for s in slots if s.kind is SurfaceSlotKind.INTERNAL_IDENTIFIER)
    content_slot = next(s for s in slots if s.authority is SurfaceAuthority.OPAQUE_USER_VALUE)
    assert binder_slot.slot_id == "openui:binder:title"
    assert binder_slot.semantic_symbol_id == "title"
    assert binder_slot.authority is SurfaceAuthority.SURFACE_ONLY
    assert content_slot.slot_id == "openui:content::hero.title"
    assert content_slot.opaque_region_id == "openui:content::hero.title"


def test_root_binder_is_not_surface_only() -> None:
    """The program root is syntactically required to be spelled 'root'."""
    pack = get_pack("openui")
    slots = resolve_surface_slot_extractor(pack)(HERO)
    assert not any(s.semantic_symbol_id == "root" for s in slots)


def test_unknown_fields_default_to_not_extracted() -> None:
    """Component names, property keys, and operators remain semantic by omission."""
    pack = get_pack("openui")
    slots = resolve_surface_slot_extractor(pack)(HERO)
    slot_kinds = {s.kind for s in slots}
    assert SurfaceSlotKind.STRUCTURED_STRING not in slot_kinds
    assert SurfaceSlotKind.EXTERNALLY_OBSERVABLE_NAME not in slot_kinds


# ---------------------------------------------------------------------------
# Deterministic realizer
# ---------------------------------------------------------------------------


def test_deterministic_realizer_assigns_canonical_binder_names() -> None:
    slot = SurfaceSlot(
        slot_id="s1",
        kind=SurfaceSlotKind.INTERNAL_IDENTIFIER,
        authority=SurfaceAuthority.SURFACE_ONLY,
        ast_path=(),
        semantic_symbol_id="title",
        opaque_region_id=None,
        constraints=SurfaceConstraint(),
        current_value_digest=None,
    )
    request = SurfaceRealizationRequest(
        pack_id="openui",
        constraint_version="v1",
        semantic_ir_fingerprint="fp",
        slots=(slot,),
        context={},
    )
    realizer = DeterministicSurfaceRealizer()
    assignments = realizer.realize(request)
    assert len(assignments) == 1
    assert assignments[0].value == "v0"
    assert assignments[0].provenance == "deterministic:canonical_name"


def test_deterministic_realizer_rejects_structured_and_observable_slots() -> None:
    for kind in (
        SurfaceSlotKind.STRUCTURED_STRING,
        SurfaceSlotKind.EXTERNALLY_OBSERVABLE_NAME,
    ):
        slot = SurfaceSlot(
            slot_id="s1",
            kind=kind,
            authority=SurfaceAuthority.SURFACE_ONLY,
            ast_path=(),
            semantic_symbol_id=None,
            opaque_region_id=None,
            constraints=SurfaceConstraint(),
            current_value_digest=None,
        )
        request = SurfaceRealizationRequest(
            pack_id="openui",
            constraint_version="v1",
            semantic_ir_fingerprint="fp",
            slots=(slot,),
            context={},
        )
        with pytest.raises(ValueError, match="cannot be freely realized"):
            DeterministicSurfaceRealizer().realize(request)


def test_deterministic_realizer_rejects_comment_docstring() -> None:
    for kind in (SurfaceSlotKind.COMMENT, SurfaceSlotKind.DOCSTRING):
        slot = SurfaceSlot(
            slot_id="s1",
            kind=kind,
            authority=SurfaceAuthority.SURFACE_ONLY,
            ast_path=(),
            semantic_symbol_id=None,
            opaque_region_id=None,
            constraints=SurfaceConstraint(),
            current_value_digest=None,
        )
        request = SurfaceRealizationRequest(
            pack_id="openui",
            constraint_version="v1",
            semantic_ir_fingerprint="fp",
            slots=(slot,),
            context={},
        )
        with pytest.raises(ValueError, match="unsupported"):
            DeterministicSurfaceRealizer().realize(request)


def test_deterministic_realizer_skips_opaque_user_value_slots() -> None:
    slot = SurfaceSlot(
        slot_id="s1",
        kind=SurfaceSlotKind.DECORATIVE_TEXT,
        authority=SurfaceAuthority.OPAQUE_USER_VALUE,
        ast_path=(),
        semantic_symbol_id=None,
        opaque_region_id="r1",
        constraints=SurfaceConstraint(),
        current_value_digest=None,
    )
    request = SurfaceRealizationRequest(
        pack_id="openui",
        constraint_version="v1",
        semantic_ir_fingerprint="fp",
        slots=(slot,),
        context={},
    )
    assignments = DeterministicSurfaceRealizer().realize(request)
    assert assignments == ()


def test_deterministic_realizer_enforces_reserved_words() -> None:
    slot = SurfaceSlot(
        slot_id="s1",
        kind=SurfaceSlotKind.INTERNAL_IDENTIFIER,
        authority=SurfaceAuthority.SURFACE_ONLY,
        ast_path=(),
        semantic_symbol_id="title",
        opaque_region_id=None,
        constraints=SurfaceConstraint(reserved=("v0",)),
        current_value_digest=None,
    )
    request = SurfaceRealizationRequest(
        pack_id="openui",
        constraint_version="v1",
        semantic_ir_fingerprint="fp",
        slots=(slot,),
        context={},
    )
    assignments = DeterministicSurfaceRealizer().realize(request)
    assert assignments[0].value == "v0_1"


# ---------------------------------------------------------------------------
# End-to-end surface realization
# ---------------------------------------------------------------------------


def test_internal_binder_renaming_is_alpha_equivalent() -> None:
    pack = get_pack("openui")
    result = realize_surface_and_verify(
        NO_OPAQUE,
        pack=pack,
        semantic_ir_fingerprint="fp",
        prior_status="solved",
    )
    assert result.status == "solved"
    assert result.source is not None
    assert result.semantic_equivalence is not None
    assert result.semantic_equivalence["alpha_equivalent"] is True
    assert "v0" in result.source


def test_content_placeholder_routes_through_opaque_region_path() -> None:
    pack = get_pack("openui")
    result = realize_surface_and_verify(
        HERO,
        pack=pack,
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
    assert 'TextContent(":user.title")' in result.source
    assert result.diagnostics["opaque_user_value_assignments"] == 1


def test_literal_materialization_capability_probe_rejects_real_content() -> None:
    pack = get_pack("openui")
    result = realize_surface_and_verify(
        HERO,
        pack=pack,
        opaque_bindings={
            "openui:content::hero.title": _openui_binding(
                "openui:content::hero.title", 'Welcome "back"\nToday'
            )
        },
        semantic_ir_fingerprint="fp",
        prior_status="solved",
    )
    assert result.status == "error"
    assert result.source is None


def test_verified_template_binding_envelope_is_safe_and_deterministic() -> None:
    request = GenerationRequest(
        prompt="hero",
        slot_contract=(":hero.title",),
    )
    bindings = (CallerContentBinding("hero.title", 'Welcome "back"\\\nToday ☃'),)
    pack = get_pack("openui")
    first = resolve_verified_template_bindings(BOUND_HERO, request, bindings, pack=pack)
    second = resolve_verified_template_bindings(BOUND_HERO, request, bindings, pack=pack)

    assert first == second
    assert json.dumps(first.to_dict(), sort_keys=True, ensure_ascii=False).encode() == json.dumps(
        second.to_dict(), sort_keys=True, ensure_ascii=False
    ).encode()
    assert first.status == "resolved"
    assert first.template_verification == "pack_verified"
    assert first.materialized_source is None
    assert first.realization_mode == "template_plus_bindings"
    assert len(first.template_fingerprint or "") == 64
    assert len(first.fingerprint) == 64
    assert first.bindings[0].internal_slot == 0
    assert first.bindings[0].value == bindings[0].value
    assert first.bindings[0].value_bytes == len(bindings[0].value.encode("utf-8"))
    assert bindings[0].value not in str(first.evidence_dict())


@pytest.mark.parametrize("value", ["", 'quote " slash \\ / newline\nUnicode ☃'])
def test_verified_template_binding_transports_content_without_source_injection(
    value: str,
) -> None:
    result = resolve_verified_template_bindings(
        BOUND_HERO,
        GenerationRequest(prompt="hero", slot_contract=(":hero.title",)),
        (CallerContentBinding("hero.title", value),),
        pack=get_pack("openui"),
    )
    assert result.status == "resolved"
    assert result.bindings[0].value == value
    assert result.materialized_source is None


@pytest.mark.parametrize(
    ("bindings", "message"),
    [
        ((), "missing required"),
        ((CallerContentBinding("other.title", "x"),), "unknown binding"),
        (
            (
                CallerContentBinding("hero.title", "a"),
                CallerContentBinding("hero.title", "b"),
            ),
            "duplicate binding",
        ),
    ],
)
def test_verified_template_binding_validation_fails_closed(
    bindings: tuple[CallerContentBinding, ...], message: str
) -> None:
    result = resolve_verified_template_bindings(
        BOUND_HERO,
        GenerationRequest(prompt="hero", slot_contract=(":hero.title",)),
        bindings,
        pack=get_pack("openui"),
    )
    assert result.status == "error"
    assert any(message in error for error in result.errors)


def test_verified_template_binding_rejects_alias_and_invalid_template() -> None:
    with pytest.raises(ValueError, match="unprefixed"):
        CallerContentBinding(":hero.title", "x")
    result = resolve_verified_template_bindings(
        "root = Broken(",
        GenerationRequest(prompt="hero", slot_contract=(":hero.title",)),
        (CallerContentBinding("hero.title", "x"),),
        pack=get_pack("openui"),
    )
    assert result.status == "error"
    assert result.template_verification == "canonicalization_failed"
    external_name_result = resolve_verified_template_bindings(
        HERO,
        GenerationRequest(prompt="hero", slot_contract=(":hero.title",)),
        (CallerContentBinding("hero.title", "x"),),
        pack=get_pack("openui"),
    )
    assert external_name_result.status == "error"
    assert any("undeclared model slot" in error for error in external_name_result.errors)


def test_verified_template_repeated_binding_covers_every_occurrence() -> None:
    repeated = (
        'root = Stack([title, subtitle], "column")\n'
        'title = TextContent(":hero.title")\n'
        'subtitle = TextContent(":hero.title")'
    )
    result = resolve_verified_template_bindings(
        repeated.replace(":hero.title", ":slot_0"),
        GenerationRequest(prompt="hero", slot_contract=(":hero.title",)),
        (CallerContentBinding("hero.title", "same value"),),
        pack=get_pack("openui"),
    )
    assert result.status == "resolved"
    assert result.bindings[0].occurrence_count == 2


def test_missing_required_opaque_value_fails_closed() -> None:
    pack = get_pack("openui")
    result = realize_surface_and_verify(
        HERO,
        pack=pack,
        semantic_ir_fingerprint="fp",
        prior_status="solved",
    )
    assert result.status == "error"
    assert any("missing required" in err for err in result.errors)


def test_unknown_assignment_fails_closed() -> None:
    pack = get_pack("openui")

    class BadRealizer:
        def realize(self, request: SurfaceRealizationRequest) -> tuple[SurfaceAssignment, ...]:
            return (SurfaceAssignment(slot_id="no-such-slot", value="x", provenance="test"),)

    result = realize_surface_and_verify(
        NO_OPAQUE,
        pack=pack,
        realizer=BadRealizer(),
        semantic_ir_fingerprint="fp",
        prior_status="solved",
    )
    assert result.status == "error"
    assert any("unknown slot" in err for err in result.errors)


def test_duplicate_assignment_fails_closed() -> None:
    pack = get_pack("openui")

    class BadRealizer:
        def realize(self, request: SurfaceRealizationRequest) -> tuple[SurfaceAssignment, ...]:
            return (
                SurfaceAssignment(slot_id="openui:binder:title", value="a", provenance="test"),
                SurfaceAssignment(slot_id="openui:binder:title", value="b", provenance="test"),
            )

    result = realize_surface_and_verify(
        NO_OPAQUE,
        pack=pack,
        realizer=BadRealizer(),
        semantic_ir_fingerprint="fp",
        prior_status="solved",
    )
    assert result.status == "error"
    assert any("duplicate assignment" in err for err in result.errors)


def test_tampered_assignment_cannot_return_certified_output() -> None:
    """An adversarial realizer that emits an invalid identifier is rejected."""
    pack = get_pack("openui")

    class BadRealizer:
        def realize(self, request: SurfaceRealizationRequest) -> tuple[SurfaceAssignment, ...]:
            return (SurfaceAssignment(slot_id="openui:binder:title", value="123bad", provenance="test"),)

    result = realize_surface_and_verify(
        NO_OPAQUE,
        pack=pack,
        realizer=BadRealizer(),
        semantic_ir_fingerprint="fp",
        prior_status="solved",
    )
    assert result.status == "error"
    assert result.source is None
    assert any("does not match" in err for err in result.errors)


def test_failed_verifier_returns_no_certified_result() -> None:
    """A program that cannot survive canonicalization/verification returns no source."""
    pack = get_pack("openui")
    # A source with an unbalanced component call will be rejected by the oracle.
    broken = 'root = Broken('
    result = realize_surface_and_verify(
        broken,
        pack=pack,
        semantic_ir_fingerprint="fp",
        prior_status="solved",
    )
    assert result.status in {"error", "rejected"}
    assert result.source is None


def test_typed_oracle_failure_is_not_mislabeled_solved() -> None:
    base_pack = get_pack("openui")
    rejected_report = SimpleNamespace(failing_gate="policy", ok=False)
    pack = SimpleNamespace(
        pack_id="openui",
        canonicalize=lambda source: source,
        oracle=lambda _source: rejected_report,
        opaque_region_extractor=base_pack.opaque_region_extractor,
    )
    surface_result = realize_surface_and_verify(
        NO_OPAQUE,
        pack=pack,
        semantic_ir_fingerprint="fp",
        prior_status="verified",
    )
    opaque_result = realize_opaque_regions(
        HERO,
        {
            "openui:content::hero.title": _openui_binding(
                "openui:content::hero.title", ":user.title"
            )
        },
        pack=pack,
    )
    assert surface_result.status == "rejected"
    assert opaque_result.status == "rejected"


# ---------------------------------------------------------------------------
# Preconditions and honest boundaries
# ---------------------------------------------------------------------------


def test_missing_fingerprint_fails_closed() -> None:
    pack = get_pack("openui")
    result = realize_surface_and_verify(
        NO_OPAQUE,
        pack=pack,
        semantic_ir_fingerprint="",
        prior_status="solved",
    )
    assert result.status == "error"
    assert any("semantic_ir_fingerprint" in err for err in result.errors)


def test_invalid_prior_status_fails_closed() -> None:
    pack = get_pack("openui")
    result = realize_surface_and_verify(
        NO_OPAQUE,
        pack=pack,
        semantic_ir_fingerprint="fp",
        prior_status="in_progress",
    )
    assert result.status == "error"
    assert any("prior_status" in err for err in result.errors)


def test_pack_without_surface_extractor_fails_closed() -> None:
    pack = get_pack("toy-layout")
    result = realize_surface_and_verify(
        'root = row(title, action)',
        pack=pack,
        semantic_ir_fingerprint="fp",
        prior_status="solved",
    )
    assert result.status == "error"
    assert any("surface_slot_extractor" in err for err in result.errors)


# ---------------------------------------------------------------------------
# Serialization / round-trip
# ---------------------------------------------------------------------------


def test_result_survives_json_round_trip() -> None:
    pack = get_pack("openui")
    result = realize_surface_and_verify(
        NO_OPAQUE,
        pack=pack,
        semantic_ir_fingerprint="fp",
        prior_status="solved",
    )
    data = result.to_dict()
    assert data["status"] == "solved"
    assert isinstance(data["assignments"], list)
    assert data["diagnostics"]["total_slots"] >= 1


def test_slot_survives_dict_round_trip() -> None:
    slot = SurfaceSlot(
        slot_id="s1",
        kind=SurfaceSlotKind.INTERNAL_IDENTIFIER,
        authority=SurfaceAuthority.SURFACE_ONLY,
        ast_path=("statement", 0),
        semantic_symbol_id="title",
        opaque_region_id=None,
        constraints=SurfaceConstraint(max_bytes=32),
        current_value_digest="abcd",
    )
    recovered = SurfaceSlot.from_dict(slot.to_dict())
    assert recovered == slot


# ---------------------------------------------------------------------------
# Historical compatibility
# ---------------------------------------------------------------------------


def test_historical_opaque_regions_still_work() -> None:
    """The new surface hooks do not change existing opaque-region behavior."""
    pack = get_pack("openui")
    extractor = pack.require("opaque_region_extractor")
    regions = extractor(HERO)
    binding = OpaqueRegionBinding(
        region_id=regions[0].region_id,
        scalar_value=":user.title",
    )
    result = realize_opaque_regions(
        HERO, {binding.region_id: binding}, pack=pack
    )
    assert result.status == "solved"
    assert result.source is not None


def test_canonicalize_input_helper_uses_pack_canonicalize() -> None:
    pack = get_pack("openui")
    canonical = canonicalize_input(HERO, pack)
    assert "root = Stack" in canonical
