from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace

import pytest

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    CompilerCoverage,
    EffectDeltaKind,
    EffectDeltaV1,
    OperatorArgumentSlotV1,
    OperatorAuthorityError,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorReplayError,
    OperatorStateV1,
    RefKind,
    RegisteredOperatorV1,
    ValueRef,
)
from slm_training.dsl.pack import PackSlotUnavailable, get_pack

SOURCE = 'root = TextContent(":hero.title")'
UPDATED = 'root = TextContent(":hero.body")'


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _declaration(operator_id: str = "openui.set_fixture_value") -> AstOperatorV1:
    return AstOperatorV1(
        operator_id=operator_id,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(
            OperatorArgumentSlotV1("value", RefKind.VALUE, BindingPhase.APPLICATION),
        ),
        preconditions=(),
        effect_signature=(EffectDeltaKind.PROPERTY,),
        locality="node",
        cost=1.0,
    )


def _executor(
    state: OperatorStateV1, arguments: tuple[BoundArgumentV1, ...]
) -> OperatorMutationV1:
    ref = arguments[0].value
    return OperatorMutationV1(
        source=state.source.replace(":hero.title", ":hero.body"),
        effect=ActionEffectV1(
            property_deltas=(
                EffectDeltaV1(
                    EffectDeltaKind.PROPERTY,
                    ref,
                    ":hero.title",
                    ":hero.body",
                ),
            ),
            compiler_coverage=CompilerCoverage.EXACT,
            estimated_completion_cost=1.0,
        ),
    )


def _invalid_executor(
    state: OperatorStateV1, arguments: tuple[BoundArgumentV1, ...]
) -> OperatorMutationV1:
    return OperatorMutationV1(
        source="root = Broken(",
        effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
    )


def _library(executor=_executor) -> OperatorLibraryV1:
    return OperatorLibraryV1((RegisteredOperatorV1(_declaration(), executor),))


def _pack(library: OperatorLibraryV1):
    return replace(get_pack("openui"), operator_library=library)


def _provenance() -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.compiler",
        compiler_version="2026.07",
        source_artifact_digest=_sha(SOURCE),
        request_id="req-1",
    )


def _arguments() -> tuple[BoundArgumentV1, ...]:
    return (BoundArgumentV1("value", ValueRef("req-1", "v1")),)


def test_registry_fingerprint_and_lookup_are_order_invariant() -> None:
    first = RegisteredOperatorV1(_declaration("openui.first"), _executor)
    second = RegisteredOperatorV1(_declaration("openui.second"), _executor)
    left = OperatorLibraryV1((first, second))
    right = OperatorLibraryV1((second, first))
    assert left.registry_fingerprint == right.registry_fingerprint
    assert left.lookup("openui.first").fingerprint == first.declaration.fingerprint
    with pytest.raises(FrozenInstanceError):
        left.registry_fingerprint = "mutable"  # type: ignore[misc]
    with pytest.raises(KeyError, match="unsupported operator"):
        left.lookup("openui.missing")


def test_pack_owned_apply_and_dry_run_share_exact_outcome() -> None:
    library = _library()
    pack = _pack(library)
    state = OperatorStateV1.from_source(pack, SOURCE)
    before = replace(state)

    dry = library.dry_run(
        pack, state, "openui.set_fixture_value", _arguments(), _provenance()
    )
    applied = library.apply(
        pack, state, "openui.set_fixture_value", _arguments(), _provenance()
    )

    assert applied.succeeded
    assert applied.application.application_id == dry.application_id
    assert applied.state is not None
    assert applied.state.source == pack.canonicalize(UPDATED)
    assert pack.oracle(applied.state.source).ok
    assert state == before
    assert applied.application.proof is not None
    assert set(applied.application.proof.checks) == {
        "pack.parse",
        "pack.static_schema",
        "pack.scope",
        "pack.property_order",
        "pack.canonical",
        "pack.roundtrip",
        "input.immutable",
    }


def test_argument_type_and_required_slots_fail_closed() -> None:
    library = _library()
    pack = _pack(library)
    state = OperatorStateV1.from_source(pack, SOURCE)
    missing = library.apply(pack, state, "openui.set_fixture_value", (), _provenance())
    assert not missing.succeeded
    assert missing.application.rejection is not None
    assert missing.application.rejection.code == "operator.arguments_rejected"


def test_invalid_rewrite_is_typed_rejection_in_both_paths() -> None:
    library = _library(_invalid_executor)
    pack = _pack(library)
    state = OperatorStateV1.from_source(pack, SOURCE)
    dry = library.dry_run(
        pack, state, "openui.set_fixture_value", _arguments(), _provenance()
    )
    applied = library.apply(
        pack, state, "openui.set_fixture_value", _arguments(), _provenance()
    )
    assert not dry.succeeded
    assert not applied.succeeded
    assert dry.application_id == applied.application.application_id
    assert applied.application.after_state_digest is None
    assert applied.application.rejection is not None
    assert applied.application.rejection.code == "operator.authority_rejected"


def test_unsupported_operator_returns_typed_rejection() -> None:
    library = _library()
    pack = _pack(library)
    state = OperatorStateV1.from_source(pack, SOURCE)
    result = library.apply(pack, state, "openui.missing", _arguments(), _provenance())
    assert not result.succeeded
    assert result.application.rejection is not None
    assert result.application.rejection.code == "operator.unsupported"


def test_successful_application_replays_exactly() -> None:
    library = _library()
    pack = _pack(library)
    state = OperatorStateV1.from_source(pack, SOURCE)
    result = library.apply(
        pack, state, "openui.set_fixture_value", _arguments(), _provenance()
    )
    replayed = library.replay(pack, state, result.application)
    assert replayed.application.application_id == result.application.application_id
    assert replayed.state == result.state

    with pytest.raises(OperatorReplayError, match="before state"):
        library.replay(
            pack,
            replace(state, state_digest="f" * 64),
            result.application,
        )


def test_registry_cannot_execute_through_another_pack_or_library() -> None:
    library = _library()
    pack = _pack(library)
    state = OperatorStateV1.from_source(pack, SOURCE)
    other = _library()
    with pytest.raises(OperatorAuthorityError, match="not owned"):
        other.apply(
            pack, state, "openui.set_fixture_value", _arguments(), _provenance()
        )
    with pytest.raises(OperatorAuthorityError, match="identity"):
        library.apply(
            pack,
            replace(state, state_digest="f" * 64),
            "openui.set_fixture_value",
            _arguments(),
            _provenance(),
        )


def test_partial_pack_fails_closed_before_operator_execution() -> None:
    toy = get_pack("toy-layout")
    with pytest.raises(PackSlotUnavailable, match="operator_library"):
        toy.require("operator_library")
    library = _library()
    with pytest.raises(PackSlotUnavailable, match="canonicalize"):
        OperatorStateV1.from_source(
            replace(toy, operator_library=library),
            'root = text(":hero.title")',
        )
