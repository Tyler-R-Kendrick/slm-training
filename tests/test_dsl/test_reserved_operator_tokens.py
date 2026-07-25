from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    CompilerCoverage,
    OperatorArgumentSlotV1,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    RegisteredOperatorV1,
    ReservedOperatorDisposition,
    ReservedOperatorTargetMode,
    ReservedOperatorTokenConfigV1,
    apply_reserved_operator_target,
    build_reference_table,
    enumerate_operator_legal_set,
    reserved_operator_checkpoint_metadata,
    serialize_reserved_operator_target,
    validate_reserved_operator_checkpoint,
)
from slm_training.dsl.pack import get_pack

SOURCE = 'root = TextContent(":before")'
RESULT = 'root = TextContent(":after")'
OPERATOR_ID = "openui.fixture_reserved"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _fixture():
    base = get_pack("openui")
    state = OperatorStateV1.from_source(base, SOURCE)
    descriptor = ReferenceDescriptorV1(
        ref_kind=RefKind.VALUE,
        semantic_fingerprint=_sha("after"),
        value_type="openui.string",
    )
    table = build_reference_table(
        request_id="reserved-test",
        state_digest=state.state_digest,
        branch_digest=_sha("branch"),
        descriptors=(descriptor,),
        seed=3,
    )
    allowed_ref = table.entries[0].ref

    def execute(operator_state, arguments):
        if arguments != (BoundArgumentV1("value", allowed_ref),):
            raise OperatorRejectedError("fixture.not_legal")
        return OperatorMutationV1(
            source=RESULT,
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    declaration = AstOperatorV1(
        operator_id=OPERATOR_ID,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(
            OperatorArgumentSlotV1(
                "value", RefKind.VALUE, BindingPhase.APPLICATION
            ),
        ),
        preconditions=(),
        effect_signature=(),
        locality="node",
        cost=1.0,
    )
    library = OperatorLibraryV1((RegisteredOperatorV1(declaration, execute),))
    pack = replace(base, operator_library=library)
    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="reserved-test",
        compiler_version="v1",
        source_artifact_digest=_sha(SOURCE),
        request_id="reserved-test",
    )
    legal_set = enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=state,
        reference_table=table,
        provenance=provenance,
    )
    return pack, library, state, provenance, legal_set


def test_reserved_tokens_are_default_off_and_checkpoint_incompatible() -> None:
    *_, legal_set = _fixture()
    action = legal_set.operator_actions[0]
    with pytest.raises(ValueError, match="disabled"):
        serialize_reserved_operator_target(
            action=action,
            result_ast=RESULT,
            mode=ReservedOperatorTargetMode.OPERATOR_ONLY,
        )

    enabled = ReservedOperatorTokenConfigV1(enabled=True)
    with pytest.raises(ValueError, match="lacks"):
        validate_reserved_operator_checkpoint(None, enabled)
    disabled_metadata = reserved_operator_checkpoint_metadata()
    with pytest.raises(ValueError, match="config mismatch"):
        validate_reserved_operator_checkpoint(disabled_metadata, enabled)
    validate_reserved_operator_checkpoint(
        reserved_operator_checkpoint_metadata(enabled), enabled
    )


@pytest.mark.parametrize(
    "mode",
    [
        ReservedOperatorTargetMode.OPERATOR_ONLY,
        ReservedOperatorTargetMode.OPERATOR_PLUS_RESULT,
    ],
)
def test_live_legal_member_applies_only_through_compiler(mode) -> None:
    pack, library, state, provenance, legal_set = _fixture()
    config = ReservedOperatorTokenConfigV1(enabled=True)
    target = serialize_reserved_operator_target(
        action=legal_set.operator_actions[0],
        result_ast=RESULT,
        mode=mode,
        config=config,
    )
    decision, result = apply_reserved_operator_target(
        value=target,
        config=config,
        pack=pack,
        library=library,
        state=state,
        legal_set=legal_set,
        provenance=provenance,
    )
    assert decision.disposition is ReservedOperatorDisposition.APPLY
    assert decision.result_ast == RESULT
    assert result is not None and result.state is not None
    assert result.state.source == RESULT


def test_invalid_or_mismatched_targets_never_gain_legal_membership() -> None:
    pack, library, state, provenance, legal_set = _fixture()
    config = ReservedOperatorTokenConfigV1(enabled=True)
    action = legal_set.operator_actions[0]
    target = serialize_reserved_operator_target(
        action=action,
        result_ast="root = Separator()",
        mode=ReservedOperatorTargetMode.OPERATOR_PLUS_RESULT,
        config=config,
    )
    decision, _ = apply_reserved_operator_target(
        value=target,
        config=config,
        pack=pack,
        library=library,
        state=state,
        legal_set=legal_set,
        provenance=provenance,
    )
    assert decision.disposition is ReservedOperatorDisposition.REJECT
    assert decision.reason == "operator.result_ast_mismatch"

    nonmember = serialize_reserved_operator_target(
        action=(
            "OPERATOR openui.other "
            "value=value:reserved-test:not-a-live-reference"
        ),
        result_ast=RESULT,
        mode=ReservedOperatorTargetMode.OPERATOR_ONLY,
        config=config,
    )
    rejected, result = apply_reserved_operator_target(
        value=nonmember,
        config=config,
        pack=pack,
        library=library,
        state=state,
        legal_set=legal_set,
        provenance=provenance,
    )
    assert rejected.disposition is ReservedOperatorDisposition.REJECT
    assert rejected.reason == "operator.not_in_live_legal_set"
    assert result is None

    malformed, result = apply_reserved_operator_target(
        value="<|openui_operator:v2|>{}<|end_openui_operator|>",
        config=config,
        pack=pack,
        library=library,
        state=state,
        legal_set=legal_set,
        provenance=provenance,
    )
    assert malformed.disposition is ReservedOperatorDisposition.DEFER
    assert result is None


def test_result_only_target_defers_to_ordinary_exact_path() -> None:
    pack, library, state, provenance, legal_set = _fixture()
    config = ReservedOperatorTokenConfigV1(enabled=True)
    target = serialize_reserved_operator_target(
        action=legal_set.operator_actions[0],
        result_ast=RESULT,
        mode=ReservedOperatorTargetMode.RESULT_AST_ONLY,
        config=config,
    )
    decision, result = apply_reserved_operator_target(
        value=target,
        config=config,
        pack=pack,
        library=library,
        state=state,
        legal_set=legal_set,
        provenance=provenance,
    )
    assert decision.disposition is ReservedOperatorDisposition.DEFER
    assert decision.reason == "operator.result_only_ordinary_path"
    assert result is None
