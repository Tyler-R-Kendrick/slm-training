from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from slm_training.dsl.operators import (
    MOVE_NODE,
    SET_PROPERTY,
    ApplicationProvenanceV1,
    OperatorStateV1,
    branch_fingerprint,
    build_openui_topology_operator_context,
    build_openui_topology_operator_library,
    enumerate_operator_legal_set,
)
from slm_training.dsl.operators.topology_apply import (
    StaleTopologyStateError,
    TopologyApplyConfigV1,
    TopologyApplySession,
    TopologyEditKind,
    map_operator_to_edit_kind,
    materialize_topology_tree,
)
from slm_training.dsl.operators.topology import (
    CONTRACT_SUBTREE,
    DUPLICATE_SUBTREE,
    EXPAND_TEMPLATE,
    REPARENT_NODE,
    UNWRAP_NODE,
    WRAP_NODE,
)
from slm_training.dsl.pack import get_pack
from slm_training.dsl.production_codec import parse_statement_bindings

SOURCE = 'root = Card([Stack([TextContent(":source.value")]), Stack([])], "clear")'
REQUEST_ID = "topology-apply-test"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _provenance(request_id: str = REQUEST_ID) -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="topology-apply-test",
        compiler_version="v1",
        source_artifact_digest=_sha(SOURCE),
        request_id=request_id,
    )


def _fixture(source: str = SOURCE, *, request_id: str = REQUEST_ID, seed: int = 11):
    base = get_pack("openui")
    state = OperatorStateV1.from_source(base, source)
    branch = branch_fingerprint(state.state_digest, _sha(request_id))
    context = build_openui_topology_operator_context(
        base,
        state,
        request_id=request_id,
        branch_digest=branch,
        seed=seed,
        values=("clear", "scroll", [1, 0]),
    )
    library = build_openui_topology_operator_library(context)
    pack = replace(base, operator_library=library)
    legal_set = enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=state,
        reference_table=context.local.reference_table,
        provenance=_provenance(request_id),
    )
    return pack, state, context, library, legal_set


def _session(mode: str = "topology_apply", request_id: str = REQUEST_ID):
    session = TopologyApplySession(
        TopologyApplyConfigV1(mode=mode),  # type: ignore[arg-type]
        request_id=request_id,
        branch_digest=branch_fingerprint(
            OperatorStateV1.from_source(get_pack("openui"), SOURCE).state_digest,
            _sha(request_id),
        ),
    )
    return session


def _first_action(legal_set, operator_id: str):
    return next(
        action
        for action in legal_set.operator_actions
        if action.operator_id == operator_id
    )


def _apply(pack, state, library, action, request_id: str = REQUEST_ID):
    result = library.apply(
        pack, state, action.operator_id, action.arguments, _provenance(request_id)
    )
    assert result.succeeded and result.state is not None
    return result


class TestEditKindMapping:
    def test_move_family_maps_to_move(self):
        assert map_operator_to_edit_kind(MOVE_NODE) is TopologyEditKind.MOVE
        assert map_operator_to_edit_kind(REPARENT_NODE) is TopologyEditKind.MOVE

    def test_expand_contract_rewrite_mapping(self):
        assert map_operator_to_edit_kind(DUPLICATE_SUBTREE) is TopologyEditKind.EXPAND
        assert map_operator_to_edit_kind(EXPAND_TEMPLATE) is TopologyEditKind.EXPAND
        assert map_operator_to_edit_kind(WRAP_NODE) is TopologyEditKind.EXPAND
        assert map_operator_to_edit_kind(CONTRACT_SUBTREE) is TopologyEditKind.CONTRACT
        assert map_operator_to_edit_kind(UNWRAP_NODE) is TopologyEditKind.CONTRACT
        assert map_operator_to_edit_kind(SET_PROPERTY) is TopologyEditKind.REWRITE

    def test_unknown_operator_has_no_mapping(self):
        assert map_operator_to_edit_kind("openui.nonexistent") is None


class TestDefaultOff:
    def test_off_mode_always_defers(self):
        pack, state, context, library, legal_set = _fixture()
        session = _session(mode="off")
        session.begin(state, parse_statement_bindings(state.source, dsl="openui"))
        passes_before = session.stats.node_passes
        action = _first_action(legal_set, MOVE_NODE)
        result = _apply(pack, state, library, action)
        applied = session.apply_direct(
            result=result,
            action=action,
            legal_set=legal_set,
            context=context,
            new_bindings=parse_statement_bindings(result.state.source, dsl="openui"),
        )
        assert applied is False
        assert session.stats.node_passes == passes_before
        assert session.stats.direct_applies == 0

    def test_config_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown topology-apply mode"):
            TopologyApplyConfigV1(mode="always")  # type: ignore[arg-type]

    def test_config_roundtrip(self):
        config = TopologyApplyConfigV1(mode="topology_apply")
        assert TopologyApplyConfigV1.from_dict(config.to_dict()) == config


class TestDirectApply:
    def test_direct_apply_matches_authoritative_tree_with_fewer_passes(self):
        pack, state, context, library, legal_set = _fixture()
        session = _session()
        bindings = parse_statement_bindings(state.source, dsl="openui")
        session.begin(state, bindings)
        _, _, full_passes = materialize_topology_tree(bindings)

        action = _first_action(legal_set, MOVE_NODE)
        result = _apply(pack, state, library, action)
        new_bindings = parse_statement_bindings(result.state.source, dsl="openui")
        before = session.stats.node_passes
        applied = session.apply_direct(
            result=result,
            action=action,
            legal_set=legal_set,
            context=context,
            new_bindings=new_bindings,
        )
        assert applied is True
        _, authoritative_roots, authoritative_passes = materialize_topology_tree(
            new_bindings
        )
        assert session.roots == authoritative_roots
        direct_passes = session.stats.node_passes - before
        assert 0 < direct_passes < authoritative_passes + full_passes
        assert session.stats.remask_phases == 1
        assert session.stats.direct_applies == 1
        assert session.edits[-1].kind is TopologyEditKind.MOVE

    def test_sequential_direct_applies_stay_exact(self):
        pack, state, context, library, legal_set = _fixture()
        session = _session()
        session.begin(state, parse_statement_bindings(state.source, dsl="openui"))
        for _ in range(3):
            actions = sorted(
                legal_set.operator_actions, key=lambda action: action.semantic_id
            )
            action = actions[0]
            result = _apply(pack, state, library, action)
            new_bindings = parse_statement_bindings(result.state.source, dsl="openui")
            session.apply_direct(
                result=result,
                action=action,
                legal_set=legal_set,
                context=context,
                new_bindings=new_bindings,
            )
            _, authoritative_roots, _ = materialize_topology_tree(new_bindings)
            assert session.roots == authoritative_roots
            state = result.state
            pack, state, context, library, legal_set = _fixture(state.source)


class TestStaleInvalidation:
    def test_stale_state_digest_fails_closed(self):
        pack, state, context, library, legal_set = _fixture()
        session = _session()
        session.begin(state, parse_statement_bindings(state.source, dsl="openui"))
        action = _first_action(legal_set, MOVE_NODE)
        result = _apply(pack, state, library, action)
        # Advance the session to the post-apply state, then replay the stale
        # application recorded against the old state.
        session.begin(
            result.state, parse_statement_bindings(result.state.source, dsl="openui")
        )
        with pytest.raises(StaleTopologyStateError, match="stale state"):
            session.apply_direct(
                result=result,
                action=action,
                legal_set=legal_set,
                context=context,
                new_bindings=parse_statement_bindings(
                    result.state.source, dsl="openui"
                ),
            )

    def test_cross_request_effect_fails_closed(self):
        pack, state, context, library, legal_set = _fixture()
        other_pack, _, other_context, other_library, other_legal = _fixture(
            request_id="other-request"
        )
        session = _session()
        session.begin(state, parse_statement_bindings(state.source, dsl="openui"))
        action = _first_action(other_legal, MOVE_NODE)
        result = _apply(other_pack, state, other_library, action, "other-request")
        with pytest.raises(StaleTopologyStateError, match="another request"):
            session.apply_direct(
                result=result,
                action=action,
                legal_set=other_legal,
                context=other_context,
                new_bindings=parse_statement_bindings(
                    result.state.source, dsl="openui"
                ),
            )

    def test_stale_legal_set_fails_closed(self):
        pack, state, context, library, legal_set = _fixture()
        session = _session()
        session.begin(state, parse_statement_bindings(state.source, dsl="openui"))
        action = _first_action(legal_set, MOVE_NODE)
        result = _apply(pack, state, library, action)
        # Legal set enumerated for the *next* state is stale relative to the
        # session's current (pre-apply) state.
        _, next_state, next_context, _, next_legal = _fixture(result.state.source)
        with pytest.raises(StaleTopologyStateError, match="stale"):
            session.apply_direct(
                result=result,
                action=action,
                legal_set=next_legal,
                context=next_context,
                new_bindings=parse_statement_bindings(
                    result.state.source, dsl="openui"
                ),
            )

    def test_apply_before_begin_fails_closed(self):
        pack, state, context, library, legal_set = _fixture()
        session = _session()
        action = _first_action(legal_set, MOVE_NODE)
        result = _apply(pack, state, library, action)
        with pytest.raises(StaleTopologyStateError, match="before session begin"):
            session.apply_direct(
                result=result,
                action=action,
                legal_set=legal_set,
                context=context,
                new_bindings=parse_statement_bindings(
                    result.state.source, dsl="openui"
                ),
            )

    def test_non_member_action_fails_closed(self):
        pack, state, context, library, legal_set = _fixture()
        _, _, _, _, other_legal = _fixture(seed=29)
        session = _session()
        session.begin(state, parse_statement_bindings(state.source, dsl="openui"))
        foreign = next(
            action
            for action in other_legal.operator_actions
            if action.application_id
            not in {a.application_id for a in legal_set.operator_actions}
        )
        result = _apply(
            pack, state, library, _first_action(legal_set, MOVE_NODE)
        )
        with pytest.raises(ValueError, match="not a member of the live legal set"):
            session.apply_direct(
                result=result,
                action=foreign,
                legal_set=legal_set,
                context=context,
                new_bindings=parse_statement_bindings(
                    result.state.source, dsl="openui"
                ),
            )

    def test_branch_change_invalidates_prior_state(self):
        pack, state, context, library, legal_set = _fixture()
        session = _session()
        session.begin(state, parse_statement_bindings(state.source, dsl="openui"))
        action = _first_action(legal_set, MOVE_NODE)
        result = _apply(pack, state, library, action)
        # Opening a new branch on the *next* state invalidates every node (and
        # every recorded application) of the previous branch.
        session.begin_branch(
            branch_fingerprint(result.state.state_digest, _sha("branch-2")),
            result.state,
            parse_statement_bindings(result.state.source, dsl="openui"),
        )
        with pytest.raises(StaleTopologyStateError, match="stale state"):
            session.apply_direct(
                result=result,
                action=action,
                legal_set=legal_set,
                context=context,
                new_bindings=parse_statement_bindings(
                    result.state.source, dsl="openui"
                ),
            )
