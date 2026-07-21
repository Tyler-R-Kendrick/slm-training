"""Focused contracts for the persistent typed ScopeEnv core."""

from __future__ import annotations

import pytest

from slm_training.dsl.scope_env import (
    ForwardReferencePolicy,
    ScopeEnv,
    ShadowingPolicy,
    StableSymbolId,
    SurfaceAliasMap,
    SymbolNamespace,
)


def _declare(
    env: ScopeEnv,
    aliases: SurfaceAliasMap,
    namespace: SymbolNamespace,
    alias: str,
    *,
    shadowing: ShadowingPolicy = ShadowingPolicy.FORBID,
):
    return env.declare(
        namespace,
        aliases,
        alias=alias,
        shadowing=shadowing,
    )


def test_stable_ids_and_model_fingerprint_do_not_depend_on_aliases() -> None:
    left, left_aliases, left_symbol = _declare(
        ScopeEnv(), SurfaceAliasMap(), SymbolNamespace.CONTENT, "hero.title"
    )
    right, right_aliases, right_symbol = _declare(
        ScopeEnv(), SurfaceAliasMap(), SymbolNamespace.CONTENT, "marketing.heading"
    )

    assert (
        left_symbol.symbol_id
        == right_symbol.symbol_id
        == StableSymbolId(SymbolNamespace.CONTENT, 0)
    )
    assert left_symbol.symbol_id.canonical == "content:0000"
    assert left.to_dict() == right.to_dict()
    assert left.fingerprint == right.fingerprint
    assert left_aliases.fingerprint != right_aliases.fingerprint
    assert "hero.title" not in str(left.to_dict())


def test_parent_linked_scope_is_persistent_and_shadowing_is_explicit() -> None:
    root, aliases, outer = _declare(
        ScopeEnv(), SurfaceAliasMap(), SymbolNamespace.BINDER, "item"
    )
    child = root.enter_scope("child")
    shadowed, aliases, inner = _declare(
        child,
        aliases,
        SymbolNamespace.BINDER,
        "item",
        shadowing=ShadowingPolicy.ALLOW,
    )

    assert root.current.parent is None
    assert child.current.declarations == ()
    assert shadowed.current.parent is root.current
    assert (
        shadowed.resolve(
            "item",
            SymbolNamespace.BINDER,
            aliases,
            forward_references=ForwardReferencePolicy.FORBID,
        )
        == inner
    )
    exited = shadowed.exit_scope()
    assert (
        exited.resolve(
            "item",
            SymbolNamespace.BINDER,
            aliases,
            forward_references=ForwardReferencePolicy.FORBID,
        )
        == outer
    )


def test_shadowing_forbid_and_same_frame_duplicate_fail_closed() -> None:
    env, aliases, _ = _declare(
        ScopeEnv(), SurfaceAliasMap(), SymbolNamespace.BINDER, "item"
    )
    child = env.enter_scope("child")
    with pytest.raises(ValueError, match="already declared"):
        _declare(child, aliases, SymbolNamespace.BINDER, "item")
    with pytest.raises(ValueError, match="already declared"):
        _declare(
            env,
            aliases,
            SymbolNamespace.BINDER,
            "item",
            shadowing=ShadowingPolicy.ALLOW,
        )


def test_namespaces_are_disjoint_even_when_aliases_match() -> None:
    env, aliases, content = _declare(
        ScopeEnv(), SurfaceAliasMap(), SymbolNamespace.CONTENT, "target"
    )
    env, aliases, state = _declare(env, aliases, SymbolNamespace.STATE, "target")

    assert content.symbol_id.canonical == "content:0000"
    assert state.symbol_id.canonical == "state:0000"
    assert (
        env.resolve(
            "target",
            SymbolNamespace.CONTENT,
            aliases,
            forward_references=ForwardReferencePolicy.FORBID,
        )
        == content
    )
    assert (
        env.resolve(
            "target",
            SymbolNamespace.STATE,
            aliases,
            forward_references=ForwardReferencePolicy.FORBID,
        )
        == state
    )


def test_forward_reference_requires_predeclaration_and_explicit_policy() -> None:
    env, aliases, reserved = ScopeEnv().predeclare(
        SymbolNamespace.QUERY,
        SurfaceAliasMap(),
        alias="results",
        shadowing=ShadowingPolicy.FORBID,
        semantic_type="rows",
    )
    with pytest.raises(LookupError, match="forward reference"):
        env.resolve(
            "results",
            SymbolNamespace.QUERY,
            aliases,
            forward_references=ForwardReferencePolicy.FORBID,
        )
    assert (
        env.resolve(
            "results",
            SymbolNamespace.QUERY,
            aliases,
            forward_references=ForwardReferencePolicy.ALLOW_PREDECLARED,
        )
        == reserved
    )

    env, aliases, declared = env.declare(
        SymbolNamespace.QUERY,
        aliases,
        alias="results",
        shadowing=ShadowingPolicy.FORBID,
        semantic_type="rows",
        predeclared_id=reserved.symbol_id,
    )
    assert declared == reserved
    assert (
        env.resolve(
            "results",
            SymbolNamespace.QUERY,
            aliases,
            forward_references=ForwardReferencePolicy.FORBID,
        )
        == declared
    )


def test_visible_hides_shadowed_parent_and_keeps_declaration_order() -> None:
    env, aliases, outer = _declare(
        ScopeEnv(), SurfaceAliasMap(), SymbolNamespace.BINDER, "item"
    )
    env, aliases, sibling = _declare(env, aliases, SymbolNamespace.BINDER, "other")
    env = env.enter_scope("child")
    env, aliases, inner = _declare(
        env,
        aliases,
        SymbolNamespace.BINDER,
        "item",
        shadowing=ShadowingPolicy.ALLOW,
    )

    assert env.visible(SymbolNamespace.BINDER, aliases) == (inner, sibling)
    assert outer not in env.visible(SymbolNamespace.BINDER, aliases)


def test_scope_and_alias_json_round_trip_with_deterministic_fingerprints() -> None:
    env, aliases, _ = _declare(
        ScopeEnv(), SurfaceAliasMap(), SymbolNamespace.CONTENT, "hero.title"
    )
    env = env.enter_scope("component")
    env, aliases, _ = _declare(env, aliases, SymbolNamespace.COMPILER_LOCAL, "tmp")

    loaded_env = ScopeEnv.from_dict(env.to_dict())
    loaded_aliases = SurfaceAliasMap.from_dict(aliases.to_dict())
    loaded_env.validate_aliases(loaded_aliases)
    assert loaded_env == env
    assert loaded_aliases == aliases
    assert loaded_env.fingerprint == env.fingerprint
    assert loaded_aliases.fingerprint == aliases.fingerprint


def test_invalid_scope_transitions_and_unknown_symbols_fail_closed() -> None:
    with pytest.raises(ValueError, match="root"):
        ScopeEnv().exit_scope()
    with pytest.raises(LookupError, match="unknown"):
        ScopeEnv().resolve(
            "missing",
            SymbolNamespace.ACTION,
            SurfaceAliasMap(),
            forward_references=ForwardReferencePolicy.FORBID,
        )
    with pytest.raises(ValueError, match="unknown predeclared"):
        ScopeEnv().declare(
            SymbolNamespace.ACTION,
            SurfaceAliasMap(),
            alias="submit",
            shadowing=ShadowingPolicy.FORBID,
            predeclared_id=StableSymbolId(SymbolNamespace.ACTION, 7),
        )
    with pytest.raises(TypeError, match="explicit ShadowingPolicy"):
        ScopeEnv().declare(
            SymbolNamespace.ACTION,
            SurfaceAliasMap(),
            alias="submit",
            shadowing="forbid",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="explicit ForwardReferencePolicy"):
        ScopeEnv().resolve(
            "submit",
            SymbolNamespace.ACTION,
            SurfaceAliasMap(),
            forward_references="forbid",  # type: ignore[arg-type]
        )


def test_deserialization_rejects_reusable_symbol_ordinals() -> None:
    env, _aliases, _ = _declare(
        ScopeEnv(), SurfaceAliasMap(), SymbolNamespace.CONTENT, "hero.title"
    )
    payload = env.to_dict()
    payload["next_ordinals"]["content"] = 0
    with pytest.raises(ValueError, match="must exceed"):
        ScopeEnv.from_dict(payload)


def test_exit_discards_child_predeclarations_without_recycling_ids() -> None:
    child = ScopeEnv().enter_scope("caller-name")
    child, aliases, pending = child.predeclare(
        SymbolNamespace.COMPILER_LOCAL,
        SurfaceAliasMap(),
        alias="temporary",
        shadowing=ShadowingPolicy.FORBID,
    )
    exited = child.exit_scope()
    assert pending not in exited.predeclared
    assert exited.next_ordinals == ((SymbolNamespace.COMPILER_LOCAL, 1),)
    with pytest.raises(ValueError, match="unknown symbol"):
        exited.validate_aliases(aliases)


def test_model_fingerprint_ignores_scope_labels_and_aliases() -> None:
    left = ScopeEnv().enter_scope("hero.title")
    right = ScopeEnv().enter_scope("marketing.heading")
    assert left.to_dict() != right.to_dict()
    assert left.fingerprint == right.fingerprint


def test_deserialization_rejects_predeclaration_in_inactive_frame() -> None:
    env = ScopeEnv().enter_scope("child")
    env, _aliases, _pending = env.predeclare(
        SymbolNamespace.QUERY,
        SurfaceAliasMap(),
        alias="query",
        shadowing=ShadowingPolicy.FORBID,
    )
    payload = env.to_dict()
    payload["frames"] = payload["frames"][:1]
    with pytest.raises(ValueError, match="inactive frame"):
        ScopeEnv.from_dict(payload)
