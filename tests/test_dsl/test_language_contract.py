"""Tests for the OpenUI language-contract identity (F0)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from slm_training.data.contract import GenerationRequest, RuntimeSymbol
from slm_training.dsl import lang_core
from slm_training.dsl.language_contract import (
    LANG_SPEC,
    OUTPUT_CONTRACT_VERSION,
    SYMBOLIC_SURFACE_POLICY_VERSION,
    LanguageContract,
    OutputContractError,
    SurfaceCategory,
    SurfaceDecision,
    SymbolicSurfacePolicyV1,
    assert_symbol_only_output,
    contract_id,
    current_contract,
)
from slm_training.dsl.pack import get_pack
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import load_jsonl
from slm_training.dsl.scope_env import (
    ScopeEnv,
    ShadowingPolicy,
    SurfaceAliasMap,
    SymbolNamespace,
)


def test_current_contract_pins_installed_langcore() -> None:
    contract = current_contract()
    versions = dict(contract.openui_versions)
    # The installed language layer is the 0.2.x subset (lang-core caps at 0.2.9).
    assert versions["@openuidev/lang-core"] == "0.2.9"
    assert contract.lang_spec == LANG_SPEC == "openui-lang-0.2.x"


def test_contract_id_is_deterministic_16_hex() -> None:
    first = contract_id()
    second = current_contract().contract_id
    assert first == second
    assert len(first) == 16
    int(first, 16)  # raises if not hex


def test_contract_id_changes_when_surface_changes() -> None:
    base = current_contract()
    bumped = LanguageContract(
        lang_spec=base.lang_spec,
        openui_versions=base.openui_versions,
        grammar_sha256=base.grammar_sha256,
        tokenizer_version=base.tokenizer_version + 1,
        dsl_tokenizer_version=base.dsl_tokenizer_version,
    )
    assert bumped.contract_id != base.contract_id


def test_to_dict_round_trips_fields() -> None:
    contract = current_contract()
    data = contract.to_dict()
    assert data["contract_id"] == contract.contract_id
    assert data["lang_spec"] == contract.lang_spec
    assert set(data["openui_versions"]) == {name for name, _ in contract.openui_versions}
    assert data["output_contract_version"] == OUTPUT_CONTRACT_VERSION == 2


def test_symbol_only_output_contract_rejects_free_form_text() -> None:
    assert_symbol_only_output('root = Stack([TextContent(":hero.title")], "column")')
    with pytest.raises(OutputContractError, match="free-form strings"):
        assert_symbol_only_output('root = TextContent("Welcome back")')
    with pytest.raises(OutputContractError, match="free-form strings"):
        assert_symbol_only_output('root = TagBlock(["New Item Alpha"])')


def test_symbolic_surface_policy_preserves_closed_terms_and_declared_markers() -> None:
    source = 'root = Stack([item], "column")\nitem = TextContent(":hero.title")'
    report = SymbolicSurfacePolicyV1().evaluate(
        source,
        runtime_symbols=(
            RuntimeSymbol(surface=":hero.title", role="external_entity"),
        ),
    )

    assert report.admitted
    assert report.violations == ()
    assert report.policy_version == SYMBOLIC_SURFACE_POLICY_VERSION
    assert len(report.pack_version) == 64
    legacy_request = GenerationRequest(
        prompt="render a title",
        slot_contract=(":hero.title",),
    )
    assert SymbolicSurfacePolicyV1().evaluate_request(source, legacy_request).admitted


def test_symbolic_surface_policy_reports_typed_open_and_undeclared_surfaces() -> None:
    source = 'root = Stack([TextContent("Welcome")], "column")\n# note\nx = Slider(5)'
    report = SymbolicSurfacePolicyV1().evaluate(source)

    observed = {
        (item.category, item.decision, item.suggested_marker_role)
        for item in report.violations
    }
    assert (
        SurfaceCategory.OPEN_STRING,
        SurfaceDecision.TEMPLATE,
        "external_entity",
    ) in observed
    assert (
        SurfaceCategory.OPEN_NUMBER,
        SurfaceDecision.REJECT,
        None,
    ) in observed
    assert (
        SurfaceCategory.COMMENT_PROSE,
        SurfaceDecision.REJECT,
        None,
    ) in observed
    assert all(source[item.start : item.end] == item.surface for item in report.violations)
    assert all(item.pack_id == "openui" for item in report.violations)


def test_symbolic_surface_policy_requires_declared_marker_and_state_roles() -> None:
    source = 'root = TextContent(":hero.title")\n$session = root'
    report = SymbolicSurfacePolicyV1().evaluate(
        source,
        runtime_symbols=(
            RuntimeSymbol(surface=":other", role="external_entity"),
            RuntimeSymbol(surface="$other", role="state"),
        ),
    )

    assert [
        (item.category, item.decision, item.suggested_marker_role)
        for item in report.violations
    ] == [
        (
            SurfaceCategory.EXTERNAL_REF,
            SurfaceDecision.REJECT,
            "external_entity",
        ),
        (SurfaceCategory.STATE_REF, SurfaceDecision.REJECT, "state"),
    ]
    with pytest.raises(OutputContractError, match="rejected staged target"):
        SymbolicSurfacePolicyV1().require_admitted(source)


def test_alpha_renaming_preserves_policy_and_canonical_ast_meaning() -> None:
    first = 'root = Stack([item], "column")\nitem = TextContent(":hero.title")'
    second = 'root = Stack([copy], "column")\ncopy = TextContent(":hero.title")'
    symbols = (RuntimeSymbol(surface=":hero.title", role="external_entity"),)
    policy = SymbolicSurfacePolicyV1()

    assert policy.evaluate(first, runtime_symbols=symbols).admitted
    assert policy.evaluate(second, runtime_symbols=symbols).admitted
    pack = get_pack("openui")
    assert pack.canonicalize(first) == pack.canonicalize(second)


def test_marker_alias_permutation_preserves_opaque_scope_identity() -> None:
    first_env, first_aliases = ScopeEnv(), SurfaceAliasMap()
    first_env, first_aliases, _ = first_env.declare(
        SymbolNamespace.CONTENT,
        first_aliases,
        alias=":hero.title",
        shadowing=ShadowingPolicy.FORBID,
    )
    first_env, first_aliases, _ = first_env.declare(
        SymbolNamespace.CONTENT,
        first_aliases,
        alias=":hero.body",
        shadowing=ShadowingPolicy.FORBID,
    )
    second_env, second_aliases = ScopeEnv(), SurfaceAliasMap()
    second_env, second_aliases, _ = second_env.declare(
        SymbolNamespace.CONTENT,
        second_aliases,
        alias=":hero.body",
        shadowing=ShadowingPolicy.FORBID,
    )
    second_env, second_aliases, _ = second_env.declare(
        SymbolNamespace.CONTENT,
        second_aliases,
        alias=":hero.title",
        shadowing=ShadowingPolicy.FORBID,
    )
    first = 'root = Stack([TextContent(":hero.title"), TextContent(":hero.body")])'
    second = 'root = Stack([TextContent(":hero.body"), TextContent(":hero.title")])'
    symbols = (
        RuntimeSymbol(surface=":hero.title", role="external_entity"),
        RuntimeSymbol(surface=":hero.body", role="external_entity"),
    )

    assert first_env.fingerprint == second_env.fingerprint
    assert SymbolicSurfacePolicyV1().evaluate(first, runtime_symbols=symbols).admitted
    assert SymbolicSurfacePolicyV1().evaluate(second, runtime_symbols=symbols).admitted

    def stable_surface(source: str, aliases: SurfaceAliasMap) -> str:
        for entry in aliases.entries:
            source = source.replace(entry.alias, entry.symbol_id.canonical)
        return source

    assert stable_surface(first, first_aliases) == stable_surface(
        second, second_aliases
    )


def test_symbolic_surface_policy_uses_second_pack_schema_authority_offline() -> None:
    source = "query GetPost($id: ID!) { post(id: $id) { title } }"
    report = SymbolicSurfacePolicyV1("graphql").evaluate(
        source,
        runtime_symbols=(RuntimeSymbol(surface="$id", role="state"),),
    )

    assert report.admitted
    assert report.pack_id == "graphql"


def test_symbolic_surface_evidence_hashes_match_committed_sources() -> None:
    payload = json.loads(
        Path(
            "docs/design/dsh0-02-symbolic-surface-policy-20260723.json"
        ).read_text(encoding="utf-8")
    )

    for artifact in payload["source_identities"]:
        actual = hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest()
        assert actual == artifact["sha256"]


def test_canonical_eval_seeds_are_symbol_only_and_declared() -> None:
    records = load_jsonl(Path("src/slm_training/resources/test_seeds.jsonl"))
    for record in records:
        assert_symbol_only_output(record.openui, output_kind=record.target_kind)
        assert set(extract_placeholders(record.openui)).issubset(record.placeholders), record.id


def test_library_schema_falls_back_to_committed_snapshot(monkeypatch) -> None:
    lang_core._RESULT_CACHE.pop("schema", None)
    monkeypatch.setattr(
        lang_core,
        "_invoke",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    schema = lang_core.library_schema(refresh=True)
    assert len(schema["properties"]) == 54
    assert schema["$defs"]["Card"]["required"] == ["children"]
    assert list(schema["$defs"]["RadioItem"]["properties"]) == [
        "label",
        "description",
        "value",
    ]


def test_bridge_uses_matching_git_common_checkout_dependencies(
    tmp_path, monkeypatch
) -> None:
    worktree = tmp_path / "worktree"
    common = tmp_path / "common"
    work_bridge = worktree / "src/apps/openui_bridge"
    common_bridge = common / "src/apps/openui_bridge"
    for root in (work_bridge, common_bridge):
        root.mkdir(parents=True)
        for name in lang_core._BRIDGE_SOURCES:
            (root / name).write_text(name)
    (common_bridge / "node_modules/@openuidev/lang-core").mkdir(parents=True)
    monkeypatch.delenv("OPENUI_BRIDGE_CLI", raising=False)
    monkeypatch.setattr(lang_core, "REPO_ROOT", worktree)
    monkeypatch.setattr(lang_core, "DEFAULT_BRIDGE_DIR", work_bridge)
    monkeypatch.setattr(
        lang_core, "checkout_roots", lambda root: (root, common)
    )
    lang_core._bridge_dir.cache_clear()

    assert lang_core._bridge_dir() == common_bridge
    assert lang_core._bridge_cli() == common_bridge / "cli.mjs"

    lang_core._bridge_dir.cache_clear()


def test_committed_schema_matches_pinned_library() -> None:
    if not lang_core.bridge_available():
        pytest.skip("OpenUI bridge dependencies are unavailable")
    live = lang_core.library_schema(refresh=True, allow_snapshot=False)
    assert live == lang_core._schema_snapshot()
