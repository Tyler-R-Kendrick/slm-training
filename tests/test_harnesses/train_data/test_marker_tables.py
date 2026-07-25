import pytest

from slm_training.data.contract import GenerationRequest, RuntimeSymbol
from slm_training.harnesses.train_data.marker_tables import (
    MarkerAuditCode,
    MarkerTableBundleV1,
    assert_role_not_ordinal_recoverable,
    audit_marker_table,
    canonical_marker_payload,
    materialize_marker_table,
    permute_marker_surfaces,
    require_clean,
    resolve_marker,
    split_permutation_seed,
)
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1


def _root_for(split: str) -> str:
    policy = RootFamilySplitPolicyV1()
    for index in range(10_000):
        value = f"family-{split}-{index}"
        if policy.assign(value) == split:
            return value
    raise AssertionError(f"no root for {split}")


def _request() -> GenerationRequest:
    return GenerationRequest(
        prompt="build a settings row",
        slot_contract=(":title",),
        runtime_symbols=(
            RuntimeSymbol(
                surface="root",
                role="alpha_binder",
                semantic_type="element",
                scope="canvas",
            ),
            RuntimeSymbol(
                surface="row",
                role="alpha_binder",
                semantic_type="element",
                scope="canvas",
                signature="row(text)",
            ),
            RuntimeSymbol(
                surface=":title",
                role="external_entity",
                semantic_type="text",
                semantic_role="content",
            ),
            RuntimeSymbol(
                surface="$expanded",
                role="state",
                semantic_type="bool",
                scope="session",
            ),
            RuntimeSymbol(surface="draft", role="fresh_binder"),
        ),
    )


def _bundle(split: str, seed: int = 7) -> MarkerTableBundleV1:
    root = _root_for(split)
    return materialize_marker_table(
        _request(), root_family_id=root, split=split, master_seed=seed
    )


def test_materialize_is_deterministic() -> None:
    left = _bundle("train")
    right = _bundle("train")
    assert left.table.to_dict() == right.table.to_dict()
    assert left.table.table_id == right.table.table_id


def test_rows_carry_only_inference_visible_facts() -> None:
    bundle = _bundle("train")
    logicals = {entry.logical_surface for entry in bundle.provenance.entries}
    for row in bundle.table.rows:
        assert row.surface not in logicals
        payload = row.to_dict()
        assert set(payload) <= {
            "surface",
            "role",
            "ordinal",
            "semantic_type",
            "semantic_role",
            "scope",
            "signature",
        }
    # typed authority survives the projection
    by_role = {row.role: row for row in bundle.table.rows}
    assert by_role["binder"].semantic_type == "element"
    assert by_role["external_entity"].semantic_role == "content"
    assert by_role["state_ref"].scope == "session"


def test_runtime_symbol_projection_round_trips_v8() -> None:
    bundle = _bundle("train")
    symbols = bundle.table.runtime_symbols()
    roles = {symbol.role for symbol in symbols}
    assert roles == {"alpha_binder", "external_entity", "state", "fresh_binder"}
    for symbol in symbols:  # RuntimeSymbol codec validation passes
        resolved = resolve_marker(bundle.table, symbol.surface)
        assert resolved.surface == symbol.surface


def test_independent_split_permutations_are_disjoint() -> None:
    train = _bundle("train")
    test = _bundle("test")
    assert set(train.table.surfaces()).isdisjoint(test.table.surfaces())
    assert train.table.permutation_digest != test.table.permutation_digest
    assert split_permutation_seed(7, _root_for("train"), "train") != (
        split_permutation_seed(7, _root_for("train"), "test")
    )


def test_alpha_renaming_preserves_canonical_identity() -> None:
    bundle = _bundle("train")
    renamed = permute_marker_surfaces(bundle, seed=1234)
    assert set(renamed.table.surfaces()).isdisjoint(bundle.table.surfaces())
    assert canonical_marker_payload(renamed.table) == canonical_marker_payload(
        bundle.table
    )
    # provenance still maps the same logical surfaces
    assert {e.logical_surface for e in renamed.provenance.entries} == {
        e.logical_surface for e in bundle.provenance.entries
    }
    require_clean(renamed, reference=bundle.table)


def test_alpha_mismatch_detected() -> None:
    bundle = _bundle("train")
    other = materialize_marker_table(
        GenerationRequest(prompt="other", slot_contract=(":different",)),
        root_family_id=_root_for("train"),
        split="train",
        master_seed=7,
    )
    findings = audit_marker_table(other, reference=bundle.table)
    assert MarkerAuditCode.ALPHA_MISMATCH in {item.code for item in findings}


def test_unknown_marker_fails_closed() -> None:
    bundle = _bundle("train")
    findings = audit_marker_table(bundle, used_surfaces=("mystery",))
    assert [item.code for item in findings] == [MarkerAuditCode.UNKNOWN]
    with pytest.raises(ValueError, match="marker_unknown"):
        resolve_marker(bundle.table, "mystery")
    with pytest.raises(ValueError, match="fail"):
        require_clean(bundle, used_surfaces=("mystery",))


def test_out_of_range_marker_fails_closed() -> None:
    bundle = _bundle("train")
    findings = audit_marker_table(bundle, used_surfaces=("b_ffffff99",))
    assert [item.code for item in findings] == [MarkerAuditCode.OUT_OF_RANGE]
    with pytest.raises(ValueError, match="marker_out_of_range"):
        resolve_marker(bundle.table, "b_ffffff99")


def test_duplicate_marker_use_fails_closed() -> None:
    bundle = _bundle("train")
    surface = bundle.table.rows[0].surface
    findings = audit_marker_table(bundle, used_surfaces=(surface, surface))
    assert [item.code for item in findings] == [MarkerAuditCode.DUPLICATE]
    with pytest.raises(ValueError, match="marker_duplicate"):
        require_clean(bundle, used_surfaces=(surface, surface))


def test_exact_copy_surface_fails_closed() -> None:
    bundle = _bundle("train")
    provenance = bundle.provenance
    entry = provenance.entries[0]
    # forge a table row whose surface equals its logical surface
    from slm_training.harnesses.train_data.marker_tables import (
        MarkerProvenanceEntryV1,
        MarkerRowV1,
        MarkerTableProvenanceV1,
        MarkerTableV1,
    )

    row = bundle.table.rows[0]
    forged_row = MarkerRowV1(
        surface=entry.logical_surface
        if not entry.logical_surface.startswith((":", "$"))
        else row.surface,
        role=row.role,
        ordinal=row.ordinal,
    )
    if forged_row.surface == row.surface:
        forged_row = MarkerRowV1(surface="root", role="binder", ordinal=row.ordinal)
    forged_table = MarkerTableV1(
        table_id=bundle.table.table_id,
        root_family_id=bundle.table.root_family_id,
        split=bundle.table.split,
        policy_version=bundle.table.policy_version,
        permutation_digest=bundle.table.permutation_digest,
        rows=(forged_row, *bundle.table.rows[1:]),
    )
    forged_prov = MarkerTableProvenanceV1(
        table_id=provenance.table_id,
        master_seed=provenance.master_seed,
        entries=(
            MarkerProvenanceEntryV1(
                logical_surface=forged_row.surface,
                marker_surface=forged_row.surface,
            ),
            *provenance.entries[1:],
        ),
    )
    forged = MarkerTableBundleV1(table=forged_table, provenance=forged_prov)
    findings = audit_marker_table(forged)
    assert MarkerAuditCode.EXACT_COPY in {item.code for item in findings}


def test_role_not_recoverable_from_ordinal() -> None:
    for split in ("train", "validation", "test"):
        bundle = _bundle(split)
        assert_role_not_ordinal_recoverable(bundle)
        require_clean(bundle)
        # ordinal stream is deranged relative to declaration order
        ordinals = {
            entry.marker_surface: row.ordinal
            for entry, row in zip(
                bundle.provenance.entries,
                sorted(bundle.table.rows, key=lambda r: r.surface),
            )
        }
        by_surface = {row.surface: row.ordinal for row in bundle.table.rows}
        for index, entry in enumerate(bundle.provenance.entries):
            assert by_surface[entry.marker_surface] != index
        assert ordinals  # sanity: mapping constructed


def test_split_policy_enforced() -> None:
    with pytest.raises(ValueError, match="belongs to"):
        materialize_marker_table(
            _request(),
            root_family_id=_root_for("train"),
            split="test",
            master_seed=7,
        )


def test_logical_mapping_only_in_provenance() -> None:
    bundle = _bundle("train")
    table_payload = str(bundle.table.to_dict())
    for entry in bundle.provenance.entries:
        logical = entry.logical_surface.lstrip(":$")
        if logical and logical not in {"root"}:  # 'root' can appear in digests
            assert entry.logical_surface not in [
                row.surface for row in bundle.table.rows
            ]
    assert "logical_surface" not in table_payload
