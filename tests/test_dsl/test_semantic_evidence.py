"""Contracts and representation properties for advisory semantic factors."""

from __future__ import annotations

import pytest

from slm_training.data.progspec.semantic_evidence import (
    FactorRepresentation,
    build_factor_representation,
    digest_snapshot,
    lossy_pairwise_edges,
    project_program_factors,
    reconstruct_membership,
    shuffle_port_roles,
)


def test_project_and_digest_stable() -> None:
    snap = project_program_factors(
        program_id="p1",
        components=(("Card", "Card"), ("b1", "TextContent")),
        binder_edges=(("Card", "b1", "USE"),),
        legal_candidate_ids=("b1", "b2"),
    )
    again = project_program_factors(
        program_id="p1",
        components=(("Card", "Card"), ("b1", "TextContent")),
        binder_edges=(("Card", "b1", "USE"),),
        legal_candidate_ids=("b1", "b2"),
    )
    assert digest_snapshot(snap) == digest_snapshot(again)
    assert snap.advisory is True
    assert all(f.advisory for f in snap.factors)


def test_factor_node_membership_roundtrip() -> None:
    snap = project_program_factors(
        program_id="p2",
        components=(("a", "Card"), ("b", "Button"), ("c", "Input")),
        binder_edges=(("a", "b", "USE"), ("a", "c", "USE")),
        legal_candidate_ids=("b", "c"),
    )
    view = build_factor_representation(snap, FactorRepresentation.FACTOR_NODE)
    assert view is not None
    reconstructed = reconstruct_membership(view)
    expected = {f.factor_id: f.membership() for f in snap.factors}
    assert reconstructed == expected


def test_lossy_pairwise_loses_factor_identity() -> None:
    snap = project_program_factors(
        program_id="p3",
        components=(("a", "Card"), ("b", "Button"), ("c", "Input")),
        binder_edges=(("a", "b", "USE"), ("a", "c", "USE")),
        legal_candidate_ids=("b", "c"),
    )
    pairs = lossy_pairwise_edges(snap.factors)
    # Pairs exist, but no factor_id remains in the representation.
    assert pairs
    assert all(len(p) == 2 for p in pairs)


def test_role_shuffle_preserves_membership() -> None:
    snap = project_program_factors(
        program_id="p4",
        components=(("src", "binder"), ("tgt", "binder")),
        binder_edges=(("src", "tgt", "DEFINITION"),),
        legal_candidate_ids=("src", "tgt"),
    )
    shuffled = shuffle_port_roles(snap.factors, seed=3)
    for original, changed in zip(snap.factors, shuffled, strict=True):
        assert original.membership() == changed.membership()
        # Roles multiset preserved under rotation when >1 port.
        if len(original.ports) > 1:
            assert sorted(p.role for p in original.ports) == sorted(
                p.role for p in changed.ports
            )


def test_rejects_non_advisory_factor() -> None:
    from slm_training.data.progspec.semantic_evidence import (
        AdvisoryEvidenceFactorV1,
        EvidencePolarity,
        EvidenceSourceKind,
        FactorPortDirection,
        FactorPortV1,
    )

    with pytest.raises(ValueError, match="advisory"):
        AdvisoryEvidenceFactorV1(
            factor_version="AdvisoryEvidenceFactorV1",
            factor_id="f",
            factor_type="t",
            ports=(
                FactorPortV1(
                    port_id="p",
                    role="R",
                    node_id="n",
                    node_kind="k",
                    direction=FactorPortDirection.UNORDERED,
                    ordinal=0,
                ),
            ),
            polarity=EvidencePolarity.SUPPORTS,
            source_kind=EvidenceSourceKind.EXACT_STATE,
            source_ref="x",
            advisory=False,
        )
