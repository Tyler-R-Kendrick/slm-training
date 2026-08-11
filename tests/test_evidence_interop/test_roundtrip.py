"""EVID-12 / SLM-543 evidence interop round-trip and authority law tests."""

from __future__ import annotations

import copy

import pytest

from slm_training.evidence_interop import (
    AuthorityRoundtripError,
    UnsupportedFieldError,
    authority_ids_equal,
    export_prov_o,
    extract_authority,
    import_projection,
    import_prov_o,
    roundtrip,
    roundtrip_all,
)
from slm_training.formal.objects import FormalObjectKind, FormalObjectV1
from slm_training.harness_core.evidence_bundle import ArtifactRef, EvidenceBundleV1
from slm_training.harness_core.lineage.records import DataSnapshot


def _formal() -> FormalObjectV1:
    from slm_training.formal.objects import _object_digest

    statement = {"family": "fixture", "theorem_id": "t.fixture"}
    payload = {"theorem_id": "t.fixture"}
    oid = "formal-fixture:evid12"
    digest = _object_digest(FormalObjectKind.LEAN_CLAIM, oid, statement, payload)
    return FormalObjectV1(
        kind=FormalObjectKind.LEAN_CLAIM,
        object_id=oid,
        statement=statement,
        payload=payload,
        content_digest=digest,
        required_checkers=("python_structural", "python_reference"),
        tier="fixture",
    )


def _bundle() -> EvidenceBundleV1:
    return EvidenceBundleV1(
        run_id="run-evid12",
        claim_class="fixture",
        source_commit="a" * 40,
        source_dirty=False,
        config_sha256="b" * 64,
        seeds=(1, 2),
        environment_digest="c" * 64,
        corpus_sha256="d" * 64,
        suite_hashes={
            "smoke": ArtifactRef(
                name="smoke",
                sha256="e" * 64,
                uri="cas://sha256/" + "e" * 64,
            ),
        },
        checkpoint=ArtifactRef(
            name="ckpt",
            sha256="f" * 64,
            uri="cas://sha256/" + "f" * 64,
        ),
        raw_envelopes=(
            ArtifactRef(name="raw", sha256="1" * 64, uri="cas://sha256/" + "1" * 64),
        ),
        evaluator_version="eval/v1",
        gate_version="gate/v1",
    )


def _snapshot() -> DataSnapshot:
    return DataSnapshot(
        snapshot_id="snap-evid12",
        sources=("fixture",),
        records_sha="2" * 64,
        record_count=3,
        target_token_count=10,
        created_at="1970-01-01T00:00:00Z",
        annotations_sha="3" * 64,
    )


def _campaign():
    """Minimal duck-typed campaign stand-in (avoids full StrictModel fixture)."""

    class _Campaign:
        schema_version = "ExperimentCampaignV1"
        campaign_id = "camp-evid12"
        locked_eval_manifest_sha256 = "4" * 64
        source_commit = "5" * 40

    return _Campaign()


@pytest.mark.parametrize(
    "factory",
    [_formal, _bundle, _snapshot, _campaign],
    ids=["formal", "bundle", "snapshot", "campaign"],
)
def test_roundtrip_all_preserves_authority(factory):
    obj = factory()
    views = roundtrip_all(obj)
    expected = extract_authority(obj)
    assert set(views) == {
        "prov-o-jsonld",
        "ro-crate",
        "spdx3",
        "intoto",
        "oci-referrer",
    }
    for view in views.values():
        assert view.semantic_authority is False
        assert authority_ids_equal(view.authority_ids, expected.authority_ids)
        assert view.kind == expected.kind


def test_import_never_upgrades_semantic_authority():
    doc = export_prov_o(_formal())
    doc = copy.deepcopy(doc)
    doc["slm:semantic_authority"] = True
    view = import_prov_o(doc)
    assert view.semantic_authority is False


def test_import_without_authority_block_fails():
    doc = export_prov_o(_formal())
    doc = copy.deepcopy(doc)
    del doc["slm:authority"]
    with pytest.raises(ValueError, match="slm:authority"):
        import_projection(doc)


def test_unsupported_field_raise():
    doc = export_prov_o(_formal())
    doc = copy.deepcopy(doc)
    doc["ex:unknownPredicate"] = "nope"
    with pytest.raises(UnsupportedFieldError):
        import_projection(doc, raise_on_unsupported=True)
    view = import_projection(doc, raise_on_unsupported=False)
    assert "ex:unknownPredicate" in view.unsupported_fields
    assert view.semantic_authority is False


def test_roundtrip_detects_authority_tamper():
    import importlib

    obj = _formal()
    doc = export_prov_o(obj)
    doc = copy.deepcopy(doc)
    doc["slm:authority"]["authority_ids"]["content_digest"] = "0" * 64

    def _bad_export(_obj):
        return doc

    rt_mod = importlib.import_module("slm_training.evidence_interop.roundtrip")
    original = rt_mod.EXPORTERS["prov-o-jsonld"]
    rt_mod.EXPORTERS["prov-o-jsonld"] = _bad_export
    try:
        with pytest.raises(AuthorityRoundtripError):
            roundtrip(obj, format="prov-o-jsonld")
    finally:
        rt_mod.EXPORTERS["prov-o-jsonld"] = original


def test_projection_embeds_profile_and_non_authority_flag():
    doc = export_prov_o(_bundle())
    assert doc["slm:profile"] == "slm-evidence-interop/v1"
    assert doc["slm:semantic_authority"] is False
    assert doc["slm:projection_only"] is True
    assert doc["slm:authority"]["authority_ids"]["run_id"] == "run-evid12"
