"""Regression tests for VSS2-04 opaque-region splicing (SLM-68)."""

from __future__ import annotations

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.dsl.opaque_regions import (
    OpaqueRegion,
    OpaqueRegionBinding,
    OpaqueRegionKind,
    OpaqueRegionSummary,
    realize_opaque_regions,
)
from slm_training.dsl.pack import get_pack


HERO = 'root = Stack([title], "column")\ntitle = TextContent(":hero.title")'


def _spec_with_regions(*regions: OpaqueRegion) -> ProgramSpec:
    return ProgramSpec(
        id="opaque-test",
        ast={"type": "root"},
        canonical_openui=HERO,
        facts={},
        contract_id="0123456789abcdef",
        program_family_id="fam-1",
        lineage_id="line-1",
        split_group_id="sg-1",
        opaque_regions=regions,
    )


def test_programspec_round_trips_opaque_regions() -> None:
    region = OpaqueRegion(
        region_id="r1",
        kind=OpaqueRegionKind.CONTENT_VALUE,
        placeholder=":hero.title",
        source_digest="abcd1234",
        summary=OpaqueRegionSummary(
            input_bindings=("ctx.hero.title",),
            effects=("display",),
        ),
    )
    spec = _spec_with_regions(region)
    recovered = ProgramSpec.from_dict(spec.to_dict())
    assert recovered.opaque_regions == spec.opaque_regions
    assert recovered.opaque_regions[0].summary.effects == ("display",)


def test_historical_programspec_without_opaque_regions_round_trips() -> None:
    spec = ProgramSpec(
        id="legacy",
        ast={"type": "root"},
        canonical_openui=HERO,
        facts={},
        contract_id="0123456789abcdef",
        program_family_id="fam-1",
        lineage_id="line-1",
        split_group_id="sg-1",
    )
    recovered = ProgramSpec.from_dict(spec.to_dict())
    assert recovered.opaque_regions == ()


def test_openui_pack_extracts_content_placeholders_as_regions() -> None:
    pack = get_pack("openui")
    extractor = pack.require("opaque_region_extractor")
    regions = extractor(HERO)
    assert len(regions) == 1
    assert regions[0].region_id == "openui:content::hero.title"
    assert regions[0].kind is OpaqueRegionKind.CONTENT_VALUE
    assert regions[0].placeholder == ":hero.title"


def test_realize_content_placeholders_solves_and_verifies() -> None:
    pack = get_pack("openui")
    extractor = pack.require("opaque_region_extractor")
    regions = extractor(HERO)
    spec = _spec_with_regions(*regions)
    binding = OpaqueRegionBinding(
        region_id=regions[0].region_id,
        scalar_value=":user.title",
    )
    result = realize_opaque_regions(spec, {binding.region_id: binding}, pack=pack)
    assert result.status == "solved"
    assert result.source is not None
    assert 'TextContent(":user.title")' in result.source
    assert result.errors == ()
    assert regions[0].region_id in result.source_map


def test_missing_required_region_fails() -> None:
    pack = get_pack("openui")
    region = OpaqueRegion(
        region_id="r1",
        kind=OpaqueRegionKind.CONTENT_VALUE,
        placeholder=":missing",
        required=True,
    )
    spec = _spec_with_regions(region)
    result = realize_opaque_regions(spec, {}, pack=pack)
    assert result.status == "error"
    assert any("missing required" in err for err in result.errors)


def test_unknown_binding_fails() -> None:
    pack = get_pack("openui")
    region = OpaqueRegion(
        region_id="r1",
        kind=OpaqueRegionKind.CONTENT_VALUE,
        placeholder=":x",
        required=True,
    )
    spec = _spec_with_regions(region)
    binding = OpaqueRegionBinding(region_id="no-such-region", scalar_value="x")
    result = realize_opaque_regions(spec, {"no-such-region": binding}, pack=pack)
    assert result.status == "error"
    assert any("unknown region" in err for err in result.errors)


def test_duplicate_binding_fails() -> None:
    pack = get_pack("openui")
    region = OpaqueRegion(
        region_id="r1",
        kind=OpaqueRegionKind.CONTENT_VALUE,
        placeholder=":x",
        required=True,
    )
    spec = _spec_with_regions(region)
    binding = OpaqueRegionBinding(region_id="r1", scalar_value="a")
    result = realize_opaque_regions(
        spec,
        {"r1": binding, "r1-dup": OpaqueRegionBinding(region_id="r1", scalar_value="b")},
        pack=pack,
    )
    assert result.status == "error"
    assert any("duplicate binding" in err for err in result.errors)


def test_unsupported_region_kind_fails_closed() -> None:
    pack = get_pack("openui")
    region = OpaqueRegion(
        region_id="r1",
        kind=OpaqueRegionKind.EXPRESSION,
        placeholder=None,
        required=True,
    )
    spec = _spec_with_regions(region)
    binding = OpaqueRegionBinding(region_id="r1", source_fragment="1 + 1")
    result = realize_opaque_regions(spec, {"r1": binding}, pack=pack)
    assert result.status == "error"
    # OpenUI pack has no expression splicer.
    assert any("splicing" in err or "fragment" in err for err in result.errors)


def test_region_digests_do_not_leak_raw_text() -> None:
    pack = get_pack("openui")
    extractor = pack.require("opaque_region_extractor")
    regions = extractor(HERO)
    spec = _spec_with_regions(*regions)
    binding = OpaqueRegionBinding(
        region_id=regions[0].region_id,
        scalar_value=":user.title",
    )
    result = realize_opaque_regions(spec, {binding.region_id: binding}, pack=pack)
    digest = result.region_digests[regions[0].region_id]
    assert ":user.title" not in digest
    assert len(digest) == 32
