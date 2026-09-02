"""Certified eval suites: root-family split, decontamination, leakage.

The published certified suites and the certified train bucket are checked
against each other directly (no root family in both, no exact program
overlap, no fingerprint leak), and the sampler that grows the screening
suite is checked for determinism, exclusion and stratification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.data.leakage import (
    fingerprint_openui,
    find_leakage,
    load_train_fingerprints,
    norm_text,
)
from slm_training.dsl import bridge_available
from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.harnesses.test_data.certified import (
    CERTIFIED_TRAIN_BUCKET_DIR,
    CERTIFIED_TRAIN_BUCKET_ID,
    SUITE_FOR_SPLIT,
    bucket_of,
    exact_program_texts,
    partition_certified_corpus,
    root_family_index,
    sample_certified_candidates,
)
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

EVAL_ROOT = Path("src/slm_training/resources/data/eval")
CERTIFIED_SUITES = (
    "e938_role_safe_all_targets_smoke96_v1",
    "e938_role_safe_all_targets_heldout24_v1",
)


def _manifest(path: Path) -> dict:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _published_suites(version: str) -> dict[str, list[ExampleRecord]]:
    manifest = _manifest(EVAL_ROOT / version)
    out: dict[str, list[ExampleRecord]] = {}
    for suite, rel in manifest["suites"].items():
        path = Path(rel)
        rows = load_jsonl(path) if path.stat().st_size else []
        if rows:
            out[suite] = rows
    return out


@pytest.fixture(scope="module")
def train_bucket() -> tuple[dict, list[ExampleRecord]]:
    manifest = _manifest(CERTIFIED_TRAIN_BUCKET_DIR)
    return manifest, load_jsonl(Path(manifest["records"]))


@pytest.fixture(scope="module")
def eval_suites() -> dict[str, dict[str, list[ExampleRecord]]]:
    return {version: _published_suites(version) for version in CERTIFIED_SUITES}


def test_train_bucket_manifest_carries_leakage_fingerprints(train_bucket) -> None:
    manifest, records = train_bucket
    assert manifest["version"] == CERTIFIED_TRAIN_BUCKET_ID
    assert manifest["record_count"] == len(records) == len(manifest["ids"])
    assert len(set(manifest["ids"])) == len(records)
    assert manifest["split_policy"]["program_text_closure"] is True
    for key in (
        "root_family_ids",
        "prompt_fingerprints",
        "openui_fingerprints",
        "structure_fingerprints",
        "pair_fingerprints",
    ):
        assert manifest[key], key
    policy = RootFamilySplitPolicyV1()
    for record in records:
        assert record.meta["root_family_split"] == "train"
        assert policy.assign(record.meta["root_family_id"]) == "train"
        assert bucket_of(record.meta["root_family_id"]) < 80


def test_published_suites_have_expected_counts(eval_suites) -> None:
    assert len(eval_suites[CERTIFIED_SUITES[0]]["smoke"]) == 96
    assert len(eval_suites[CERTIFIED_SUITES[1]]["held_out"]) == 24
    for version, suites in eval_suites.items():
        manifest = _manifest(EVAL_ROOT / version)
        ids = [r.id for rows in suites.values() for r in rows]
        assert len(set(ids)) == len(ids)
        assert set(manifest["ids"]) == set(ids)


def test_no_root_family_in_train_bucket_and_any_certified_eval_suite(
    train_bucket, eval_suites
) -> None:
    manifest, records = train_bucket
    train_families = set(manifest["root_family_ids"])
    assert train_families == {r.meta["root_family_id"] for r in records}
    families_by_suite: dict[str, set[str]] = {}
    for suites in eval_suites.values():
        for suite, rows in suites.items():
            for record in rows:
                family = record.meta["root_family_id"]
                split = record.meta["root_family_split"]
                assert family not in train_families, (record.id, family)
                assert SUITE_FOR_SPLIT[split] == suite, (record.id, split, suite)
                families_by_suite.setdefault(suite, set()).add(family)
    # Smoke (validation bucket) and held-out (test bucket) never share a family.
    assert families_by_suite["smoke"].isdisjoint(families_by_suite["held_out"])


def test_no_exact_program_overlap_and_no_fingerprint_leak(train_bucket, eval_suites) -> None:
    manifest, records = train_bucket
    train_programs = exact_program_texts(records)
    train_openui = {fingerprint_openui(r.openui) for r in records}
    fps = load_train_fingerprints(CERTIFIED_TRAIN_BUCKET_DIR / "manifest.json")
    for suites in eval_suites.values():
        for rows in suites.values():
            for record in rows:
                assert norm_text(record.openui) not in train_programs, record.id
                assert fingerprint_openui(record.openui) not in train_openui, record.id
                assert find_leakage(record, fps) == [], record.id


def test_sidecars_sit_at_the_exact_decidability_floor() -> None:
    for version in CERTIFIED_SUITES:
        sidecar = json.loads(
            (EVAL_ROOT / version / "screening_sample_size.json").read_text(encoding="utf-8")
        )
        report = sidecar["report"]
        assert sidecar["eval_version"] == version
        assert sidecar["suite"] in {"smoke", "held_out"}
        assert sidecar["suite_records"] == report["suite_ceiling_n"] > 0
        assert report["verdict"] == "feasible"
        assert report["power_floor_n"] is None  # never a fabricated SD
        assert report["chosen_n"] == report["decidability_floor_n"]
        quality = json.loads(
            (EVAL_ROOT / version / "quality_report.json").read_text(encoding="utf-8")
        )
        assert quality["leakage_rejected"] == 0
        assert quality["certified"]["rejection_histogram"]


def test_family_index_closes_under_links_and_program_text() -> None:
    program = 'root = Stack([cta])\ncta = Button(":cta.label")'
    other = 'root = Stack([blurb])\nblurb = TextContent(":page.blurb")'
    rows = [
        ExampleRecord(id="a", prompt="A", openui=program, placeholders=[":cta.label"],
                      split="train", meta={"root_parent_id": "fam_a"}),
        ExampleRecord(id="a_child", prompt="A2", openui=other, placeholders=[":page.blurb"],
                      split="train", meta={"parent_id": "a"}),
        ExampleRecord(id="b", prompt="B", openui=program + "\n", placeholders=[":cta.label"],
                      split="train", meta={"root_parent_id": "fam_b"}),
        ExampleRecord(id="c", prompt="C", openui=other + " ", placeholders=[":page.blurb"],
                      split="train", meta={"root_parent_id": "fam_c"}),
    ]
    linked = root_family_index(rows, close_under_program_text=False)
    assert linked["a"] == linked["a_child"] == "fam_a"
    assert linked["b"] == "fam_b" and linked["c"] == "fam_c"
    closed = root_family_index(rows)
    # b shares a's program, c shares a_child's: one family, smallest root wins.
    assert {closed[k] for k in ("a", "a_child", "b", "c")} == {"fam_a"}


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)
def test_sampler_is_deterministic_stratified_and_excludes_existing() -> None:
    partition = partition_certified_corpus()
    assert partition.link_families > partition.families  # closure merged families
    assert partition.by_split["train"] and partition.by_split["validation"]
    first = sample_certified_candidates(existing_ids=set(), need=12, partition=partition)
    second = sample_certified_candidates(existing_ids=set(), need=12, partition=partition)
    ids = [r.id for r in first.records]
    assert ids == [r.id for r in second.records]
    assert len(ids) == 12 == len(set(ids))
    assert len(first.stratification) > 1
    assert all(r.split == "smoke" and r.meta["suite"] == "smoke" for r in first.records)
    reserved = set(ids[:6])
    later = sample_certified_candidates(existing_ids=reserved, need=12, partition=partition)
    assert reserved.isdisjoint(r.id for r in later.records)
    assert later.excluded_existing >= 6
    with pytest.raises(ValueError):
        sample_certified_candidates(existing_ids=set(), need=1, suite="adversarial")
