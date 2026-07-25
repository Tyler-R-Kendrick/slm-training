"""AP-018 (SLM-306): bottleneck warm-up records from SemanticPlanV1 and verbal plans."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.data.abstract_cot.warmup import (
    DEFAULT_ATTENTION_BLOCK,
    SEGMENT_ORDER,
    AbstractCotWarmupConfig,
    AbstractCotWarmupRecordV1,
    SegmentLayoutV1,
    WarmupExclusionV1,
    build_abstract_cot_warmup_corpus,
    load_locked_test_keys,
    verbalize_semantic_plan,
)
from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.semantic_plan.extract import OpenUISemanticPlanExtractor
from slm_training.dsl.abstract_plan import AbstractPlanV1
from slm_training.dsl.pack import get_pack
from slm_training.dsl.schema import ExampleRecord

_PACK = get_pack("openui")


def _record(
    id: str,
    prompt: str,
    openui: str,
    *,
    placeholders: list[str] | None = None,
    split: str = "train",
) -> ExampleRecord:
    return ExampleRecord(
        id=id,
        prompt=prompt,
        openui=openui,
        placeholders=placeholders or [],
        split=split,
    )


def _fixture_records() -> list[ExampleRecord]:
    return [
        _record(
            "warmup_0_binders",
            "Show a plain text label.",
            'root = TextContent("Hello")',
        ),
        _record(
            "warmup_1_binder",
            "Show a labeled button in a column layout.",
            'root = Stack([item], "column")\nitem = Button(":item.label")',
            placeholders=[":item.label"],
        ),
        _record(
            "warmup_2_binder",
            "Show a card with a title and body.",
            'root = Card([title, body])\ntitle = TextContent(":hero.title")\nbody = TextContent(":hero.body")',
            placeholders=[":hero.title", ":hero.body"],
        ),
        _record(
            "warmup_3_binder",
            "Login screen with email, password, and continue button.",
            (
                'root = Stack([title, email, password, continueBtn], "column")\n'
                'title = TextContent(":auth.title")\n'
                'email = Input("email", ":auth.email", "email")\n'
                'password = Input("password", ":auth.password", "password")\n'
                'continueBtn = Button(":auth.continue")'
            ),
            placeholders=[":auth.title", ":auth.email", ":auth.password", ":auth.continue"],
        ),
        _record(
            "warmup_row_layout",
            "Show two buttons side by side.",
            'root = Stack([a, b], "row")\na = Button(":x.y")\nb = Button(":z.w")',
            placeholders=[":x.y", ":z.w"],
        ),
    ]


# --- SegmentLayoutV1 bottleneck contract -------------------------------------


def test_default_segment_layout_blocks_target_from_privileged_plan() -> None:
    layout = SegmentLayoutV1()
    assert layout.order == SEGMENT_ORDER
    assert "privileged_plan" not in layout.attention_block["target"]
    assert layout.loss_segments == ("target",)


def test_segment_layout_rejects_target_attending_privileged_plan() -> None:
    bad_block = dict(DEFAULT_ATTENTION_BLOCK)
    bad_block["target"] = ("prompt", "privileged_plan", "abstract_span", "target")
    with pytest.raises(ValueError, match="must never attend to privileged_plan"):
        SegmentLayoutV1(attention_block=bad_block)


def test_segment_layout_requires_self_attention() -> None:
    bad_block = dict(DEFAULT_ATTENTION_BLOCK)
    bad_block["prompt"] = ()
    with pytest.raises(ValueError, match="must attend to itself"):
        SegmentLayoutV1(attention_block=bad_block)


def test_segment_layout_round_trip() -> None:
    layout = SegmentLayoutV1()
    restored = SegmentLayoutV1.from_dict(json.loads(json.dumps(layout.to_dict())))
    assert restored == layout


# --- verbalization is deterministic ------------------------------------------


def test_verbalization_is_deterministic() -> None:
    spec = ProgramSpec.from_openui(
        id="v1",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")',
        facts={},
        program_family_id="test",
        lineage_id="v1",
        split_group_id="v1",
    )
    plan = OpenUISemanticPlanExtractor().extract(spec, _PACK)
    first = verbalize_semantic_plan(plan)
    second = verbalize_semantic_plan(plan)
    assert first == second
    assert "Binders: 1 binding(s)" in first


# --- corpus builder: acceptance, coverage, determinism -----------------------


def test_builder_accepts_all_well_formed_fixture_rows() -> None:
    corpus = build_abstract_cot_warmup_corpus(_fixture_records())
    assert corpus.eligible_count == 5
    assert corpus.accepted_count == 5
    assert corpus.acceptance_rate == 1.0
    assert corpus.emitted_count == 5
    assert not corpus.exclusions
    # Iron Law: every input row has an outcome (accepted, here) -- nothing silently dropped.
    assert corpus.eligible_count == corpus.accepted_count + len(corpus.exclusions)


def test_every_accepted_record_has_the_required_fields() -> None:
    corpus = build_abstract_cot_warmup_corpus(_fixture_records())
    for record in corpus.records:
        assert record.prompt
        assert record.target
        assert record.privileged_plan_factors
        assert record.abstract_span_tokens[0] == "<beginabstract>"
        assert record.abstract_span_tokens[-1] == "<endabstract>"
        assert len(record.abstract_span_token_ids) == len(record.abstract_span_tokens)
        assert record.codebook_version == AbstractPlanV1().codebook_version
        assert record.segments.order == SEGMENT_ORDER
        assert record.plan_fingerprint
        assert record.source_content_hash
        assert record.verifier_hash


def test_iteration0_abstract_span_is_deterministic_across_runs() -> None:
    records = _fixture_records()
    first = build_abstract_cot_warmup_corpus(records, AbstractCotWarmupConfig(seed=7))
    second = build_abstract_cot_warmup_corpus(records, AbstractCotWarmupConfig(seed=7))
    for a, b in zip(first.records, second.records, strict=True):
        assert a.abstract_span_token_ids == b.abstract_span_token_ids
        assert a.abstract_span_tokens == b.abstract_span_tokens


def test_iteration0_abstract_span_does_not_depend_on_processing_order() -> None:
    records = _fixture_records()
    forward = build_abstract_cot_warmup_corpus(records, AbstractCotWarmupConfig(seed=3))
    reversed_records = list(reversed(records))
    backward = build_abstract_cot_warmup_corpus(reversed_records, AbstractCotWarmupConfig(seed=3))
    forward_by_id = {r.source_record_id: r.abstract_span_token_ids for r in forward.records}
    backward_by_id = {r.source_record_id: r.abstract_span_token_ids for r in backward.records}
    assert forward_by_id == backward_by_id


def test_different_seed_changes_at_least_one_span() -> None:
    records = _fixture_records()
    a = build_abstract_cot_warmup_corpus(records, AbstractCotWarmupConfig(seed=1))
    b = build_abstract_cot_warmup_corpus(records, AbstractCotWarmupConfig(seed=2))
    a_ids = [r.abstract_span_token_ids for r in a.records]
    b_ids = [r.abstract_span_token_ids for r in b.records]
    assert a_ids != b_ids


def test_binder_bucket_counts_cover_multiple_buckets() -> None:
    corpus = build_abstract_cot_warmup_corpus(_fixture_records())
    assert set(corpus.binder_bucket_counts) >= {"0", "1", "2"}
    assert sum(corpus.binder_bucket_counts.values()) == corpus.accepted_count


# --- exclusions: nothing dropped silently ------------------------------------


def test_empty_prompt_is_excluded_with_reason() -> None:
    records = _fixture_records() + [
        ExampleRecord(id="blank", prompt=" ", openui='root = TextContent("x")')
    ]
    corpus = build_abstract_cot_warmup_corpus(records)
    reasons = {item.source_record_id: item.reason for item in corpus.exclusions}
    assert reasons["blank"] == "empty_prompt_or_openui"
    assert corpus.eligible_count == len(records)


def test_unparseable_openui_is_excluded_with_reason() -> None:
    records = _fixture_records() + [
        _record("broken", "A broken program.", "root = NotARealComponent(")
    ]
    corpus = build_abstract_cot_warmup_corpus(records)
    reasons = {item.source_record_id: item.reason for item in corpus.exclusions}
    assert reasons["broken"] == "openui_parse_failed"


def test_locked_test_rows_are_excluded_by_id(tmp_path: Path) -> None:
    manifest = {
        "schema": "LockedPromotionManifestV1",
        "manifest_sha256": "unused-in-this-fixture",
        "metadata": {},
        "rows": [
            {
                "partition": "locked_test",
                "complexity": {},
                "record": {
                    "id": "warmup_1_binder",
                    "prompt": "Show a labeled button in a column layout.",
                    "openui": 'root = Stack([item], "column")\nitem = Button(":item.label")',
                    "placeholders": [":item.label"],
                },
            }
        ],
    }
    manifest_path = tmp_path / "locked.jsonl"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    ids, fingerprints = load_locked_test_keys(manifest_path)
    assert "warmup_1_binder" in ids
    assert fingerprints  # the fixture program parses, so a fingerprint is derived too

    corpus = build_abstract_cot_warmup_corpus(
        _fixture_records(),
        AbstractCotWarmupConfig(locked_manifest_path=manifest_path),
    )
    reasons = {item.source_record_id: item.reason for item in corpus.exclusions}
    assert reasons["warmup_1_binder"] == "locked_test_excluded"
    assert corpus.accepted_count == 4


def test_missing_locked_manifest_path_is_not_fatal(tmp_path: Path) -> None:
    ids, fingerprints = load_locked_test_keys(tmp_path / "does_not_exist.jsonl")
    assert ids == set()
    assert fingerprints == set()
    corpus = build_abstract_cot_warmup_corpus(
        _fixture_records(),
        AbstractCotWarmupConfig(locked_manifest_path=tmp_path / "does_not_exist.jsonl"),
    )
    assert corpus.accepted_count == 5


def test_none_locked_manifest_path_is_not_fatal() -> None:
    corpus = build_abstract_cot_warmup_corpus(
        _fixture_records(), AbstractCotWarmupConfig(locked_manifest_path=None)
    )
    assert corpus.accepted_count == 5


# --- max_records balances across binder buckets ------------------------------


def test_max_records_round_robins_across_binder_buckets() -> None:
    corpus = build_abstract_cot_warmup_corpus(
        _fixture_records(), AbstractCotWarmupConfig(max_records=3)
    )
    assert corpus.emitted_count == 3
    assert corpus.accepted_count == 5  # coverage is still measured over all input
    emitted_buckets = {record.binder_reference_bucket for record in corpus.records}
    # Round-robin across sorted bucket keys ("0", "1", "2", "3+") should reach
    # at least three distinct buckets before repeating any one bucket twice.
    assert len(emitted_buckets) >= 3


# --- config validation --------------------------------------------------------


def test_config_rejects_abstract_span_length_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="abstract_span_length"):
        AbstractCotWarmupConfig(abstract_span_length=0)
    with pytest.raises(ValueError, match="abstract_span_length"):
        AbstractCotWarmupConfig(abstract_span_length=10_000)


def test_config_rejects_bad_max_records() -> None:
    with pytest.raises(ValueError, match="max_records"):
        AbstractCotWarmupConfig(max_records=0)


# --- record round trip + jsonl I/O -------------------------------------------


def test_record_round_trip_via_dict() -> None:
    corpus = build_abstract_cot_warmup_corpus(_fixture_records())
    for record in corpus.records:
        restored = AbstractCotWarmupRecordV1.from_dict(json.loads(json.dumps(record.to_dict())))
        assert restored == record


def test_write_jsonl_round_trips(tmp_path: Path) -> None:
    corpus = build_abstract_cot_warmup_corpus(_fixture_records())
    out_path = tmp_path / "warmup_v1.jsonl"
    corpus.write_jsonl(out_path)
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == corpus.emitted_count
    restored = [AbstractCotWarmupRecordV1.from_dict(json.loads(line)) for line in lines]
    assert restored == list(corpus.records)


def test_exclusion_rejects_unknown_reason() -> None:
    with pytest.raises(ValueError, match="unknown exclusion reason"):
        WarmupExclusionV1("x", "not_a_real_reason")


def test_corpus_fingerprint_is_stable_for_identical_input() -> None:
    a = build_abstract_cot_warmup_corpus(_fixture_records())
    b = build_abstract_cot_warmup_corpus(_fixture_records())
    assert a.corpus_fingerprint == b.corpus_fingerprint
