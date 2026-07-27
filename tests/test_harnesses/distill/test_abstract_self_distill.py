"""On-policy abstract self-distillation collector tests (AP-020 / SLM-309)."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.dsl.abstract_plan import AbstractPlanV1
from slm_training.harnesses.distill.abstract_self_distill import (
    CollectionSummary,
    append_records,
    build_self_distill_record,
    collect_on_policy_traces,
    load_seen_hashes,
)
from slm_training.models.abstract_decode import (
    AbstractDecodeConfig,
    AbstractPhaseRecord,
    AbstractSamplingParams,
    AbstractTraceCapture,
    AbstractTracedGeneration,
    DecodePhase,
)
from slm_training.models.causal_lm_openui import CausalLMOpenUIConfig, CausalLMOpenUIPlugin

PLAN = AbstractPlanV1(slot_count=4, max_slot_count=4)
VALID_TEXT = "root = Separator()"
INVALID_TEXT = "not a program"


class _FakeConfig:
    pass


class _FakeModel:
    """Torch-free stand-in: compatibility_fingerprint() never imports torch."""

    config = _FakeConfig()

    def state_dict(self) -> dict:
        return {}


class _FakeTokenizer:
    """Maps fixed token-id tuples to exact program text, independent of real tokenization."""

    init_kwargs: dict = {}

    def __init__(self, decode_map: dict[tuple[int, ...], str]) -> None:
        self._decode_map = decode_map

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self._decode_map.get(tuple(int(i) for i in ids), "")


def _plugin(decode_map: dict[tuple[int, ...], str]) -> CausalLMOpenUIPlugin:
    config = CausalLMOpenUIConfig(
        base_model_id="fake",
        base_model_revision="rev",
        abstract=AbstractDecodeConfig(
            plan=PLAN,
            m_max=4,
            abstract=AbstractSamplingParams(temperature=0.0, seed=1),
            answer=AbstractSamplingParams(temperature=0.0, seed=2),
        ),
    )
    return CausalLMOpenUIPlugin(_FakeModel(), _FakeTokenizer(decode_map), config)


def _capture(
    *, abstract_ids: tuple[int, ...] = (10, 11), answer_ids: tuple[int, ...] = (1, 2, 3)
) -> AbstractTraceCapture:
    return AbstractTraceCapture(
        plan_fingerprint=PLAN.compatibility_fingerprint,
        prompt=AbstractPhaseRecord(DecodePhase.PROMPT, (0,), (None,)),
        abstract=AbstractPhaseRecord(
            DecodePhase.ABSTRACT, abstract_ids, tuple(-0.1 * i for i in range(len(abstract_ids)))
        ),
        answer=AbstractPhaseRecord(
            DecodePhase.ANSWER, answer_ids, tuple(-0.2 * i for i in range(len(answer_ids)))
        ),
        abstract_termination="sampled_end",
        forced_end=False,
        stop_reason="eos",
    )


def test_build_record_accepts_independently_valid_raw_answer() -> None:
    plugin = _plugin({(1, 2, 3): VALID_TEXT})
    capture = _capture()
    generation = AbstractTracedGeneration(text=VALID_TEXT, capture=capture, valid=True)

    record = build_self_distill_record(plugin, "Make a card", generation)

    assert record.verifier_accepted is True
    assert record.verifier_reason == ""
    assert record.raw_answer_text == VALID_TEXT
    assert record.codebook_version == PLAN.codebook_version
    assert record.plan_fingerprint == PLAN.compatibility_fingerprint
    assert record.forced_end is False
    assert record.abstract_termination == "sampled_end"
    assert record.trace_logprob == pytest.approx(-0.1 - 0.2 - 0.4)
    assert 0 <= record.shard < 16


def test_build_record_rejects_when_raw_answer_independently_invalid() -> None:
    """generation.valid is always True (certified fallback masks failure); the
    collector must re-derive verification from the raw decode itself."""
    plugin = _plugin({(1, 2, 3): INVALID_TEXT})
    capture = _capture()
    # Simulate generate_abstract_traced's fallback: text is the certified
    # fallback (always "valid"), but the raw decode independently fails.
    generation = AbstractTracedGeneration(
        text="root = Separator()", capture=capture, valid=True
    )

    record = build_self_distill_record(plugin, "Make a card", generation)

    assert record.verifier_accepted is False
    assert "not a program" not in record.verifier_reason  # reason names the exception, not raw text
    assert record.raw_answer_text == INVALID_TEXT  # rejected samples remain auditable


def test_content_hash_is_deterministic_and_prompt_sensitive() -> None:
    plugin = _plugin({(1, 2, 3): VALID_TEXT})
    capture = _capture()
    generation = AbstractTracedGeneration(text=VALID_TEXT, capture=capture, valid=True)

    a = build_self_distill_record(plugin, "Make a card", generation)
    b = build_self_distill_record(plugin, "Make a card", generation)
    c = build_self_distill_record(plugin, "Make a button", generation)

    assert a.content_hash == b.content_hash
    assert a.content_hash != c.content_hash


def test_collect_on_policy_traces_requires_enabled_abstract_config() -> None:
    plugin = CausalLMOpenUIPlugin(
        _FakeModel(), _FakeTokenizer({}), CausalLMOpenUIConfig(base_model_id="fake", base_model_revision="rev")
    )
    with pytest.raises(ValueError, match="AbstractDecodeConfig"):
        collect_on_policy_traces(plugin, ["prompt"])


def test_collect_on_policy_traces_is_deterministic_dedup_and_auditable() -> None:
    plugin = _plugin({(1, 2, 3): VALID_TEXT, (4, 5): INVALID_TEXT})

    def generate_fn(prompt: str) -> AbstractTracedGeneration:
        if prompt == "good":
            capture = _capture(answer_ids=(1, 2, 3))
        else:
            capture = _capture(abstract_ids=(12,), answer_ids=(4, 5))
        return AbstractTracedGeneration(text="root = Separator()", capture=capture, valid=True)

    prompts = ["good", "bad", "good"]  # "good" repeats -> must dedup on the second pass
    records, summary = collect_on_policy_traces(plugin, prompts, generate_fn=generate_fn)

    assert isinstance(summary, CollectionSummary)
    assert summary.attempted == 3
    assert summary.duplicate == 1
    assert summary.accepted == 1
    assert summary.rejected == 1
    assert len(records) == 2  # duplicate "good" was skipped, not appended
    # Failed/rejected samples remain auditable, not dropped.
    rejected = [r for r in records if not r.verifier_accepted]
    assert len(rejected) == 1
    assert rejected[0].raw_answer_text == INVALID_TEXT

    # Determinism: an identical second run (fresh seen-set) yields identical hashes.
    records_2, _ = collect_on_policy_traces(plugin, prompts, generate_fn=generate_fn)
    assert [r.content_hash for r in records] == [r.content_hash for r in records_2]


def test_collect_on_policy_traces_resumes_via_seen_hashes() -> None:
    plugin = _plugin({(1, 2, 3): VALID_TEXT})

    def generate_fn(_prompt: str) -> AbstractTracedGeneration:
        return AbstractTracedGeneration(
            text="root = Separator()", capture=_capture(), valid=True
        )

    first_records, _ = collect_on_policy_traces(plugin, ["a prompt"], generate_fn=generate_fn)
    seen = {record.content_hash for record in first_records}

    resumed_records, resumed_summary = collect_on_policy_traces(
        plugin, ["a prompt"], seen_content_hashes=seen, generate_fn=generate_fn
    )

    assert resumed_records == []
    assert resumed_summary.duplicate == 1
    assert resumed_summary.accepted == 0


def test_append_records_and_load_seen_hashes_round_trip(tmp_path: Path) -> None:
    plugin = _plugin({(1, 2, 3): VALID_TEXT})
    generation = AbstractTracedGeneration(
        text="root = Separator()", capture=_capture(), valid=True
    )
    record = build_self_distill_record(plugin, "Make a card", generation)
    path = tmp_path / "self_distill.jsonl"

    assert load_seen_hashes(path) == set()  # missing file resumes empty, not an error

    append_records(path, [record])
    append_records(path, [record])  # append-only: never rewrites existing rows

    seen = load_seen_hashes(path)
    assert seen == {record.content_hash}
    assert path.read_text().count("\n") == 2


def test_self_distill_training_batch_reuses_forward_with_segments() -> None:
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    # Build the record with the fake (JSON-serializable) tokenizer first --
    # build_self_distill_record's compatibility_fingerprint() call chokes on
    # a real HF tokenizer's non-JSON-serializable init_kwargs (a pre-existing
    # artifact_identity() limitation, unrelated to this collector). Swap in
    # the real model only for the forward pass this test actually needs.
    plugin = _plugin({(1, 2, 3): VALID_TEXT})
    generation = AbstractTracedGeneration(
        text="root = Separator()",
        capture=_capture(abstract_ids=(10, 11), answer_ids=(1, 2, 3)),
        valid=True,
    )
    record = build_self_distill_record(plugin, "Make a card", generation)
    assert record.verifier_accepted is True

    plugin.model = transformers.AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )

    from slm_training.harnesses.distill.abstract_self_distill import (
        self_distill_training_batch,
    )

    result = self_distill_training_batch(plugin, (5, 6), record)

    assert result["effective_token_count"] == len(record.abstract_token_ids) + len(
        record.answer_token_ids
    )
    assert result["loss"] == result["loss"]  # not NaN


def test_self_distill_training_batch_rejects_unverified_record() -> None:
    plugin = _plugin({(1, 2, 3): INVALID_TEXT})
    generation = AbstractTracedGeneration(
        text="root = Separator()", capture=_capture(), valid=True
    )
    record = build_self_distill_record(plugin, "Make a card", generation)
    assert record.verifier_accepted is False

    from slm_training.harnesses.distill.abstract_self_distill import (
        self_distill_training_batch,
    )

    with pytest.raises(ValueError, match="rejected"):
        self_distill_training_batch(plugin, (5, 6), record)
