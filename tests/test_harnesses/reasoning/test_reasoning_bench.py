"""G4 (SLM-36): reasoning bench invariants + tiny end-to-end smoke."""

from __future__ import annotations

import pytest

from tests.casefiles import case_values

from slm_training.dsl.packs import get_pack
from slm_training.dsl.packs.arith_sketch import evaluate_answer
from slm_training.harnesses.reasoning import (
    score_direct_output,
    score_sketch_output,
)


def test_evaluator_is_the_single_oracle() -> None:
    """Validity and answer share one code path; every invalid shape rejects."""
    assert evaluate_answer("x = 3\ny = x * 4\nroot = y + 2") == 14.0
    assert evaluate_answer("root = 10 / 4") == pytest.approx(2.5)
    for bad in (
        "x = 3",                      # no root
        "root = y + 1",               # undefined ref
        "x = y\ny = x\nroot = x",     # cycle
        "root = 1 / 0",               # division by zero
        "root = ",                    # parse failure
    ):
        with pytest.raises(ValueError):
            evaluate_answer(bad)


def test_generator_gold_matches_oracle_and_is_deterministic() -> None:
    pack = get_pack("arith-sketch")
    records = pack.corpus_generator(12, 3)
    again = pack.corpus_generator(12, 3)
    assert [r.openui for r in records] == [r.openui for r in again]
    for record in records:
        assert evaluate_answer(record.openui) == record.meta["gold_answer"]


def test_scoring_is_fail_closed() -> None:
    gold = 14.0
    assert score_sketch_output("x = 3\ny = x * 4\nroot = y + 2", gold)["correct"]
    invalid = score_sketch_output("x = 3\nroot = z + 1", gold)
    assert not invalid["valid"] and not invalid["correct"]
    wrong = score_sketch_output("root = 13", gold)
    assert wrong["valid"] and not wrong["correct"]
    assert score_direct_output("14", gold)["correct"]
    assert not score_direct_output("nope", gold)["correct"]
    assert not score_direct_output("", gold)["correct"]


def test_bench_end_to_end_tiny(tmp_path) -> None:
    """Both arms train, decode, and score through the one oracle. Accuracy is
    not asserted (2 steps of training proves wiring, not skill)."""
    pytest.importorskip("torch")

    from slm_training.harnesses.reasoning import (
        ReasoningBenchConfig,
        run_reasoning_bench,
    )

    summary = run_reasoning_bench(
        ReasoningBenchConfig(
            n_train=8,
            n_test=3,
            steps=2,
            d_model=32,
            denoiser_layers=1,
            seed=1,
            campaign_id="g4_unit",
            output_root=tmp_path,
        )
    )
    assert summary["n_test"] == 3
    for arm in ("sketch", "direct"):
        assert 0.0 <= summary[arm]["answer_accuracy"] <= 1.0
        assert len(summary[arm]["outputs"]) == 3
    # Gold answers re-verify through the same oracle used for scoring.
    assert all(isinstance(g, float) for g in summary["golds"])
    assert summary["grammar_constrained"] is True
    assert summary["training_pack_id"] == "arith-sketch"
    assert summary["sketch"]["trace_validity_rate"] == 1.0
    for source, score in zip(summary["sketch"]["outputs"], summary["sketch"]["scores"], strict=True):
        assert evaluate_answer(source) == score["value"]


@pytest.fixture
def arithmetic_scope(monkeypatch):
    from slm_training.models import grammar
    from slm_training.dsl.grammar import backends

    previous, backend = grammar._ACTIVE_DSL, backends._DEFAULT_ID
    monkeypatch.setenv("SLM_GRAMMAR_DSL", "arith-sketch")
    grammar.set_active_dsl("arith-sketch")
    try:
        yield
    finally:
        grammar.set_active_dsl(previous)
        backends.set_default_backend(backend)


def _arithmetic_record(source="root = 2"):
    from slm_training.dsl.schema import ExampleRecord

    return ExampleRecord(id="arithmetic", prompt="compute", openui=source,
                         placeholders=[], meta={"pack_id": "arith-sketch"})


def _arithmetic_config():
    from slm_training.models.twotower import TwoTowerConfig

    return TwoTowerConfig(training_pack_id="arith-sketch", output_tokenizer="lexer",
                          context_backend="scratch", d_model=32, n_heads=4,
                          context_layers=1, denoiser_layers=1, max_target_len=32,
                          grammar_constrained=True, grammar_finalize_validate=True,
                          grammar_ltr_primary=True, compiler_decode_mode="off")


def test_arithmetic_ast_expansion_preserves_operations_and_answer():
    from slm_training.dsl.pack import get_pack as canonical_pack

    pack = canonical_pack("arith-sketch")
    source = "x = 4\ny = 11 * 3\nroot = x + y"
    expanded = pack.canonicalize(source)
    assert expanded == "root = (4 + (11 * 3))"
    assert pack.canonicalize(expanded) == expanded
    assert evaluate_answer(expanded) == evaluate_answer(source) == 37
    for record in pack.corpus_generator(12, 3):
        assert record.openui == pack.canonicalize(record.openui)


def test_model_build_factory_binds_canonical_pack(tmp_path):
    from slm_training.harnesses.model_build.config import ModelBuildConfig
    from slm_training.harnesses.model_build.factory import _twotower_config_from_build, apply_runtime_overrides

    config = ModelBuildConfig(train_dir=tmp_path, grammar_dsl="arith-sketch")
    assert _twotower_config_from_build(config).training_pack_id == "arith-sketch"
    config.grammar_dsl = "openui-lark"
    assert _twotower_config_from_build(config).training_pack_id == "openui"
    config.grammar_dsl = "forged-pack"
    with pytest.raises(KeyError):
        _twotower_config_from_build(config)
    with pytest.raises(KeyError):
        apply_runtime_overrides(object(), config)


def test_training_requires_resolved_pack_representation():
    from slm_training.models.twotower import TwoTowerModel

    raw = _arithmetic_record("x = 2\nroot = x + 3")
    with pytest.raises(ValueError, match="canonical pack targets"):
        TwoTowerModel.from_records([raw], config=_arithmetic_config())


def test_direct_arm_does_not_score_a_trace_as_a_bare_answer():
    assert not score_direct_output("14 + 1", 14)["correct"]


@pytest.mark.parametrize("pack_id", [None, "", "unknown", "default", "toy-layout"])
def test_training_rejects_missing_unknown_or_unsupported_pack_authority(pack_id):
    from slm_training.data.record_admission import assert_training_record

    record = _arithmetic_record()
    record.meta["pack_id"] = pack_id
    with pytest.raises((ValueError, KeyError, NotImplementedError)):
        assert_training_record(record)


@pytest.mark.parametrize("source", case_values(__file__, "test_invalid_arithmetic_rejected_before_model_build_and_training"))
def test_invalid_arithmetic_rejected_before_model_build_and_training(source):
    from slm_training.models.twotower import TwoTowerModel

    cfg = _arithmetic_config()
    invalid = _arithmetic_record(source)
    with pytest.raises(ValueError):
        TwoTowerModel.from_records([invalid], config=cfg)
    model = TwoTowerModel.from_records([_arithmetic_record()], config=cfg)
    with pytest.raises(ValueError):
        model.training_loss([invalid])


def test_openui_admission_and_model_pack_binding_stay_strict():
    from slm_training.data.record_admission import assert_training_record
    from slm_training.models.twotower import TwoTowerModel

    record = _arithmetic_record("root = (4 + 3)")
    record.meta.clear()
    with pytest.raises(ValueError):
        assert_training_record(record)
    with pytest.raises(ValueError, match="does not match model pack"):
        TwoTowerModel.from_records([_arithmetic_record()])
    cfg = _arithmetic_config()
    cfg.output_tokenizer = "compositional"
    with pytest.raises(ValueError, match="free-form-capable"):
        TwoTowerModel.from_records([_arithmetic_record()], config=cfg)


@pytest.mark.parametrize("source", ["", "root", "root =", "root = (1 /", "root = (1 + 2)"])
def test_every_arithmetic_domain_candidate_has_bounded_replayable_witness(source):
    from slm_training.dsl.pack import get_pack as canonical_pack
    from slm_training.dsl.grammar_capabilities import CompletionDomainRequestV1
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    prefix = tuple(tok.encode(source, add_special=False))
    domain = canonical_pack("arith-sketch").grammar_capability_authority.completion_domain(
        CompletionDomainRequestV1(prefix, tok, remaining_tokens=24))
    assert domain.status == "complete" and domain.candidates
    for candidate in domain.candidates:
        assert candidate.terminal_witness[:len(candidate.token_ids)] == candidate.token_ids
        assert len(candidate.terminal_witness) <= 24
        evaluate_answer(tok.decode([*prefix, *candidate.terminal_witness]))


def test_zero_divisor_and_exhausted_proof_fail_closed(monkeypatch):
    from slm_training.dsl import arith_completion
    from slm_training.dsl.pack import get_pack as canonical_pack
    from slm_training.dsl.grammar_capabilities import CompletionDomainRequestV1
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    authority = canonical_pack("arith-sketch").grammar_capability_authority
    prefix = tuple(tok.encode("root = 1 / 0", add_special=False))
    domain = authority.completion_domain(CompletionDomainRequestV1(prefix, tok, remaining_tokens=24))
    assert domain.status == "complete" and not domain.candidates
    monkeypatch.setattr(arith_completion, "_MAX_WITNESS_NODES", 0)
    unknown = authority.completion_domain(CompletionDomainRequestV1((), tok, remaining_tokens=24))
    assert unknown.status == "incomplete" and not unknown.candidates


def test_arithmetic_singletons_bypass_neural_forward(arithmetic_scope, monkeypatch):
    import torch
    from slm_training.models.twotower import TwoTowerModel

    model = TwoTowerModel.from_records([_arithmetic_record()], config=_arithmetic_config())
    original = model.denoiser.forward
    forwards_at_singletons = 0
    forwards = 0

    def observed_forward(ids, *args, **kwargs):
        nonlocal forwards, forwards_at_singletons
        forwards += 1
        prefix = ids[0].tolist()
        prefix = prefix[:prefix.index(model.tokenizer.mask_id)]
        if len(prefix) < 3:
            forwards_at_singletons += 1
        return original(ids, *args, **kwargs)

    monkeypatch.setattr(model.denoiser, "forward", observed_forward)
    ids = model._greedy_ltr_decode_batch(torch.zeros(1, 1, 32), torch.zeros(1, 1, dtype=torch.bool), 8)
    assert model.tokenizer.decode(ids[0, :3].tolist()).strip() == "root ="
    assert forwards > 0  # Ambiguous numeric/expression choice still uses this model.
    assert forwards_at_singletons == 0


@pytest.mark.parametrize("source", case_values(__file__, "test_numeric_canonicalization_preserves_grammar_and_value"))
def test_numeric_canonicalization_preserves_grammar_and_value(source, arithmetic_scope):
    from slm_training.dsl.arith_sketch import canonical_training_source
    from slm_training.models.twotower_surface import canonical_valid_openui

    canonical = canonical_training_source(source)
    assert "e" not in canonical
    assert evaluate_answer(canonical) == evaluate_answer(source)
    assert canonical_training_source(canonical) == canonical
    assert canonical_valid_openui(source) == canonical


def test_finalizer_rejects_invalid_canonical_output(arithmetic_scope, monkeypatch):
    from dataclasses import replace
    from slm_training.dsl import pack as registry
    from slm_training.models.twotower_surface import canonical_valid_openui

    pack = registry.get_pack("arith-sketch")
    corrupted = replace(pack, canonicalize=lambda source: "root = 1e-07")
    monkeypatch.setattr(registry, "get_pack", lambda: corrupted)
    assert canonical_valid_openui("root = 0.0000001") is None
