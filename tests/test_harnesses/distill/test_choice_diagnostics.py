"""Real recorder/decode integration and adversarial choice evidence fixtures."""

import pytest

from tests.casefiles import case_values

from slm_training.harnesses.distill.choice_diagnostics import (
    compare_legal_controls,
    decision_diagnostics,
    legal_logits,
    rank_single_hole,
)
from slm_training.harnesses.distill.grammar_trace import (
    GrammarTraceRecorder,
    normalize_legal_probs,
)


def test_singleton_controls_never_call_any_ranker():
    def forbidden(_):
        pytest.fail("ranker called on a proven singleton")

    result = compare_legal_controls(
        ["="], coverage="complete", scorers={"learned": forbidden, "expert": forbidden}
    )
    assert result["scorer_calls"] == 0


def test_controls_receive_identical_legal_domain():
    domains = []

    def score(values):
        domains.append(values)
        return [0.1, 0.8]

    result = compare_legal_controls(
        ["Button", "TextContent"],
        coverage="complete",
        scorers={"learned": score, "train_only": score},
        oracle_targets=["Button"],
    )
    assert domains[0] == domains[1] == ("Button", "TextContent")
    assert result["expert_upper_bound"] == "Button"
    assert result["expert_deployable"] is False


@pytest.mark.parametrize(
    "targets,selected,outcome,cause",
    case_values(__file__, "test_attribution_requires_target_evidence"),
)
def test_attribution_requires_target_evidence(targets, selected, outcome, cause):
    row = {
        "legal_action_ids": ["a", "b"],
        "compiler_coverage": "complete",
        "target_action_ids": targets,
        "selected_action_id": selected,
        "verification_outcome": outcome,
    }
    assert decision_diagnostics(row)["failure_attribution"] == cause


def test_full_vocabulary_logits_are_not_a_legal_distribution():
    with pytest.raises(ValueError, match="aligned"):
        decision_diagnostics(
            {"legal_action_ids": ["a", "b"], "logits_or_energies": [1.0, 2.0, 100.0]}
        )


def test_actual_legal_projection_excludes_illegal_highest_logit():
    from types import SimpleNamespace

    import torch

    tok = SimpleNamespace(id_to_token={0: "illegal", 1: "b", 2: "a"})
    values = legal_logits(tok, ["a", "b"], torch.tensor([1000.0, 1.0, 2.0]))
    assert values == [2.0, 1.0]


def test_energy_normalization_is_stable_for_large_finite_span():
    assert normalize_legal_probs([0.0, 1000.0], convention="energy") == (1.0, 0.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_score_cannot_become_choice_probability(value):
    with pytest.raises(ValueError, match="finite"):
        normalize_legal_probs([0.0, value])


def test_actual_nonprefix_singleton_trace_spends_zero_forwards(monkeypatch):
    from slm_training.models.decode_stats import collect_decode_stats
    from tests.test_models.test_step_commit_histogram import EXACT_SOURCE, _tiny_model

    model = _tiny_model(
        gen_steps=1, grammar_fastpath_mode="ltr", allow_unconstrained_fallback=False
    )
    tok = model.tokenizer
    canvas = [tok.bos_id, *tok.encode(EXACT_SOURCE, add_special=False), tok.eos_id]
    canvas[canvas.index(tok.token_to_id["="])] = tok.mask_id
    ctx, pad = model._encode_context(["Hero card"])
    recorder = GrammarTraceRecorder(capture_logits=True)
    model.grammar_trace_recorder = recorder

    def forbidden(*args, **kwargs):
        pytest.fail("singleton instrumentation called neural forward")

    monkeypatch.setattr(model, "_denoiser_forward", forbidden)
    with collect_decode_stats() as stats:
        model._generate_maskgit_one(
            ctx,
            pad,
            len(canvas),
            use_grammar=True,
            slot_contract=[":slot_0"],
            seed_ids=canvas,
        )
    rows = recorder.finalize()
    assert stats.forwards_count == 0
    assert len(rows) == 1
    assert rows[0]["state_signature_version"] == "diffusion_canvas/v1"
    assert rows[0]["choice_diagnostics"]["forced"]
    assert rows[0]["logits_or_energies"] is None


def test_exhaustive_singleton_diagnostic_never_encodes_or_scores(monkeypatch):
    from tests.test_models.test_step_commit_histogram import EXACT_SOURCE, _tiny_model

    model = _tiny_model()
    tok = model.tokenizer
    canvas = tok.encode(EXACT_SOURCE, add_special=True)
    position = canvas.index(tok.token_to_id["="])
    canvas[position] = tok.mask_id

    def forbidden(*args, **kwargs):
        pytest.fail("proven single-hole singleton spent inference")

    monkeypatch.setattr(model, "_encode_context", forbidden)
    monkeypatch.setattr(model, "_denoiser_forward", forbidden)
    _, trace = rank_single_hole(
        model, "unused", canvas, position, context_controls=True
    )
    assert trace["forwards"] == 0
    assert tok.token_to_id["="] in trace["legal_token_ids"]
    assert trace["distinct_certified_programs"] == 1
    assert trace["equivalence_certificate"] == "identical_decoded_program/v1"
