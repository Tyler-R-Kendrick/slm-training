"""Tests for VAR3-01 PromptedTeacherV1 (SLM-429).

Mirrors the synthetic stand-in's unit-test bar with recorded provider
responses (injected transport -- tests never touch the network):

* adapter conformance against the existing ``OperatorTeacherAdapter``
  Protocol (same ``TeacherQueryV1`` in, ``RankingV1`` out, probabilities
  normalized, reliability metadata honored);
* the anti-leak assertions: the rendered prompt can never contain opaque
  application ids, ``current_scores`` values, or
  ``accepted_application_ids``;
* fail-closed behavior on malformed provider replies;
* content-addressed caching: a repeated query never re-queries the
  provider;
* perturbation surface: candidate order / opaque-id relabeling /
  description-length / prompt-template variation reach the prompt, and the
  response maps back to application ids through the presented labels.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from slm_training.harnesses.distill import operator_teacher_ceiling as otc
from slm_training.harnesses.distill.prompted_teacher import (
    PromptedTeacherV1,
    render_teacher_prompt,
)
from tests.test_harnesses.distill.test_operator_teacher_ceiling import (
    _build,
    _sha,
)


def _response_for(labels: list[str], scores: list[float]) -> Mapping[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"ranking": [{"label": label, "score": score} for label, score in zip(labels, scores)]}
                    )
                }
            }
        ]
    }


class _RecordedTransport:
    """Recorded-response transport: answers every request with a ranking
    derived *only* from the labels present in the prompt, so tests stay
    offline and deterministic. Counts calls to prove cache behavior."""

    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append(payload)
        user = payload["messages"][1]["content"]
        labels = [
            line.split(":", 1)[0].removeprefix("- ").strip()
            for line in user.splitlines()
            if line.startswith("- ")
        ]
        # Deterministic "teacher": prefers later labels (a non-trivial,
        # label-order-independent rule would be needed to be a good teacher;
        # for conformance testing any deterministic rule suffices).
        scores = [float(i + 1) for i in range(len(labels))]
        return _response_for(labels, scores)


def _teacher(tmp_path, transport) -> PromptedTeacherV1:
    return PromptedTeacherV1(cache_dir=tmp_path / "cache", transport=transport)


# --------------------------------------------------------------------------- #
# Adapter conformance
# --------------------------------------------------------------------------- #


def test_prompted_teacher_returns_valid_ranking_v1(tmp_path) -> None:
    fx = _build(trace_id="pt-conform", n_candidates=5, accepted_indices=(2,))
    teacher = _teacher(tmp_path, _RecordedTransport())
    query = otc.default_teacher_query(fx.trace)
    ranking = teacher.rank(query)
    assert ranking is not None
    assert ranking.source_id == teacher.source_id
    assert set(ranking.application_order) == {
        a.application_id for a in fx.trace.legal_set.operator_actions
    }
    assert ranking.probabilities is not None
    assert abs(sum(ranking.probabilities) - 1.0) < 1e-9
    assert ranking.reliability is otc.RankingReliability.EXACT
    assert "prompted_llm_ranking" in ranking.reliability_reason


def test_prompted_teacher_empty_legal_set(tmp_path) -> None:
    fx = _build(trace_id="pt-empty", n_candidates=0)
    teacher = _teacher(tmp_path, _RecordedTransport())
    ranking = teacher.rank(otc.default_teacher_query(fx.trace))
    assert ranking is not None
    assert ranking.application_order == ()


def test_prompted_teacher_marks_partial_coverage_approximate(tmp_path) -> None:
    fx = _build(
        trace_id="pt-approx", n_candidates=5, max_combinations_per_operator=1, approximate=True
    )
    teacher = _teacher(tmp_path, _RecordedTransport())
    ranking = teacher.rank(otc.default_teacher_query(fx.trace))
    assert ranking is not None
    assert ranking.approximate is True
    assert ranking.reliability is otc.RankingReliability.APPROXIMATE


# --------------------------------------------------------------------------- #
# Anti-leak assertions
# --------------------------------------------------------------------------- #


def test_prompt_never_contains_evaluator_only_or_opaque_fields(tmp_path) -> None:
    fx = _build(
        trace_id="pt-antileak",
        n_candidates=5,
        accepted_indices=(2,),
        current_scores_by_index={0: 0.111, 1: 0.222, 2: 0.333, 3: 0.444, 4: 0.555},
    )
    trace = fx.trace
    prompt = render_teacher_prompt(otc.default_teacher_query(trace))
    # Opaque application ids must never reach the teacher.
    for action in trace.legal_set.operator_actions:
        assert action.application_id not in prompt
        assert action.serialized not in prompt
    # Gold/evaluator-only labels must never reach the teacher.
    for accepted_id in trace.accepted_application_ids:
        assert accepted_id not in prompt
    for score_text in ("0.111", "0.222", "0.333", "0.444", "0.555"):
        assert score_text not in prompt
    # The allowed surface must be present: operator id, arity, proof checks,
    # and a semantic_id-derived identity term.
    assert "openui.dsh4_02_test_fixture" in prompt
    assert "arity=1" in prompt
    assert "proof_checks=" in prompt
    assert "identity=" in prompt


def test_teacher_does_not_read_current_scores_or_accepted_ids(tmp_path) -> None:
    """Two traces differing only in evaluator-only labels must produce the
    identical provider request payload (and hence the same cached response)."""
    fx_a = _build(
        trace_id="pt-blind-a",
        n_candidates=5,
        accepted_indices=(0,),
        current_scores_by_index={0: 0.9, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1},
    )
    fx_b = _build(
        trace_id="pt-blind-b",
        n_candidates=5,
        accepted_indices=(3,),
        current_scores_by_index={0: 0.1, 1: 0.1, 2: 0.1, 3: 0.9, 4: 0.1},
    )
    teacher = _teacher(tmp_path, _RecordedTransport())
    payload_a = teacher._request_payload(otc.default_teacher_query(fx_a.trace))
    payload_b = teacher._request_payload(otc.default_teacher_query(fx_b.trace))
    # The candidate vocabularies of the two fixture traces are identical, so
    # blind-to-labels prompts must be byte-identical.
    assert payload_a == payload_b


# --------------------------------------------------------------------------- #
# Fail closed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "content",
    [
        "not json at all",
        json.dumps({"no_ranking_key": True}),
        json.dumps({"ranking": []}),
        json.dumps({"ranking": [{"label": "candidate-0", "score": -1.0}]}),
        json.dumps({"ranking": [{"label": "candidate-0", "score": 0.0}]}),  # missing labels
        json.dumps(
            {"ranking": [{"label": "candidate-0", "score": 0.5}] * 2}
        ),  # duplicate labels
        json.dumps({"ranking": [{"label": "candidate-0", "score": 0.0}]}),
    ],
)
def test_malformed_replies_fail_closed(tmp_path, content: str) -> None:
    fx = _build(trace_id="pt-malformed", n_candidates=2)
    transport = lambda payload: {"choices": [{"message": {"content": content}}]}  # noqa: E731
    teacher = _teacher(tmp_path, transport)
    assert teacher.rank(otc.default_teacher_query(fx.trace)) is None


def test_all_zero_scores_fail_closed(tmp_path) -> None:
    fx = _build(trace_id="pt-zeros", n_candidates=2)
    query = otc.default_teacher_query(fx.trace)
    labels = [query.opaque_id_labels[a.application_id] for a in fx.trace.legal_set.operator_actions]
    transport = lambda payload: _response_for(labels, [0.0, 0.0])  # noqa: E731
    teacher = _teacher(tmp_path, transport)
    assert teacher.rank(query) is None


def test_provider_error_fails_closed(tmp_path) -> None:
    fx = _build(trace_id="pt-httperror", n_candidates=2)

    def transport(payload):
        raise RuntimeError("provider down")

    teacher = _teacher(tmp_path, transport)
    assert teacher.rank(otc.default_teacher_query(fx.trace)) is None


# --------------------------------------------------------------------------- #
# Content-addressed caching
# --------------------------------------------------------------------------- #


def test_repeated_query_hits_cache_without_requerying(tmp_path) -> None:
    fx = _build(trace_id="pt-cache", n_candidates=5, accepted_indices=(1,))
    transport = _RecordedTransport()
    teacher = _teacher(tmp_path, transport)
    query = otc.default_teacher_query(fx.trace)
    first = teacher.rank(query)
    second = teacher.rank(query)
    assert first is not None and second is not None
    assert first.application_order == second.application_order
    assert first.probabilities == second.probabilities
    assert len(transport.calls) == 1
    cache_files = list((tmp_path / "cache").glob("*.json"))
    assert len(cache_files) == 1
    record = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert record["schema"] == "prompted_teacher_exchange/v1"
    assert "request" in record and "response" in record


def test_cache_is_shared_across_teacher_instances(tmp_path) -> None:
    fx = _build(trace_id="pt-cache-share", n_candidates=5)
    transport = _RecordedTransport()
    query = otc.default_teacher_query(fx.trace)
    assert _teacher(tmp_path, transport).rank(query) is not None
    # A fresh instance over the same cache dir must not re-query.
    assert _teacher(tmp_path, transport).rank(query) is not None
    assert len(transport.calls) == 1


# --------------------------------------------------------------------------- #
# Perturbation surface: labels map back; presentation reaches the prompt
# --------------------------------------------------------------------------- #


def test_opaque_id_relabel_maps_back_to_application_ids(tmp_path) -> None:
    fx = _build(trace_id="pt-relabel", n_candidates=5, accepted_indices=(2,))
    teacher = _teacher(tmp_path, _RecordedTransport())
    canonical = teacher.rank(otc.default_teacher_query(fx.trace))
    perturbed = teacher.rank(
        otc.perturb_opaque_id_labels(otc.default_teacher_query(fx.trace), seed=7)
    )
    assert canonical is not None and perturbed is not None
    # The recorded transport ranks by label position in the prompt, so under
    # relabeling the *application-id* order must be identical (only the
    # presentation names changed).
    assert canonical.application_order == perturbed.application_order


def test_candidate_order_perturbation_reaches_the_prompt(tmp_path) -> None:
    fx = _build(trace_id="pt-order", n_candidates=5)
    canonical_prompt = render_teacher_prompt(otc.default_teacher_query(fx.trace))
    shuffled = render_teacher_prompt(
        otc.perturb_candidate_order(otc.default_teacher_query(fx.trace), seed=3)
    )
    assert canonical_prompt != shuffled  # presentation changed
    # The candidate *set* presented is unchanged.
    assert sorted(canonical_prompt.splitlines()) == sorted(shuffled.splitlines())


def test_description_length_perturbation_truncates_descriptions(tmp_path) -> None:
    fx = _build(trace_id="pt-desc", n_candidates=3)
    canonical = render_teacher_prompt(otc.default_teacher_query(fx.trace))
    truncated = render_teacher_prompt(
        otc.perturb_description_length(otc.default_teacher_query(fx.trace), seed=5)
    )
    assert "identity=" in canonical
    assert canonical != truncated or all(
        len(line) < 60 for line in truncated.splitlines() if line.startswith("- ")
    )


def test_prompt_template_hash_selects_among_preregistered_templates(tmp_path) -> None:
    fx = _build(trace_id="pt-template", n_candidates=3)
    prompts = {
        render_teacher_prompt(
            otc.default_teacher_query(fx.trace, prompt_template_hash=_sha(f"template-{i}"))
        )
        for i in range(8)
    }
    # At least two distinct preregistered paraphrases are exercised.
    assert len(prompts) >= 2


def test_end_to_end_comparison_accepts_prompted_teacher(tmp_path) -> None:
    """The unchanged harness consumes PromptedTeacherV1 exactly like the
    synthetic stand-in: full report shape, all comparators present."""
    training = [
        _build(trace_id=f"pt-e2e-train-{i}", n_candidates=5, accepted_indices=(2,)).trace
        for i in range(6)
    ]
    eval_traces = [
        _build(
            trace_id=f"pt-e2e-eval-{i}",
            n_candidates=5,
            accepted_indices=(2,),
            current_scores_by_index={0: 0.1, 1: 0.1, 2: 0.6, 3: 0.1, 4: 0.1},
        ).trace
        for i in range(6)
    ]
    teacher = _teacher(tmp_path, _RecordedTransport())
    report = otc.compare_operator_teacher_ceiling(
        eval_traces=eval_traces,
        frequency_training_traces=training,
        teacher=teacher,
        n_bootstrap=100,
    )
    assert report.teacher_source_id == teacher.source_id
    assert teacher.source_id in report.metrics_overall
    assert set(report.metrics_by_family)  # family slices ran
    assert set(report.perturbation.kind_distances) == {
        "candidate_order",
        "opaque_id_relabel",
        "description_length",
        "prompt_template",
    }
    assert isinstance(report.stop_rule, otc.StopRuleVerdictV1)
