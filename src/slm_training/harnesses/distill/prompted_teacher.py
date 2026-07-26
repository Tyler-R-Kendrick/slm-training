"""SLM-429 (VAR3-01): a real, prompted-LLM teacher behind ``OperatorTeacherAdapter``.

``operator_teacher_ceiling.py`` documents three preconditions for a
production KD-readiness claim: (1) a real teacher behind
``OperatorTeacherAdapter``, (2) re-running ``compare_operator_teacher_ceiling``
unchanged against it, and (3) an honest stop-rule read of that real output.
This module is precondition (1): ``PromptedTeacherV1`` implements the same
narrow Protocol (``source_id: str`` + ``rank(query) -> RankingV1 | None``)
that ``SyntheticDescriptorTeacherV1`` implements, so the comparison harness
requires no change to consume it (``compare_operator_teacher_ceiling``'s only
touch point is its ``teacher=`` argument).

Honesty scoping
----------------
Whether preconditions (2)/(3) are actually satisfied by a *given run* of this
module depends entirely on whether a real, authenticated LLM client was
injected. ``PromptedTeacherV1`` accepts a dependency-injected ``client``
(mirroring ``slm_training.autoresearch.providers.OpenAIResearchProvider``'s
convention) so it can be exercised in tests, or with a fixture/fake client,
without ever claiming that run used a live model. A caller publishing results
from this module must record, alongside the report, whether ``client`` was a
real ``openai.OpenAI()`` instance or an injected stand-in -- see
``scripts/run_var3_01_real_teacher_ceiling.py``.

Anti-leak surface (matches ``SyntheticDescriptorTeacherV1`` exactly)
----------------------------------------------------------------------
The prompt this module renders exposes only ``operator_id``, argument arity,
``proof_checks``, and a ``semantic_id``-derived identity term -- the same
structural features ``SyntheticDescriptorTeacherV1`` reads. It never renders
``trace.current_scores``, ``trace.accepted_application_ids``, or any real
``application_id``; per-candidate labels come from ``query.opaque_id_labels``
(a caller-controlled, perturbable presentation label), never from the
comparator-internal id. Unlike the synthetic teacher, this module's *prompt
text* (not its scoring authority) legitimately varies with
``query.candidate_order``, ``query.opaque_id_labels``,
``query.description_lengths``, and ``query.prompt_template_hash`` -- that is
the whole point of ``run_perturbation_robustness()``: a real prompted teacher
may show non-zero order/label/length/template sensitivity, and the harness
measures that honestly rather than assuming it away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from slm_training.dsl.operators.legal_set import LegalOperatorActionV1
from slm_training.harnesses.distill.operator_decision_state import (
    OperatorDecisionStateTraceV1,
)
from slm_training.harnesses.distill.operator_teacher_ceiling import (
    RankingV1,
    TeacherQueryV1,
    _empty_ranking,
    _trace_reliability,
)

__all__ = [
    "PromptedTeacherResponseV1",
    "PromptedTeacherV1",
    "build_prompt",
]


class PromptedTeacherResponseV1(BaseModel):
    """The one JSON shape a prompted teacher call must return.

    ``ranked_candidate_labels`` must be a permutation of the presented
    ``candidate-N``/relabeled opaque labels the prompt showed, best-first.
    Labels the query's ``opaque_id_labels`` cannot reverse-map, or a
    non-permutation (missing/duplicate/extra labels), make the response
    unusable -- ``PromptedTeacherV1.rank`` returns ``None`` rather than
    fabricate a ranking from a malformed reply.
    """

    ranked_candidate_labels: list[str] = Field(min_length=1)
    confidence: list[float] | None = Field(default=None)


def _action_by_label(
    query: TeacherQueryV1,
) -> tuple[dict[str, LegalOperatorActionV1], dict[str, str]]:
    actions_by_id = {
        action.application_id: action for action in query.trace.legal_set.operator_actions
    }
    label_to_id = {label: aid for aid, label in query.opaque_id_labels.items()}
    return actions_by_id, label_to_id


def _render_candidate(
    action: LegalOperatorActionV1, *, label: str, description_length: int
) -> str:
    # Deterministic filler standing in for an upstream natural-language
    # description of the declared length -- there is no real per-candidate
    # NL description on LegalOperatorActionV1 (DSH3-06 is digest-only), so a
    # real prompted-teacher pipeline's rendered description length is itself
    # part of what SLM-387's description-length perturbation exercises.
    filler = ("..." * ((description_length // 3) + 1))[:description_length]
    return (
        f"- label: {label}\n"
        f"  operator: {action.operator_id}\n"
        f"  argument_arity: {len(action.arguments)}\n"
        f"  proof_checks: {list(action.proof_checks)}\n"
        f"  identity: {action.semantic_id[:16]}\n"
        f"  description_filler: {filler!r}"
    )


def build_prompt(query: TeacherQueryV1) -> str:
    """Render ``query`` into a ranking prompt.

    Pure function -- no client, no network -- so its anti-leak surface is
    directly unit-testable: it must never contain a real ``application_id``,
    ``trace.current_scores``, or ``trace.accepted_application_ids`` value.
    """
    actions_by_id, _ = _action_by_label(query)
    blocks = []
    for aid in query.candidate_order:
        action = actions_by_id[aid]
        label = query.opaque_id_labels[aid]
        blocks.append(
            _render_candidate(
                action, label=label, description_length=query.description_lengths.get(aid, 0)
            )
        )
    return (
        "You are ranking candidate compiler operator actions by how likely a "
        "user intends them, from most to least likely. Use only the "
        "structural fields shown for each candidate -- you are not given, "
        "and must not guess at, any score, acceptance label, or real "
        "identifier beyond what is listed.\n\n"
        f"prompt_template: {query.prompt_template_hash}\n\n"
        "Candidates (presentation order is not meaningful evidence):\n"
        + "\n".join(blocks)
        + "\n\nReturn every candidate's label in `ranked_candidate_labels`, "
        "best-first, with no omissions, duplicates, or invented labels."
    )


@dataclass
class PromptedTeacherV1:
    """A real, prompted-LLM ``OperatorTeacherAdapter``.

    ``client`` is dependency-injected (mirroring
    ``autoresearch.providers.OpenAIResearchProvider``): pass a real
    ``openai.OpenAI()`` instance (or anything exposing a compatible
    ``.responses.parse(model=..., input=..., text_format=...)`` method) for a
    genuine run, or a fake/fixture object in tests. With ``client=None`` the
    constructor lazily imports and constructs ``openai.OpenAI()``, raising
    ``RuntimeError`` if the ``openai`` package is not installed -- the same
    fail-closed convention every other real-provider adapter in this repo
    uses; it never silently degrades to a fixture.
    """

    model: str = "gpt-5.6-sol"
    client: Any = None
    source_id: str = "teacher:prompted_v1"

    def __post_init__(self) -> None:
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency error
                raise RuntimeError(
                    "install slm-training[research] for OpenAI research"
                ) from exc
            self.client = OpenAI()

    def rank(self, query: TeacherQueryV1) -> RankingV1 | None:
        trace: OperatorDecisionStateTraceV1 = query.trace
        actions_by_id, label_to_id = _action_by_label(query)
        if not actions_by_id:
            return _empty_ranking(trace, self.source_id, "empty_legal_set")

        prompt = build_prompt(query)
        try:
            response = self.client.responses.parse(
                model=self.model,
                input=prompt,
                text_format=PromptedTeacherResponseV1,
            )
            parsed = response.output_parsed
            if not isinstance(parsed, PromptedTeacherResponseV1):
                parsed = PromptedTeacherResponseV1.model_validate(parsed)
        except Exception:  # noqa: BLE001 - any client/parse failure is a non-ranking, not a crash
            return None

        resolved_ids: list[str] = []
        for label in parsed.ranked_candidate_labels:
            aid = label_to_id.get(label)
            if aid is None:
                return None
            resolved_ids.append(aid)
        if len(set(resolved_ids)) != len(resolved_ids) or set(resolved_ids) != set(
            actions_by_id
        ):
            return None

        probabilities: tuple[float, ...] | None = None
        if parsed.confidence is not None and len(parsed.confidence) == len(resolved_ids):
            if all(value >= 0.0 for value in parsed.confidence):
                total = sum(parsed.confidence)
                if total > 0.0:
                    probabilities = tuple(value / total for value in parsed.confidence)

        reliability, approximate, coverage_reason = _trace_reliability(trace)
        return RankingV1(
            trace_id=trace.trace_id,
            source_id=self.source_id,
            application_order=tuple(resolved_ids),
            probabilities=probabilities,
            reliability=reliability,
            reliability_reason=f"prompted_llm_response; {coverage_reason}",
            approximate=approximate,
        )
