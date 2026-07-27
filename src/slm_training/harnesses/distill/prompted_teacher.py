"""VAR3-01 (SLM-429): a real prompted LLM teacher behind ``OperatorTeacherAdapter``.

DSH4-02 (SLM-387, ``operator_teacher_ceiling.py``) fired its stop rule as
``no_go_defer`` because the only wired teacher was the deterministic,
non-learned ``SyntheticDescriptorTeacherV1`` stand-in. This module wires a
real teacher -- a prompted instruction-tuned LLM reached over an
OpenAI-compatible chat-completions endpoint -- behind the *existing*
``OperatorTeacherAdapter`` Protocol. **The comparison harness is not
modified**; ``compare_operator_teacher_ceiling`` consumes this teacher
exactly as it consumed the synthetic stand-in.

Anti-leak contract (identical to the synthetic stand-in)
---------------------------------------------------------
The prompt is built only from fields a teacher is allowed to see:

* ``operator_id``, argument arity, ``proof_checks``, and a
  ``semantic_id``-derived identity term (a truncated content-identity
  digest, never an opaque id);

plus *presentation-only* metadata from ``TeacherQueryV1`` (candidate order,
opaque presentation labels, description-length limits, prompt-template
hash) -- exactly the perturbation surface SLM-387 requires robustness
tests over. The teacher never sees ``application_id`` / opaque refs,
``trace.current_scores``, or ``trace.accepted_application_ids``: those
values are never read here, and the unit tests assert the rendered prompt
cannot contain them.

Reproducibility
---------------
Every provider exchange is cached content-addressed under ``cache_dir``:
the cache key is the SHA-256 of the exact request payload (model, system
prompt, user prompt, temperature, max tokens), so re-running the
comparison never re-queries the provider. Requests use ``temperature=0``.

Fail closed
-----------
A provider/transport error, an unparseable reply, a reply whose labels do
not exactly cover the presented candidate set, or non-positive scores makes
``rank`` return ``None`` (the harness then honestly reports missing data,
which the stop rule counts as insufficient -- never as a win). Uncertified
text is never returned as a ranking.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from slm_training.dsl.operators.legal_set import LegalOperatorActionV1
from slm_training.harnesses.distill.operator_teacher_ceiling import (
    RankingV1,
    TeacherQueryV1,
    _empty_ranking,
    _trace_reliability,
)

__all__ = [
    "DEFAULT_ENDPOINT",
    "DEFAULT_MODEL",
    "PromptedTeacherV1",
    "TeacherTransport",
    "resolve_hf_token",
]

DEFAULT_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

#: What the teacher is asked to do. Deliberately narrow: rank the presented
#: candidates by which a competent solver would apply next, and emit a JSON
#: probability distribution. No acceptance labels or scorer values exist in
#: the prompt, so the model cannot parrot the evaluator back.
_SYSTEM_PROMPT = (
    "You are an expert ranking candidates for the next operator application "
    "in a grammar-constrained UI compiler. You are given one operator and "
    "several candidate applications of it, each named by a label. Rank the "
    "candidates from most to least plausible as the next accepted step, "
    "based only on the structural information shown. Reply with a single "
    "JSON object of the form "
    '{"ranking": [{"label": "<label>", "score": <number in [0,1]>}]} '
    "covering every label exactly once, best first. Scores are relative "
    "plausibilities and need not sum to 1. Output JSON only."
)

#: Preregistered prompt-template paraphrases. ``TeacherQueryV1.prompt_template_hash``
#: selects one; a compliant teacher's *ranking* must not move when the
#: template changes (that is the prompt-template perturbation test).
_USER_TEMPLATES: tuple[str, ...] = (
    "Operator: {operator_id}\nCandidates:\n{candidates}\n\nRank all {n} candidates.",
    "You must choose the next {operator_id} application. Here are the {n} "
    "legal candidates:\n{candidates}\n\nProvide your ranking.",
    "Legal candidates for operator {operator_id} ({n} total):\n{candidates}\n\n"
    "Which should be applied next? Rank them all.",
    "Below are {n} candidates of {operator_id}.\n{candidates}\n\n"
    "Return the full best-first ranking as JSON.",
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_hf_token() -> str | None:
    """Resolve an HF credential the standard way (env first, then the CLI
    cache). Never logged or embedded in any artifact."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    cache = Path.home() / ".cache" / "huggingface" / "token"
    try:
        value = cache.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


#: Injected provider call: request payload (model/messages/temperature/
#: max_tokens) -> raw decoded JSON response. Tests inject recorded responses;
#: the default talks to an OpenAI-compatible endpoint over stdlib urllib.
TeacherTransport = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _identity_term(action: LegalOperatorActionV1) -> str:
    """A ``semantic_id``-derived identity term: content-identity, stable
    under opaque-id relabeling and presentation order, and never itself an
    opaque id (application ids and refs are not readable from it)."""
    return action.semantic_id[:16]


def _candidate_description(
    action: LegalOperatorActionV1, *, max_length: int
) -> str:
    text = (
        f"arity={len(action.arguments)}; "
        f"proof_checks={len(action.proof_checks)} ({', '.join(action.proof_checks)}); "
        f"identity={_identity_term(action)}"
    )
    # ``description_lengths`` bounds the description: 0 = no bound (the
    # canonical query), N > 0 = truncate to N characters (the perturbation).
    if max_length > 0:
        text = text[:max_length]
    return text


def render_teacher_prompt(query: TeacherQueryV1) -> str:
    """Render the user prompt for one query -- presentation metadata in,
    evaluator-only labels out.

    Candidates are presented in ``query.candidate_order`` under
    ``query.opaque_id_labels`` (presentation only; the response is mapped
    back to application ids through the same labels). ``description_lengths``
    bounds each candidate's structural description. ``prompt_template_hash``
    selects a preregistered template paraphrase.
    """
    actions_by_id = {
        action.application_id: action for action in query.trace.legal_set.operator_actions
    }
    lines: list[str] = []
    operator_ids = {
        actions_by_id[aid].operator_id
        for aid in query.candidate_order
        if aid in actions_by_id
    }
    for aid in query.candidate_order:
        action = actions_by_id[aid]
        label = query.opaque_id_labels[aid]
        max_length = query.description_lengths.get(aid, 0)
        lines.append(
            f"- {label}: {_candidate_description(action, max_length=max_length)}"
        )
    template_index = int(_sha(str(query.prompt_template_hash))[:8], 16) % len(
        _USER_TEMPLATES
    )
    template = _USER_TEMPLATES[template_index]
    return template.format(
        operator_id=sorted(operator_ids)[0] if len(operator_ids) == 1 else "multiple",
        candidates="\n".join(lines),
        n=len(lines),
    )


def _default_transport_factory(
    *, endpoint: str, token: str, timeout_seconds: float
) -> TeacherTransport:
    def transport(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        wire = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=wire,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in (429, 500, 502, 503, 504):
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        delay = float(retry_after) if retry_after else 2.0**attempt
                    except ValueError:
                        delay = 2.0**attempt
                    time.sleep(min(delay, 60.0))
                    continue
                break
            except (urllib.error.URLError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(2.0**attempt)
        raise RuntimeError(
            f"teacher provider request failed after retries: "
            f"{type(last_error).__name__}"
        ) from last_error

    return transport


def _parse_ranking_content(content: str) -> list[tuple[str, float]]:
    """Extract ``[(label, score), ...]`` best-first from the model's JSON
    reply. Raises ``ValueError`` on anything malformed -- the caller fails
    closed rather than guessing."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text)
    rows = data["ranking"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("teacher reply carries no ranking rows")
    parsed: list[tuple[str, float]] = []
    for row in rows:
        label = row["label"]
        score = float(row["score"])
        if score < 0.0:
            raise ValueError("teacher reply carries a negative score")
        parsed.append((str(label), score))
    return parsed


@dataclass(frozen=True)
class PromptedTeacherV1:
    """A real prompted LLM teacher behind ``OperatorTeacherAdapter`` (VAR3-01).

    Provider exchanges are content-addressed under ``cache_dir``; a second
    run over the same queries is fully offline. ``transport`` is injectable
    so tests and recorded replays never touch the network. When no
    ``transport`` is given, the default stdlib transport requires a
    resolvable HF token (see ``resolve_hf_token``).
    """

    cache_dir: Path | str
    model: str = DEFAULT_MODEL
    endpoint: str = DEFAULT_ENDPOINT
    source_id: str = "teacher:prompted_qwen3_4b_v1"
    max_output_tokens: int = 512
    timeout_seconds: float = 60.0
    transport: TeacherTransport | None = field(default=None, compare=False)

    def _request_payload(self, query: TeacherQueryV1) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": render_teacher_prompt(query)},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_output_tokens,
        }

    def _exchange(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Content-addressed provider exchange: cache hit -> no network."""
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _sha(json.dumps(payload, sort_keys=True))
        cache_path = cache_dir / f"{key}.json"
        if cache_path.exists():
            record = json.loads(cache_path.read_text(encoding="utf-8"))
            return record["response"]
        if self.transport is not None:
            response = dict(self.transport(payload))
        else:
            token = resolve_hf_token()
            if not token:
                raise RuntimeError(
                    "no teacher credential resolvable (HF_TOKEN unset and no "
                    "~/.cache/huggingface/token); refusing to fabricate a teacher"
                )
            response = dict(
                _default_transport_factory(
                    endpoint=self.endpoint,
                    token=token,
                    timeout_seconds=self.timeout_seconds,
                )(payload)
            )
        cache_path.write_text(
            json.dumps({"schema": "prompted_teacher_exchange/v1",
                        "request": payload, "response": response},
                       indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return response

    def rank(self, query: TeacherQueryV1) -> RankingV1 | None:
        trace = query.trace
        actions = trace.legal_set.operator_actions
        if not actions:
            return _empty_ranking(trace, self.source_id, "empty_legal_set")

        try:
            response = self._exchange(self._request_payload(query))
            content = response["choices"][0]["message"]["content"]
            parsed = _parse_ranking_content(content)
        except Exception as exc:  # fail closed: no uncertified ranking
            print(
                f"PromptedTeacherV1: trace {trace.trace_id}: provider/parse "
                f"failure ({type(exc).__name__}); returning no ranking",
                file=sys.stderr,
            )
            return None

        label_to_id = {query.opaque_id_labels[aid]: aid for aid in query.candidate_order}
        presented = set(label_to_id)
        returned_labels = [label for label, _ in parsed]
        if len(set(returned_labels)) != len(returned_labels) or set(returned_labels) != presented:
            print(
                f"PromptedTeacherV1: trace {trace.trace_id}: reply labels do "
                "not cover the presented candidate set; returning no ranking",
                file=sys.stderr,
            )
            return None

        scores = [score for _, score in parsed]
        total = sum(scores)
        if total <= 0.0:
            print(
                f"PromptedTeacherV1: trace {trace.trace_id}: all-zero scores; "
                "returning no ranking",
                file=sys.stderr,
            )
            return None
        order = tuple(label_to_id[label] for label in returned_labels)
        probabilities = tuple(score / total for score in scores)

        reliability, approximate, coverage_reason = _trace_reliability(trace)
        return RankingV1(
            trace_id=trace.trace_id,
            source_id=self.source_id,
            application_order=order,
            probabilities=probabilities,
            reliability=reliability,
            reliability_reason=(
                f"prompted_llm_ranking model={self.model} temperature=0 "
                f"content_addressed_cache; {coverage_reason}"
            ),
            approximate=approximate,
        )


def prefetch_queries(
    teacher: PromptedTeacherV1, queries: Sequence[TeacherQueryV1], *, workers: int = 8
) -> tuple[int, int]:
    """Warm the content-addressed cache for ``queries`` concurrently.

    Returns ``(n_cached, n_failed)``. Failures are not retried here -- the
    subsequent harness run fails closed on them (``rank`` returns ``None``).
    Concurrency exists only to keep the run inside the repo's hard run cap;
    cache keys make the result identical to a sequential warm-up.
    """
    from concurrent.futures import ThreadPoolExecutor

    payloads = [teacher._request_payload(query) for query in queries]

    def warm(payload: Mapping[str, Any]) -> bool:
        try:
            teacher._exchange(payload)
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(warm, payloads))
    return sum(results), len(results) - sum(results)
