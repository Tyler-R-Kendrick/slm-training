"""Nearest-neighbor OpenUI skeleton retrieval by prompt overlap."""

from __future__ import annotations

from dataclasses import dataclass

from slm_training.data.leakage import norm_text
from slm_training.dsl.schema import ExampleRecord


@dataclass(frozen=True)
class SkeletonHit:
    prompt: str
    openui: str
    score: float
    record_id: str = ""


def build_skeleton_bank(records: list[ExampleRecord]) -> list[tuple[str, str, str]]:
    """Return (norm_prompt, openui, id) entries."""
    bank: list[tuple[str, str, str]] = []
    for record in records:
        prompt = norm_text(record.prompt or "")
        openui = (record.openui or "").strip()
        if prompt and openui:
            bank.append((prompt, openui, record.id))
    return bank


def _overlap_score(query_tokens: set[str], candidate: str) -> float:
    c = set(candidate.split())
    if not query_tokens or not c:
        return 0.0
    return len(query_tokens & c) / len(query_tokens | c)


def nearest_skeletons(
    bank: list[tuple[str, str, str]],
    query: str,
    *,
    k: int = 1,
    exclude_id: str | None = None,
) -> list[SkeletonHit]:
    q = norm_text(query or "")
    q_tokens = set(q.split())
    scored: list[SkeletonHit] = []
    for prompt, openui, rid in bank:
        if exclude_id and rid == exclude_id:
            continue
        score = _overlap_score(q_tokens, prompt)
        if score <= 0.0:
            continue
        scored.append(SkeletonHit(prompt=prompt, openui=openui, score=score, record_id=rid))
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored[: max(0, int(k))]


def format_retrieved_skeleton(openui: str, *, budget: int = 400) -> str:
    text = (openui or "").strip()
    if len(text) > budget:
        text = text[:budget].rsplit("\n", 1)[0]
    return text
