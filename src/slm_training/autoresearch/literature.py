"""Hugging Face Daily Papers and paper-search acquisition."""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from typing import Any

from slm_training.autoresearch.schemas import ResearchSource, utc_now


def categorical_discovery_source() -> ResearchSource:
    """Reviewed source that defines the harness's candidate-novelty boundary."""
    return ResearchSource(
        source_id="arxiv-2606.01444",
        kind="hf_paper_search",
        title=(
            "Self-Revising Discovery Systems for Science: "
            "A Categorical Framework for Agentic Artificial Intelligence"
        ),
        uri="https://arxiv.org/abs/2606.01444",
        published_at="2026-05-31",
        summary=(
            "Discovery is a verified regime transition that preserves old artifacts, "
            "transports them by left Kan extension, and identifies accepted residual "
            "content outside transport; pre-experiment novelty remains a candidate."
        ),
        metadata={"implementation_status": "Adapted"},
    )


class HuggingFacePapersClient:
    def __init__(self, *, client: Any | None = None, token: str | None = None) -> None:
        if client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - dependency error
                raise RuntimeError("install slm-training[research] for HF papers") from exc
            client = httpx.Client(timeout=30.0)
        self.client = client
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def daily(self, *, days: int = 7, limit_per_day: int = 20) -> list[ResearchSource]:
        sources: list[ResearchSource] = []
        for offset in range(max(1, days)):
            day = date.today() - timedelta(days=offset)
            response = self.client.get(
                "https://huggingface.co/api/daily_papers",
                params={
                    "p": 0,
                    "limit": min(100, max(1, limit_per_day)),
                    "date": day.isoformat(),
                    "sort": "publishedAt",
                },
                headers=self.headers,
            )
            response.raise_for_status()
            sources.extend(self._rows(response.json(), kind="hf_daily_paper"))
        return _dedupe(sources)

    def search(self, query: str, *, limit: int = 20) -> list[ResearchSource]:
        response = self.client.get(
            "https://huggingface.co/api/papers/search",
            params={"q": query[:250], "limit": min(120, max(1, limit))},
            headers=self.headers,
        )
        response.raise_for_status()
        return self._rows(response.json(), kind="hf_paper_search")

    @staticmethod
    def _rows(payload: Any, *, kind: str) -> list[ResearchSource]:
        if isinstance(payload, dict):
            payload = payload.get("papers") or payload.get("items") or []
        result = []
        for wrapper in payload if isinstance(payload, list) else []:
            row = wrapper.get("paper", wrapper) if isinstance(wrapper, dict) else {}
            paper_id = str(row.get("id") or row.get("paperId") or row.get("arxivId") or "")
            title = str(row.get("title") or "Untitled paper")
            summary = str(row.get("summary") or row.get("ai_summary") or row.get("aiSummary") or "")
            uri = str(row.get("url") or (f"https://huggingface.co/papers/{paper_id}" if paper_id else ""))
            if not uri:
                continue
            source_id = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
            result.append(
                ResearchSource(
                    source_id=f"hf-{source_id}",
                    kind=kind,  # type: ignore[arg-type]
                    title=title,
                    uri=uri,
                    retrieved_at=utc_now(),
                    published_at=str(row.get("publishedAt") or row.get("published_at") or "") or None,
                    summary=summary[:4000],
                    metadata={
                        "paper_id": paper_id,
                        "upvotes": wrapper.get("numUpvotes") if isinstance(wrapper, dict) else None,
                    },
                )
            )
        return result


def _dedupe(sources: list[ResearchSource]) -> list[ResearchSource]:
    return list({source.uri: source for source in sources}.values())
