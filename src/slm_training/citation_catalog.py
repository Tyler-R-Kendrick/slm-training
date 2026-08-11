"""RESEARCH-01 research citation catalog loader (SLM-518).

Machine-readable owner:
``src/slm_training/resources/research_citation_catalog.json``.

Human-readable owner: ``docs/design/research-citation-catalog.md``.

Repo-head links in the catalog are **delta baselines**, not static truth —
callers must re-audit latest ``main`` before treating a row as current.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "research_citation_catalog/v1"
CATALOG_RELATIVE = "src/slm_training/resources/research_citation_catalog.json"
SOURCE_TYPES = frozenset(
    {
        "foundational_research",
        "solver_certificate",
        "lean_tooling",
        "provenance_standard",
        "repository_evidence",
    }
)
TRUST_LEVELS = frozenset(
    {"primary", "secondary", "repository_baseline", "adjacent"}
)
REQUIRED_SOURCE_KEYS = frozenset(
    {
        "source_id",
        "source_type",
        "title",
        "authors",
        "year",
        "url",
        "claim_supported",
        "does_not_prove",
        "repo_relevance",
        "repo_application",
        "assumption",
        "falsifier",
        "trust_level",
        "derivative_keys",
        "delta_baseline",
    }
)
REPO_ONLY_SOURCE_ID = "repo-only-hypothesis"


def catalog_path(*, root: Path | None = None) -> Path:
    """Absolute path to the committed catalog JSON."""
    base = root if root is not None else Path(__file__).resolve().parents[2]
    return base / CATALOG_RELATIVE


@lru_cache(maxsize=4)
def _load_raw(path_str: str) -> dict[str, Any]:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def load_catalog(*, root: Path | None = None) -> dict[str, Any]:
    """Load the catalog document (schema ``research_citation_catalog/v1``)."""
    path = catalog_path(root=root)
    doc = _load_raw(str(path))
    if doc.get("schema") != SCHEMA:
        raise ValueError(
            f"unexpected catalog schema {doc.get('schema')!r}; expected {SCHEMA!r}"
        )
    return doc


def iter_sources(*, root: Path | None = None) -> list[dict[str, Any]]:
    """Return the catalog ``sources`` list."""
    return list(load_catalog(root=root)["sources"])


def source_by_id(source_id: str, *, root: Path | None = None) -> dict[str, Any]:
    """Return one source row or raise ``KeyError``."""
    for row in iter_sources(root=root):
        if row["source_id"] == source_id:
            return row
    raise KeyError(source_id)


def require_grounded_citations(
    citation_ids: list[str] | tuple[str, ...],
    *,
    root: Path | None = None,
) -> None:
    """Fail closed unless every id exists, or ``repo-only-hypothesis`` is used.

    Derivative experiments must pass either grounded external/repo evidence ids
    or the explicit repository-only marker. An empty citation list is rejected.
    """
    if not citation_ids:
        raise ValueError(
            "experiment citations empty: cite grounded catalog source_id(s) "
            f"or {REPO_ONLY_SOURCE_ID!r}"
        )
    known = {row["source_id"] for row in iter_sources(root=root)}
    missing = [sid for sid in citation_ids if sid not in known]
    if missing:
        raise KeyError(f"unknown research catalog source_id(s): {missing}")


def normalize_url(url: str) -> str:
    """Normalize URL text for duplicate detection."""
    text = url.strip()
    if text.endswith("/"):
        text = text[:-1]
    return text.lower()


def validate_source_row(row: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when a source row is incomplete or mistyped."""
    missing = sorted(REQUIRED_SOURCE_KEYS - set(row))
    if missing:
        raise ValueError(f"source missing keys {missing}: {row.get('source_id')!r}")
    sid = row["source_id"]
    if not isinstance(sid, str) or not sid.strip():
        raise ValueError("source_id must be a non-empty string")
    if row["source_type"] not in SOURCE_TYPES:
        raise ValueError(f"{sid}: unknown source_type {row['source_type']!r}")
    if row["trust_level"] not in TRUST_LEVELS:
        raise ValueError(f"{sid}: unknown trust_level {row['trust_level']!r}")
    if not isinstance(row["authors"], list) or not row["authors"]:
        raise ValueError(f"{sid}: authors must be a non-empty list")
    if not isinstance(row["year"], int) or row["year"] < 1900:
        raise ValueError(f"{sid}: year must be an int >= 1900")
    url = row["url"]
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError(f"{sid}: url must be http(s)")
    if not str(row["claim_supported"]).strip():
        raise ValueError(f"{sid}: claim_supported empty")
    dnp = row["does_not_prove"]
    if not isinstance(dnp, list) or not dnp or any(not str(x).strip() for x in dnp):
        raise ValueError(f"{sid}: does_not_prove must be a non-empty string list")
    for key in (
        "repo_relevance",
        "repo_application",
        "assumption",
        "falsifier",
    ):
        if not str(row[key]).strip():
            raise ValueError(f"{sid}: {key} empty")
    keys = row["derivative_keys"]
    if not isinstance(keys, list) or not keys:
        raise ValueError(f"{sid}: derivative_keys must be a non-empty list")
    baseline = row["delta_baseline"]
    if not isinstance(baseline, Mapping) or not str(baseline.get("path", "")).strip():
        raise ValueError(f"{sid}: delta_baseline.path required")


__all__ = [
    "CATALOG_RELATIVE",
    "REPO_ONLY_SOURCE_ID",
    "SCHEMA",
    "SOURCE_TYPES",
    "TRUST_LEVELS",
    "catalog_path",
    "iter_sources",
    "load_catalog",
    "normalize_url",
    "require_grounded_citations",
    "source_by_id",
    "validate_source_row",
]
