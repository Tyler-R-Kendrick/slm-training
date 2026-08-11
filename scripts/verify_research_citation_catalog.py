#!/usr/bin/env python3
"""CI certificate for the RESEARCH-01 research citation catalog (SLM-518).

Static checks over the committed catalog + design doc:

1. Schema / issue key / design_doc / verified_by pointers resolve.
2. Every source row has required metadata (authors/year/url/claims/…).
3. ``source_id`` values and normalized URLs are unique (dedupe gate).
4. ``source_type`` and ``trust_level`` stay inside the declared enums.
5. ``does_not_prove`` is non-empty for every row (anti-overclaim).
6. Design markdown lists every catalog ``source_id``.
7. Delta-baseline paths exist in the tree (repo-head links as baselines).
8. Core RESEARCH-01 literature IDs are present.

Run: ``PYTHONPATH=src python -m scripts.verify_research_citation_catalog``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = "src/slm_training/resources/research_citation_catalog.json"
DESIGN_DOC = "docs/design/research-citation-catalog.md"
LINEAGE_DOC = "docs/design/research-lineage.md"
LOADER = "src/slm_training/citation_catalog.py"
SCHEMA = "research_citation_catalog/v1"
ISSUE_KEY = "RESEARCH-01"
LINEAR_ID = "SLM-518"

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
REQUIRED_SOURCE_KEYS = (
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
)
CORE_SOURCE_IDS = frozenset(
    {
        "simpson-sosoa",
        "arxiv-1804.05495",
        "arxiv-2212.00489",
        "mathlib-partrec",
        "buss-marktoberdorf95",
        "arxiv-2601.15571",
        "arxiv-2410.15986",
        "arxiv-1707.03202",
        "arxiv-2607.00815",
        "arxiv-2602.08692",
        "szs-ontology",
        "leandojo",
        "arxiv-2605.20244",
        "w3c-prov-o",
        "spdx-ai-model",
        "repo-research-lineage",
        "repo-revmath-policy",
        "repo-only-hypothesis",
    }
)


class ResearchCatalogError(AssertionError):
    """Catalog claim no longer matches the live repository."""


def _read(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise ResearchCatalogError(f"missing required file: {relative}")
    return path.read_text(encoding="utf-8")


def _load_catalog() -> dict[str, Any]:
    return json.loads(_read(CATALOG_PATH))


def _normalize_url(url: str) -> str:
    text = url.strip()
    if text.endswith("/"):
        text = text[:-1]
    return text.lower()


def check_document(doc: dict[str, Any]) -> None:
    if doc.get("schema") != SCHEMA:
        raise ResearchCatalogError(f"schema must be {SCHEMA!r}, got {doc.get('schema')!r}")
    if doc.get("issue_key") != ISSUE_KEY:
        raise ResearchCatalogError(f"issue_key must be {ISSUE_KEY!r}")
    if doc.get("linear_id") != LINEAR_ID:
        raise ResearchCatalogError(f"linear_id must be {LINEAR_ID!r}")
    if doc.get("design_doc") != DESIGN_DOC:
        raise ResearchCatalogError(f"design_doc must be {DESIGN_DOC!r}")
    if doc.get("verified_by") != "scripts/verify_research_citation_catalog.py":
        raise ResearchCatalogError("verified_by pointer mismatch")
    if "delta baselines" not in str(doc.get("base_sha_note", "")).lower():
        raise ResearchCatalogError("base_sha_note must state delta-baseline law")
    declared_types = set(doc.get("source_types") or [])
    if declared_types != set(SOURCE_TYPES):
        raise ResearchCatalogError(f"source_types drift: {sorted(declared_types)}")
    declared_trust = set(doc.get("trust_levels") or [])
    if declared_trust != set(TRUST_LEVELS):
        raise ResearchCatalogError(f"trust_levels drift: {sorted(declared_trust)}")


def check_sources(doc: dict[str, Any]) -> list[str]:
    sources = doc.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ResearchCatalogError("sources must be a non-empty list")
    seen_ids: set[str] = set()
    seen_urls: dict[str, str] = {}
    present_types: set[str] = set()
    for row in sources:
        if not isinstance(row, dict):
            raise ResearchCatalogError("each source must be an object")
        for key in REQUIRED_SOURCE_KEYS:
            if key not in row:
                raise ResearchCatalogError(
                    f"source missing {key!r}: {row.get('source_id')!r}"
                )
        sid = row["source_id"]
        if not isinstance(sid, str) or not sid.strip():
            raise ResearchCatalogError("empty source_id")
        if sid in seen_ids:
            raise ResearchCatalogError(f"duplicate source_id: {sid!r}")
        seen_ids.add(sid)
        st = row["source_type"]
        if st not in SOURCE_TYPES:
            raise ResearchCatalogError(f"{sid}: unknown source_type {st!r}")
        present_types.add(st)
        tl = row["trust_level"]
        if tl not in TRUST_LEVELS:
            raise ResearchCatalogError(f"{sid}: unknown trust_level {tl!r}")
        authors = row["authors"]
        if not isinstance(authors, list) or not authors:
            raise ResearchCatalogError(f"{sid}: authors must be non-empty list")
        year = row["year"]
        if not isinstance(year, int) or year < 1900:
            raise ResearchCatalogError(f"{sid}: invalid year {year!r}")
        url = row["url"]
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ResearchCatalogError(f"{sid}: url must be http(s)")
        norm = _normalize_url(url)
        if norm in seen_urls:
            raise ResearchCatalogError(
                f"duplicate url {url!r} on {sid!r} and {seen_urls[norm]!r}"
            )
        seen_urls[norm] = sid
        if not str(row["claim_supported"]).strip():
            raise ResearchCatalogError(f"{sid}: claim_supported empty")
        dnp = row["does_not_prove"]
        if not isinstance(dnp, list) or not dnp:
            raise ResearchCatalogError(f"{sid}: does_not_prove must be non-empty")
        if any(not str(item).strip() for item in dnp):
            raise ResearchCatalogError(f"{sid}: empty does_not_prove entry")
        for key in (
            "title",
            "repo_relevance",
            "repo_application",
            "assumption",
            "falsifier",
        ):
            if not str(row[key]).strip():
                raise ResearchCatalogError(f"{sid}: {key} empty")
        keys = row["derivative_keys"]
        if not isinstance(keys, list) or not keys:
            raise ResearchCatalogError(f"{sid}: derivative_keys must be non-empty")
        baseline = row["delta_baseline"]
        if not isinstance(baseline, dict):
            raise ResearchCatalogError(f"{sid}: delta_baseline must be object")
        bpath = baseline.get("path")
        if not isinstance(bpath, str) or not bpath.strip():
            raise ResearchCatalogError(f"{sid}: delta_baseline.path required")
        # Repo-head delta baselines must exist; URLs alone are not enough.
        if not bpath.startswith(("http://", "https://")):
            if not (ROOT / bpath).exists():
                raise ResearchCatalogError(
                    f"{sid}: delta_baseline path missing: {bpath}"
                )
        note = str(baseline.get("note", "")).lower()
        if "delta" not in note and "baseline" not in note:
            raise ResearchCatalogError(
                f"{sid}: delta_baseline.note must mention delta/baseline"
            )
    missing_core = sorted(CORE_SOURCE_IDS - seen_ids)
    if missing_core:
        raise ResearchCatalogError(f"missing core source_id(s): {missing_core}")
    missing_types = sorted(SOURCE_TYPES - present_types)
    if missing_types:
        raise ResearchCatalogError(
            f"catalog must include every source_type; missing {missing_types}"
        )
    return sorted(seen_ids)


def check_docs(source_ids: list[str]) -> None:
    design = _read(DESIGN_DOC)
    if "Delta baseline law" not in design and "delta baseline" not in design.lower():
        raise ResearchCatalogError(f"{DESIGN_DOC}: missing delta-baseline law")
    if "does **not** prove" not in design and "does not prove" not in design.lower():
        raise ResearchCatalogError(f"{DESIGN_DOC}: missing does-not-prove section")
    if ISSUE_KEY not in design or LINEAR_ID not in design:
        raise ResearchCatalogError(f"{DESIGN_DOC}: must cite {ISSUE_KEY}/{LINEAR_ID}")
    if CATALOG_PATH not in design:
        raise ResearchCatalogError(f"{DESIGN_DOC}: must link {CATALOG_PATH}")
    for sid in source_ids:
        if f"`{sid}`" not in design and sid not in design:
            raise ResearchCatalogError(
                f"{DESIGN_DOC}: missing catalog source_id {sid!r}"
            )
    lineage = _read(LINEAGE_DOC)
    if "research-citation-catalog.md" not in lineage:
        raise ResearchCatalogError(
            f"{LINEAGE_DOC}: must point at research-citation-catalog.md"
        )
    if not (ROOT / LOADER).is_file():
        raise ResearchCatalogError(f"missing loader: {LOADER}")


def certify() -> dict[str, Any]:
    doc = _load_catalog()
    check_document(doc)
    source_ids = check_sources(doc)
    check_docs(source_ids)
    return {
        "schema": SCHEMA,
        "issue_key": ISSUE_KEY,
        "linear_id": LINEAR_ID,
        "source_count": len(source_ids),
        "source_ids": source_ids,
        "design_doc": DESIGN_DOC,
        "catalog": CATALOG_PATH,
    }


def main(argv: list[str] | None = None) -> int:
    del argv  # unused; keeps CLI shape stable for check_changed runners
    try:
        result = certify()
    except (ResearchCatalogError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"research citation catalog FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        "research citation catalog OK: "
        f"{result['source_count']} sources ({result['issue_key']}/{result['linear_id']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
