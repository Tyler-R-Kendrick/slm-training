"""RESEARCH-01 research citation catalog must fail on real regressions (SLM-518)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import verify_research_citation_catalog as verifier

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_passes_on_the_real_repository():
    result = verifier.certify()
    assert result["schema"] == "research_citation_catalog/v1"
    assert result["issue_key"] == "RESEARCH-01"
    assert result["linear_id"] == "SLM-518"
    assert result["source_count"] >= 18
    assert "simpson-sosoa" in result["source_ids"]
    assert "repo-only-hypothesis" in result["source_ids"]


def test_loader_require_grounded_citations():
    from slm_training.citation_catalog import (
        REPO_ONLY_SOURCE_ID,
        require_grounded_citations,
        source_by_id,
    )

    row = source_by_id("simpson-sosoa")
    assert row["source_type"] == "foundational_research"
    assert row["does_not_prove"]
    require_grounded_citations(["repo-revmath-policy", "simpson-sosoa"])
    require_grounded_citations([REPO_ONLY_SOURCE_ID])
    with pytest.raises(ValueError):
        require_grounded_citations([])
    with pytest.raises(KeyError):
        require_grounded_citations(["not-a-real-source"])


def test_catches_duplicate_source_id():
    doc = json.loads((REPO_ROOT / verifier.CATALOG_PATH).read_text(encoding="utf-8"))
    doc["sources"] = [doc["sources"][0], dict(doc["sources"][0])]
    with pytest.raises(verifier.ResearchCatalogError, match="duplicate source_id"):
        verifier.check_sources(doc)


def test_catches_duplicate_url(tmp_path):
    doc = json.loads((REPO_ROOT / verifier.CATALOG_PATH).read_text(encoding="utf-8"))
    a, b = dict(doc["sources"][0]), dict(doc["sources"][1])
    b["source_id"] = "dup-url-row"
    b["url"] = a["url"]
    doc["sources"] = [a, b]
    with pytest.raises(verifier.ResearchCatalogError, match="duplicate url"):
        verifier.check_sources(doc)


def test_catches_empty_does_not_prove():
    doc = json.loads((REPO_ROOT / verifier.CATALOG_PATH).read_text(encoding="utf-8"))
    row = dict(doc["sources"][0])
    row["does_not_prove"] = []
    doc["sources"] = [row]
    with pytest.raises(verifier.ResearchCatalogError, match="does_not_prove"):
        verifier.check_sources(doc)


def test_catches_missing_design_source_id(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    design = tmp_path / verifier.DESIGN_DOC
    design.parent.mkdir(parents=True, exist_ok=True)
    design.write_text(
        "RESEARCH-01 SLM-518 delta baseline does not prove\n"
        f"{verifier.CATALOG_PATH}\n",
        encoding="utf-8",
    )
    lineage = tmp_path / verifier.LINEAGE_DOC
    lineage.parent.mkdir(parents=True, exist_ok=True)
    lineage.write_text("research-citation-catalog.md\n", encoding="utf-8")
    loader = tmp_path / verifier.LOADER
    loader.parent.mkdir(parents=True, exist_ok=True)
    loader.write_text("# stub\n", encoding="utf-8")
    with pytest.raises(verifier.ResearchCatalogError, match="missing catalog source_id"):
        verifier.check_docs(["simpson-sosoa"])
