from pathlib import Path

from scripts.autoresearch import _load_sources


_ALLOWED_STATUSES = {"Faithful", "Adapted", "Adjacent", "Diagnostic"}


def test_semantic_planning_source_manifest_is_complete() -> None:
    path = Path(
        "src/slm_training/resources/autoresearch/semantic-planning-sources.json"
    )
    rows = _load_sources(path)

    assert len(rows) == 12
    assert len({row.source_id for row in rows}) == 12
    assert len({row.uri for row in rows}) == 12
    assert all(row.uri.startswith("https://") for row in rows)
    assert all(row.metadata.get("arxiv_id") or row.metadata.get("canonical_id") for row in rows)
    assert all(row.metadata.get("authors") for row in rows)
    assert all(row.metadata.get("category") == "semantic_plan" for row in rows)
    assert all(row.metadata.get("feedback_claim") for row in rows)
    assert all(row.metadata.get("repo_relevance") for row in rows)
    assert all(row.metadata.get("implementation_status") in _ALLOWED_STATUSES for row in rows)
    assert all(row.metadata.get("limitations") for row in rows)
    assert all(row.metadata.get("novel_hypothesis") for row in rows)
