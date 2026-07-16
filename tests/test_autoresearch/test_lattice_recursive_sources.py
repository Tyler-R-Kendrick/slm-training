import json
from pathlib import Path


def test_lattice_recursive_source_manifest_is_complete() -> None:
    from scripts.autoresearch import _load_sources

    path = Path(
        "src/slm_training/resources/autoresearch/lattice-recursive-sources.json"
    )
    manifest = json.loads(path.read_text())
    rows = _load_sources(path)

    assert "6a58edb7-547c-83ea-ae8a-b6f62d3b283a" in manifest["source_scope"]
    assert len(rows) == 27
    assert len({row.source_id for row in rows}) == 27
    assert len({row.uri for row in rows}) == 27
    assert {row.source_id.split("-", 1)[0] for row in rows} == {
        f"R{index}" for index in range(27)
    }
    assert sum(row.metadata.get("evidence_class") == "academic" for row in rows) == 25
    assert all(row.metadata.get("feedback_claim") for row in rows)
    assert all(row.metadata.get("repo_relevance") for row in rows)
    assert all(row.metadata.get("implementation_status") for row in rows)
    assert all(row.metadata.get("limitations") for row in rows)
    assert all(row.metadata.get("novel_hypothesis") for row in rows)
