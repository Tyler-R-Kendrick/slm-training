"""End-to-end tests for scripts/publish_planning_result.py (AP-037 / SLM-338)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.publish_planning_result import main
from slm_training.harnesses.autoresearch.planning_result import (
    AbstractPlanningResultV1,
    render_markdown,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "planning_result"
    / "ap037_fixture.json"
)
DOCS_DESIGN = Path(__file__).resolve().parents[2] / "docs" / "design"


def _publish(tmp_path: Path, *extra: str) -> int:
    return main([str(FIXTURE), "--output", str(tmp_path / "evidence"), *extra])


def test_fixture_publishes_json_md_and_snippet_from_one_object(tmp_path: Path) -> None:
    snippet = tmp_path / "row.md"
    assert _publish(tmp_path, "--snippet-out", str(snippet)) == 0

    canonical = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "evidence.md").read_text(encoding="utf-8")
    row = snippet.read_text(encoding="utf-8")

    assert canonical["schema"] == "AbstractPlanningResultV1"
    result = AbstractPlanningResultV1.from_dict(canonical)
    assert canonical["content_sha256"] == result.content_sha256()

    # Every headline number in the Markdown comes from the JSON payload.
    constrained = next(m for m in canonical["metrics"] if m["path"] == "constrained")
    assert (
        f"| constrained | {constrained['n']} | {constrained['meaning_v2']:.4f} "
        in markdown
    )
    assert f"{canonical['latency']['total_seconds']:.4f}" in markdown
    assert canonical["campaign_manifest_sha256"] in markdown

    # Model-card row is derived, not hand-written.
    assert row.startswith("| ap-037-fixture abstract-planning result |")
    assert "not promotable or ship" in row  # diagnostic honesty tag


def test_markdown_is_a_deterministic_render_of_the_json(tmp_path: Path) -> None:
    assert _publish(tmp_path) == 0
    canonical = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert render_markdown(canonical) == (tmp_path / "evidence.md").read_text(
        encoding="utf-8"
    )


def test_blockers_fail_closed_with_reasons(tmp_path: Path, capsys) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["source_commit"] = "UNKNOWN"
    payload["latency"]["total_seconds"] = None
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")
    rc = main([str(broken), "--output", str(tmp_path / "nope")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "missing_provenance:source_commit" in err
    assert "missing_total_latency" in err
    assert not (tmp_path / "nope.json").exists()


def test_disposition_append_is_idempotent_per_campaign_hash(tmp_path: Path) -> None:
    doc = tmp_path / "disposition.md"
    assert _publish(tmp_path, "--append-disposition", str(doc)) == 0
    assert _publish(tmp_path, "--append-disposition", str(doc)) == 0
    text = doc.read_text(encoding="utf-8")
    manifest_sha = json.loads(FIXTURE.read_text(encoding="utf-8"))[
        "campaign_manifest_sha256"
    ]
    assert text.count(f"## Campaign `{manifest_sha}`") == 1


def test_from_dir_merges_harness_output_fragments(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    harness_dir = tmp_path / "harness_out"
    harness_dir.mkdir()
    (harness_dir / "10_result.json").write_text(
        json.dumps(
            {k: v for k, v in payload.items() if k not in {"metrics", "latency"}}
        )
    )
    (harness_dir / "20_metrics.json").write_text(
        json.dumps({"metrics": payload["metrics"], "latency": payload["latency"]})
    )
    rc = main([str(harness_dir), "--from-dir", "--output", str(tmp_path / "merged")])
    assert rc == 0
    canonical = json.loads((tmp_path / "merged.json").read_text(encoding="utf-8"))
    assert len(canonical["metrics"]) == 3


def test_publisher_writes_only_its_own_artifacts(tmp_path: Path) -> None:
    doc = tmp_path / "disposition.md"
    before = {p: p.read_bytes() for p in DOCS_DESIGN.glob("iter-*") if p.is_file()}
    assert (
        _publish(
            tmp_path,
            "--snippet-out",
            str(tmp_path / "row.md"),
            "--append-disposition",
            str(doc),
        )
        == 0
    )
    after = {p: p.read_bytes() for p in DOCS_DESIGN.glob("iter-*") if p.is_file()}
    assert before == after  # historical iter docs untouched
    written = {p.name for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {"evidence.json", "evidence.md", "row.md", "disposition.md"}
