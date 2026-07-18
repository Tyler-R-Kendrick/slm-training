"""Historical checkpoint-reference backfill (honest, no fabrication)."""

from __future__ import annotations

from pathlib import Path

from scripts.backfill_checkpoint_references import build_backfill, render_markdown

_MODEL_CARD = """# Model card

## Current checkpoint roster

| Role | Run id | Kind | Location | Status |
| --- | --- | --- | --- | --- |
| Playground demo | `playground_demo` | Fixture | `src/slm_training/resources/checkpoints/playground_demo/last.pt` (git) | demo only |
| Local scratch | `qx_e240_compiler_tree_control` | scratch | `outputs/runs/qx_e240_compiler_tree_control/checkpoints/last.pt` (local) | not ship |
| Tmp scratch | `overnight_1000` | scratch | `/tmp/x/outputs/runs/overnight_1000/checkpoints/last.pt` (local) | not ship |
| Production HF ship | — | — | `hf://buckets/TKendrick/OpenUI/checkpoints/<run_id>/` | None registered yet |
"""


def test_backfill_classifies_rows_honestly(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "MODEL_CARD.md").write_text(_MODEL_CARD, encoding="utf-8")

    report = build_backfill(root=tmp_path)
    by_run = {e["run_id"]: e for e in report["entries"]}

    assert by_run["playground_demo"]["classification"] == "tracked_local"
    assert by_run["playground_demo"]["resolvable_from_clone"] is True

    unresolved = by_run["qx_e240_compiler_tree_control"]
    assert unresolved["classification"] == "unresolved_local"
    assert unresolved["resolvable_from_clone"] is False
    # A concrete remediation command is recorded, no remote is invented.
    assert "sync_checkpoints.py" in unresolved["remediation"]
    assert unresolved["durable_uri"] is None

    assert by_run["overnight_1000"]["classification"] == "unresolved_local"

    # The hf:// template row is not fabricated into a real remote.
    template = [e for e in report["entries"] if e["classification"] == "template"]
    assert len(template) == 1

    assert report["counts"]["unresolved_local"] == 2
    assert report["unresolved_count"] == 2
    # Champion registry is recorded as unresolvable, never dropped.
    assert report["champion_registry"]["resolvable_from_clone"] is False


def test_backfill_markdown_lists_unresolved(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "MODEL_CARD.md").write_text(_MODEL_CARD, encoding="utf-8")
    md = render_markdown(build_backfill(root=tmp_path))
    assert "Unresolved historical checkpoints" in md
    assert "qx_e240_compiler_tree_control" in md
