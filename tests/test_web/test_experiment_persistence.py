"""Focused persistence checks for newly committed experiment evidence."""

from pathlib import Path

from slm_training.lineage.store import LineageStore
from slm_training.web.observability import Readers


def test_e646_neutral_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e646-root-slot-references-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.5
    assert suite["structural_similarity"] == 0.581675
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e647_rejected_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e647-role-plan-completion-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.25
    assert suite["reward_score"] == 0.884
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints
