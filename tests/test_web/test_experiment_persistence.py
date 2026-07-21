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


def test_e648_rejected_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e648-root-only-role-plans-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.5
    assert suite["structural_similarity"] == 0.48835
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e649_rejected_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e649-bound-role-plans-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.25
    assert suite["placeholder_fidelity"] == 0.7583333333333333
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e650_retained_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e650-role-obligation-margin-r1"
    run = readers.run(run_id)
    assert run["provenance"] == "committed"
    suite = run["scoreboard"]["suites"]["ood"]
    assert suite["binding_aware_meaningful_v2_rate_strict"] == 0.75
    assert suite["structural_similarity"] == 0.605625
    checkpoints = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_id not in checkpoints


def test_e701_retained_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e701-schema-aware-role-capacity-r6"
    run = readers.run(run_id)

    assert run["provenance"] == "committed"
    held_out = run["scoreboard"]["suites"]["held_out"]
    assert held_out["binding_aware_meaningful_v2_rate_strict"] == 1.0
    assert held_out["structural_similarity"] == 0.81042
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e702_rejected_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e702-joint-role-cardinality-r1"
    run = readers.run(run_id)

    assert run["provenance"] == "committed"
    rico = run["scoreboard"]["suites"]["rico_held"]
    assert rico["binding_aware_meaningful_v2_rate_strict"] == 0.0
    assert rico["structural_similarity"] == 0.4678333333333333
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e703_neutral_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e703-enum-safe-repeated-slots-r1"
    run = readers.run(run_id)

    assert run["provenance"] == "committed"
    rico = run["scoreboard"]["suites"]["rico_held"]
    assert rico["binding_aware_meaningful_v2_rate_strict"] == 0.0
    assert rico["structural_similarity"] == 0.7611333333333334
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e704_rejected_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e704-schema-value-weight8-r1"
    run = readers.run(run_id)

    assert run["provenance"] == "committed"
    rico = run["scoreboard"]["suites"]["rico_held"]
    assert rico["binding_aware_meaningful_v2_rate_strict"] == 0.0
    assert rico["placeholder_fidelity"] == 0.875
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e705_partial_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e705-schema5-root-margin0-r1"
    run = readers.run(run_id)

    assert run["provenance"] == "committed"
    rico = run["scoreboard"]["suites"]["rico_held"]
    assert rico["binding_aware_meaningful_v2_rate_strict"] == 0.0
    assert rico["structural_similarity"] == 0.5098333333333334
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e706_rejected_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e706-bounded-slot-carrier-r1"
    run = readers.run(run_id)

    assert run["provenance"] == "committed"
    rico = run["scoreboard"]["suites"]["rico_held"]
    assert rico["binding_aware_meaningful_v2_rate_strict"] == 0.0
    assert rico["placeholder_fidelity"] == 0.875
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e707_rejected_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e707-carrier-root-reference-margin2-r2"
    run = readers.run(run_id)

    assert run["provenance"] == "committed"
    rico = run["scoreboard"]["suites"]["rico_held"]
    assert rico["placeholder_fidelity"] == 1.0
    assert rico["reward_score"] == 1.0
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
