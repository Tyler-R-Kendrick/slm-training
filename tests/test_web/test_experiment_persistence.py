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


def test_e708_retained_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e708-carrier-reference-obligation-r4"
    run = readers.run(run_id)

    assert run["provenance"] == "committed"
    rico = run["scoreboard"]["suites"]["rico_held"]
    assert rico["placeholder_fidelity"] == 1.0
    assert rico["structural_similarity"] == 0.7915333333333333
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }


def test_e709_retained_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e709-final-schema-value-margin-r5"
    run = readers.run(run_id)

    assert run["provenance"] == "committed"
    rico = run["scoreboard"]["suites"]["rico_held"]
    assert rico["binding_aware_meaningful_v2_rate_strict"] == 1.0
    assert rico["placeholder_fidelity"] == 1.0
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    control_id = "e709-rebased-v181-control-r1"
    control = readers.run(control_id)
    assert control["provenance"] == "committed"
    assert (
        control["scoreboard"]["suites"]["rico_held"][
            "binding_aware_meaningful_v2_rate_strict"
        ]
        == 0.0
    )
    visible_run_ids = {row.get("run_id") for row in readers.runs()["runs"]}
    assert {run_id, control_id} <= visible_run_ids


def test_e710_rejected_runs_persist_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {
        "e710-role-binding-negative-margin-r1",
        "e710-role-binding-negative-margin-r2",
        "e710-role-binding-negative-margin-r3",
    }

    visible_run_ids = {row.get("run_id") for row in readers.runs()["runs"]}
    assert run_ids <= visible_run_ids
    assert readers.run("e710-role-binding-negative-margin-r2")["provenance"] == (
        "committed"
    )
    assert readers.run("e710-role-binding-negative-margin-r2")["scoreboard"][
        "suites"
    ]["held_out"]["placeholder_fidelity"] == 0.96
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_ids.isdisjoint(checkpoint_ids)


def test_e711_retained_runs_persist_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {
        "e711-required-role-capacity-r1",
        "e711-required-role-capacity-r3",
        "e711-required-role-capacity-r4",
    }

    visible_run_ids = {row.get("run_id") for row in readers.runs()["runs"]}
    assert run_ids <= visible_run_ids
    retained = readers.run("e711-required-role-capacity-r4")
    assert retained["provenance"] == "committed"
    assert (
        retained["scoreboard"]["suites"]["ood"][
            "binding_aware_meaningful_v2_rate_strict"
        ]
        == 1.0
    )
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_ids.isdisjoint(checkpoint_ids)


def test_e712_retained_runs_persist_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_ids = {"e712-component-count-phrases-r3"}

    visible_run_ids = {row.get("run_id") for row in readers.runs()["runs"]}
    assert run_ids <= visible_run_ids
    retained = readers.run("e712-component-count-phrases-r3")
    assert retained["provenance"] == "committed"
    assert (
        retained["scoreboard"]["suites"]["adversarial"][
            "binding_aware_meaningful_v2_rate_strict"
        ]
        == 0.5
    )
    checkpoint_ids = {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
    assert run_ids.isdisjoint(checkpoint_ids)


def test_e720_checkpoint_and_scoreboard_persist_without_outputs(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e720-symbol-only-component-inventory600-r1"

    run = readers.run(run_id)

    assert run["provenance"] == "committed"
    smoke = run["scoreboard"]["suites"]["smoke"]
    assert smoke["binding_aware_meaningful_v2_rate_strict"] == 0.0
    assert smoke["placeholder_fidelity"] == 0.8055555555555555
    assert run["train_summary"]["checkpoint_sha256"] == (
        "842a1a21fb9897fe5ee594d9c9d2835315d63d4a12905e3c3640eec348f91a11"
    )
    assert run["train_summary"]["checkpoint_synced"] is False
    checkpoint = next(
        row for row in readers.checkpoints()["checkpoints"]
        if row.get("run_id") == run_id
    )
    assert "842a1a21" in checkpoint["status"]


def test_e721_checkpoint_and_scoreboard_persist_without_outputs(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e721-symbol-only-component-plan190-r4"

    run = readers.run(run_id)

    assert run["provenance"] == "committed"
    smoke = run["scoreboard"]["suites"]["smoke"]
    assert smoke["parse_rate"] == 1.0
    assert smoke["binding_aware_meaningful_v2_rate_strict"] == 0.0
    assert run["train_summary"]["checkpoint_sha256"] == (
        "c30fd565fced08626f39af5e9e23d233d88c26e0dac3b031928105b97c20f530"
    )
    assert run["train_summary"]["checkpoint_synced"] is False


def test_e722_checkpoint_and_scoreboard_persist_without_outputs(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e722-symbol-only-component-edge150-r1"

    run = readers.run(run_id)

    assert run["provenance"] == "committed"
    smoke = run["scoreboard"]["suites"]["smoke"]
    assert smoke["parse_rate"] == 1.0
    assert smoke["binding_aware_meaningful_v2_rate_strict"] == 0.0
    assert smoke["structural_similarity"] == 0.2861
    assert run["train_summary"]["checkpoint_sha256"] == (
        "08873bf0940eec19d0e90f50bfbd801f8547b45e450fa6379abe21c90a25597d"
    )
    assert run["train_summary"]["checkpoint_synced"] is False


def test_e723_checkpoint_and_scoreboard_persist_without_outputs(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e723-symbol-only-slot-owner140-r1"

    run = readers.run(run_id)

    assert run["provenance"] == "committed"
    assert run["scoreboard"]["suites"]["smoke"]["meaningful_program_rate"] == (
        0.6666666666666666
    )
    assert run["scoreboard"]["suites"]["held_out"]["structural_similarity"] == 0.394
    assert run["train_summary"]["checkpoint_sha256"] == (
        "787d2d21d7c29d56637355fd364f16a0d67b1f452fc0f4ce3a7d486b2bd62795"
    )
    assert run["train_summary"]["checkpoint_synced"] is False


def test_e724_no_effect_run_persists_without_new_checkpoint(tmp_path: Path) -> None:
    readers = Readers(Path(__file__).parents[2])
    readers.outputs = tmp_path / "missing-outputs"
    readers.lineage = LineageStore(readers.outputs / "lineage")
    run_id = "e724-slot-coverage-close2-smoke-r1"

    run = readers.run(run_id)

    assert run["provenance"] == "committed"
    smoke = run["scoreboard"]["suites"]["smoke"]
    assert smoke["meaningful_program_rate"] == 0.6666666666666666
    assert smoke["binding_aware_meaningful_v2_rate_strict"] == 0.0
    assert run_id not in {
        row.get("run_id") for row in readers.checkpoints()["checkpoints"]
    }
