from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_tasks import main as evaluate_tasks_main
from scripts.run_mixture_search import main as mixture_search_main
from slm_training.dsl.schema import ExampleRecord, write_jsonl

PROGRAM = 'root = Stack([cta])\ncta = Button(":cta.label")'


def test_evaluate_tasks_cli_writes_scoreboard(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "id": "case-a",
                "task": "generation",
                "gold": PROGRAM,
                "prediction": PROGRAM,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "scoreboard.json"
    assert evaluate_tasks_main(["--cases", str(cases), "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["task_scoreboard"]["n"] == 1
    assert payload["task_scoreboard"]["tasks"]["generation"]["n"] == 1
    assert payload["agentv"]["summary"]["failed"] == 1


def test_mixture_search_dry_run_profiles_task_corpus(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    write_jsonl(
        train_dir / "records.jsonl",
        [
            ExampleRecord(
                id="a",
                prompt="button",
                openui=PROGRAM,
                meta={
                    "source_family": "programspec_generated",
                    "task": "generation",
                    "program_family_id": "button-family",
                },
            )
        ],
    )
    out = tmp_path / "mixtures"
    assert (
        mixture_search_main(
            [
                "--out",
                str(out),
                "--train-dir",
                str(train_dir),
                "--limit-probes",
                "3",
            ]
        )
        == 0
    )
    summary = json.loads((out / "search_summary.json").read_text(encoding="utf-8"))
    assert summary["scored"] is False
    assert summary["task_weights"] == {
        "generation": pytest.approx(1 / 6),
        "identity_echo": pytest.approx(1 / 6),
        "noop_adversarial": pytest.approx(1 / 6),
        "patch_edit": pytest.approx(1 / 6),
        "repair_completion_inpaint": pytest.approx(1 / 6),
        "state_behavior": pytest.approx(1 / 6),
    }
    assert summary["corpus_diagnostics"]["unique_program_families"] == 1
    assert len(summary["probes"]) == 3
