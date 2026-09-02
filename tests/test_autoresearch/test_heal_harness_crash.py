"""harness_crash playbook: crash triage is evidence, never a claimed heal."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.autoresearch import heal
from slm_training.autoresearch.heal.classify import classify_blocker
from slm_training.autoresearch.heal.playbooks import harness_crash
from slm_training.autoresearch.heal.playbooks.harness_crash import (
    PLAYBOOK,
    build_repair_action,
    capture_crash_evidence,
    extract_traceback,
    harness_family_for_module,
    mirrored_test_file,
)
from slm_training.autoresearch.schemas import AutotrainActionV1

LOOP = "continuous-openui-local"
CAMPAIGN = "continuous-loop-20260821-continuous-openui-local-8c0b60dd-c543"
CONTROL = "c20260821-continuous-openui-local-8c0b60dd-c543-control"

_TRACEBACK = """some harness output
Traceback (most recent call last):
  File "/repo/scripts/evaluate_model.py", line 12, in <module>
    main()
  File "/repo/src/slm_training/harnesses/model_build/eval_runner.py", line 40, in main
    scoreboard = build(records)
KeyError: 'smoke.eval_nll'
"""


def _campaign(root: Path, *, exit_code: int, stderr: str = _TRACEBACK) -> Path:
    camp = root / CAMPAIGN
    (camp / "artifacts" / "outcomes").mkdir(parents=True)
    (camp / "sdlc_delivery.json").write_text(
        json.dumps(
            {
                "campaign_id": CAMPAIGN,
                "arm_exits": {CONTROL: exit_code},
                "reasons": [
                    f"measurement_incomplete:{CONTROL}:missing_scoreboard",
                    f"harness_failure:{CONTROL}:experiment_failed",
                ],
                "measurement_complete": False,
            }
        ),
        encoding="utf-8",
    )
    (camp / "artifacts" / "outcomes" / f"{CONTROL}.json").write_text(
        json.dumps(
            {
                "experiment_id": CONTROL,
                "campaign_id": CAMPAIGN,
                "status": "failed",
                "exit_code": exit_code,
                "error": "experiment_failed",
                "stage_telemetry": [
                    {
                        "command": ["python", "-m", "scripts.evaluate_model"],
                        "exit_code": exit_code,
                        "stderr": stderr,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runs = camp / "runs" / CONTROL
    runs.mkdir(parents=True)
    (runs / "stderr.log").write_text(stderr, encoding="utf-8")
    return camp


def _blocker(kind: str = "repair_harness") -> dict:
    return {
        "campaign_id": CAMPAIGN,
        "index": 0,
        "kind": kind,
        "reason": f"harness_failure:{CONTROL}:experiment_failed",
    }


class TestEvidence:
    def test_extract_traceback_returns_last_block(self) -> None:
        text = "junk\n" + _TRACEBACK + "\nmore\n" + _TRACEBACK.replace(
            "KeyError", "ValueError"
        )
        block = extract_traceback(text)
        assert block.startswith("Traceback (most recent call last):")
        assert block.rstrip().endswith("ValueError: 'smoke.eval_nll'")

    def test_capture_names_arm_exit_module_and_family(self, tmp_path: Path) -> None:
        camp = _campaign(tmp_path, exit_code=2)
        evidence = capture_crash_evidence(camp)
        assert evidence.arm_id == CONTROL
        assert evidence.exit_code == 2
        assert "KeyError: 'smoke.eval_nll'" in evidence.traceback
        assert evidence.module == "slm_training/harnesses/model_build/eval_runner.py"
        assert evidence.harness_family == "model_build"
        assert "sdlc_delivery.json" in evidence.sources
        assert any(s.startswith("runs/") for s in evidence.sources)

    def test_family_mapping_covers_action_schema(self) -> None:
        assert harness_family_for_module("slm_training/harnesses/train_data/x.py") == (
            "train_data"
        )
        assert harness_family_for_module("slm_training.autoresearch.engine") == (
            "autoresearch"
        )
        assert harness_family_for_module("scripts/build_test_data.py") == "test_data"
        assert harness_family_for_module("") == "model_build"

    def test_test_file_mirror(self, tmp_path: Path) -> None:
        assert mirrored_test_file("slm_training/autoresearch/heal/classify.py", tmp_path) is None
        target = tmp_path / "tests" / "test_autoresearch" / "test_heal_classify.py"
        target.parent.mkdir(parents=True)
        target.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        assert mirrored_test_file(
            "slm_training/autoresearch/heal/classify.py", tmp_path
        ) == target


class TestExecute:
    def test_exit_two_arm_yields_repair_action_and_attempted_receipt(
        self, tmp_path: Path
    ) -> None:
        camp = _campaign(tmp_path, exit_code=2)
        blocker = _blocker()
        assert classify_blocker(blocker["kind"], blocker["reason"]) == "code"
        assert PLAYBOOK.matches(blocker)
        receipt = harness_crash.execute(
            blocker,
            cwd=tmp_path,
            root=tmp_path,
            loop_id=LOOP,
            campaign_id=CAMPAIGN,
            run_tests=False,
        )
        assert receipt.outcome == "attempted"
        assert receipt.verify_result is None
        assert receipt.playbook_id == "harness_crash/v1"
        # Receipt row persisted in the loop's heal ledger.
        rows = heal.load_heal_receipts(tmp_path, LOOP)
        assert [r.outcome for r in rows] == ["attempted"]
        # Typed action carrying the traceback, next to the ledger.
        evidence_files = list((tmp_path / "loops" / LOOP / "heal_harness_crash").glob("*.json"))
        assert len(evidence_files) == 1
        payload = json.loads(evidence_files[0].read_text(encoding="utf-8"))
        action = AutotrainActionV1.model_validate(payload["action"])
        assert action.kind == "repair_harness"
        assert action.harness_family == "model_build"
        assert "KeyError: 'smoke.eval_nll'" in action.reason
        assert "exit=2" in action.reason
        assert f"campaign:{CAMPAIGN}" in action.evidence_ids
        assert payload["evidence"]["exit_code"] == 2
        assert camp.is_dir()

    def test_module_tests_run_once_and_recorded(self, tmp_path: Path) -> None:
        _campaign(tmp_path, exit_code=2)
        test_dir = tmp_path / "tests" / "test_harnesses" / "model_build"
        test_dir.mkdir(parents=True)
        (test_dir / "test_eval_runner.py").write_text(
            "def test_fails():\n    assert False\n", encoding="utf-8"
        )
        receipt = harness_crash.execute(
            _blocker(), cwd=tmp_path, root=tmp_path, loop_id=LOOP, campaign_id=CAMPAIGN
        )
        assert receipt.outcome == "attempted"
        step = receipt.step_results[-1]
        assert step.step_id == "module_tests"
        assert step.outcome == "completed"
        assert step.returncode != 0

    def test_exit_124_is_noted_as_timeout_not_crash(self, tmp_path: Path) -> None:
        _campaign(tmp_path, exit_code=124, stderr="killed by wall budget\n")
        receipt = harness_crash.execute(
            _blocker(), cwd=tmp_path, root=tmp_path, loop_id=LOOP, campaign_id=CAMPAIGN,
            run_tests=False,
        )
        assert receipt.outcome == "attempted"
        assert "arm_exit_124_is_a_timeout_not_a_crash" in receipt.note

    def test_build_action_without_traceback_is_still_typed(self) -> None:
        evidence = capture_crash_evidence(Path("/nonexistent/campaign"))
        action = build_repair_action(evidence, campaign_id="c1")
        assert action.kind == "repair_harness"
        assert action.harness_family == "model_build"
        assert "module=unresolved" in action.reason


class TestRunnerCompatibility:
    def test_discovered_and_never_healed_via_plan(self, tmp_path: Path) -> None:
        ids = {p.playbook_id for p in heal.discovered_playbooks()}
        assert "harness_crash/v1" in ids
        blocker = {**_blocker(), "_root": tmp_path, "_loop_id": LOOP}
        plan = PLAYBOOK.plan(blocker, cwd=tmp_path)
        assert plan is not None
        assert plan.blocker_class == "code"
        assert plan.steps[0].step_id == "diagnose_crash"
        # The verify probe must always fail: a code crash has no self-verify.
        import subprocess

        probe = subprocess.run(list(plan.verify.argv), capture_output=True, text=True)
        assert probe.returncode != 0

    def test_run_playbooks_records_non_healed_receipt(self, tmp_path: Path) -> None:
        _campaign(tmp_path, exit_code=2)
        blocker = {**_blocker(), "_root": tmp_path, "_loop_id": LOOP}
        receipts = heal.run_playbooks(
            root=tmp_path,
            loop_id=LOOP,
            campaign_id=CAMPAIGN,
            blockers=[blocker],
            cwd=tmp_path,
            playbooks=(PLAYBOOK,),
        )
        assert len(receipts) == 1
        assert receipts[0].outcome != "healed"
        assert receipts[0].outcome in {"verify_failed", "attempted"}
