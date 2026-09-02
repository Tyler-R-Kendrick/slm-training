"""data_rebuild playbook: the rebuild seam is healed only when records grow."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.autoresearch import heal
from slm_training.autoresearch.heal.classify import classify_blocker
from slm_training.autoresearch.heal.playbooks import data_rebuild
from slm_training.autoresearch.heal.playbooks.data_rebuild import (
    PLAYBOOK,
    count_data_records,
)

LOOP = "continuous-openui-local"
CAMPAIGN = "continuous-loop-c9"


def _blocker(reason: str = "rebuild_data: train records below unique_root_target") -> dict:
    return {"campaign_id": CAMPAIGN, "index": 1, "kind": "rebuild_data", "reason": reason}


def _write_records(cwd: Path, version: str, n: int) -> None:
    train = cwd / "outputs" / "data" / "train" / version
    train.mkdir(parents=True, exist_ok=True)
    (train / "records.jsonl").write_text(
        "".join(json.dumps({"id": f"{version}-{i}"}) + "\n" for i in range(n)),
        encoding="utf-8",
    )


class TestCount:
    def test_counts_train_test_and_seed_records(self, tmp_path: Path) -> None:
        assert count_data_records(tmp_path, tmp_path, CAMPAIGN) == 0
        _write_records(tmp_path, "v1", 3)
        seed = tmp_path / "src" / "slm_training" / "resources" / "test_seeds.jsonl"
        seed.parent.mkdir(parents=True)
        seed.write_text('{"id": "s1"}\n\n{"id": "s2"}\n', encoding="utf-8")
        assert count_data_records(tmp_path, tmp_path, CAMPAIGN) == 5


class TestExecute:
    def test_stub_seam_adding_zero_records_is_postcondition_failure(
        self, tmp_path: Path
    ) -> None:
        _write_records(tmp_path, "v1", 4)
        calls: list[dict] = []

        def seam(**kwargs):
            calls.append(kwargs)
            return "rebuild_data_noop"

        blocker = _blocker()
        assert classify_blocker(blocker["kind"], blocker["reason"]) == "data"
        assert PLAYBOOK.matches(blocker)
        receipt = data_rebuild.execute(
            blocker, cwd=tmp_path, root=tmp_path, loop_id=LOOP, campaign_id=CAMPAIGN,
            seam=seam,
        )
        assert calls == [
            {"cwd": tmp_path, "root": tmp_path, "loop_id": LOOP, "campaign_id": CAMPAIGN}
        ]
        assert receipt.outcome == "postcondition_failed"
        assert receipt.note.startswith("heal_postcondition_failed:")
        assert "records_before=4 records_after=4" in receipt.note
        assert receipt.verify_result is not None
        assert receipt.verify_result.returncode == 1
        rows = heal.load_heal_receipts(tmp_path, LOOP)
        assert [r.outcome for r in rows] == ["postcondition_failed"]

    def test_seam_growing_records_is_healed(self, tmp_path: Path) -> None:
        _write_records(tmp_path, "v1", 4)

        def seam(**kwargs):
            _write_records(kwargs["cwd"], "v2", 6)
            return "rebuild_data_local"

        receipt = data_rebuild.execute(
            _blocker(), cwd=tmp_path, root=tmp_path, loop_id=LOOP, campaign_id=CAMPAIGN,
            seam=seam,
        )
        assert receipt.outcome == "healed"
        assert receipt.verify_result is not None
        assert receipt.verify_result.returncode == 0
        assert "records_before=4 records_after=10" in receipt.note

    def test_seam_crash_is_step_failed_never_healed(self, tmp_path: Path) -> None:
        def seam(**kwargs):
            raise RuntimeError("build_train_data exploded")

        receipt = data_rebuild.execute(
            _blocker(), cwd=tmp_path, root=tmp_path, loop_id=LOOP, campaign_id=CAMPAIGN,
            seam=seam,
        )
        assert receipt.outcome == "step_failed"
        assert "build_train_data exploded" in receipt.note

    def test_custom_counter_drives_postcondition(self, tmp_path: Path) -> None:
        counts = iter([10, 10])
        receipt = data_rebuild.execute(
            _blocker(), cwd=tmp_path, root=tmp_path, loop_id=LOOP, campaign_id=CAMPAIGN,
            seam=lambda **_: None,
            count_records=lambda *_: next(counts),
            write_receipt=False,
        )
        assert receipt.outcome == "postcondition_failed"
        assert heal.load_heal_receipts(tmp_path, LOOP) == ()


class TestRunnerCompatibility:
    def test_discovered_and_plan_shape(self, tmp_path: Path) -> None:
        ids = {p.playbook_id for p in heal.discovered_playbooks()}
        assert "data_rebuild/v1" in ids
        blocker = {**_blocker(), "_root": tmp_path / "outputs", "_loop_id": LOOP}
        plan = PLAYBOOK.plan(blocker, cwd=tmp_path)
        assert plan is not None
        assert plan.blocker_class == "data"
        assert plan.steps[0].step_id == "rebuild_data_seam"
        assert "--verify-state" in plan.verify.argv
        assert PLAYBOOK.plan(_blocker(), cwd=tmp_path) is None

    def test_verify_state_cli_decides_on_growth(self, tmp_path: Path) -> None:
        state = tmp_path / "state.json"
        state.write_text(
            json.dumps({"records_before": 4, "records_after": 4}), encoding="utf-8"
        )
        assert data_rebuild._main(["--verify-state", str(state)]) == 1
        state.write_text(
            json.dumps({"records_before": 4, "records_after": 9}), encoding="utf-8"
        )
        assert data_rebuild._main(["--verify-state", str(state)]) == 0
