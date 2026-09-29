"""Regression: only the exact current data predicate authorizes healing."""
from __future__ import annotations

import json

import pytest

from slm_training.autoresearch import heal
from slm_training.autoresearch.heal.playbooks import data_rebuild
from slm_training.autoresearch.heal.escalation import blocker_fingerprint
from slm_training.harnesses.train_data.repair import publish_repair
from tests.test_harnesses.train_data.readiness_fixtures import record, request_fixture, stage_rows

LOOP = "test-loop"
CAMPAIGN = "campaign"


def _blocker(request=None):
    result = {"campaign_id": CAMPAIGN, "index": 1, "kind": "rebuild_data", "reason": "role-safety"}
    if request:
        result["data_readiness_request"] = request.model_dump()
    return result


def _execute(root, request, seam):
    return data_rebuild.execute(_blocker(request), cwd=root, root=root, loop_id=LOOP,
                                campaign_id=CAMPAIGN, seam=seam)


def test_legacy_growth_counter_cannot_heal_or_invoke_unspecified_builder(tmp_path):
    calls = []
    receipt = data_rebuild.execute(_blocker(), cwd=tmp_path, root=tmp_path, loop_id=LOOP,
                                  campaign_id=CAMPAIGN, seam=lambda **kw: calls.append(kw),
                                  count_records=lambda *_: 100000)
    assert receipt.outcome == "postcondition_failed"
    assert "missing_data_readiness_contract" in receipt.note
    assert calls == []


@pytest.mark.parametrize("effect", ["unrelated", "malformed", "copy", "assert_fixed"])
def test_unrelated_growth_or_claimed_success_is_not_healing(tmp_path, effect):
    request = request_fixture(tmp_path)
    def seam(**_):
        unrelated = tmp_path / "outputs/data/train/unrelated"
        unrelated.mkdir()
        original = tmp_path / request.original.directory / "records.jsonl"
        (unrelated / "records.jsonl").write_text(
            original.read_text() if effect == "copy" else "{garbage\n" * 30)
        return {"healed": True, "records_after": 9999, "effect": effect}
    receipt = _execute(tmp_path, request, seam)
    assert receipt.outcome != "healed"
    assert receipt.verify_result.returncode != 0


def test_same_count_actual_repair_and_current_state_reverification(tmp_path):
    request = request_fixture(tmp_path)
    def seam(**kw):
        return publish_repair(kw["request"], root=kw["cwd"],
                              staged=stage_rows(tmp_path, request, [record()]))
    receipt = _execute(tmp_path, request, seam)
    assert receipt.outcome == "healed", receipt
    assert receipt.verify_result.step_id == "original_data_predicate"
    fingerprint = blocker_fingerprint("rebuild_data", "role-safety", data_request=request)
    assert receipt.blocker_fingerprint == fingerprint
    state = data_rebuild.state_path(tmp_path, LOOP, fingerprint)
    assert data_rebuild._verify_state(state, cwd=tmp_path, request_sha=request.sha256) == 0
    assert data_rebuild._verify_state(state, cwd=tmp_path, request_sha="0" * 64) == 1
    destination = tmp_path / "outputs/data/train/successor/records.jsonl"
    destination.write_text("{}\n")
    assert data_rebuild._verify_state(state, cwd=tmp_path, request_sha=request.sha256) == 1
    # Saved 'healed' is historical, not a current acknowledgment.
    assert json.loads(state.read_text())["outcome"] == "healed"
    assert len(heal.load_heal_receipts(tmp_path, LOOP)) == 1


def test_seam_crash_remains_failed(tmp_path):
    request = request_fixture(tmp_path)
    def seam(**_):
        raise RuntimeError("build_train_data exploded")
    receipt = _execute(tmp_path, request, seam)
    assert receipt.outcome == "step_failed"
    assert "exploded" in receipt.note


def test_plan_binds_original_request_and_legacy_has_no_guessed_plan(tmp_path):
    request = request_fixture(tmp_path)
    blocker = {**_blocker(request), "_root": tmp_path, "_loop_id": LOOP}
    plan = data_rebuild.PLAYBOOK.plan(blocker, cwd=tmp_path)
    assert plan is not None
    assert plan.steps[0].step_id == "rebuild_data_seam"
    assert request.sha256 in plan.verify.argv
    assert "--request-payload" in plan.steps[0].argv
    assert data_rebuild.PLAYBOOK.plan(_blocker(), cwd=tmp_path) is None
    assert "data_rebuild/v1" in {p.playbook_id for p in heal.discovered_playbooks()}


def test_old_count_state_never_verifies(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"records_before": 4, "records_after": 9999}))
    assert data_rebuild._main(["--verify-state", str(state), "--cwd", str(tmp_path)]) == 1


def test_wrong_campaign_contract_never_executes(tmp_path):
    request = request_fixture(tmp_path).model_copy(update={"campaign_id": "wrong"})
    calls = []
    receipt = _execute(tmp_path, request, lambda **kw: calls.append(kw))
    assert receipt.outcome != "healed" and not calls
