"""Authenticated canonical merge producer to repair acceptance; simulated isolation."""

import json
import shutil
from dataclasses import replace

import pytest

from scripts import merge_verification as merge
from scripts.merge_verification_evidence import ReceiptCache
from slm_training.autoresearch.heal.repair_acceptance import VerificationWorkspace
from tests.test_autoresearch.test_repair_acceptance import source_gate_fixture


@pytest.fixture
def source_gate(tmp_path, monkeypatch, request):
    base = tmp_path / "base"
    (base / "tests").mkdir(parents=True)
    (base / "module.py").write_text("value = 0\n")
    (base / "tests/test_module.py").write_text("def test_fixture():\n    assert True\n")
    candidate = tmp_path / "candidate"
    shutil.copytree(base, candidate)
    (candidate / "module.py").write_text("value = 1\n")
    workspace = VerificationWorkspace(base, candidate)
    return workspace, source_gate_fixture(tmp_path, monkeypatch, workspace, complete=getattr(request, "param", True))


def test_complete_real_pytest_producer_is_authenticated_and_content_bound(source_gate):
    workspace, gate = source_gate
    summary = gate.read(workspace)
    assert summary["verification_complete"]
    assert not summary["release_authorized"], "tests alone never authorize source activation"


def test_worker_changed_pass_claim_has_no_valid_mac(source_gate):
    workspace, gate = source_gate
    path = gate.state_dir / (gate.identity + ".json")
    envelope = json.loads(path.read_text())
    envelope["payload"]["passed_nodes"] = ["forged::test"]
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="unauthenticated"):
        gate.read(workspace)


def test_partial_authenticated_collection_is_still_pending(source_gate):
    workspace, gate = source_gate
    cache = ReceiptCache(gate.state_dir, gate.root)
    state = cache.load(gate.identity)
    state["passed_nodes"] = []
    state["attempts"] = [row for row in state["attempts"] if row["kind"] == "collection"]
    cache.save(state)
    assert gate.read(workspace) is None


def test_stale_identity_cannot_reuse_complete_evidence(source_gate):
    workspace, gate = source_gate
    assert replace(gate, identity="0" * 64).read(workspace) is None


def test_tested_source_drift_invalidates_the_authenticated_journal(source_gate):
    workspace, gate = source_gate
    (gate.root / "module.py").write_text("value = 88\n")
    with pytest.raises(ValueError, match="binding_changed"):
        gate.read(workspace)


def test_absent_journal_does_not_create_an_issuer(source_gate, tmp_path):
    workspace, gate = source_gate
    directory = tmp_path / "missing-cache"
    assert replace(gate, state_dir=directory).read(workspace) is None
    assert not directory.exists()


def test_pending_materialization_is_not_synchronous_work(source_gate, tmp_path, monkeypatch):
    workspace, gate = source_gate
    monkeypatch.setattr(merge, "verification_binding", lambda *args, **kwargs: pytest.fail("pending materialization inspected"))
    waiting = replace(gate, root=tmp_path / "not-materialized", state_dir=tmp_path / "not-issued")
    assert waiting.read(workspace) is None
    assert not waiting.root.exists() and not waiting.state_dir.exists()


def test_wrong_candidate_cannot_borrow_successful_test_evidence(source_gate, tmp_path):
    workspace, gate = source_gate
    other = tmp_path / "other-candidate"
    shutil.copytree(workspace.candidate, other)
    (other / "module.py").write_text("value = 999\n")
    with pytest.raises(ValueError, match="candidate_mismatch"):
        gate.read(replace(workspace, candidate=other))


def test_changed_base_cannot_reclassify_frozen_coverage(source_gate):
    workspace, gate = source_gate
    (workspace.base / "module.py").write_text("value = 999\n")
    with pytest.raises(ValueError, match="base_mismatch"):
        gate.read(workspace)


def test_repair_change_omitted_by_comparison_base_is_refused(source_gate, monkeypatch):
    workspace, gate = source_gate
    cache = ReceiptCache(gate.state_dir, gate.root)
    state = cache.load(gate.identity)
    binding = dict(state["binding"], changed_paths=["tests/test_module.py"])
    state.update(binding=binding, identity=merge.digest(binding))
    cache.save(state)
    monkeypatch.setattr(merge, "verification_binding", lambda *args, **kwargs: binding)
    with pytest.raises(ValueError, match="omits_repair_changes"):
        replace(gate, identity=state["identity"]).read(workspace)


@pytest.mark.parametrize("mode,edit_bytes,accepted", [(0o644, False, True), (0o604, False, False),
                                                     (0o755, False, False), (0o644, True, False)])
def test_only_exact_owner_write_preparation_can_be_absent_from_git_coverage(
    tmp_path, monkeypatch, mode, edit_bytes, accepted,
):
    base, candidate = tmp_path / "base", tmp_path / "candidate"
    (base / "tests").mkdir(parents=True)
    (base / "tests/test_fixture.py").write_text("def test_fixture():\n    assert True\n")
    (base / "untouched.py").write_text("VALUE = 1\n")
    (base / "untouched.py").chmod(0o444)
    shutil.copytree(base, candidate)
    (candidate / "untouched.py").chmod(mode)
    if edit_bytes:
        (candidate / "untouched.py").write_text("VALUE = 2\n")
    workspace = VerificationWorkspace(base, candidate)
    gate = source_gate_fixture(tmp_path, monkeypatch, workspace)
    cache = ReceiptCache(gate.state_dir, gate.root)
    state = cache.load(gate.identity)
    binding = dict(state["binding"], changed_paths=["tests/test_fixture.py"])
    state.update(binding=binding, identity=merge.digest(binding))
    cache.save(state)
    monkeypatch.setattr(merge, "verification_binding", lambda *args, **kwargs: binding)
    reader = replace(gate, identity=state["identity"])
    if accepted:
        assert reader.read(workspace)["verification_complete"]
    else:
        with pytest.raises(ValueError, match="omits_repair_changes"):
            reader.read(workspace)


def test_local_feedback_cannot_be_relabelled_isolated(source_gate):
    workspace, gate = source_gate
    cache = ReceiptCache(gate.state_dir, gate.root)
    state = cache.load(gate.identity)
    state["binding"]["isolation_enforced"] = False
    cache.save(state)
    with pytest.raises(ValueError, match="binding mismatch"):
        gate.read(workspace)


@pytest.mark.parametrize("exposure", ["candidate", "base", "runtime"])
def test_controller_issuer_cannot_be_in_a_workload_mount(source_gate, exposure):
    workspace, gate = source_gate
    if exposure == "runtime":
        workspace = replace(workspace, runtime_roots=(gate.state_dir,))
    else:
        gate = replace(gate, state_dir=getattr(workspace, exposure) / "issuer")
    with pytest.raises(ValueError, match="issuer_exposed_to_workload"):
        gate.read(workspace)


@pytest.mark.parametrize("source_gate", [False], indirect=True)
def test_pending_canonical_journal_resumes_in_a_separate_invocation(source_gate, tmp_path, monkeypatch):
    workspace, pending_gate = source_gate
    assert pending_gate.read(workspace) is None
    cache = ReceiptCache(pending_gate.state_dir, pending_gate.root)
    before = cache.load(pending_gate.identity)
    completed_gate = source_gate_fixture(tmp_path, monkeypatch, workspace)
    assert completed_gate.identity == pending_gate.identity
    after = cache.load(completed_gate.identity)
    assert [row for row in before["attempts"] if row["kind"] == "collection"] == [
        row for row in after["attempts"] if row["kind"] == "collection"
    ]
    assert len(after["attempts"]) > len(before["attempts"])
    assert completed_gate.read(workspace)["verification_complete"]
