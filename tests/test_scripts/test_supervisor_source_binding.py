"""Startup shares one scan; subsequent worker validation observes drift."""

from pathlib import Path

import pytest

from tests.test_autoresearch.test_locked_prereg_supervisor import _fixture


def test_locked_startup_shares_validation_but_worker_boundary_rechecks(tmp_path, monkeypatch):
    from scripts import run_autotrain_supervisor as supervisor
    from scripts import autotrain_repair_activation as activation
    from scripts import merge_verification_evidence as evidence
    from scripts.autotrain_supervisor_operations import validate_operation_identity
    from slm_training.harness_core import execution_release as release

    plan, path, root, store, _ = _fixture(tmp_path, monkeypatch)
    source = Path(plan["source_path"])
    monkeypatch.chdir(source)
    scans = []
    original = release._files
    def scan(path):
        scans.append(path)
        return original(path)
    monkeypatch.setattr(release, "_files", scan)
    monkeypatch.setattr(evidence, "environment_identity", lambda: {})
    monkeypatch.setattr(activation, "recover_release", lambda *a, **k: None)
    def supervise(args, runtime, common):
        assert scans == [tmp_path / "release", source]
        assert common["source_digest"] == plan["source_digest"]
        # Same length and restored mtime cannot hide changed writable source.
        target = source / "fixture.py"
        info = target.stat()
        target.write_text("modified bytes\n")
        import os
        os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
        with pytest.raises(ValueError, match="execution_source_drift"):
            validate_operation_identity(common, supervisor._source_identity, "before launch")
        return 10
    monkeypatch.setattr(supervisor, "_supervise", supervise)
    assert supervisor.main([
        "--loop-id", plan["campaign_id"], "--root", str(root),
        "--locked-preregistration", str(path), "--train-version", "fixture-train",
        "--steps", "6", "--primary-metric", "smoke.eval_nll",
        "--continuation-grant", store.load_campaign().budget.continuation_grant.model_dump_json(),
        "--max-cycles", "1",
    ]) == 10
