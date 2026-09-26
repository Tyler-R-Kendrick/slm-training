"""Bounded shard admission and real verified-collection resumption."""

from functools import partial
import subprocess
import sys

from scripts import check_changed, merge_verification
from scripts.merge_verification import _shard_seconds, run_release_gate
from scripts.verify_merge_ready import Step, run_step
from tests.test_scripts.test_merge_verification import _fixture


def test_stale_duration_cannot_park_a_bounded_shard(monkeypatch) -> None:
    node = "tests/test_case.py::test_case"
    monkeypatch.setattr(
        check_changed, "_test_file_durations", lambda: {"tests/test_case.py": 99999.0}
    )
    assert _shard_seconds({"nodes": [node], "attempts": []}, [node], 10.0) == 10.0


def test_real_workload_resumes_verified_collection_without_duplicate_tests(
    tmp_path, monkeypatch
) -> None:
    root, control = _fixture(tmp_path, "def test_case(): pass\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    # Only base-tree lookup is a fixture; collection, execution, signing and restart are real.
    monkeypatch.setattr(
        merge_verification,
        "changed_paths",
        lambda *_: ("base-fixture", ["tests/test_case.py"]),
    )
    steps = (Step("static-canary", (sys.executable, "-c", "pass")),)
    execute = merge_verification._run_shards
    monkeypatch.setattr(merge_verification, "_run_shards", lambda *args: None)
    verify = partial(
        run_release_gate,
        steps,
        root=root,
        base_ref="HEAD",
        state_dir=control,
        step_seconds=60,
        run_step=run_step,
        local_feedback=True,
    )
    first = verify()
    assert first["status"] == "pending"
    assert first["node_counts"]["pending"] == 1
    monkeypatch.setattr(merge_verification, "_run_shards", execute)
    second = verify()
    assert second["verification_complete"] is True, second
    assert second["release_authorized"] is False
    assert [row.get("kind") for row in second["steps"]].count("collection") == 1
    assert [row.get("kind") for row in second["steps"]].count("shard") == 1
    third = verify(step_seconds=30)
    assert third["identity"] == second["identity"]
    assert third["spent_seconds"] == second["spent_seconds"]
    (root / "fixture.json").write_text("new fixture identity")
    fourth = verify()
    assert fourth["identity"] != third["identity"]
    assert fourth["verification_complete"] is True, fourth
