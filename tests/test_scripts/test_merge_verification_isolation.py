"""Controller manifests stay outside candidate authority, including failure paths."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.merge_verification import run_release_gate, run_workload
from scripts.merge_verification_evidence import ReceiptCache, validate_cached_state
from scripts.merge_verification_isolation import isolated_static, runtime_command
from scripts.verify_merge_ready import Step
from slm_training.autoresearch.heal.isolation import IsolationUnavailable


def _candidate(tmp_path, code):
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "test_case.py").write_text(code)
    control = tmp_path / "controller"
    control.mkdir(mode=0o700)
    (control / "secret").write_text("controller-only")
    return root, control


def test_missing_boundary_refuses_before_candidate_or_cache_access(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from slm_training.autoresearch.heal import isolation

    monkeypatch.setattr(
        isolation,
        "probe_isolation",
        lambda: SimpleNamespace(available=False, reason="denied"),
    )
    with pytest.raises(IsolationUnavailable, match="denied"):
        run_release_gate(
            (),
            root=tmp_path,
            base_ref="HEAD",
            state_dir=tmp_path / "cache",
            step_seconds=1,
            run_step=lambda *args: pytest.fail("executed"),
        )
    assert not (tmp_path / "cache").exists()


def test_real_isolated_manifest_cache_and_source_are_protected(tmp_path):
    root, control = _candidate(tmp_path, "")
    code = f"""from pathlib import Path
import pytest
def test_boundary():
    assert not Path({str(control / "secret")!r}).exists()
    assert not Path('/home/codex').exists()
    for path in ['/workspace/control/request.json', '/workspace/control/worker.py', __file__]:
        with pytest.raises(OSError):
            Path(path).write_text('forged')
    assert not Path('/workspace/candidate/.git').exists()
"""
    (root / "test_case.py").write_text(code)
    result = run_workload(
        root,
        ["test_case.py::test_boundary"],
        collect_only=False,
        seconds=15,
        directory=control,
        isolated=True,
        runtimes=(Path(sys.prefix),),
    )
    assert result["status"] == "ok", result
    assert result["evidence_class"] == "isolated_process"
    assert (control / "secret").read_text() == "controller-only"
    assert (root / "test_case.py").read_text() == code


def test_real_isolated_zero_exit_without_result_is_not_success(tmp_path):
    root, control = _candidate(tmp_path, "import os\nos._exit(0)\n")
    result = run_workload(
        root,
        ["test_case.py"],
        collect_only=True,
        seconds=15,
        directory=control,
        isolated=True,
        runtimes=(Path(sys.prefix),),
    )
    assert result["status"] == "failed"
    assert "workload_sha256" not in result


def test_granted_bridge_requires_matching_candidate_sources(tmp_path):
    from scripts.merge_verification_isolation import _workload_argv

    runtime = tmp_path / "openui_bridge"
    runtime.mkdir()
    (runtime / "cli.mjs").write_text("unrelated source")
    with pytest.raises(ValueError, match="bridge_runtime_source_mismatch"):
        _workload_argv(tmp_path / "candidate", (runtime,), ["python"])


def test_real_static_failure_cannot_write_control_store(tmp_path):
    root, control = _candidate(tmp_path, "")
    step = Step(
        "attack",
        (
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(control / 'secret')!r}).write_text('forged')",
        ),
    )
    result = isolated_static(
        step, budget_seconds=10, root=root, runtimes=(Path(sys.prefix),)
    )
    assert result["status"] == "failed"
    assert (control / "secret").read_text() == "controller-only"


def test_isolated_static_exposes_approved_runtime_bin(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import scripts.merge_verification_isolation as isolation

    root, _ = _candidate(tmp_path, "")
    captured = {}

    def fake_run(spec, argv):
        captured["spec"] = spec
        return SimpleNamespace(
            returncode=0,
            timed_out=False,
            interrupted=False,
            killed=False,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(isolation, "run_isolated", fake_run)
    result = isolation.isolated_static(
        Step("ruff", (sys.executable, "-c", "pass")),
        budget_seconds=10,
        root=root,
        runtimes=(Path(sys.prefix),),
    )

    assert result["status"] == "ok"
    assert captured["spec"].environment == (
        ("PATH", "/runtime/0/bin:/usr/bin:/bin"),
        ("RUFF_CACHE_DIR", "/tmp/ruff-cache"),
    )
    isolation.isolated_static(
        Step("compileall", (sys.executable, "-c", "pass")),
        budget_seconds=10,
        root=root,
        runtimes=(Path(sys.prefix),),
    )
    assert captured["spec"].environment[-1] == (
        "PYTHONPYCACHEPREFIX", "/tmp/pycache"
    )


def test_executable_requires_approved_runtime():
    with pytest.raises(ValueError, match="approved"):
        runtime_command(("/home/attacker/python", "-c", "pass"), ())


@pytest.mark.parametrize("identity", ["../escape", "a" * 63, "A" * 64, "a" * 64 + "/x"])
def test_cache_identity_cannot_escape_namespace(tmp_path, identity):
    cache = ReceiptCache(tmp_path / "control", tmp_path / "candidate")
    with pytest.raises(ValueError, match="SHA256"):
        cache.save({"identity": identity})
    with pytest.raises(ValueError, match="SHA256"):
        cache.load(identity)


def test_cached_pass_counter_is_not_proof():
    from scripts.merge_verification_evidence import digest

    binding = {"static_commands": []}
    state = {
        "binding": binding,
        "identity": digest(binding),
        "nodes": [],
        "passed_nodes": ["forged"],
    }
    with pytest.raises(ValueError, match="without collection"):
        validate_cached_state(state, binding)


def test_quarantined_cache_cannot_be_reused_even_after_source_is_restored():
    with pytest.raises(ValueError, match="quarantined"):
        validate_cached_state({"invalidated": "source_changed_during_verification"}, {})


@pytest.mark.parametrize(
    "targets",
    [[], ["-p", "evil"], ["../test.py"], ["/host/test.py"], ["test.py", "test.py"]],
)
def test_candidate_targets_cannot_inject_options_or_host_paths(targets, tmp_path):
    with pytest.raises(ValueError):
        run_workload(
            tmp_path, targets, collect_only=True, seconds=1, directory=tmp_path
        )


def test_exhausted_obligation_does_not_get_another_process(monkeypatch):
    from scripts import merge_verification

    nodes = ["test_case.py::test_case"]
    state = {
        "binding": {"max_attempts_per_obligation": 3},
        "shards": [nodes],
        "passed_nodes": [],
        "attempts": [
            {"kind": "shard", "nodes": nodes, "status": "failed"} for _ in range(3)
        ],
    }
    monkeypatch.setattr(
        merge_verification,
        "run_workload",
        lambda *args, **kwargs: pytest.fail("unbounded retry"),
    )
    merge_verification._run_shards(
        state, Path("unused"), Path("unused"), lambda: 20, lambda: None
    )
    assert len(state["attempts"]) == 3


def test_private_metadata_preparation_failure_cannot_be_a_green_static(
    tmp_path, monkeypatch
):
    import time
    from types import SimpleNamespace

    from scripts import merge_verification_isolation as owner

    monkeypatch.setattr(
        owner,
        "run_bounded_process",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            timed_out=False,
            interrupted=False,
            killed=False,
            stderr="fixture failure",
        ),
    )
    with pytest.raises(IsolationUnavailable, match="private_git_preparation_failed"):
        owner._private_git(
            tmp_path, tmp_path / "controller" / "git", 1, time.monotonic()
        )


def test_private_git_command_never_writes_source_refs(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace

    from scripts import merge_verification_isolation as owner

    commands = []
    destination = tmp_path / "controller" / "git"
    destination.parent.mkdir()

    def execute(command, **kwargs):
        commands.append((command, kwargs))
        destination.mkdir(exist_ok=True)
        return SimpleNamespace(
            returncode=0, timed_out=False, interrupted=False, killed=False
        )

    monkeypatch.setattr(owner, "run_bounded_process", execute)
    owner._private_git(tmp_path / "source", destination, 10, time.monotonic())
    assert "--no-local" in commands[0][0] and "--no-hardlinks" in commands[0][0]
    assert commands[0][0][-1] == str(destination)
    assert commands[1][0] == ["git", "--git-dir", str(destination), "read-tree", "HEAD"]
    assert commands[0][1]["env"]["GIT_ALLOW_PROTOCOL"] == "file"
    assert "remote" not in (destination / "config").read_text()


def test_real_private_git_metadata_is_readonly_and_independent(tmp_path):
    """Copy existing history into disposable metadata; never edit authoring refs."""
    import time

    from scripts import merge_verification_isolation as owner
    from slm_training.autoresearch.heal.isolation import IsolationSpec, run_isolated

    source = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    metadata = tmp_path / "control" / "git"
    owner._private_git(source, metadata, 40, time.monotonic())
    assert not (metadata / "objects/info/alternates").exists()
    assert not (metadata / "hooks").exists()
    result = run_isolated(
        IsolationSpec(tmp_path, timeout_seconds=10),
        ["git", "--git-dir=/workspace/control/git", "rev-parse", "HEAD"],
    )
    assert result.returncode == 0 and result.stdout.strip() == head, result
    indexed = subprocess.check_output(
        ["git", "--git-dir", str(metadata), "ls-files"], text=True
    )
    assert "scripts/verify_merge_ready.py" in indexed.splitlines()
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        == head
    )


def test_snapshot_excludes_ignored_unbound_inputs(tmp_path, monkeypatch):
    from scripts import merge_verification_isolation as owner

    source = tmp_path / "source"
    source.mkdir()
    (source / ".git").mkdir()
    (source / ".env").write_text("PRIVATE=not-for-worker")
    (source / "test_case.py").write_text("def test_case(): pass\n")
    monkeypatch.setattr(owner, "source_paths", lambda root: ["test_case.py"])
    copy = owner._candidate_snapshot(source, tmp_path / "copy")
    assert (copy / "test_case.py").is_file()
    assert not (copy / ".env").exists()
    assert (source / ".env").read_text() == "PRIVATE=not-for-worker"


def test_real_extra_runtime_and_failed_collection_preserve_evidence(tmp_path):
    root, control = _candidate(tmp_path, "import approved_extra\nassert False, 'original-collection-fault'\n")
    runtime = tmp_path / "dependencies"
    runtime.mkdir()
    (runtime / "approved_extra.py").write_text("VALUE = 1\n")
    result = run_workload(
        root, ["test_case.py"], collect_only=True, seconds=15,
        directory=control, isolated=True, runtimes=(Path(sys.prefix), runtime),
    )
    assert result["status"] == "failed"
    assert "original-collection-fault" in result["output_tail"]
    assert "No module named" not in result["output_tail"]
    assert result["workload_observation"]["collection_errors"] == ["test_case.py"]
    assert "workload_sha256" not in result


def test_real_collection_plans_fresh_invocations_not_remaining_tail(tmp_path, monkeypatch):
    from scripts import merge_verification as owner

    root, control = _candidate(tmp_path, "import pytest\n@pytest.mark.parametrize('value', range(60))\ndef test_value(value): assert value >= 0\n")
    monkeypatch.setattr(owner.check_changed, "_test_file_durations", lambda: {})
    state = {"binding": {"targets": ["test_case.py"], "isolation_enforced": False,
                         "runtime_roots": [], "max_attempts_per_obligation": 3},
             "shard_budget_seconds": 60, "attempts": []}
    assert owner._collect(state, root, control, lambda: 12, lambda: None)
    assert len(state["nodes"]) == 60
    assert len(state["shards"]) == 2
    assert sorted(node for shard in state["shards"] for node in shard) == sorted(state["nodes"])


def test_shard_waits_for_declared_slice_without_spending_retry(tmp_path, monkeypatch):
    from scripts import merge_verification as owner

    nodes = ["test_case.py::test_value"]
    state = {"binding": {"max_attempts_per_obligation": 3, "isolation_enforced": False,
                         "runtime_roots": []}, "shards": [nodes],
             "passed_nodes": [], "attempts": [], "shard_budget_seconds": 60}
    monkeypatch.setattr(owner, "run_workload", lambda *a, **k: pytest.fail("launched on inadequate tail"))
    owner._run_shards(state, tmp_path, tmp_path, lambda: 12, lambda: None)
    assert state["attempts"] == state["passed_nodes"] == []


def test_real_feedback_worker_reuses_readonly_caches(tmp_path, monkeypatch):
    root, control = _candidate(tmp_path, "import os\ndef test_environment():\n assert 'PYTHONPYCACHEPREFIX' not in os.environ\n assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'\n")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(tmp_path / "inherited-cache"))
    result = run_workload(root, ["test_case.py::test_environment"], collect_only=False,
                          seconds=15, directory=control)
    assert result["status"] == "ok", result
    assert not (root / "__pycache__").exists()
    assert not (tmp_path / "inherited-cache").exists()
