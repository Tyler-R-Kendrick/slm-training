"""Actual command construction and cancellation plumbing; no agent inference."""

import hashlib
import json
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from slm_training.autoresearch.heal.isolated_agent import BubblewrapAgentRunner
from slm_training.autoresearch.heal import isolation_workspace
from slm_training.autoresearch.heal.isolation import build_isolated_command
from slm_training.autoresearch.heal.isolation_workspace import (
    manifest_digest,
    tree_manifest,
)
from tests.test_autoresearch.test_repair_dispatch import (
    repair_request as base_repair_request,
)


@pytest.fixture
def repair_request(tmp_path):
    return base_repair_request.__wrapped__(tmp_path)


@pytest.mark.parametrize("cancel", [False, True])
def test_runner_mounts_exact_grants_for_atomic_replace_and_cancels_independently(
    tmp_path, repair_request, monkeypatch, cancel
):
    source = tmp_path / "source"
    (source / "src/package").mkdir(parents=True)
    (source / "src/package/module.py").write_text("answer = 0\n")
    (source / "src/package/sibling.py").write_text("protected = True\n")
    (source / "tests").mkdir()
    (source / "tests/test_original.py").write_text("assert True\n")
    request = repair_request.model_copy(
        update={
            "allowed_paths": ("src/package/module.py", "tests/test_added.py"),
            "blocker": repair_request.blocker.model_copy(
                update={"source_digest": manifest_digest(tree_manifest(source))}
            ),
        }
    )
    runner = BubblewrapAgentRunner(source=source, attempt_root=tmp_path / "attempts")
    monkeypatch.setattr(runner, "capability", lambda _: None)
    monkeypatch.setattr(runner, "_argv", lambda argv: argv)
    cancelled = threading.Event()

    def process(spec, argv, **kwargs):
        helper = spec.workspace / "repair-input/proposal-digest.py"
        assert helper.read_bytes() == Path(isolation_workspace.__file__).read_bytes()
        assert stat.S_IMODE(helper.stat().st_mode) == 0o444
        instructions = json.loads(
            (spec.workspace / "repair-input/repair-instructions.json").read_text()
        )
        assert "/usr/bin/python3 -I -S repair-input/proposal-digest.py" in instructions[
            "digest_contract"
        ]
        command = build_isolated_command(spec, argv, "/usr/bin/bwrap")
        mounts = [
            command[i + 1 : i + 3] for i, arg in enumerate(command) if arg == "--bind"
        ]
        assert spec.writable_paths == (
            "src/package/module.py", "tests/test_added.py", "repair-output/proposal.json"
        )
        assert spec.writable_dirs == ()
        assert mounts == [
            [str(spec.workspace / relative), f"/workspace/{relative}"]
            for relative in ("repair-output", "src/package", "tests")
        ]
        for relative in ("src/package/sibling.py", "tests/test_original.py"):
            overlay = ["--ro-bind", str(spec.workspace / relative), f"/workspace/{relative}"]
            parent_bind = [
                "--bind", str(spec.workspace / Path(relative).parent),
                f"/workspace/{Path(relative).parent.as_posix()}",
            ]
            assert any(command[i:i + 3] == overlay for i in range(len(command)))
            assert command.index(overlay[1]) > command.index(parent_bind[1])
        assert "--unshare-all" in command and "--clearenv" in command
        if cancel:
            cancelled.set()
            assert kwargs["cancel_event"].wait(1), (
                "cancellation must not depend on heartbeat"
            )
        else:
            for relative, content in (
                ("src/package/module.py", "answer = 1\n"),
                ("tests/test_added.py", "assert True\n"),
                ("repair-output/proposal.json", '{"fixture":true}'),
            ):
                target = spec.workspace / relative
                temporary = target.with_suffix(".tmp")
                temporary.write_text(content)
                temporary.replace(target)
        return SimpleNamespace(
            outcome=SimpleNamespace(value="cancelled" if cancel else "completed"),
            returncode=-2 if cancel else 0,
            duration_seconds=0.1,
            stdout="native diagnostic tail", stderr="native failure",
            stdout_truncated=True, stderr_truncated=False,
        )

    monkeypatch.setattr(
        "slm_training.autoresearch.heal.isolated_agent.run_isolated", process
    )
    result = runner.run(
        request,
        ("/bin/true",),
        inputs={
            "proposal-schema.json": {},
            "repair-instructions.json": {"digest_contract": "Hash the candidate."},
        },
        progress=lambda: None,
        cancelled=cancelled.is_set,
    )
    assert result.outcome == ("cancelled" if cancel else "completed")
    if not cancel:
        assert result.final_json == '{"fixture":true}'
    attempt = runner.attempt_root / request.digest()
    assert (attempt / "output/proposal.json").is_file()
    assert not (attempt / "candidate/repair-output").exists()
    receipt = json.loads((attempt / "execution/receipt.json").read_text())
    assert receipt["acceptance_authority"] is False
    assert receipt["outcome"] == result.outcome
    assert receipt["stdout_truncated"] is True
    assert receipt["stdout_sha256"] == hashlib.sha256(b"native diagnostic tail").hexdigest()
    assert (attempt / "execution/stdout.txt").read_text() == "native diagnostic tail"
    assert not (attempt / "candidate/execution").exists()


def test_existing_tests_cannot_be_writable_source(tmp_path):
    from slm_training.autoresearch.heal.isolated_agent import _create_regression_mounts

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_original.py").write_text("assert True\n")
    with pytest.raises(ValueError, match="protected"):
        _create_regression_mounts(tmp_path, ("tests/test_original.py",))


def test_cancelled_process_preserves_spent_compute(repair_request, monkeypatch):
    from slm_training.autoresearch.heal.agent_executor import AgentRun
    from tests.test_autoresearch.test_repair_dispatch import executor as make_executor

    executor = make_executor.__wrapped__(monkeypatch)
    monkeypatch.setattr(
        executor.runner, "run", lambda *a, **k: AgentRun("cancelled", -2, 0.5, "")
    )
    result = executor.execute(
        repair_request, progress=lambda: None, cancelled=lambda: False
    )
    assert result.status == "cancelled" and result.spent_seconds == 0.5


@pytest.mark.parametrize("fails", [False, True])
def test_empty_codex_mount_target_is_removed_after_execution(tmp_path, monkeypatch, fails):
    from slm_training.autoresearch.heal import isolation

    monkeypatch.setattr(isolation, "probe_isolation", lambda: SimpleNamespace(available=True, executable="bwrap"))
    spec = isolation.IsolationSpec(tmp_path, codex_mount_target=True)
    result = object()

    def run(*args):
        assert (tmp_path / ".git").is_dir()
        assert not list((tmp_path / ".git").iterdir())
        if fails:
            raise OSError("launch failure")
        return result

    monkeypatch.setattr(isolation, "_run_checked", run)
    if fails:
        with pytest.raises(OSError, match="launch failure"):
            isolation.run_isolated(spec, ("/bin/true",))
    else:
        assert isolation.run_isolated(spec, ("/bin/true",)) is result
    assert not (tmp_path / ".git").exists()


def test_codex_mount_target_never_accepts_existing_git(tmp_path):
    from slm_training.autoresearch.heal.isolation import IsolationSpec
    from slm_training.autoresearch.heal.isolation_workspace import IsolationViolation

    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/config").write_text("existing metadata")
    with pytest.raises(IsolationViolation, match="Git metadata"):
        build_isolated_command(IsolationSpec(tmp_path, codex_mount_target=True), ("/bin/true",), "bwrap")
    assert (tmp_path / ".git/config").read_text() == "existing metadata"
