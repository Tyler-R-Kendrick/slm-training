"""Actual command construction and cancellation plumbing; no agent inference."""

import threading
from types import SimpleNamespace

import pytest

from slm_training.autoresearch.heal.isolated_agent import BubblewrapAgentRunner
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
def test_runner_mounts_only_result_file_and_cancels_independently(
    tmp_path, repair_request, monkeypatch, cancel
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("answer = 0\n")
    request = repair_request.model_copy(
        update={
            "allowed_paths": ("module.py", "tests/test_added.py"),
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
        command = build_isolated_command(spec, argv, "/usr/bin/bwrap")
        mounts = [
            command[i + 1 : i + 3] for i, arg in enumerate(command) if arg == "--bind"
        ]
        assert [
            str(spec.workspace / "repair-output/proposal.json"),
            "/workspace/repair-output/proposal.json",
        ] in mounts
        assert all(pair[1] != "/workspace/repair-output" for pair in mounts)
        assert "--unshare-all" in command and "--clearenv" in command
        if cancel:
            cancelled.set()
            assert kwargs["cancel_event"].wait(1), (
                "cancellation must not depend on heartbeat"
            )
        else:
            (spec.workspace / "repair-output/proposal.json").write_text(
                '{"fixture":true}'
            )
        return SimpleNamespace(
            outcome=SimpleNamespace(value="cancelled" if cancel else "completed"),
            returncode=-2 if cancel else 0,
            duration_seconds=0.1,
        )

    monkeypatch.setattr(
        "slm_training.autoresearch.heal.isolated_agent.run_isolated", process
    )
    result = runner.run(
        request,
        ("/bin/true",),
        inputs={"proposal-schema.json": {}},
        progress=lambda: None,
        cancelled=cancelled.is_set,
    )
    assert result.outcome == ("cancelled" if cancel else "completed")
    attempt = runner.attempt_root / request.digest()
    assert (attempt / "output/proposal.json").is_file()
    assert not (attempt / "candidate/repair-output").exists()


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
