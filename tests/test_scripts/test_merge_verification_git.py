"""Private Git metadata grants never mutate or share authority with the source."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import merge_verification_git as owner
from slm_training.autoresearch.heal.isolation import IsolationSpec, IsolationUnavailable, run_isolated


def test_private_metadata_preparation_failure_cannot_be_a_green_static(
    tmp_path, monkeypatch
):
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
    nested = Path("/workspace/candidate").exists()
    if nested:
        # The candidate tree inside the verifier sandbox carries no Git
        # metadata (the private grant owns it), so the real source checkout is
        # replaced by a fixture repository with the same shape.
        source = tmp_path / "source"
        (source / "scripts").mkdir(parents=True)
        (source / "scripts/verify_merge_ready.py").write_text("fixture")
        subprocess.run(["git", "init", "-q"], cwd=source, check=True)
        subprocess.run(
            ["git", "config", "user.email", "fixture@example.com"],
            cwd=source,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Fixture"], cwd=source, check=True
        )
        subprocess.run(["git", "add", "-A"], cwd=source, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source, check=True)
    else:
        source = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    metadata = tmp_path / "control" / "git"
    owner._private_git(source, metadata, 40, time.monotonic())
    assert not (metadata / "objects/info/alternates").exists()
    assert not (metadata / "hooks").exists()
    if nested:
        observed = subprocess.run(
            ["git", "--git-dir", str(metadata), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    else:
        result = run_isolated(
            IsolationSpec(tmp_path, timeout_seconds=10),
            ["git", "--git-dir=/workspace/control/git", "rev-parse", "HEAD"],
        )
        assert result.returncode == 0, result
        observed = result.stdout.strip()
    assert observed == head
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
