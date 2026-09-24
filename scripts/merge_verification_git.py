from __future__ import annotations
import shutil
import time
from pathlib import Path
from slm_training.autoresearch.heal.isolation import IsolationUnavailable
from slm_training.harness_core.bounded_process import run_bounded_process
from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS
def _remaining(seconds: float, started: float) -> float:
    remaining = seconds - (time.monotonic() - started)
    if remaining <= 0:
        raise IsolationUnavailable("verification_preparation_budget_exhausted")
    return min(remaining, INTERRUPT_AFTER_SECONDS)
def _private_git(
    source: Path, destination: Path, seconds: float, started: float
) -> None:
    """Only a new disposable metadata copy is written; source refs never change.

    Non-local file transport copies reachable objects, not hooks/config,
    hardlinks or alternates. No credential or network protocol is available.
    Git-dependent static checks must see real base/index history, not an empty
    `ls-files` result from a metadata-free snapshot.
    """
    destination.parent.mkdir(exist_ok=True)
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(destination.parent),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_ALLOW_PROTOCOL": "file",
        "GIT_TERMINAL_PROMPT": "0",
    }
    commands = [
        [
            "git",
            "clone",
            "--bare",
            "--depth",
            "2",
            "--single-branch",
            "--no-local",
            "--no-hardlinks",
            "--",
            source.resolve().as_uri(),
            str(destination),
        ],
        ["git", "--git-dir", str(destination), "read-tree", "HEAD"],
    ]
    for command in commands:
        result = run_bounded_process(
            command,
            env=env,
            interrupt_after_seconds=_remaining(seconds, started),
            kill_grace_seconds=KILL_GRACE_SECONDS,
        )
        if result.timed_out or result.interrupted:
            # A bounded slice too small for the clone is a transient budget
            # wait, not a capability failure: park for a fresh invocation.
            raise IsolationUnavailable("verification_preparation_budget_exhausted")
        if result.returncode != 0 or result.killed:
            raise IsolationUnavailable(
                "private_git_preparation_failed: " + result.stderr[-1000:]
            )
    (destination / "config").write_text(
        "[core]\nrepositoryformatversion = 0\nbare = false\n"
    )
    shutil.rmtree(destination / "hooks", ignore_errors=True)
    if (destination / "objects/info/alternates").exists():
        raise ValueError("private Git metadata retained shared object authority")
def _git_wrapper(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    wrapper = directory / "git"
    wrapper.write_text(
        "#!/bin/sh\n"
        "# The private candidate metadata grant binds only to commands whose\n"
        "# working repository is the candidate tree. A command that explicitly\n"
        "# targets another repository (-C outside the candidate, or a new-repo\n"
        "# init/clone) must never see the read-only control metadata.\n"
        "target=\"$PWD\"\n"
        "subcommand=\"\"\n"
        "previous=\"\"\n"
        "for argument in \"$@\"; do\n"
        "  if [ \"$previous\" = \"-C\" ]; then target=\"$argument\"; fi\n"
        "  case \"$argument\" in\n"
        "    -*) ;;\n"
        "    *) if [ -z \"$subcommand\" ] && [ \"$previous\" != \"-C\" ]; then subcommand=\"$argument\"; fi ;;\n"
        "  esac\n"
        "  previous=\"$argument\"\n"
        "done\n"
        "case \"$subcommand\" in init|clone) target=\"/\";; esac\n"
        "case \"$target\" in /*) ;; *) target=\"$PWD/$target\";; esac\n"
        "case \"$target\" in /workspace/candidate|/workspace/candidate/*) ;; *) unset GIT_DIR GIT_WORK_TREE GIT_CONFIG_NOSYSTEM GIT_CONFIG_GLOBAL GIT_OPTIONAL_LOCKS;; esac\n"
        "exec /usr/bin/git \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
