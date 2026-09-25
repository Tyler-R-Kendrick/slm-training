"""Real OS boundary tests plus fail-closed capability and scope regressions."""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from slm_training.autoresearch.heal.isolation import (
    IsolationSpec,
    IsolationUnavailable,
    build_isolated_command,
    probe_isolation,
    run_isolated,
)
from slm_training.autoresearch.heal import isolation as isolation_module
from slm_training.autoresearch.heal.isolation_workspace import (
    IsolationViolation,
    private_snapshot,
    private_snapshot_with_disposable_dirs,
    scope_changes,
    tree_manifest,
)
from slm_training.levers import INTERRUPT_AFTER_SECONDS


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "private"
    root.mkdir()
    (root / "editable.py").write_text("old\n")
    (root / "protected.py").write_text("already dirty\n")
    return root


@pytest.fixture
def real_isolation() -> None:
    capability = probe_isolation()
    if not capability.available:
        if os.environ.get("SLM_REQUIRE_ISOLATION") == "1":
            pytest.fail(capability.reason)
        pytest.skip(f"real isolation unavailable: {capability.reason}")


def test_repeated_edit_of_already_dirty_path_is_a_scope_violation(
    workspace: Path,
) -> None:
    before = tree_manifest(workspace)
    (workspace / "protected.py").write_text("second mutation\n")
    assert scope_changes(before, tree_manifest(workspace), ("editable.py",)) == (
        "protected.py",
    )


def test_mode_and_deletion_are_checked(workspace: Path) -> None:
    before = tree_manifest(workspace)
    (workspace / "protected.py").chmod(0o700)
    (workspace / "editable.py").unlink()
    assert scope_changes(before, tree_manifest(workspace), ()) == (
        "editable.py",
        "protected.py",
    )


@pytest.mark.parametrize(
    "relative", ["../private", "/tmp", "./editable.py", "editable.py/../protected.py"]
)
def test_noncanonical_mounts_rejected(workspace: Path, relative: str) -> None:
    with pytest.raises(IsolationViolation):
        build_isolated_command(
            IsolationSpec(workspace, (relative,)), ("/bin/true",), "/usr/bin/bwrap"
        )


def test_symlink_mount_alias_rejected(workspace: Path) -> None:
    (workspace / "alias").symlink_to("protected.py")
    with pytest.raises(IsolationViolation, match="symlink"):
        build_isolated_command(
            IsolationSpec(workspace, ("alias",)), ("/bin/true",), "bwrap"
        )


def test_external_symlink_and_hardlink_rejected(workspace: Path) -> None:
    alias = workspace / "alias"
    alias.symlink_to("/etc/passwd")
    with pytest.raises(IsolationViolation, match="external symlink"):
        tree_manifest(workspace)
    alias.unlink()
    os.link(workspace / "protected.py", alias)
    with pytest.raises(IsolationViolation, match="hardlink"):
        tree_manifest(workspace)


def test_runtime_descendant_hardlink_rejected_even_after_prior_validation(tmp_path):
    runtime = tmp_path / "runtime"
    nested = runtime / "lib"
    nested.mkdir(parents=True)
    (nested / "module.py").write_text("safe runtime\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = IsolationSpec(workspace, runtime_roots=(runtime,))

    isolation_module._runtime_mounts(spec)
    os.link(nested / "module.py", tmp_path / "outside.py")
    with pytest.raises(IsolationViolation, match="hardlinked runtime file"):
        isolation_module._runtime_mounts(spec)


def test_node_modules_runtime_mount_targets_matching_immutable_candidate(tmp_path):
    from slm_training.autoresearch.heal.isolation import _node_module_mounts

    runtime = tmp_path / "runtime/src/apps/openui_bridge/node_modules"
    runtime.mkdir(parents=True)
    (runtime.parent / "package.json").write_text('{"name":"openui-bridge"}')
    candidate_app = tmp_path / "workspace/candidate/src/apps/openui_bridge"
    candidate_app.mkdir(parents=True)
    (candidate_app / "package.json").write_text('{"name":"openui-bridge"}')

    mounts = _node_module_mounts(tmp_path / "workspace", (runtime,))

    assert mounts == [
        "--ro-bind",
        str(runtime),
        "/workspace/candidate/src/apps/openui_bridge/node_modules",
    ]
    assert (candidate_app / "node_modules").is_dir()


def test_socket_mount_rejected(workspace: Path) -> None:
    with socket.socket(socket.AF_UNIX) as server:
        try:
            server.bind(str(workspace / "host.sock"))
        except PermissionError:
            pytest.skip("host denies pathname AF_UNIX fixtures")
        with pytest.raises(IsolationViolation, match="special file"):
            tree_manifest(workspace)


def test_live_git_metadata_refused(workspace: Path) -> None:
    (workspace / ".git").write_text("gitdir: /some/human/worktree")
    with pytest.raises(IsolationViolation, match="Git metadata"):
        build_isolated_command(IsolationSpec(workspace), ("/bin/true",), "bwrap")


def test_private_snapshot_does_not_copy_shared_metadata(
    workspace: Path, tmp_path: Path
) -> None:
    (workspace / ".git").write_text("gitdir: /some/human/worktree")
    copied = private_snapshot(workspace, tmp_path / "copy")
    assert not (copied / ".git").exists()
    assert (copied / "editable.py").stat().st_ino != (
        workspace / "editable.py"
    ).stat().st_ino


def test_snapshot_destination_alias_cannot_recurse_into_source(workspace, tmp_path):
    (tmp_path / "alias").symlink_to(workspace, target_is_directory=True)
    with pytest.raises(IsolationViolation, match="overlaps"):
        private_snapshot(workspace, tmp_path / "alias/copy")
    assert not (workspace / "copy").exists()


def test_filtered_snapshot_keeps_only_bound_files_and_empty_mounts(workspace, tmp_path):
    (workspace / "src").mkdir()
    (workspace / "src/bound.py").write_text("bound")
    (workspace / "src/unbound.py").write_text("unbound")
    copied = private_snapshot_with_disposable_dirs(
        workspace, tmp_path / "copy", ("node_modules",), ("src/bound.py",)
    )
    assert set(tree_manifest(copied)) == {"src", "src/bound.py", "node_modules"}
    assert (copied / "src/bound.py").read_text() == "bound"


@pytest.mark.parametrize("paths", [("../escape",), ("/tmp/escape",), ("./alias",)])
def test_filtered_snapshot_rejects_traversal_before_copy(workspace, tmp_path, paths):
    with pytest.raises(IsolationViolation, match="noncanonical"):
        private_snapshot_with_disposable_dirs(workspace, tmp_path / "copy", paths, ())
    assert not (tmp_path / "copy").exists()


def test_disposable_mount_cannot_follow_source_symlink(workspace, tmp_path):
    (workspace / "real").mkdir()
    (workspace / "alias").symlink_to("real", target_is_directory=True)
    with pytest.raises(IsolationViolation, match="symlink"):
        private_snapshot_with_disposable_dirs(
            workspace, tmp_path / "copy", ("alias/mount",), ("editable.py",)
        )
    assert not (tmp_path / "copy/real/mount").exists()


@pytest.mark.parametrize(
    "budget", [float("nan"), float("inf"), 0, INTERRUPT_AFTER_SECONDS + 1]
)
def test_cap_is_not_optional(workspace: Path, budget: float) -> None:
    with pytest.raises(IsolationViolation, match="timeout"):
        build_isolated_command(
            IsolationSpec(workspace, timeout_seconds=budget), ("true",), "bwrap"
        )


def test_missing_bwrap_never_executes_unsandboxed(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "slm_training.autoresearch.heal.isolation.shutil.which", lambda _: None
    )
    with pytest.raises(IsolationUnavailable, match="bwrap_not_installed"):
        run_isolated(IsolationSpec(workspace), ("/bin/touch", "protected.py"))
    assert (workspace / "protected.py").read_text() == "already dirty\n"


def test_real_denied_access_and_exact_writable_mount(
    workspace: Path, real_isolation: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ISOLATION_HOST_SECRET", "must-not-inherit")
    script = (
        "from pathlib import Path; import os; "
        "assert 'ISOLATION_HOST_SECRET' not in os.environ; "
        "assert not Path('/home/codex').exists(); "
        "assert not Path('/run').exists(); "
        "assert not Path('/workspace/.git').exists(); "
        "Path('editable.py').write_text('repaired'); "
        "\ntry: Path('protected.py').write_text('bad')\n"
        "except OSError: pass\n"
        "else: raise AssertionError('protected write succeeded')\n"
        "print('boundary-observed')"
    )
    result = run_isolated(
        IsolationSpec(workspace, ("editable.py",)), ("/usr/bin/python3", "-c", script)
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "boundary-observed\n"
    assert (workspace / "protected.py").read_text() == "already dirty\n"
    assert (workspace / "editable.py").read_text() == "repaired"


def test_real_failure_cannot_mutate_protected_file(
    workspace: Path, real_isolation: None
) -> None:
    result = run_isolated(
        IsolationSpec(workspace), ("/bin/sh", "-c", "echo corrupt > protected.py")
    )
    assert result.returncode != 0
    assert (workspace / "protected.py").read_text() == "already dirty\n"


def test_real_host_socket_and_result_channel_are_inaccessible(
    workspace: Path,
    real_isolation: None,
    tmp_path: Path,
) -> None:
    controller = tmp_path / "controller"
    controller.mkdir()
    receipt = controller / "receipt.json"
    receipt.write_text("controller-owned")
    with socket.socket(socket.AF_UNIX) as server:
        # Linux sockaddr_un has a 108-byte path limit. Keep the actual host
        # socket in the same fixture directory but address it through a short
        # directory FD path; the sandbox cannot resolve the host process FD.
        directory_fd = os.open(controller, os.O_RDONLY | os.O_DIRECTORY)
        address = f"/proc/{os.getpid()}/fd/{directory_fd}/control.sock"
        try:
            server.bind(address)
        except PermissionError:
            os.close(directory_fd)
            pytest.skip("host denies pathname AF_UNIX fixtures")
        server.listen()
        script = (
            "import socket,sys; from pathlib import Path; "
            "assert not Path(sys.argv[1]).exists(); "
            "assert not Path(sys.argv[2]).exists(); "
            "s=socket.socket(socket.AF_UNIX); "
            "\ntry: s.connect(sys.argv[1])\n"
            "except OSError: pass\nelse: raise AssertionError('socket exposed')\n"
            "try: Path(sys.argv[2]).write_text('forged')\n"
            "except OSError: pass\nelse: raise AssertionError('receipt writable')\n"
            "print('controller-inaccessible')"
        )
        result = run_isolated(
            IsolationSpec(workspace),
            ("/usr/bin/python3", "-c", script, address, str(receipt)),
        )
        os.close(directory_fd)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "controller-inaccessible\n"
    assert receipt.read_text() == "controller-owned"


def test_real_file_limit_is_hard(workspace: Path, real_isolation: None) -> None:
    script = (
        "import resource; from pathlib import Path; "
        "assert resource.getrlimit(resource.RLIMIT_FSIZE) == (1024,1024); "
        "Path('editable.py').write_bytes(b'x'*2048)"
    )
    result = run_isolated(
        IsolationSpec(workspace, ("editable.py",), scratch_bytes=1024),
        ("/usr/bin/python3", "-c", script),
    )
    assert result.returncode != 0
    assert (workspace / "editable.py").stat().st_size <= 1024


def test_real_hung_descendants_terminated(
    workspace: Path, real_isolation: None
) -> None:
    import uuid

    marker = "isolation-child-" + uuid.uuid4().hex
    script = (
        "import os,signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); "
        "os.fork(); time.sleep(60)"
    )
    result = run_isolated(
        IsolationSpec(workspace, timeout_seconds=0.1),
        ("/usr/bin/python3", "-c", script, marker),
    )
    assert result.timed_out and result.outcome != "completed"
    assert result.duration_seconds < 15
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        assert marker.encode() not in command, f"owned descendant survived: {path}"
