"""Nested controller CLI shares its live lease; scientific children do not."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from io import StringIO
from pathlib import Path
import sys
import os
from types import SimpleNamespace

_REQUEST = ContextVar("nested_controller_request", default=None)


@contextmanager
def execution_scope(request):
    token = _REQUEST.set(request)
    try:
        yield
    finally:
        _REQUEST.reset(token)


def current_controller():
    """Read declaration only; each consumer validates at its execution boundary."""
    from scripts.autotrain_controller_execution import parent_controller

    request = _REQUEST.get()
    if request is None:
        return parent_controller()
    return request.get("controller_execution") or (request.get("controller_launch") or {}).get("controller_execution")


def workload_cwd(store, default):
    from scripts.autotrain_cycle_lock import load_context
    from scripts.autotrain_execution_transition import verify_locked_execution

    request = _REQUEST.get()
    if not request or current_controller() is None:
        return default
    cwd = Path(request["cwd"]).resolve()
    if cwd != Path.cwd().resolve():
        raise ValueError("nested_workload_cwd_changed")
    value = load_context(store)
    if not request.get("logical_continuation"):
        from scripts.autotrain_controller_execution import operation_controller
        from scripts.autoresearch_continuation_identity import resolved_continuation_grant

        operation_controller(request)
        actual = resolved_continuation_grant(cwd, value["total_seconds"])
        if (value["cwd"] != str(cwd) or actual.execution_identity != value["execution_identity"]
                or value["loop_id"] != request["loop_id"]
                or store.root.parent.resolve() != Path(request["root"]).resolve()):
            raise ValueError("controller_upgrade_changed_locked_workload")
        return cwd
    checked = verify_locked_execution(store, cwd, value)
    if any(checked.get(key) != value for key, value in request.items()
           if key not in {"lease", "parent_event", "controller_launch"}):
        raise ValueError("nested_operation_request_changed")
    return cwd


def logical_source(store):
    """Original proof remains the scientific evidence subject, not the controller."""
    from scripts.autotrain_cycle_lock import load_context

    request = _REQUEST.get()
    if not request or current_controller() is None:
        return None
    workload_cwd(store, None)
    return Path(load_context(store)["cwd"])


def run_nested(cmd, *, store, cwd, cli):
    """Execute only the locked canonical run CLI in the delegated worker itself."""
    request = _REQUEST.get()
    if not request or current_controller() is None:
        return None
    argv = _nested_argv(cmd, store)
    if workload_cwd(store, None) != Path(cwd).resolve():
        raise ValueError("nested_execution_workload_changed")
    controller = Path(current_controller()["cwd"]).resolve()
    if Path(cli.__file__).resolve() != controller / "scripts/autoresearch.py":
        raise ValueError("nested_execution_controller_import_changed")
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(argv)
    return SimpleNamespace(returncode=code, stdout=out.getvalue(),
                           stderr=err.getvalue(), timed_out=False)


def validate_logical_source(store, campaign, diagnostic_receipt, validator, *, default):
    from slm_training.harness_core.execution_release import runtime_git_provenance
    from scripts.autotrain_cycle_lock import load_context

    cwd = workload_cwd(store, default)
    if not _REQUEST.get() or current_controller() is None:
        validator(campaign, diagnostic_receipt)
        return cwd
    proof = runtime_git_provenance(Path(load_context(store)["cwd"]))
    if (not proof or proof["upstream_commit"] != campaign.upstream_commit
            or proof["integration_commit"] != campaign.integration_commit
            or (proof["code_dirty"] and diagnostic_receipt is None)):
        raise ValueError("nested_original_source_proof_changed")
    return cwd


def startup_source(cwd):
    """Read original commit proof only after checking this operation's migration."""
    from slm_training.autoresearch.storage import CampaignStore
    from scripts.autotrain_cycle_lock import active_reference

    request = _REQUEST.get()
    if not request or not request.get("logical_continuation"):
        return Path(cwd)
    root = Path(request["root"])
    journal = CampaignStore("runtime", root / "loops" / request["loop_id"])
    reference = active_reference(journal)
    if reference is None:
        raise ValueError("nested_startup_requires_original_driver_lock")
    return logical_source(CampaignStore(reference["campaign_id"], root))


def _nested_argv(cmd, store):
    if (len(cmd) < 6 or cmd[1:4] != ["-m", "scripts.autoresearch", "--root"]
            or cmd[5] != "run" or Path(cmd[4]).resolve() != store.root.parent.resolve()):
        raise ValueError("nested_execution_requires_canonical_run_cli")
    if os.path.abspath(cmd[0]) != os.path.abspath(sys.executable):
        raise ValueError("nested_controller_interpreter_differs_from_locked_driver")
    return cmd[3:]


def diagnostic_provenance(store, stamp):
    """Stamp diagnostic controller code separately from locked measurement proof."""
    from slm_training.harness_core.execution_release import runtime_git_provenance
    from scripts.autotrain_cycle_lock import load_context

    source = logical_source(store)
    if source is None:
        return {"version_stamp": stamp}
    request = _REQUEST.get()
    binding = request.get("controller_execution") or request["controller_launch"]["controller_execution"]
    proof = runtime_git_provenance(Path(binding["cwd"]))
    if proof is None:
        raise ValueError("diagnostic_controller_provenance_missing")
    controller_stamp = {k: v for k, v in stamp.items() if k != "source_authority"}
    controller_stamp.update(code_commit=proof["integration_commit"], code_dirty=proof["code_dirty"])
    return {"version_stamp": controller_stamp, "controller_execution": binding,
            "measurement_source": load_context(store)["publication_source"]}
