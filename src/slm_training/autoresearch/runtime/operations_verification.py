"""Bounded authentication of canonical gate evidence and execution mapping."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from slm_training.harness_core.bounded_process import (
    run_bounded_process,
    ProcessOutcome,
)
from slm_training.harness_core.execution_environment import verification_environment as verification_environment
from slm_training.levers import KILL_GRACE_SECONDS


def require_local_gate(host, *, seconds=30.0):
    """Hash source/dependencies in a capped trusted child, never the supervisor."""
    plan = host.verification_plan
    if not plan:
        raise ValueError("authenticated_local_merge_gate_missing")
    plan = {"execution": str(Path.cwd()), **plan}
    env = dict(os.environ)
    env.update(verification_environment())  # Absolute PYTHONPATH before chdir.
    owner = Path(__file__).resolve().parents[4]
    code = (
        "import sys; sys.path[:0]="
        + repr([str(owner / "src"), str(owner)])
        + "; import json; from pathlib import Path; from types import SimpleNamespace; "
        "from slm_training.autoresearch.runtime.operations_verification import _validate_local_gate; "
        "_validate_local_gate(SimpleNamespace(**json.loads(Path(sys.argv[1]).read_text())))"
    )
    with tempfile.TemporaryDirectory(prefix="slm-delivery-proof-") as temporary:
        request = Path(temporary) / "request.json"
        request.write_text(
            json.dumps(
                {
                    "verification_plan": plan,
                    "base_ref": getattr(host, "base_ref", ""),
                    "source_digest": getattr(host, "source_digest", ""),
                }
            )
        )
        request.chmod(0o600)
        result = run_bounded_process(
            [sys.executable, "-c", code, str(request)],
            cwd=plan["source"],
            env=env,
            interrupt_after_seconds=min(30.0, seconds),
            kill_grace_seconds=KILL_GRACE_SECONDS,
        )
    if result.outcome != ProcessOutcome.COMPLETED or result.returncode != 0:
        raise ValueError(
            "authenticated_local_merge_gate_rejected: " + result.stderr[-1500:]
        )


def _validate_local_gate(host):
    """Authenticate canonical isolated evidence, independent of hosted CI policy."""
    from scripts.merge_verification import _summary
    from scripts.merge_verification_evidence import ReceiptCache, validate_cached_state

    plan = host.verification_plan
    if not plan:
        raise ValueError("authenticated_local_merge_gate_missing")
    source = Path(plan["source"])
    if not (Path(plan["state_dir"]) / "issuer.key").is_file():
        raise ValueError("authenticated_local_merge_gate_missing")
    state = ReceiptCache(Path(plan["state_dir"]), source).load(plan["identity"])
    if state is None:
        raise ValueError("authenticated_local_merge_gate_missing")
    validate_cached_state(state, state["binding"])
    if (
        state["binding"].get("base_ref")
        not in {host.base_ref, host.base_ref + "^{commit}"}
        or not state["binding"].get("isolation_enforced")
        or not _summary(state)["verification_complete"]
        or _verification_context_changed(host, state["binding"], source)
    ):
        raise ValueError("authenticated_local_merge_gate_incomplete_or_changed")
    _require_execution_mapping(host, source, state["binding"])


def _require_execution_mapping(host, source, binding):
    """Map a Git source proof into the release's normalized file manifest."""
    from scripts.merge_verification_evidence import source_paths
    from slm_training.harness_core.execution_release import (
        MARKER,
        _files,
        document_successor_manifest,
        runtime_source_identity,
    )

    execution = Path(host.verification_plan.get("execution", Path.cwd())).resolve()
    runtime_digest = runtime_source_identity(execution)
    documents = host.verification_plan.get("delivery_documents_sha256", {})
    if runtime_digest is None:
        actual = _files(source)
        if (
            execution != source.resolve()
            or binding["candidate_tree_sha256"] != host.source_digest
            or document_successor_manifest(actual, documents) != actual
            or not set(actual) <= set(source_paths(source))
        ):
            raise ValueError("local_gate_execution_source_mismatch")
        return
    manifest = json.loads((execution / MARKER).read_text())
    expected = document_successor_manifest(
        manifest["files"], documents
    )
    if (
        runtime_digest != host.source_digest
        or _files(source) != expected
        or not set(expected) <= set(source_paths(source))
    ):
        raise ValueError("local_gate_execution_manifest_mismatch")


def _verification_context_changed(host, binding, source):
    """Controller delivery is not a claim of scientific environment equivalence."""
    from scripts.merge_verification_evidence import (
        digest,
        environment_identity,
        runtime_identity,
        source_identity,
    )

    if (
        source_identity(source) != binding["candidate_tree_sha256"]
        or runtime_identity(tuple(Path(p) for p in binding["runtime_roots"]))
        != binding["runtime_identity"]
    ):
        return True
    recorded = dict(binding["environment"])
    current = environment_identity()
    expected_env = recorded.pop("execution_environment_sha256")
    actual_env = current.pop("execution_environment_sha256")
    execution = Path(host.verification_plan.get("execution", Path.cwd())).resolve()

    def relocate(path):
        value = Path(path)
        return (
            str(source.resolve() / value.relative_to(execution))
            if value.is_relative_to(execution)
            else path
        )

    current["command_files"] = {
        key: [relocate(value[0]), value[1]] if value else None
        for key, value in current["command_files"].items()
    }
    if current != recorded:
        return True  # Python, packages, approved commands and platform still bind.
    if actual_env == expected_env:
        return False
    captured = host.verification_plan.get("environment")
    if not isinstance(captured, dict) or digest(captured) != expected_env:
        return True
    actual = verification_environment()
    actual["PYTHONPATH"] = os.pathsep.join(
        relocate(p) for p in actual.get("PYTHONPATH", "").split(os.pathsep)
    )
    # Only named controller-role changes are permitted. Python dependency paths,
    # import flags, thread settings and every other captured variable still bind.
    allowed = {"PATH", "SLM_CONTROLLER_ROLE", "PYTHONDONTWRITEBYTECODE"}
    return any(
        actual.get(key) != captured.get(key)
        for key in (actual.keys() | captured.keys()) - allowed
    )
