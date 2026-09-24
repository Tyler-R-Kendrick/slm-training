"""Authenticated merge receipts must map to the exact executable release."""

import pytest
import os
import subprocess
import shutil
import hashlib
from types import SimpleNamespace
from scripts import merge_verification as verifier
from scripts.merge_verification_evidence import ReceiptCache, source_identity
from slm_training.harness_core.execution_release import prepare_release
from slm_training.autoresearch.runtime.operations_reconciliation import (
    require_local_gate,
)


def test_signed_isolated_gate_maps_exact_execution_release(tmp_path, monkeypatch):
    """Real collection/shard receipts, Git source hashing and release manifest."""
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True, timeout=10)
    (source / "tests").mkdir()
    test = source / "tests/test_case.py"
    test.write_text("def test_case():\n    assert 2 + 2 == 4\n")
    (source / ".gitignore").write_text("ignored.py\n")
    # Supply only fixture Git metadata: no commit/ref mutations. Source hashing,
    # test selection, isolated execution, workload proofs and MACs are real.
    monkeypatch.setattr(
        verifier, "changed_paths", lambda *_: ("a" * 40, ["tests/test_case.py"])
    )
    from scripts import merge_verification_isolation as boundary

    monkeypatch.setattr(
        boundary,
        "_private_git",
        lambda root, destination, *_: shutil.copytree(root / ".git", destination),
    )
    from slm_training.autoresearch.runtime.operations_verification import (
        verification_environment,
    )

    monkeypatch.setenv("PYTHONPATH", str(source / "src") + os.pathsep + str(source))
    captured = verification_environment()
    cache = tmp_path / "cache"
    if os.environ.get("SLM_REQUIRE_ISOLATION") == "1":
        from slm_training.autoresearch.heal.isolation import IsolationUnavailable

        with pytest.raises(IsolationUnavailable):
            verifier.run_release_gate(
                (),
                root=source,
                base_ref="a" * 40,
                state_dir=cache,
                step_seconds=40,
                run_step=None,
            )
        return  # The outer verifier forbids nested namespaces; fail closed.
    summary = verifier.run_release_gate(
        (),
        root=source,
        base_ref="a" * 40,
        state_dir=cache,
        step_seconds=40,
        run_step=None,
    )
    assert summary["verification_complete"], summary
    state = ReceiptCache(cache, source).load(summary["identity"])
    assert state["binding"]["isolation_enforced"] is True
    assert {row["kind"] for row in state["attempts"]} >= {"collection", "shard"}
    execution = tmp_path / "execution"
    manifest = prepare_release(
        source, tmp_path / "release", execution, tmp_path / "outputs"
    )
    assert source_identity(source) != manifest["source_digest"]
    host = SimpleNamespace(
        source_digest=manifest["source_digest"],
        base_ref="a" * 40,
        verification_plan={
            "source": str(source),
            "execution": str(execution),
            "state_dir": str(cache),
            "identity": summary["identity"],
            "environment": captured,
        },
    )
    monkeypatch.setenv(
        "PYTHONPATH", str(execution / "src") + os.pathsep + str(execution)
    )
    monkeypatch.setenv("SLM_CONTROLLER_ROLE", "delivery")
    monkeypatch.setenv("PATH", "/opt/service-bin:" + os.environ["PATH"])
    require_local_gate(host)
    _reject_dependency_environment_drift(host, monkeypatch, tmp_path)
    # Signed evidence cannot authorize changed authoring or execution contents.
    test.write_text("def test_case():\n    assert False\n")
    with pytest.raises(ValueError, match="incomplete_or_changed"):
        require_local_gate(host)
    test.write_text("def test_case():\n    assert 2 + 2 == 4\n")
    (execution / "tests/test_case.py").write_text("tampered")
    with pytest.raises(ValueError, match="execution_source_drift"):
        require_local_gate(host)
    # A Git-ignored file is invisible to the signed candidate snapshot but
    # copied by prepare_release: reject this otherwise matching second release.
    (source / "ignored.py").write_text("unverified = True\n")
    second = tmp_path / "execution2"
    manifest = prepare_release(
        source, tmp_path / "release2", second, tmp_path / "outputs2"
    )
    host.source_digest = manifest["source_digest"]
    host.verification_plan["execution"] = str(second)
    monkeypatch.setenv("PYTHONPATH", str(second / "src") + os.pathsep + str(second))
    with pytest.raises(ValueError, match="execution_manifest_mismatch"):
        require_local_gate(host)


def _reject_dependency_environment_drift(host, monkeypatch, root):
    for key in ("PYTHONHOME", "PYTHONUSERBASE"):
        with monkeypatch.context() as changed:
            changed.setenv(key, str(root / "unapproved-python"))
            with pytest.raises(
                ValueError, match="authenticated_local_merge_gate_rejected"
            ):
                require_local_gate(host)


def test_document_successor_mapping_preserves_all_executable_source(tmp_path):
    from slm_training.autoresearch.runtime.operations_verification import _require_execution_mapping
    from slm_training.harness_core.execution_release import document_successor_manifest

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True, timeout=10)
    code = source / "code.py"
    code.write_text("value = 1\n")
    execution = tmp_path / "execution"
    manifest = prepare_release(source, tmp_path / "release", execution, tmp_path / "outputs")
    document = source / "docs/design/campaign-results.md"
    document.parent.mkdir(parents=True)
    document.write_text("# Measured result\n")
    documents = {"docs/design/campaign-results.md": hashlib.sha256(document.read_bytes()).hexdigest()}
    host = SimpleNamespace(source_digest=manifest["source_digest"], verification_plan={
        "execution": str(execution), "delivery_documents_sha256": documents,
    })
    # This checks only source mapping; the caller independently requires a
    # complete authenticated gate over the successor, including this document.
    _require_execution_mapping(host, source, {})
    document.write_text("unverified document change\n")
    with pytest.raises(ValueError, match="manifest_mismatch"):
        _require_execution_mapping(host, source, {})
    document.write_text("# Measured result\n")
    code.write_text("value = 2\n")
    with pytest.raises(ValueError, match="manifest_mismatch"):
        _require_execution_mapping(host, source, {})
    for name in ("code.py", "docs/design/../../code.py", "docs/design/helper.py"):
        with pytest.raises(ValueError, match="invalid_delivery_document_manifest"):
            document_successor_manifest(manifest["files"], {name: "a" * 64})


def test_ordinary_checkout_requires_exact_verified_documents(tmp_path):
    from slm_training.autoresearch.runtime.operations_verification import _require_execution_mapping

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, timeout=10)
    readme = tmp_path / "README.md"
    readme.write_text("verified bytes\n")
    digest = source_identity(tmp_path)
    host = SimpleNamespace(source_digest=digest, verification_plan={
        "execution": str(tmp_path), "delivery_documents_sha256": {"README.md": "a" * 64},
    })
    with pytest.raises(ValueError, match="local_gate_execution_source_mismatch"):
        _require_execution_mapping(host, tmp_path, {"candidate_tree_sha256": digest})
    host.verification_plan["delivery_documents_sha256"]["README.md"] = hashlib.sha256(readme.read_bytes()).hexdigest()
    _require_execution_mapping(host, tmp_path, {"candidate_tree_sha256": digest})
    host.verification_plan["delivery_documents_sha256"] = {"source.py": "a" * 64}
    with pytest.raises(ValueError, match="invalid_delivery_document_manifest"):
        _require_execution_mapping(host, tmp_path, {"candidate_tree_sha256": digest})
