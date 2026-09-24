"""Production CLI and domain adapter fail closed without a trusted host."""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.casefiles import case_values

from scripts import github_delivery_adapter as adapter
from slm_training.harness_core.activity_contract import contract_digest
from slm_training.harness_core.github_connector import ConnectorConfig, DeliveryWaiting
from tests.test_autoresearch.test_delivery_reconciliation import document_wait


@pytest.fixture
def subject(tmp_path, monkeypatch):
    from scripts.autotrain_docs import with_evidence_ledger
    from slm_training.autoresearch.action_dependencies import campaign_prerequisites

    store, handoff, wait, files = document_wait(tmp_path)
    # Append a current producer materialization; retain the old bundle/event.
    old = (
        store.root
        / "artifacts/delivery_documents"
        / (wait["artifact_sha256"] + ".json")
    )
    payload = json.loads(old.read_text())
    files = with_evidence_ledger(tmp_path, files)
    payload["files"] = files
    artifact = store.write_artifact("delivery_documents", payload)
    store.append_event("documentation_waiting_delivery", artifact_sha256=artifact.stem)
    _, waits = campaign_prerequisites(tmp_path, handoff)
    wait = waits[0]
    host = adapter.DeliveryHost(
        command=("/trusted/writer",),
        executable_sha256="a" * 64,
        repository="owner/repo",
        base_ref="b" * 40,
        source_digest="c" * 64,
        expires_at=time.time() + 300,
        authorized=True,
        required_checks=(),
        verification_plan={
            "source": str(tmp_path),
            "delivery_documents_sha256": {
                name: hashlib.sha256(value.encode()).hexdigest()
                for name, value in files.items()
            },
        },
    )
    config = adapter.AdapterConfig(
        host=host,
        transport=ConnectorConfig(),
        runtime_root=str(tmp_path / "loops/lab/runtime"),
        state_dir=str(tmp_path / "journal"),
        reviewers=("reviewer",),
        review_note_authors=("writer",),
    )
    key = "delivery:" + contract_digest(
        {
            "input": contract_digest(wait),
            "source": host.source_digest,
            "repository": host.repository,
            "base_ref": host.base_ref,
        }
    )
    request = {
        "schema_version": "connector_delivery_request/v1",
        "operation": "deliver_and_reconcile_squash_merge",
        "repository": host.repository,
        "base_ref": host.base_ref,
        "source_digest": host.source_digest,
        "grant_expires_at": host.expires_at,
        "runtime_root": config.runtime_root,
        "idempotency_key": key,
        "wait": wait,
        "lease": {"expires_at": time.time() + 150},
    }
    state = SimpleNamespace(spec=SimpleNamespace(input_digest=contract_digest(wait)))
    monkeypatch.setattr(adapter, "current_lease", lambda *args: state)
    from scripts import merge_verification_evidence
    from slm_training.harness_core import github_delivery_tree

    monkeypatch.setattr(merge_verification_evidence, "source_paths", lambda _: [])
    # Whole-tree comparisons have independent code/mode/link/deletion regressions.
    monkeypatch.setattr(github_delivery_tree, "document_tree", lambda *args: "d" * 40)
    return config, request, files


def test_materialized_binding_is_exact_and_has_verified_tree(subject):
    config, request, files = subject
    result = adapter.document_binding(config, request)
    assert result["files"] == files and result["candidate_git_tree"] == "d" * 40
    assert result["required_checks"] == []
    ledger = (
        "src/slm_training/resources/experiments/autotrain_climb/evidence_ledger.v1.json"
    )
    assert ledger in result["files"]
    assert config.host.verification_plan["delivery_documents_sha256"][ledger] == (
        hashlib.sha256(result["files"][ledger].encode()).hexdigest()
    )


@pytest.mark.parametrize("change", ["legacy", "extra_resource", "wrong_ledger"])
def test_only_canonical_derived_ledger_is_in_materialized_scope(
    subject, monkeypatch, change
):
    config, request, _ = subject
    parts = adapter._delivery_subject(
        Path(config.runtime_root).resolve().parents[2], request["wait"]
    )
    files = parts[3]["files"]
    ledger = (
        "src/slm_training/resources/experiments/autotrain_climb/evidence_ledger.v1.json"
    )
    if change == "legacy":
        files.pop(ledger)
    elif change == "extra_resource":
        files["src/slm_training/resources/versions.json"] = "{}\n"
    else:
        files[ledger] = "tampered\n"
    monkeypatch.setattr(adapter, "_delivery_subject", lambda *args: parts)
    pattern = (
        "legacy_document_ledger_rematerialization_required"
        if change == "legacy"
        else "authorized_scope|exact_document_successor"
    )
    with pytest.raises(ValueError, match=pattern):
        adapter.document_binding(config, request)


@pytest.mark.parametrize("change", ["extra", "missing", "content"])
def test_delivery_map_must_exactly_match_current_materialization(subject, change):
    config, request, files = subject
    scope = config.host.verification_plan["delivery_documents_sha256"]
    name = next(iter(files))
    if change == "extra":
        scope["docs/design/unrelated.md"] = "f" * 64
    elif change == "missing":
        scope.pop(name)
    else:
        scope[name] = "f" * 64
    with pytest.raises(DeliveryWaiting, match="exact_document_successor"):
        adapter.document_binding(config, request)


@pytest.mark.parametrize(
    "field",
    case_values(__file__, "test_request_identity_cannot_widen_host_grant"),
)
def test_request_identity_cannot_widen_host_grant(subject, field):
    config, request, _ = subject
    request[field] = "forged"
    with pytest.raises(ValueError, match="request_binding_mismatch"):
        adapter.document_binding(config, request)


def test_reader_restricts_tools_repository_and_binds_response(subject):
    config, _, _ = subject
    calls = []

    async def connector(tool, args):
        calls.append((tool, args))
        return {"structuredContent": {"merged": True}}

    request = {
        "schema_version": "connector_read_request/v1",
        "lease": {},
        "tool": "github_get_pr_info",
        "arguments": {"repository_full_name": "owner/repo", "pr_number": 7},
    }
    response = asyncio.run(adapter.read_request(config, request, connector))
    assert (
        response["request_digest"] == contract_digest(request)
        and response["read_only"] is True
    )
    request["tool"] = "github_merge_pull_request"
    with pytest.raises(ValueError, match="method_not_allowed"):
        asyncio.run(adapter.read_request(config, request, connector))
    request["tool"] = "github_get_pr_info"
    request["arguments"]["repository_full_name"] = "other/repo"
    with pytest.raises(ValueError, match="repository_mismatch"):
        asyncio.run(adapter.read_request(config, request, connector))
    assert len(calls) == 1


def test_cli_persists_honest_wait_without_endpoint(subject, tmp_path):
    config, request, _ = subject
    path, source, output = (
        tmp_path / name for name in ("host.json", "request.json", "result.json")
    )
    path.write_text(config.model_dump_json())
    source.write_text(json.dumps(request))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        adapter.main(
            [
                "--mode",
                "writer",
                "--config",
                str(path),
                "--config-sha256",
                digest,
                "--request",
                str(source),
                "--output",
                str(output),
            ]
        )
        == 10
    )
    result = json.loads(output.read_text())
    assert result["state"] == "waiting_capability"
    assert result["reason"] == "trusted_github_connector_endpoint_unavailable"
    assert result["request_digest"] == contract_digest(request)
    assert not (tmp_path / "journal").exists()
    with pytest.raises(ValueError, match="host_config_changed"):
        adapter.load_config(path, "0" * 64)


def test_real_lease_lookup_rejects_missing_authority(tmp_path):
    host = adapter.DeliveryHost(
        command=("/trusted/writer",),
        executable_sha256="a" * 64,
        repository="owner/repo",
        base_ref="b" * 40,
        source_digest="c" * 64,
        expires_at=time.time() + 300,
        authorized=True,
    )
    config = adapter.AdapterConfig(
        host=host,
        transport=ConnectorConfig(),
        runtime_root=str(tmp_path / "loops/lab/runtime"),
        state_dir=str(tmp_path / "journal"),
        reviewers=("reviewer",),
        review_note_authors=("writer",),
    )
    lease = {
        "activity_id": "absent",
        "attempt_id": "attempt",
        "epoch": "epoch",
        "generation": 1,
        "token": "token",
        "owner_identity": "owner",
        "expires_at": time.time() + 30,
    }
    with pytest.raises(ValueError, match="lease_not_current"):
        adapter.current_lease(config, {"lease": lease})


def test_cli_configuration_error_cannot_echo_secrets(tmp_path, capsys):
    path = tmp_path / "host.json"
    path.write_text('{"secret":"do-not-print-secret"}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        adapter.main(
            [
                "--mode",
                "writer",
                "--config",
                str(path),
                "--config-sha256",
                digest,
                "--request",
                str(tmp_path / "absent"),
                "--output",
                str(tmp_path / "result"),
            ]
        )
        == 20
    )
    captured = capsys.readouterr()
    assert "do-not-print-secret" not in captured.err and "Traceback" not in captured.err
    assert json.loads(captured.err)["reason"] == "connector_host_or_request_failed"
