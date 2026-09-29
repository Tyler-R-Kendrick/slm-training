"""Real runtime and pinned child processes: source receipt precedes success."""

import asyncio
import hashlib
import json
import sys
import time

import pytest

from scripts import github_source_delivery as source
from slm_training.autoresearch.runtime import operations_reconciliation
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.runtime.operations_delivery import DeliveryHost, consume_delivery
from slm_training.harness_core.activity_contract import ActivitySpec, contract_digest
from tests.test_scripts.test_github_source_delivery import accepted as accepted, ReadRemote
from tests.test_autoresearch import test_repair_delivered as activation_fixtures

publication = activation_fixtures.publication
accepted_source = activation_fixtures.accepted_source
accepted_execution = activation_fixtures.accepted_execution
reconciled = activation_fixtures.reconciled


@pytest.mark.parametrize("failure", [None, "tree", "receipt", "activation", "capability"])
def test_consumer_requires_independent_bound_source_proof_and_replays(
    accepted, tmp_path, monkeypatch, failure
):
    store, wait, configured, _ = accepted
    binding = source.source_binding(store, wait, configured)
    responses = {}
    remote = ReadRemote(binding)

    async def capture(tool, arguments):
        result = await remote(tool, arguments)
        responses[tool + json.dumps(arguments, sort_keys=True)] = result
        return result

    proposal = remote.proposal
    asyncio.run(source.verify_source_remote(capture, binding, proposal, ()))
    if failure == "tree":
        key = next(key for key in responses if "/git/trees/" in key)
        data = json.loads(responses[key]["structuredContent"]["content"])
        data["tree"][0]["mode"] = "100755"
        responses[key]["structuredContent"]["content"] = json.dumps(data)
    writer, reader = protocol_children(tmp_path, responses, proposal, failure)
    host = DeliveryHost(command=(str(writer),), executable_sha256=hashlib.sha256(writer.read_bytes()).hexdigest(),
        read_command=(str(reader),), reader_sha256=hashlib.sha256(reader.read_bytes()).hexdigest(),
        repository=configured.repository, base_ref=configured.base_ref, source_digest=configured.source_digest,
        expires_at=time.time() + 120, authorized=True, required_checks=(), verification_plan=configured.verification_plan)
    # Source gate internals have separate signed-cache tests; exercise real
    # runtime/child/reader receipt composition without a concurrent repo gate.
    monkeypatch.setattr(operations_reconciliation, "require_local_gate", lambda *args, **kwargs: None)
    from slm_training.autoresearch.heal import repair_delivered
    activations = []

    def activate(runtime, lease, given, reconciliation):
        assert runtime.snapshot()[lease.activity_id].status == "running"
        assert source.source_completion(runtime.store, given, reconciliation)["source_digest"] == host.source_digest
        if failure == "activation":
            raise ValueError("delivered_activation_failed")
        activations.append(lease.attempt_id)
        return {"publication_id": given["publication_id"], "attempt_id": lease.attempt_id}

    monkeypatch.setattr(repair_delivered, "record_delivered_activation", activate)
    with ActivityRuntime(store) as runtime:
        runtime.register(ActivitySpec(activity_id="source-delivery", family="lab", kind="delivery",
            source_digest=host.source_digest, environment_digest="a" * 64, input_digest=contract_digest(wait),
            output_namespace="attempts/source-delivery", capabilities=("local_process", "authorized_github_connector_delivery")))
        result = consume_delivery(runtime, "source-delivery", wait, host)
        if failure:
            assert result["state"] == ("waiting_capability" if failure == "capability" else "waiting_repair")
            if failure == "capability":
                assert result["reason"] == "trusted_github_connector_endpoint_unavailable"
            assert any(event["event_type"] == "verified_repair_source_delivered" for event in store.verify_event_chain()) == (failure == "activation")
        else:
            assert result["state"] == "succeeded"
            proof = source.source_completion(store, wait, result["reconciliation"])
            assert proof["publication_id"] == wait["publication_id"]
            assert proof["base_ref"] == host.base_ref and proof["merge_sha"] == proposal["merge_sha"]
            again = consume_delivery(runtime, "source-delivery", wait, host)
            assert again == result
            assert len(activations) == 1 and result["reconciliation"]["delivered_activation"]["publication_id"] == wait["publication_id"]
            assert runtime.snapshot()["source-delivery"].attempts == 1
            from pathlib import Path
            artifact = Path(result["reconciliation"]["verification_artifact"])
            artifact.write_text("{}")
            with pytest.raises(ValueError, match="completion_artifact_mismatch"):
                consume_delivery(runtime, "source-delivery", wait, host)


def protocol_children(tmp_path, responses, proposal, failure):
    preamble = f"#!{sys.executable}\n" + """import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');a=p.parse_args()
r=json.loads(Path(a.request).read_text())
digest=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
"""
    writer, reader = tmp_path / "writer", tmp_path / "reader"
    writer.write_text(preamble + f"proposal={proposal!r}\nfailure={failure!r}\n" + """out={
'schema_version':'connector_delivery_receipt/v1','request_digest':digest,'provider':'github_connector',
'repository':r['repository'],'source_digest':r['source_digest'],'merged':True,
'checks_verified':True,'reviews_resolved':True,'content_verified':True,**proposal}
if failure=='receipt':out['source_digest']='0'*64
if failure=='capability':out={'schema_version':'connector_delivery_wait/v1','request_digest':digest,
'provider':'github_connector','state':'waiting_capability','reason':'trusted_github_connector_endpoint_unavailable'}
Path(a.output).write_text(json.dumps(out))
if failure=='capability':raise SystemExit(10)
""")
    reader.write_text(preamble + f"responses={responses!r}\n" + """key=r['tool']+json.dumps(r['arguments'],sort_keys=True)
Path(a.output).write_text(json.dumps({'provider':'github_connector','request_digest':digest,
'read_only':True,'result':responses[key]}))
""")
    for path in (writer, reader):
        path.chmod(0o700)
    return writer, reader


def test_reconciliation_materializes_real_delivered_release_before_finish(reconciled, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from slm_training.autoresearch.heal import repair_delivered
    from slm_training.harness_core.execution_release import runtime_git_provenance

    runtime, lease, wait, existing, accepted = reconciled
    remote = existing["receipt"]["remote"]
    # Remote observations and the already-tested binding are fixtures. Receipt
    # publication, immutable materialization, lease fencing and activation are real.
    monkeypatch.setattr(source, "source_binding", lambda *args: dict(remote))

    async def observed(*args):
        return dict(remote)

    monkeypatch.setattr(source, "verify_source_remote", observed)
    host = SimpleNamespace(required_checks=())
    proof = asyncio.run(source.reconcile_source_delivery(runtime, lease, wait, {}, host, None))
    ref = proof["delivered_activation"]
    assert runtime.snapshot()[lease.activity_id].status == "running"
    with pytest.raises(ValueError, match="requires_committed_delivery"):
        repair_delivered.resolve_delivered_activation(runtime.store, ref)
    # Replay after a crash before finish reuses the same immutable materialization.
    assert asyncio.run(source.reconcile_source_delivery(runtime, lease, wait, {}, host, None)) == proof
    activation_fixtures._finish(runtime, lease, existing, ref)
    assert operations_reconciliation.validate_completion(runtime.store, wait, proof, completed=True) == proof
    handoff = repair_delivered.resolve_delivered_activation(runtime.store, ref)
    assert handoff["successor_execution"] != accepted["successor_execution"]
    assert runtime_git_provenance(Path(handoff["successor_execution"])) == {
        "integration_commit": remote["merge_sha"], "upstream_commit": remote["merge_sha"], "code_dirty": False}
    (Path(handoff["successor_execution"]) / "fixture.py").write_text("tampered\n")
    with pytest.raises(ValueError):
        operations_reconciliation.validate_completion(runtime.store, wait, proof, completed=True)
