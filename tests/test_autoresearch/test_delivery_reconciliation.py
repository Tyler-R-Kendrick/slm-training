"""Fresh connector reads, not worker claims, discharge document prerequisites."""

import asyncio
import hashlib

import pytest

from tests.casefiles import case_values
from tests.test_harness_core.test_source_authority import initial as initial, _claim, _finish

from slm_training.autoresearch.action_dependencies import campaign_prerequisites
from slm_training.autoresearch.runtime.operations_reconciliation import (
    reconcile_document_delivery,
)
from slm_training.autoresearch.schemas import AutotrainActionV1, AutotrainCycleHandoffV1
from slm_training.autoresearch.storage import (
    CampaignStore,
    bind_autotrain_action_evidence,
)


def document_wait(tmp_path):
    action = AutotrainActionV1(
        kind="document",
        owner="documenting-experiment-results",
        reason="publish retained measurement",
        evidence_ids=("fixture",),
        dependency_scope="delivery",
    )
    handoff = AutotrainCycleHandoffV1(
        loop_id="lab",
        campaign_id="campaign",
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        cycle_role="screening",
        cycle_intent="integration fixture",
        evidence_class="fixture",
        climb_state="inconclusive",
        ship_state="blocked",
        primary_metric="eval_nll",
        actions=(action,),
    )
    store = CampaignStore("campaign", tmp_path)
    store.root.mkdir(parents=True)
    path = store.root / "cycle_handoff.json"
    path.write_text(handoff.model_dump_json())
    files = {
        "docs/design/campaign-results.md": "# fixture\n",
        "docs/design/campaign-results.json": "{}\n",
    }
    materialization = store.write_artifact(
        "delivery_documents",
        {
            "schema": "autotrain_document_materialization/v1",
            "campaign_id": "campaign",
            "handoff_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "files": files,
        },
    )
    store.append_event(
        "documentation_waiting_delivery", artifact_sha256=materialization.stem
    )
    _, waits = campaign_prerequisites(tmp_path, handoff)
    return store, handoff, waits[0], files


def connector_fixture(files, failure=None):
    async def read(tool, args):
        if tool == "github_get_pr_info":
            data = {
                "merged": failure != "merge",
                "base": "main",
                "head_sha": "c" * 40,
                "merge_commit_sha": "d" * 40,
            }
        elif tool == "github_get_commit_combined_status":
            assert args["commit_sha"] == "c" * 40
            data = {
                "statuses": [
                    {
                        "context": "required",
                        "state": "failure" if failure == "checks" else "success",
                    }
                ]
            }
        elif tool == "github_list_pull_request_review_threads":
            data = {
                "review_threads": [{"isResolved": False}]
                if failure == "threads"
                else []
            }
        elif tool == "github_list_pull_request_reviews":
            data = {"reviews": []}
        elif tool == "github_compare_commits":
            data = {"status": "ahead", "merge_base_commit": {"sha": "d" * 40}}
        elif tool == "github_fetch_file":
            assert args["ref"] == "d" * 40 and "start_line" not in args
            data = {
                "content": "wrong" if failure == "content" else files[args["path"]],
                "encoding": "utf-8",
            }
        else:
            raise AssertionError(tool)
        return {"structuredContent": data}

    return read


def reconcile(root, wait, read):
    return asyncio.run(
        reconcile_document_delivery(
            root,
            wait,
            {
                "pr_number": 12,
                "verified_head_sha": "c" * 40,
                "merge_sha": "d" * 40,
                "merged": True,
            },
            repository="owner/repo",
            required_checks=("required",),
            connector=read,
        )
    )


def test_remote_exact_content_receipt_closes_wait_idempotently(tmp_path, initial):
    store, handoff, wait, files = document_wait(tmp_path)
    _publish_document_source(store, initial)
    result = reconcile(tmp_path, wait, connector_fixture(files))
    assert campaign_prerequisites(tmp_path, handoff) == ((), [])
    again = reconcile(tmp_path, wait, connector_fixture(files))
    assert result["verification_artifact"] == again["verification_artifact"]
    assert (
        len((tmp_path / "loops/lab/action_receipts.jsonl").read_text().splitlines())
        == 1
    )
    # Tampering reopens the dependency instead of trusting the remembered ack.
    artifact = next(
        (store.root / "artifacts/connector_document_verifications").glob("*.json")
    )
    artifact.write_text("{}")
    assert len(campaign_prerequisites(tmp_path, handoff)[1]) == 1


@pytest.mark.parametrize("failure", ["merge", "checks", "threads", "content"])
def test_worker_success_cannot_discharge_failed_remote_evidence(tmp_path, failure, initial):
    store, handoff, wait, files = document_wait(tmp_path)
    _publish_document_source(store, initial)
    with pytest.raises(ValueError, match="remote_"):
        reconcile(tmp_path, wait, connector_fixture(files, failure))
    assert not (tmp_path / "loops/lab/action_receipts.jsonl").exists()
    assert len(campaign_prerequisites(tmp_path, handoff)[1]) == 1


def test_no_connector_and_unverified_artifacts_cannot_acknowledge(tmp_path, initial):
    store, handoff, wait, _ = document_wait(tmp_path)
    _publish_document_source(store, initial)
    with pytest.raises(ValueError, match="read_capability_unavailable"):
        reconcile(tmp_path, wait, None)
    artifact = store.write_artifact(
        "connector_document_verifications", {"merged": True}
    )
    with pytest.raises(ValueError, match="unverified_connector"):
        bind_autotrain_action_evidence(
            tmp_path,
            handoff,
            handoff.actions[0],
            (str(artifact.relative_to(store.root)),),
        )


@pytest.mark.parametrize("failure", [None, "content", "local_gate"])
def test_configured_supervisor_consumes_reader_before_success(
    tmp_path, monkeypatch, failure, initial
):
    import sys
    import time
    from scripts.autotrain_supervision import register_delivery_waits
    from slm_training.autoresearch.runtime import operations_reconciliation as owner
    from slm_training.autoresearch.runtime.operations_control import DeliveryHost
    from slm_training.autoresearch.runtime.operations_status import runtime_store
    from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime

    store, handoff, wait, files = document_wait(tmp_path)
    _publish_document_source(store, initial)
    preamble = (
        f"#!{sys.executable}\n"
        + """import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');a=p.parse_args()
r=json.loads(Path(a.request).read_text())
digest=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
"""
    )
    writer = tmp_path / "writer-fixture"
    writer.write_text(
        preamble
        + """out={'schema_version':'connector_delivery_receipt/v1','request_digest':digest,
'provider':'github_connector','repository':r['repository'],'source_digest':r['source_digest'],
'pr_number':12,'merged':True,'checks_verified':True,'reviews_resolved':True,'content_verified':True,
'merge_sha':'d'*40,'verified_head_sha':'c'*40}
Path(a.output).write_text(json.dumps(out))
"""
    )
    responses = {
        "github_get_pr_info": {
            "merged": True,
            "base": "main",
            "head_sha": "c" * 40,
            "merge_commit_sha": "d" * 40,
        },
        "github_get_commit_combined_status": {"statuses": []},
        "github_list_pull_request_reviews": {"reviews": []},
        "github_list_pull_request_review_threads": {"review_threads": []},
        "github_compare_commits": {
            "status": "identical",
            "merge_base_commit": {"sha": "d" * 40},
        },
    }
    reader = tmp_path / "reader-fixture"
    reader.write_text(
        preamble
        + f"responses={responses!r}\nfiles={files!r}\nfailure={failure!r}\n"
        + """tool=r['tool']
if tool=='github_fetch_file':
 assert r['arguments']['ref']=='d'*40
 data={'encoding':'utf-8','content':'wrong' if failure=='content' else files[r['arguments']['path']]}
else: data=responses[tool]
Path(a.output).write_text(json.dumps({'request_digest':digest,'provider':'github_connector',
'read_only':True,'result':{'structuredContent':data}}))
"""
    )
    for path in (writer, reader):
        path.chmod(0o700)
    host = DeliveryHost(
        command=(str(writer),),
        executable_sha256=hashlib.sha256(writer.read_bytes()).hexdigest(),
        read_command=(str(reader),),
        reader_sha256=hashlib.sha256(reader.read_bytes()).hexdigest(),
        repository="owner/repo",
        base_ref="a" * 40,
        source_digest="b" * 40,
        expires_at=time.time() + 100,
        authorized=True,
        required_checks=(),
        verification_plan={"delivery_documents_sha256": {
            name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()
        }},
    )
    # This test isolates transport/ack wiring. Real local-gate validation remains
    # mandatory in production; absence is separately exercised without a stub.
    if failure != "local_gate":
        monkeypatch.setattr(owner, "require_local_gate", lambda host, **_: None)
    config = tmp_path / "host.json"
    config.write_text(host.model_dump_json())
    common = {
        "delivery_config": str(config),
        "delivery_config_digest": hashlib.sha256(config.read_bytes()).hexdigest(),
        "source_digest": host.source_digest,
        "environment_digest": "e" * 64,
        "loop_id": "lab",
    }
    logs = []
    with ActivityRuntime(runtime_store(tmp_path, "lab")) as runtime:
        register_delivery_waits(runtime, common, [wait], logs.append)
        state = next(s for s in runtime.snapshot().values() if s.spec.kind == "delivery")
        if failure is None:
            assert state.status == "succeeded", logs
            assert campaign_prerequisites(tmp_path, handoff) == ((), [])
            register_delivery_waits(runtime, common, [wait], logs.append)
            assert next(s for s in runtime.snapshot().values() if s.spec.kind == "delivery").attempts == 1
            from slm_training.autoresearch.runtime.operations_delivery import consume_delivery
            host.base_branch = "acceptance-only/other"
            with pytest.raises(ValueError, match="base_branch_changed_requires_new_activity"):
                consume_delivery(runtime, state.spec.activity_id, wait, host)
            assert next(s for s in runtime.snapshot().values() if s.spec.kind == "delivery").attempts == 1
        else:
            assert state.status == (
                "waiting_capability" if failure == "local_gate" else "waiting_repair"
            )
            assert len(campaign_prerequisites(tmp_path, handoff)[1]) == 1
            if failure == "local_gate":
                assert state.attempts == 0
            else:
                from slm_training.autoresearch.runtime.operations_delivery import consume_delivery
                host.base_branch = "acceptance-only/other"
                with pytest.raises(ValueError, match="base_branch_changed_requires_new_activity"):
                    consume_delivery(runtime, state.spec.activity_id, wait, host)


def test_local_gate_rejects_unsigned_complete_claim(tmp_path):
    import json
    from types import SimpleNamespace
    from slm_training.autoresearch.runtime.operations_reconciliation import (
        require_local_gate,
    )

    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    cache.mkdir(mode=0o700)
    key = cache / "issuer.key"
    key.write_bytes(b"x" * 32)
    key.chmod(0o600)
    identity = "a" * 64
    (cache / (identity + ".json")).write_text(
        json.dumps(
            {
                "payload": {"identity": identity, "verification_complete": True},
                "mac": "0" * 64,
            }
        )
    )
    host = SimpleNamespace(
        verification_plan={
            "source": str(source),
            "state_dir": str(cache),
            "identity": identity,
        }
    )
    with pytest.raises(ValueError, match="unauthenticated verification cache"):
        require_local_gate(host)


@pytest.mark.parametrize(
    "states,blocked",
    case_values(__file__, "test_effective_review_ignores_superseded_requests"),
)
def test_effective_review_ignores_superseded_requests(states, blocked):
    from slm_training.autoresearch.runtime.operations_reconciliation import (
        _changes_requested,
    )

    rows = [
        {"id": i, "user": {"login": "reviewer"}, "state": state}
        for i, state in enumerate(states)
    ]
    assert _changes_requested(list(reversed(rows))) is blocked
    rows.append({"id": 100, "user": {"login": "other"}, "state": "CHANGES_REQUESTED"})
    assert _changes_requested(rows)


def _publish_document_source(store, initial):
    from scripts.github_source_authority import record_initial_release
    from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime

    host, _, connector, _ = initial
    lock = store.write_artifact("driver_cycle_inputs", {"campaign_id": store.campaign_id,
        "publication_source": {"source_digest": host.source_digest, "commit": host.base_ref,
                               "repository": host.repository}})
    store.append_event("driver_cycle_locked", artifact_sha256=lock.stem)
    with ActivityRuntime(CampaignStore("runtime", store.root.parent / "loops/lab")) as runtime:
        lease = _claim(host, runtime)
        asyncio.run(record_initial_release(runtime, lease, host, connector))
        _finish(runtime, lease)
