"""Evidence-bound publication prerequisites; scientific measurements stay intact."""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from slm_training.autoresearch.storage import CampaignStore, _sha


_REPOSITORY = ContextVar("publication_repository", default=None)


@contextmanager
def publication_repository_scope(common):
    host = _trusted_host(common)
    token = _REPOSITORY.set(host.repository if host is not None else None)
    try:
        yield
    finally:
        _REPOSITORY.reset(token)


class SourcePublicationPrerequisite(RuntimeError):
    """Typed operational yield, never a model-quality or harness-code failure."""

    def __init__(self, store, loop_id, source, measurement_complete=None, repository=None):
        request = {"schema_version": "source_publication/v1", "campaign_id": store.campaign_id,
                   "loop_id": loop_id, "source": source, "repository": repository or (source or {}).get("repository") or _REPOSITORY.get()}
        self.pending = {"schema_version": "driver_pending/v1", "outcome": "dependency",
            "reason": "source_publication_required", "measurement_complete": False,
            "diagnostic_measurement_complete": measurement_complete, "publication_complete": False,
            "campaign_id": store.campaign_id, "publication_prerequisite": request,
            "wake": {"predicate": "evidence source has authenticated main membership",
                     "source": "authorized_source_publication", "identity_digest": _sha(request)}}
        super().__init__("source_publication_required")


def capture_publication_source(cwd, commit):
    from scripts.run_autotrain_supervisor import _source_identity

    return {"source_digest": _source_identity(Path(cwd)), "commit": commit,
            "repository": _REPOSITORY.get()}


def evidence_source(store):
    """Read only pre-execution locks, never a later controller's version stamp."""
    kinds = {"promotion_finalization_locked": "promotion_finalization",
             "driver_cycle_locked": "driver_cycle_inputs"}
    for event in reversed(store.verify_event_chain()):
        if event["event_type"] not in kinds:
            continue
        sha = event["artifact_sha256"]
        path = store.root / "artifacts" / kinds[event["event_type"]] / (sha + ".json")
        value = json.loads(path.read_text())
        if path.is_symlink() or _sha(value) != sha or value["campaign_id"] != store.campaign_id:
            raise ValueError("publication_evidence_lock_changed")
        if "publication_source" in value:
            return value["publication_source"]
        if event["event_type"] == "promotion_finalization_locked":
            return {"source_digest": value["source_sha256"], "commit": value["integration_commit"],
                    "repository": value.get("publication_repository")}
    return None


def require_source_publication(store, loop_id, *, source=None, repository=None, measurement_complete=None):
    """Resolve against this loop's trusted journal and original evidence identity."""
    from slm_training.harness_core.source_authority import require_main_source_authority

    source = evidence_source(store) if source is None else source
    repository = repository or (source or {}).get("repository") or _REPOSITORY.get()
    journal = CampaignStore("runtime", store.root.parent / "loops" / loop_id)
    if source is not None and repository and source.get("repository") in {None, repository}:
        for event in reversed(journal.verify_event_chain()):
            if event["event_type"] != "source_authority_recorded":
                continue
            sha = event["artifact_sha256"]
            if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("invalid_source_authority_digest")
            path = journal.root / "artifacts/source_authorities" / (sha + ".json")
            payload = json.loads(path.read_text())
            remote = payload["remote"]
            if (payload["source_digest"] != source["source_digest"]
                    or remote["merge_sha"] != source["commit"]
                    or remote["repository"] != repository):
                continue
            reference = {"kind": payload["kind"], "artifact_sha256": sha}
            try:
                return require_main_source_authority(journal, reference, execution=payload["execution"],
                    expected_repository=repository, expected_commit=source["commit"],
                    expected_source_digest=source["source_digest"])
            except ValueError:
                continue  # Incomplete/non-main receipts cannot wake publication.
    pending = SourcePublicationPrerequisite(store, loop_id, source, measurement_complete, repository)
    request = pending.pending["publication_prerequisite"]
    artifact = journal.write_artifact("source_publication_requests", request)
    journal.append_event("source_publication_requested", artifact_sha256=artifact.stem,
                         idempotency_key="source-publication:" + artifact.stem)
    raise pending


def _trusted_host(common):
    import hashlib
    from slm_training.autoresearch.runtime.operations_delivery import load_delivery_host

    if not common.get("delivery_config"):
        return None
    path = Path(common["delivery_config"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != common["delivery_config_digest"]:
        raise ValueError("delivery configuration changed")
    host = load_delivery_host(path)
    return host if host.authorized else None


def _configured_host(common, request):
    import time

    host = _trusted_host(common)
    if host is None or request["source"] is None:
        return None
    if (not host.authorized or host.expires_at <= time.time() or host.base_branch != "main"
            or (request.get("repository") and host.repository != request["repository"])
            or host.source_digest != request["source"]["source_digest"]
            or host.base_ref != request["source"]["commit"]
            or "initial_release" not in (host.verification_plan or {})):
        return None
    return host


def _initial_readback(runtime, common, request, host):
    import asyncio
    import hashlib
    import time
    from scripts.github_source_authority import initial_source_request, record_initial_release
    from slm_training.autoresearch.runtime.operations_reconciliation import reader_connector
    from slm_training.harness_core.activity_contract import ActivityOutcome, ActivitySpec

    initial = initial_source_request(host)
    activity = "initial-source-" + _sha(initial)[:24]
    runtime.register(ActivitySpec(activity_id=activity, family=common["loop_id"], kind="verify",
        source_digest=host.source_digest, environment_digest=common["environment_digest"],
        input_digest=_sha(initial), output_namespace="attempts/" + activity,
        capabilities=("authorized_github_connector_read",)))
    lease = runtime.claim_next(activity_id=activity, capabilities={"authorized_github_connector_read"})
    if lease is None:
        return
    started = time.monotonic()
    outcome, outputs = ActivityOutcome.RETRY, {}
    try:
        connector = reader_connector(runtime, lease, host, {"source_request": initial})
        asyncio.run(record_initial_release(runtime, lease, host, connector))
        outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in runtime.attempt_dir(lease).glob("*.json")}
        outcome = ActivityOutcome.SUCCEEDED
    except Exception:
        runtime.finish(lease, outcome=outcome, outputs=outputs,
                       spent_seconds=time.monotonic() - started)
        raise
    runtime.finish(lease, outcome=outcome, outputs=outputs,
                   spent_seconds=time.monotonic() - started)


def _resolve_waits(runtime, request, proof):
    from slm_training.harness_core.activity_contract import WakeCondition

    store = CampaignStore(request["campaign_id"], runtime.store.root.parents[2])
    wake = WakeCondition.model_validate(SourcePublicationPrerequisite(store, request["loop_id"],
                                                                      request["source"], repository=request.get("repository")).pending["wake"])
    for activity, state in runtime.snapshot().items():
        if state.status == "waiting_dependency" and state.wake == wake:
            runtime.wake(activity, evidence=wake)
    artifact = runtime.store.write_artifact("source_publication_resolutions", {"request": request, "proof": proof})
    runtime.store.append_event("source_publication_resolved", artifact_sha256=artifact.stem,
        detail={"request_digest": _sha(request)}, idempotency_key="source-published:" + _sha(request))


def drain_source_publications(runtime, common, log_event):
    """One bounded configured readback, then canonical proof before waking work."""
    events = runtime.store.verify_event_chain()
    resolved = {e["detail"]["request_digest"] for e in events if e["event_type"] == "source_publication_resolved"}
    serviced = {e["detail"]["request_digest"]: index for index, e in enumerate(events)
                if e["event_type"] == "source_publication_serviced"}
    requests = [e for e in events if e["event_type"] == "source_publication_requested"
                and e["artifact_sha256"] not in resolved]
    for event in sorted(requests, key=lambda e: serviced.get(e["artifact_sha256"], -1))[:1]:
        path = runtime.store.root / "artifacts/source_publication_requests" / (event["artifact_sha256"] + ".json")
        request = json.loads(path.read_text())
        if (_sha(request) != event["artifact_sha256"] or request["loop_id"] != common["loop_id"]
                or request["schema_version"] != "source_publication/v1"):
            raise ValueError("source_publication_request_changed")
        store = CampaignStore(request["campaign_id"], Path(common["root"]))
        runtime.store.append_event("source_publication_serviced",
                                   detail={"request_digest": _sha(request)})
        host = None
        try:
            host = _configured_host(common, request)
            if host is not None:
                _initial_readback(runtime, common, request, host)
            proof = require_source_publication(store, request["loop_id"], source=request["source"],
                repository=request.get("repository") or (host.repository if host is not None else None))
        except (SourcePublicationPrerequisite, OSError, ValueError, TimeoutError):
            log_event({"event": "source_publication_wait", "request_digest": _sha(request)})
        else:
            _resolve_waits(runtime, request, proof)
        return


def committed_document_bundle(cwd, files, *, root=None, loop_id=None, campaign_id=None):
    """Both measured source and the document-bearing release need main proof."""
    from slm_training.harness_core.execution_release import runtime_git_provenance

    if not files or root is None or loop_id is None or campaign_id is None:
        return False
    store = CampaignStore(campaign_id, root)
    frozen = runtime_git_provenance(cwd)
    if frozen is None or frozen["code_dirty"]:
        return False
    try:
        authority = require_source_publication(store, loop_id)
        require_source_publication(store, loop_id, repository=authority["repository"],
            source=capture_publication_source(cwd, frozen["integration_commit"]))
    except SourcePublicationPrerequisite:
        return False
    return all((cwd / name).is_file() and (cwd / name).read_text() == content
               for name, content in files.items())

