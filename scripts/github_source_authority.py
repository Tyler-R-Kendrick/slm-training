"""Derive a successor delivery grant only from an explicitly authorized lineage."""

import json
from pathlib import Path

from slm_training.autoresearch.heal import repair_delivery
from scripts.github_source_delivery import source_completion


def successor_host(store, wait, host):
    """Static host authority delegates accepted routine successors, never dirty WIP.

    The configured source/base remain the root of authority. Each intervening
    accepted publication must already have an independent delivered receipt;
    that receipt supplies the next actual remote base without editing config.
    """
    if not host.accepted_source_successors:
        return host
    subject = repair_delivery.resolve_source_delivery(store, wait)
    base, prior = _lineage_base(store, subject, host)
    configured = host.verification_plan or {}
    namespace = Path(configured["source_delivery_root"]).resolve() / wait["publication_id"]
    plan = {"source": str(namespace / "candidate"), "state_dir": str(namespace / "cache"),
            "execution": subject["successor_execution"], "runtime_roots": configured.get("runtime_roots", ()),
            "grant": configured.get("grant"), "authority_source_digest": host.source_digest}
    if prior is None:
        plan["repository_source"] = configured["repository_source"]
    else:
        plan["remote_base_commit_object"] = prior["merge_commit_object"]
    return host.model_copy(update={"source_digest": subject["successor_source_digest"],
                                   "base_ref": base, "verification_plan": plan})


def _lineage_base(store, subject, host):
    chain, seen = [], set()
    while subject["predecessor"]["runtime_source_digest"] != host.source_digest:
        identity = subject["predecessor_publication_id"]
        if identity in seen:
            raise ValueError("source_delivery_acceptance_lineage_cycle")
        seen.add(identity)
        events = [row for row in store.verify_event_chain()
                  if row["event_type"] == "repair_release_accepted"
                  and row["detail"].get("publication_id") == identity]
        if len(events) != 1:
            raise ValueError("source_delivery_grant_predecessor_not_accepted")
        wait = events[0]["detail"]["handoff"]["source_delivery"]
        predecessor = repair_delivery.resolve_source_delivery(store, wait)
        if predecessor["successor_source_digest"] != subject["predecessor"]["runtime_source_digest"]:
            raise ValueError("source_delivery_grant_lineage_mismatch")
        chain.append(_delivery_proof(store, wait, host.repository))
        subject = predecessor
    base, prior = host.base_ref, None
    for proof in reversed(chain):
        if proof["base_ref"] != base or proof.get("base_branch", "main") != host.base_branch:
            raise ValueError("source_delivery_grant_remote_base_lineage_mismatch")
        base, prior = proof["merge_sha"], proof
    return base, prior


def _delivery_proof(store, wait, repository):
    proofs = {}
    for event in store.verify_event_chain():
        if (event["event_type"] != "verified_repair_source_delivered"
                or event["detail"].get("publication_id") != wait["publication_id"]):
            continue
        identity = event["artifact_sha256"]
        if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("source_delivery_receipt_digest_invalid")
        path = store.root / "artifacts/verified_repair_source_delivery_receipts" / (identity + ".json")
        proof = source_completion(store, wait, {"verification_artifact": str(path), "receipt": json.loads(path.read_text())})
        if proof["repository"] == repository:
            proofs[identity] = proof
    if len(proofs) != 1:
        raise ValueError("source_delivery_predecessor_receipt_missing_or_ambiguous")
    return next(iter(proofs.values()))


def initial_source_request(host):
    """Pin an initial release independently of repair or PR existence."""
    plan = (host.verification_plan or {})["initial_release"]
    return {"schema_version": "initial_source_request/v1", "repository": host.repository,
            "base_branch": host.base_branch, "commit": host.base_ref, "tree": plan["tree"],
            "source_digest": host.source_digest,
            **{name: str(Path(plan[name]).resolve()) for name in ("source", "release", "execution", "outputs")}}


async def record_initial_release(runtime, lease, host, connector):
    """Trusted readback producer; caller commits returned outputs under same lease.

    connector is the configured host capability, not worker-supplied data.
    No repair events, writes to GitHub, or inferred main membership are involved.
    """
    import asyncio
    import time
    from slm_training.harness_core.activity_contract import contract_digest
    from urllib.parse import quote
    from slm_training.harness_core.execution_release import runtime_source_identity
    from slm_training.harness_core.github_delivery_remote import fetch, verify_tree
    from slm_training.harness_core.github_git_snapshot import commit_object
    from slm_training.harness_core.source_authority import authority_record, authority_reference, publish_source_authority
    from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS

    request = initial_source_request(host)
    if not host.authorized or host.expires_at <= time.time():
        raise ValueError("initial_source_grant_expired_or_unauthorized")
    with runtime.publication(lease) as state:
        if (state.spec.kind != "verify" or state.spec.input_digest != contract_digest(request)
                or state.spec.source_digest != host.source_digest
                or "authorized_github_connector_read" not in state.spec.capabilities):
            raise ValueError("initial_source_request_lease_mismatch")
    seconds = min(INTERRUPT_AFTER_SECONDS, host.expires_at - time.time(),
                  lease.expires_at - time.time() - KILL_GRACE_SECONDS - 1)
    if seconds <= 0:
        raise ValueError("initial_source_deadline_exhausted")
    suffix = "git/ref/heads/" + quote(host.base_branch, safe="/")
    async with asyncio.timeout(seconds):
        first = await fetch(connector, host.repository, suffix)
        commit = await fetch(connector, host.repository, "git/commits/" + host.base_ref)
        raw = commit_object(commit, host.base_ref)
        if commit["tree"]["sha"] != request["tree"]:
            raise ValueError("initial_source_tree_mismatch")
        await verify_tree(connector, host.repository, request["tree"])
        last = await fetch(connector, host.repository, suffix)
    def ref(value):
        return {"ref": value["ref"], "object": {name: value["object"][name] for name in ("type", "sha")}}
    remote = {"repository": host.repository, "base_branch": host.base_branch,
              "source_digest": host.source_digest, "merge_sha": host.base_ref,
              "candidate_git_tree": request["tree"], "merge_commit_object": raw,
              "ref_before": ref(first), "ref_after": ref(last)}
    if runtime_source_identity(Path(request["source"])) != host.source_digest:
        raise ValueError("initial_source_snapshot_changed")
    payload = authority_record(runtime, lease, kind="initial_release", request=request,
                               remote=remote, execution=request["execution"])
    reference = authority_reference(payload)
    _materialize_initial(request, raw, reference)
    from slm_training.autoresearch.heal.repair_release import _durable_tree, _sync
    for name in ("release", "execution"):
        _durable_tree(Path(request[name]))
        _sync(Path(request[name]).parent)
    if host.expires_at <= time.time():
        raise ValueError("initial_source_grant_expired_or_unauthorized")
    return publish_source_authority(runtime, lease, payload)


def _materialize_initial(request, raw, reference):
    from slm_training.harness_core.execution_release import _delivered_provenance, _runtime_manifest, prepare_delivered_release

    source = (Path(request["source"]), request["source_digest"])
    commit = (raw, request["commit"])
    destinations = tuple(Path(request[key]) for key in ("release", "execution", "outputs"))
    if not any(path.exists() for path in destinations[:2]):
        prepare_delivered_release(source, destinations, commit, source_authority=reference)
        return
    # Retry may reuse only exact completed materialization; never repair partial bytes.
    manifest = _runtime_manifest(destinations[1])
    if (manifest is None or manifest.get("source_authority") != reference
            or manifest["release"] != request["release"] or manifest["outputs"] != request["outputs"]
            or manifest["source_digest"] != request["source_digest"]
            or manifest["git_provenance"] != _delivered_provenance(source, commit)):
        raise ValueError("initial_source_existing_materialization_mismatch")
