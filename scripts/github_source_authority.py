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
        if proof["base_ref"] != base:
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
