"""Run the existing finite verifier for a host-pinned remote-base candidate.

The trusted host supplies its exact candidate/cache/identity and a separate
verification grant. Acceptance of a repair does not refill that grant or turn
its synthetic-base gate into evidence for a different comparison base.
"""

from scripts.autotrain_verification import (
    authenticated_completion, dependency_plan, register_dependency,
)
from scripts.github_source_delivery import source_binding
from scripts.merge_verification_controller import execute_release_attempt
from slm_training.harness_core.activity_contract import contract_digest


def source_capable(runtime, wait, host, capable):
    import time

    if host is None or not host.authorized or host.expires_at <= time.time():
        return False
    source_binding(runtime.store, wait, host)
    return capable or advance_source_gate(runtime, wait, host)


def advance_source_gate(runtime, wait, host):
    """One bounded existing verification activity; never start a remote writer.

    Returns True only after current authenticated completion. The host's
    verification_plan must already bind the accepted candidate and real base.
    An omitted grant permits reuse of a completed gate, never execution.
    """
    from slm_training.autoresearch.runtime.operations_verification import require_local_gate
    from slm_training.autoresearch.heal.repair_acceptance import source_verification_activity_id

    source_binding(runtime.store, wait, host)
    try:
        require_local_gate(host)
        return True
    except ValueError:
        pass
    configured = host.verification_plan
    if not configured.get("grant"):
        _record(runtime, wait, "source_delivery_verification_grant_required")
        return False
    identity = configured["identity"]
    dependency = {
        "root": configured["source"], "state_dir": configured["state_dir"],
        "base_ref": host.base_ref, "verification_identity": identity,
        "activity_id": source_verification_activity_id(identity, configured["grant"]),
        "grant": configured["grant"], "runtime_roots": configured.get("runtime_roots", ()),
        "wake": {"predicate": "complete_current_source_verification",
                 "source": "source_verification_completed", "identity_digest": identity},
    }
    # Recompute the entire canonical binding, including cumulative changed paths
    # and source-owned test selection. Never edit/relabel an old signed cache.
    plan = dependency_plan(dependency)
    register_dependency(runtime, plan)
    from slm_training.autoresearch.heal.isolation import probe_isolation

    capabilities = {"local_process"}
    if probe_isolation().available:
        capabilities.add("isolated_verifier")
    lease = runtime.claim_next(activity_id=plan["activity_id"], capabilities=capabilities)
    if lease is not None:
        execute_release_attempt(runtime, plan, lease)
    if not authenticated_completion(plan):
        _record(runtime, wait, "source_delivery_full_gate_pending", identity)
        return False
    require_local_gate(host)
    return True


def _record(runtime, wait, reason, identity=None):
    detail = {"wait": wait, "predicate": reason, "verification_identity": identity,
              "writer_started": False}
    runtime.store.append_event("source_delivery_verification_wait", detail=detail,
        idempotency_key="source-delivery-gate:" + contract_digest(detail))
