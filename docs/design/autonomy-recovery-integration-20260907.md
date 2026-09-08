# Recovery and isolation integration — current verification, 2026-09-07

## Superseding update: verified successor request and production repair wiring

**46 passed, 0 skipped**, 5.50 seconds; exact command/JUnit, latest scoped hashes
and version stamp are in JSON `successor_execution_update`. Earlier sections
record earlier code snapshots, not the current continuation contract. In
particular, the earlier unchanged-request `runtime.wake` design is superseded.

The canonical supervisor now invokes `dispatch_controller_repairs` from
`repair_operation`, uses the validated immutable release for repair input, and
supplies the verified publication callback. Only trusted repair-controller
children receive `kind=control` and `controller_publication`; these are not
capabilities of the sandboxed agent. Default successor storage is a sibling
`verified-releases` directory of that immutable source; an explicit controller
request `release_root` overrides it. Publication still refuses roots inside run
outputs. No new provider, spending or service permission is inferred.

`interpret_operation_result` validates the exact envelope and maps repair waits
to capability/dependency/yield outcomes instead of success. `_operation_state`
reuses a registered successor's immutable source, input and resource grant;
altered or unregistered successor requests fail closed. Parent's existing
failure-before-finish and queue-before-inspect hooks were preserved. Scoped
complexity/import checks pass; no baseline was raised.

After parent validates publication, releases old controller ownership, and
starts a genuinely fresh successor process, it calls:

```python
successor_request = wake_verified_operation(runtime, handoff, cwd=Path.cwd())
run_operation(runtime, successor_request, sequence=sequence, log_event=log_event)
```

Despite the retained compatibility name, `wake_verified_operation` **does not
wake or mutate the old ActivitySpec**. It commits a successor plan, explicitly
cancels the predecessor, registers the new-source activity, and records
activation. The returned request contains verified `cwd`/`source_digest`,
`successor_activity_id`, `resource_grant`, and `logical_continuation` ancestry.
The new grant deducts predecessor consumed seconds and attempts; cumulative
ancestor charges and the original grant remain in the receipt. Recipe/replicate
fields remain unchanged and `scientific_replicate_increment` is zero.

Crash after predecessor cancellation but before registration replays the same
plan, activity ID and grant. `SuccessorGrantExhausted` is a resource/capability
condition, not code failure: no default grant is substituted and the old failed
activity is not cancelled. Parent must preserve finite-cycle remainder across
its restart; never translate exhausted finite cycles to `--max-cycles 0`.

Tests cover current production helper calls, typed waits without duplicate
success jobs, grant/attempt carry-forward, altered successor requests, old-import
refusal, exhausted grants, and interrupted replacement. Positive import identity
is explicitly simulated in the wake test; some workloads/delegation are mocked.
Parent-owned fresh-process launch and finite-cycle continuation remain unverified
by this sidecar. No services/processes were restarted, no live agent was invoked,
and no model improvement, promotion, shipment or full release authorization is
claimed. OPS owns the `autoresearch.runtime` move; RECOVERY changed no runtime
activity implementation.

## Latest update: failed-operation bridge and activation validation

The later `operation_bridge_update` in the JSON manifest records **66 passed,
0 skipped, 0 deselected** in 12.09 seconds. It supersedes the earlier scoped
hashes for its listed files; earlier results below remain historical snapshots.
Scoped Ruff complexity/import checks passed; no quality baseline was raised.
`verify_repair` now takes seven arguments, using
`VerificationWorkspace(base, candidate, runtime_roots)`; both callers migrated.

The tested controller call sequence is:

```python
# run_operation failure path, before finish; preserve the typed outcome.
record_operation_failure(runtime, lease, request, result, outcome=outcome)
runtime.finish(...)  # existing owner charges the failed invocation once

# Before repeating inspection, drain durable events through the same bounded seam.
for repair_request in pending_operation_repairs(runtime):
    run_operation(runtime, repair_request, sequence=sequence, log_event=log_event)

# Inside repair_operation, with a trusted control-child publication lease:
source, isolation_digest = pinned_repair_source(cwd, request["source_digest"])
# Use these two values in RecoveryContext, preserving runtime source identity.
publish = verified_release_callback(
    publisher, lease, destinations=(release_root, root)
)
dispatch_hard_pending(
    pending, context, config=config, fence_valid=fence_valid,
    publish_verified=publish,
)

# Parent validates the handoff before its fresh-process restart, not a hot import:
verified_activation_handoff(runtime.store, handoff)
# Fresh successor process, after source/import validation and controller ownership:
original_request = wake_verified_operation(runtime, handoff, cwd=cwd)
```

Parent must transport the returned **original logical request** through its
verified successor execution path. Merely changing `cwd`/source in the request
hash would mint another activity and leave the original asleep. This helper
keeps its activity ID/input digest, verifies the recorded acceptance and active
successor import location, and invokes `runtime.wake` with the exact existing
predicate. No helper sets `any_healed`, marks the trial successful, or starts a
service. Parent owns these supervisor call sites and actual restart/transport;
that full CLI path has not been established by this sidecar's tests.

`RecoveryConfig.operation_recipes` is a preapproved operation → recipe-code
mapping, default empty. Old configurations acquire no new authority. A matching
code/unknown failure runs the original probe independently, checks both its
frozen result and the failed operation's captured exit/stdout/stderr identity,
then invokes the existing executor in a subsequent bounded invocation. Diagnosis
reservations remain charged to the repair grant; interruption may retry in a new
attempt within the existing attempt/total grant, never reset the allowance.
Other typed owners (formal/data/environment/delivery) are not relabelled code.
Uncovered zero-exit/missing-output failures or mismatched observations remain
diagnosis, not an unrelated-probe success. No stderr instruction grants authority.

Evidence is real failed/diagnostic processes and real ledger/filesystem
publication, with an explicitly fake agent and mocked OS isolation. The wake
test really rejects old imports; its positive successor import identity is
simulated. No live source repair, service activation, model improvement or
whole-repository release authorization is claimed. The JSON retains the first
failed tests and their corrections. Parent's activity-package extraction remains
separate work; no activity modules were edited by this sidecar.

This report is new post-recovery evidence, not a reassertion of the recovered
historical reports. Current workspace:
`outputs/workspaces/autonomy-candidate`; base
`e0eca9f9910244ecc20eb480d363f1852417b599`; **heal v8**.
The [machine-readable manifest](autonomy-recovery-integration-20260907.json)
contains scoped source hashes, collected cases, JUnit identity, environment,
exact command, exclusions and evidence classes. This dirty tree is not a
published release.

## Implemented and exercised

- Restored dynamic RECOVERY test amendments and five historical reports,
  retaining original source paths and explicit historical/not-revalidated labels.
- Agent output uses a controller-created exact writable proposal file, not a
  forbidden directory mount. Existing tests cannot become writable repair paths.
- Cancellation reaches the independent bounded-process cancellation event;
  cancelled work retains spent-compute accounting.
- The installed Codex protocol includes `--skip-git-repo-check` for private
  snapshots without shared Git metadata. No inference was launched.
- Independent verification on a fresh lease binds the current receipt fence while
  preserving the original request. A worker's final answer cannot heal a blocker.
- The trusted dispatch callback publishes a verified implementation successor via
  the canonical activity fence and campaign history. It checks original predicates,
  identities, scope, grant/lease validity and candidate integrity; immutable staging,
  intent → pointer → acceptance reconciliation and CAS prevent mixed publication.
- A real subprocess reproducer/independent-check fixture reaches real fenced
  filesystem publication through the canonical two-invocation dispatch seam.
  Its agent is fake and its OS isolation is mocked. This is plumbing evidence,
  not live autonomous repair or model capability evidence.

## Fresh bounded results

`74 passed, 10 skipped, 1 deselected` (pytest: 4.08 seconds; JUnit: 4.066
seconds), exit 0. Default pytest marker exclusions were cleared using
`-o addopts=`; no cache authorization was reused. Scoped Ruff checks and
`git diff --check` passed. These checks do not authorize the whole candidate.

The ten skips require Bubblewrap, whose actual probe failed:

`bwrap: loopback: Failed to create NETLINK_ROUTE socket: Operation not permitted`

The initial broad run was **1 failed, 61 passed, 10 skipped**:
`test_socket_mount_rejected` could not create its host AF_UNIX socket
(`PermissionError: [Errno 1] Operation not permitted`). That test was not edited;
the final command explicitly deselects it. Neither this exclusion nor the skips
count as passing host isolation.

Actual CLI probe: `codex-cli 0.153.4`, required flags present. No explicit
provider/egress grant was supplied. Live repair is
`not_executed_capability_unavailable`; the code does not fall back to
unsandboxed execution.

Final command (run from the candidate workspace):

```sh
rtk proxy timeout -s INT -k 10 170 env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/codex/repos/slm-training/.venv/bin/python -m pytest -o addopts= -q tests/test_autoresearch/test_repair_release.py tests/test_autoresearch/test_isolated_agent.py tests/test_autoresearch/test_repair_dispatch.py tests/test_autoresearch/test_recovery_dispatch_entrypoint.py tests/test_autoresearch/test_repair_acceptance.py tests/test_autoresearch/test_repair_verifier.py tests/test_autoresearch/test_heal_isolation.py -k 'not test_socket_mount_rejected' --junitxml=outputs/runs/recovery-isolation-20260907/post-recovery-v8.xml --tb=short
```

## Parent supervisor integration contract

`dispatch_hard_pending(..., publish_verified=callback)` invokes the callback
only after controller-accepted independent verification. The callback receives
`(request, result, candidate_path)` and calls:

```python
publish_verified_repair(
    request, result, runtime=publisher, lease=lease, candidate=candidate_path,
    destinations=(release_root, run_outputs),
    expected_previous=observed_publication_id,
    authenticated=controller_authentication,
)
```

Use the canonical `DelegatedPublisher` for a trusted control child, with
`kind=control` and the `controller_publication` capability. Its store and
publisher must never be mounted into the agent or verifier workload.
Authentication is controller-owned provenance, not a flag read from worker JSON.
The release root is outside run outputs. Recovery's source is its private
immutable input snapshot and matching isolation-tree digest, not mutable
execution source containing an external outputs symlink.

The returned `release_handoff` identifies the successor execution, source,
original input/request, campaign and affected activity. It requires a **fresh
process**; publishing does not activate a service, promote a model or silently
reuse incompatible measurements. The parent owns passing this callback and
consuming the handoff to resume the exact blocked activity. This sidecar's
tests do not establish that supervisor restart integration.

No service was installed/started/restarted. No live training, scientific
comparison, promotion, shipment, cloud job, remote Git write or paid inference
was performed. Finite fault tests do not establish perpetual reliability.
