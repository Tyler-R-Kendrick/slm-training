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

## Live host-Codex repair seam — 2026-09-25

A real host Codex subscription call now traverses `CodexExecutor`, the fixed
provider bridge, and the outer Bubblewrap runner on a disposable failing
fixture. The first proposal diagnosed that Codex's nested `workspace-write`
sandbox could not create `/workspace/.agents` inside the outer runner's
read-only workspace. Codex therefore runs with `danger-full-access` **inside**
the mandatory outer Bubblewrap namespace; that outer namespace remains the
authority and exposes only the exact granted files, read-only runtimes, and one
filtered provider socket. The fixture candidate changed only `answer.py` and
its one granted new regression module. The regression-test output field is now
constrained in both the trusted prompt and JSON schema to the one exact new
test path required by the independent verifier.

The host-authenticated subscription request produced a patch whose controller-recomputed
digest matched. Independent isolated checks reproduced the original failure on
the frozen baseline, observed the new test fail on baseline and pass on the
candidate, and passed the locked original plus existing regression checks on
the candidate (**5/5 expected observations; journal chain valid**). The worker
returned `waiting_verification`, as required; this disposable fixture did not
run source-verification authorization, publication, or supervisor replay, so it
does not close production release acceptance. Credentials remained host-side
and were not emitted or mounted into the worker.

No service was installed/started/restarted. No live training, scientific
comparison, promotion, shipment, cloud job, remote Git write or paid inference
was performed. Finite fault tests do not establish perpetual reliability.

## Exact PR source reconciliation — 2026-09-25

The disposable validation checkout was compared against PR #1785 commit
`36264b992dabf5f857b5a631a52e38d5f8403c54`; six local files differed.
Their local versions were preserved before restoring the remote blob contents.
`git diff 36264b9 --stat` then returned no differences. Testing that exact
source found two unpublished regression updates: the environment-identity
stub did not accept the runtime identity keyword, and the short-shard test
still expected three seconds despite the five-second per-node estimate floor
(one second startup plus twice that floor gives eleven seconds). Both test
expectations now match the existing production contracts; no gate changed.

Validation: the autonomy-integration, verifier-scheduling, verifier-runtime,
and verifier-isolation modules passed **68 tests in 21.15 seconds**, using
the runtime-r11 Python environment, Node 22, the existing explicitly declared
OpenUI/Design MD/GraphQL bridge dependency roots, and real host Bubblewrap.
The initial restricted-shell run could not create network namespaces; the
host rerun established isolation behavior. This is focused regression
evidence, not complete source verification, model evaluation, or release
authorization. The R14 timer was observed active, last fired at 06:40 CDT,
with its older immutable source at 9/18 static obligations and no tests
collected. Its receipts are not current-PR acceptance evidence.

## Source-blocked handoff and operation continuation — 2026-09-25

An ordinary harness-failure handoff omitted the blocker code, unmet predicate,
and source-repair capability required by typed dispatch. It therefore stopped
at `waiting_diagnosis` before the configured repair executor. The producer now
provides `harness_code_failure`, `frozen_arm_measurement_complete`, and
`source_repair`, retaining the frozen manifest digest. An approved reproduction
recipe and grant remain prerequisites; descriptive error text grants nothing.

Typed source-blocked driver yields also skipped the original-request journal
event required by verified successor activation. The operation owner now
records that request only for a recognized code blocker with an explicit
source-repair capability. Publication and independent verification still govern
activation. The regression follows the original logical request into its
successor, preserves unused seconds and attempts, retains the scientific
replicate, and verifies idempotent replay. Quality-gate, external-tool-host,
data-volume, and missing-environment controls cannot request source repair.

Tests use real producer, inspection, dispatch, journal, publication, and
successor owners with simulated process/agent boundaries. They do not prove
a live production repair or authorize model promotion. The code-quality
ratchet records the one-line reduction in the continuous-driver module; no
ceiling increases. The independently discovered corrupt published baseline
was restored in commit `3036f32703a3fcb98d4f10090beebfb49b144941`; its GitHub
content was fetched back and matched the preserved valid JSON exactly.

Parent validation on the integrated source: seven-file recovery suite **84 passed in 51.53s**; handoff-focused selection **23 passed, 348 deselected in 7.66s**. These checks use simulated process/agent boundaries and do not discharge live autonomous repair acceptance.

R17 canonical verifier found `extract_test_cases` failed because the new negative-control parameter table remained inline. The existing casefile extractor moved it to `src/slm_training/resources/test_cases/test_scripts/test_self_healing_handoffs.json`; the source test now uses `case_values`. On this successor source, `extract_test_cases` passes, 11 handoff tests pass, Ruff passes, code quality passes, and version stamps pass. R17 remains a preserved failed snapshot; current-source release verification must use a successor.

The ordinary two-arm command cursor exposed a child-process import failure: a compiled `python -m scripts.train_model` child could not import `slm_training` when the parent lacked `PYTHONPATH`. The shared `engine._stage_environment` now passes this checkout's `src` path to repository CLI children and keeps scratch CPU thread limits. With the parent `PYTHONPATH` removed, real trainer and evaluator `--help` children both exit zero; 16 focused command-cursor/thread tests pass, and Ruff, code quality, test-case extraction, and version stamps pass. This proves CLI startup, not a completed train/eval campaign or AgentV publication. The `harness.autoresearch.experiment_campaign` version was bumped to v301. R18 remains an immutable prior-source verifier snapshot; current-source validation requires R19.

A clean ordinary-supervisor execution copy requires an authenticated head commit marker. The existing materializer omitted committed `.env.example` and `.serena` policy files from its tree identity and could not reconstruct an unsigned GitHub commit whose original timezone was normalized to UTC by the API. The source owner now includes the committed template and two Serena files while excluding private overrides/cache, then accepts an offset candidate only when the exact Git commit hash matches the connector-confirmed head and tree. Altered source bytes still fail the delivered-tree check. Focused materialization tests: 23 passed; Ruff, code quality, and version stamps pass. A current-head proof execution copy and fresh preregistered campaign remain required after publication.


### 2026-09-26: persistent output grants at shared isolation boundaries

R28 completed all 18 static obligations but could not start pytest collection:
its worker requested a writable root-level `workload-result.json`, which the
new narrow-parent mount policy correctly rejected. The collection and test
execution owner now precreates `workload-output/result.json` and grants that
exact file. Real Bubblewrap collection, execution, malformed-result rejection,
and protected controller/source tests exercise the same producer and consumer.

The hypothesis executor had a related directory-as-file grant. A proposed
disposable tmpfs mount also failed because its result disappeared before the
controller read it. The corrected owner precreates and grants
`proposal-output/matrix.json`. A real isolated worker regression fails on the
tmpfs implementation and passes when the controller receives the persisted
result. This check uses a deterministic executable, not a live model provider.

The five affected isolation test modules pass together: **74 passed in 17.37s**,
using the dedicated `runtime-r25-python` environment, installed AgentV runtime,
and host Bubblewrap with `SLM_REQUIRE_ISOLATION=1`. Updated agent-runner fixtures
retain cancellation and protected-sibling checks with narrow parent mounts.
These focused results do not authorize release, live repair acceptance, model
promotion, or shipment. Full current-source verification and the paired
ordinary-supervisor campaign remain required.

Operational correction: R28's user timer was live on `/run/user/1000/bus`; a
missing bus environment caused a false stopped-runner report. A duplicate shell
loop was removed. R29 initially shared hardlinks with R28 and a version-file
edit changed R28's source identity. The original R28 version file was restored
from its preserved R27 predecessor, all R29 source hardlinks were separated,
and all prior journals were retained. Evidence from different identities is
not combined. The successor verifier must use R29's immutable source and its
own authenticated state.


R30 extends the same regression set through the canonical isolated merge worker,
using exact collected node IDs and explicit approved Python, Node, bridge, and
AgentV runtime roots. This exposed a stale test assumption that nested namespaces
always fail, and two real runtime-composition defects: the SDK's Node wrapper
was selected as the native runtime, and inherited `/runtime/N` aliases overwrote
explicit slots after a nested worker reordered grants. The worker now exports
its approved native Node path through the existing test override, and runtime
mounts install aliases before explicit slots. Controller protection is tested
by the actual denied write, without assuming namespace creation is unavailable.

All **74 exact test nodes pass through canonical isolated execution**: pytest
13.02 seconds, complete worker 25.35 seconds, no skipped or deselected nodes.
The retained result is `outputs/autonomy-integration-20260921/r30-focused-worker-runtime-order.json`,
with validated workload digest
`3fb2e7de36d1faba2d76b74c7b2bdf4cabdb20f6b7416ba0d4f0069f90e28166`.
Earlier failed observations remain alongside it. This is stronger integration
evidence than host-only tests but still covers only these 74 nodes; it does not
replace the full merge gate or the live repair and scientific campaign obligations.

The retained September 23 Codex-subscription attempt did produce a proposal
(`480c08fa6d47bf3a270904b6c867f4a769c343dc0beec8b1149d6a281678aab9`).
Its controller remained parked on the original source-verification dependency
after multiple authenticated successors were activated. The previous resolver
followed only one link, so subsequent invocations never reached the runnable
verifier. The controller now resolves the complete activation chain, preserves
the proposal and remaining resource grant, and checks every identity and grant
transition before claiming work or waking repair. Activation ancestry separates
a valid return to an earlier runtime identity from ambiguous or cyclic links.

This preserves successful provider work without calling the provider again.
The retained proposal is not an accepted repair: independent verification and
publication remain incomplete, and the historical controlled attempt lacks an
original leased operation needed to prove end-to-end operation continuation.
The older September 22 editor failure remains valid historical evidence but
does not describe the later successful proposal.

The integrated successor and verifier modules pass **30 tests in 17.23 seconds**
with `SLM_REQUIRE_ISOLATION=1`, the dedicated Python runtime, and real host
Bubblewrap; no tests were skipped. The rollback regression fails before the
ancestry fix. Negative controls cover altered identities, cyclic or ambiguous
links, and residual-grant mismatches; repeated invocations retain retry timing
and charged resource usage. Independent review found the rollback case, then
verified the corrected root selection. A sandbox-only run skipped the real
isolation test and is not counted as isolation evidence.
