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

R31 addresses an observed collection scheduling failure after all 18 R30 static
checks passed. Every fresh or split batch received only 20 seconds including
sandbox preparation, even when a larger bounded allowance was available. The
worker could finish collection but still time out before the controller received
a valid result. Collection now receives the admitted remaining workload budget;
kill/finalization reserves, retry charges, and timeout rejection remain intact.
A real isolated first batch completed **1,034 collected nodes across 64 test
files in 71.37 seconds**, with workload digest
`b304155b57b8bdef1595c15fca6e6326e5a326ec8bca5b14422723c769561db4`.
This is focused collection evidence, not a passing full test suite. The retained
artifact is `outputs/autonomy-integration-20260921/r31-focused-collection.json`.
R30's failed attempts and signed journal remain preserved; their scheduling was
stopped while preparing the corrected immutable successor.

The current-source scientific attempt and its two interrupted invocations are
recorded in [the R30 measured-results report](science-lab-pr-head-r30-20260926-results.md).
No training/evaluation completion or model improvement is inferred from startup
operations or command-cursor persistence.

Integrated collection scheduling, resumption, and throughput checks pass
**51 tests in 85.41 seconds**. The existing fresh-collection assertion now checks
the full admitted allowance instead of the obsolete 20-second slice. The real
receipt-resumption fixture uses a 60-second workload allowance because the
retained failing journal showed 25.69 seconds of startup before its previous
25-second allowance could admit collection. Its explicit shard pause, duplicate
counts, changed-source rejection, and completion assertions remain unchanged;
typed pending is never accepted as successful execution.

The integrated startup/source-binding tests pass **31 tests in 16.25 seconds**.
Related operation and repaired-release tests also pass; the isolated delivery
receipt module passes **3 tests in 20.30 seconds** on real host Bubblewrap after
removing its obsolete assumption that required isolation means nested namespaces
must fail. All signature, source-drift, receipt and release-mapping assertions
remain. Independent review found no content/link/mode checks removed and no
cross-boundary identity cache introduced by the startup consolidation.

A further evidence review found that several tests in the earlier 74-node run
used local-feedback fallbacks or returned before protection assertions when
inside the outer sandbox. That run proved outer-worker execution but did not
prove every inner isolation boundary. Those shortcuts are removed. The exact
**34 isolation nodes now pass inside the canonical outer worker**, with actual
inner isolation and all protection assertions: worker 40.23 seconds, pytest
20.67 seconds, workload digest
`ccdb34a9854c3e7417d8d4939086e925b8c1c932cc315f583f5f703b084aa294`.
The **3 signed delivery-verification nodes also pass** through the same nested
worker: worker 59.29 seconds, pytest 33.38 seconds, workload digest
`a320f52b529a1c744e23334b8c9d9ca36fdf073fc5dfb0e835a6302d93aafc63`.
Their exact-node receipts remain in `r31-nested-core-verified.json` and
`r31-nested-delivery-verified.json` under the retained output directory.

The initial combined 37-node workload timed out and is not passing evidence.
A separate preparation attempt timed out listing source files; after an actual
successful listing and reduced observed host pressure, the exact delivery shard
completed. Both failed observations remain retained. The successful bounded
shards cover the full 37-node set without skipping or disabling nested isolation.

A proposed non-main delivery extension is withheld from R31: review found that
activation discarded branch authority and could relabel acceptance-only delivery
as unrestricted clean upstream provenance. Its patch is preserved for a separate
authority fix; default main-only delivery remains unchanged in this source.

## R32 pending corrections and R31 observation

The live R31 canonical gate exposed evidence-ledger drift: the committed scan count was 2,351 while rebuilding the current design documents scanned 2,354. Rebuilding changed only this count; all 401 observations and 29 arm aggregates remained identical. The rebuilt ledger passed its canonical check before adding the R31 observation report; it must be regenerated again after all R32 documents settle. Published R31 evidence remains unchanged.

The collection scheduler also returned a negative `required_seconds` hint when remaining admission time was negative. R32 retains the full positive workload requirement whenever the remaining time cannot pass the existing admission floor. Seven focused budget tests pass, including negative remaining time; no timed-out or pending work becomes successful evidence.

The [R31 measured observation](science-lab-pr-head-r31-20260926-results.md) records the incomplete first supervisor invocation. Operational serialization now prevents the retained verifier launcher and campaign launcher from competing with each other. This does not change source identity, reset the continuation grant, or establish successful scientific measurement.

### Authenticated source authority and interrupted activation

R32 introduces one resolver over two existing controller-owned producers: initial connector readback and accepted repair delivery. An execution manifest carries only an artifact reference. Resolution requires the trusted journal, committed successful activity, matching request and terminal lease, exact repository/commit/source, and validated materialization. Clean source alone does not establish membership in main. An initial main release does not require a fabricated repair receipt.

Independent review exposed an interrupted-activation defect: a new lease reused an old activation without publishing a matching authority output into the terminal attempt. The corrected path binds new requests to new immutable activation/authority records; same-request retry republishes under the live fence. The resolver also checks terminal lease binding. Four focused repair tests and 24 resolver tests pass. The complete repair-delivery module subsequently passed all 21 tests in 28.33 seconds, including real nested Bubblewrap execution under the 170-second interrupt cap. Independent read-only re-review closed this finding; production consumer wiring is still undergoing separate verification. No complete repair acceptance or original-operation continuation is inferred from these component tests.

### Agent runtime memory pressure

Five same-project Serena/Pyright instances consumed roughly 3 GB each. After identity-checked graceful termination of four older duplicate instances, all their language-server children exited and the latest loaded instance remained. Available memory reached 17,450,200 KiB. Memory full-stall pressure averaged 35.42% over 60 seconds before cleanup and 0.06% at the later verification observation. The retained R31 campaign then advanced through both control training chunks and recorded a real repair request for its interrupted evaluation cursor; memory cleanup is not presented as a complete harness fix.

The host and active project Codex configurations now omit eager `--project-from-cwd` activation. R32 carries the same project setting with an explanatory comment. This leaves Serena available and activates its language services only when `activate_project` is requested. Existing active processes are not implicitly restarted by the config edit. Private backups preserve the original host/project configuration. A bounded installed-SDK smoke proved startup exposes semantic tools without starting Pyright and accepts on-demand activation; a subsequent tiny-project semantic smoke completed: startup had no active project or new Pyright process, on-demand activation enabled `find_symbol`, and the known fixture function was found in 0.79 seconds. MCP shutdown reaped the language server. The retained evidence is `outputs/autonomy-integration-20260921/serena-lazy-start-smoke.json`.

Consumer verification passed 42 publication/delivery tests, 26 diagnostic/entrypoint tests, seven focused promotion tests, and 20 readiness tests. Re-review closed repository self-authorization and unresolved-request starvation findings: expected repository comes from trusted configuration or the pre-execution lock, and persisted servicing order rotates pending requests while dispatching at most one per invocation. These are component/integration checks; the full canonical gate and real repaired-operation continuation remain required.

After extracting file materialization from release preparation to remove a complexity regression, the combined authority and complete repair-delivery suites passed all 45 tests in 41.36 seconds with real nested isolation. Copy modes, symlinks, complete-tree checks, and fresh provenance validation remain unchanged. Agent-surface parity, repository policy, version stamps, and whitespace checks also pass on the current working candidate; final full-gate evidence is still outstanding.

### Recovery after training and before the first evaluation

The retained R31 failure exposed a concrete recovery mismatch: initial compiled evaluation argv advertises resumable records but naturally does not yet contain `--resume-run`. The former recovery helper refused this interrupted boundary even after the complete training prefix was committed. R32 verifies that entire prefix, derives the same run directory through the existing artifact-path owner, and requests explicit evaluator resume. The evaluator still validates prior rows before decoding; no interrupted execution becomes a successful measurement and all prior reservations stay charged.

The complete focused cursor-recovery module passes 13 tests in 28.57 seconds. The original R31 helper fails the new compiler-generated-command regression. Coverage includes interruption before the first evaluation, preserved partial rows, no repeated training, exact locked argv, authenticated training-resume prefixes, and rejection of altered commands. The retained real R31 cursor was inspected read-only; the patched helper derives its existing run directory. Actual repaired campaign execution still requires source-bound delivery and is not claimed from this test.

## R33 integration closure: authenticated source changes across a locked campaign

A follow-up source audit found that R32's outer operation successor does not yet transport the nested locked campaign across a verified implementation repair. The supervisor validates the original preregistration before recovering the release. Driver continuation then requires the original execution path and source/environment identity, and the command cursor digest includes both. These independent checks correctly reject arbitrary drift, but no authenticated transition joins them to the already verified operation handoff.

The real retained R31 operation supplies the counterexample: its six-update control checkpoint and original 2,700-second logical grant remain intact, but changing only the outer operation request cannot resume its nested driver/cursor. Source-repair delivery itself has no circular main-membership prerequisite; publication of original measurements still requires their own authenticated evidence source. The bounded repair-agent acceptance is held before provider execution until this integration boundary is addressed.

The successor must retain old artifacts and scientific identities while authenticating the new physical execution from existing verified release/activation receipts. It must not redefine source identity globally, rewrite old cursor inputs, repeat training, reset resource charges, or infer completion from the checkpoint alone. A fresh supervisor/worker regression must cross the same boundaries as the real operation. This is an open implementation obligation, not completed acceptance.

### Obsolete verifier ownership audit (2026-09-26)

A host process audit found orphaned R25 launcher PID 1373717 still issuing
bounded verification against superseded source while the R32 timer was active.
The launcher was stopped after checking its exact command and process start
identity. Its current bounded child was allowed to finish; all R25 journals
were retained. Inactive but enabled R26 and R27 timers were disabled to prevent
restarting obsolete verification at login. The R32 timer and its source were
not changed. This fixes duplicate compute ownership; historical results still
do not authorize the current source. Local audit artifact:
`outputs/autonomy-integration-20260921/obsolete-r25-runner-stop-20260926.json`.

At the recorded R32 observation, all 18 static checks passed and 11 of 17
collection batches completed, collecting 8,691 nodes. No test nodes had passed
yet. GitHub returned no pull-request-triggered workflow runs for commit
`79bbc8fb83e2893554d1d3a346c5b52c93a376cd`; external preview/review statuses
therefore cannot stand in for the canonical test gate.

### Controller and workload separation under implementation

The real R31 failure cannot be migrated by treating all R33 changes as an
autonomous worker repair: the integration changes include protected controller
code. The selected design pins the trusted R33 controller independently and
retains an R31-derived workload for the narrow worker repair. Fresh supervisor
and delegated operation-worker processes execute the pinned controller;
training and evaluation subprocesses execute the accepted workload. Physical
source identities remain distinct and recorded. Original scientific inputs,
logical cursor identity, checkpoints, attempt counts, and charged reservations
remain bound to their original campaign.

This design is not accepted evidence yet. Required regressions include a fresh
worker crossing the actual boundary, multi-hop continuation through cancelled
intermediate activities, rejection of copied/self-authored journals, and
rejection of any workload change affecting protected scientific behavior.

### Current-source verification findings

R32 completed collection of 12,185 selected nodes across 512 initial shards.
Execution then exposed a remaining admission-reporting defect: after a shard
used the remaining invocation budget, the persisted next-workload
`required_seconds` became negative. The earlier fix covered collection only.
R33 must fix the shared shard admission calculation without raising the run cap
or resetting retry charges. One single-node shard timed out after 57.84 seconds;
that interrupted attempt is charged and is not counted as passing evidence.

The R31 subscription transport's 21 mock-only tests passed in 1.00 seconds when
local Unix sockets were permitted. The three initial sandbox socket failures
were environmental, not provider outcomes. The real-fault reproducer now also
retains its assertion traceback and resolves the original workload without an
environment override. Neither check constitutes live-agent repair acceptance.

The shard defect also affects execution admission, not only reporting. Setting
`required = min(estimate, available)` makes the subsequent insufficient-budget
comparison ineffective. It can repeatedly admit a timed-out shard into a tail
shorter than its increased requirement. The correction must preserve the
positive estimate (bounded by the canonical workload allowance), defer when
the current tail cannot cover it, and avoid charging an attempt for that
deferral. A second two-node shard timed out after 30.89 seconds; both interrupted
attempts remain recorded rather than counted as passes.

The first R32 assertion failure was an obsolete environment-heal expectation:
`test_verified_env_heal_rewrites_to_next_experiment` assumed a successful
environment repair also acknowledged generated documentation. The production
path correctly retains the document action until authorized connector delivery.
R33 updates the regression to assert that pending action and both durable
`documentation_materialized` and `documentation_waiting_delivery` events.
No publication guard changed. All seven environment-rewrite tests passed in
1.56 seconds; this focused result does not replace the final canonical gate.

### Independent host connector capability

A dedicated host thread exposed the installed GitHub connector through Codex
app-server 0.157.1. Two independently started stdio processes resumed that
probe-owned thread and read PR 1785 with identical response digests. The probe
used `mcpServer/tool/call` directly, without a model turn, interactive login,
credential extraction, or remote mutation. The shared-daemon proxy timed out;
the explicitly selected standalone stdio path succeeded.

This removes the assumed read-capability blocker but is not complete delivery
evidence. R33 integration must provide an explicitly configured host transport
under the existing delivery adapter, retain schema and source pins, enforce
finite process lifetime, and reconcile ambiguous mutations through independent
readback. It must never silently fall back from an unavailable HTTP transport.
Unattended write servicing and genuine repair delivery remain unverified.

The R33 shard-admission change passed 52 focused verifier tests across four
files, including preservation of charges, shards, and passes while waiting for
a fresh invocation. Parent review confirmed that the implementation removes
the tail-clipping root cause. An additional fresh-invocation overhead edge is
under review before this result can support final integration acceptance.

### Monitor replacement and integration review

The replacement host monitor fired through systemd at 17:06 CDT on September
26, 2026. It invoked canonical R32 `monitor-check` and `status` against the
retained R31 campaign and read the R32 verifier report. It correctly reported
that the campaign controller was stopped while its original operation waited
for reconciliation. The monitor did not launch training or a repair provider.
The obsolete R4/R12 monitor was disabled with its logs preserved.

Parent verification confirmed a live calendar timer and the next scheduled
18:00 CDT firing, but found two configuration integration gaps: the initial
custom unit name was not the name queried by canonical monitor status, and its
service timeout plus stop grace could exceed the canonical total cap. Both
require correction before declaring monitoring fully integrated; the initial
trigger proof remains evidence of that narrower executed configuration.

Authenticated multi-hop transition tests passed: 12 tests in 41.41 seconds,
plus one unresolved-cursor regression in 16.54 seconds. The implementation
retains the original logical anchor, subtracts actual predecessor charges, and
rejects forks, cycles, missing ancestry, copied journals, cancelled endpoints,
environment drift, and changed seeds. The tests include a fresh delegated worker.
Controller activation receipt production and strict controller source-pin
validation still require integrated confirmation; these focused results do not
prove the actual retained R31 campaign has resumed.

R32 full verification also rejected SLM-298 planning fixtures that persisted
named `:x`/`:labelN` placeholders. Correcting both fixture producers to opaque
`:slot_0` exposed a real planning defect: the multi-candidate campaign omitted
its mandatory selection rule. New manifests now explicitly lock the canonical
`best_by_primary_then_smallest` rule. Existing manifest revisions continue to
produce explicit deviation records; historical observations are not re-scored.
All five CLI/protocol tests passed in 1.91 seconds. The experiment component
is v171. No training, evaluation, promotion, or scientific campaign ran during
this unit-test repair.

The configured Codex connector transport passed 113 focused tests in 8.51
seconds and three canonical `host_connector` PR-read smokes with equal response
digests. A lost-response regression retains the pending mutation journal and
prevents duplicate commit dispatch. No real mutation ran. Parent review still
requires startup time to be debited from the full invocation allowance and
identity-safe descendant cleanup after the direct child exits. Those lifecycle
conditions remain part of transport acceptance.

The monitor configuration gaps are now closed. The canonical unit
`slm-autoresearch-science-lab-r31-43327120eda9-acceptance-monitor.timer` fired
at 17:16 CDT on September 26. Canonical status reports `ready=true`, that
actual trigger, and the next production firing at 19:00 CDT. The restored
calendar is `*-*-* 00/3:00:00 UTC`, with persistence enabled. The service uses
170-second interrupt and 10-second kill grace, matching the 180-second total
cap. Both obsolete timers are disabled; their logs and definitions remain
preserved. The campaign itself remains stopped pending authenticated recovery;
monitoring liveness is not scientific progress. Evidence:
`outputs/autonomy-integration-20260921/host-progress-monitor-r32/canonical-trigger-proof-20260926.json`.

Final focused admission checkpoint: 53 tests passed after consolidating
regressions into the existing resumption test owner. The verifier is 400 lines,
scheduling tests 337, and resumption tests 363; no debt ceiling increased.
Completed static/collection phases are bypassed on shard-only continuation.
Ruff, diff checks, and version stamps passed. Actual full-run scheduling remains
part of the final-source canonical verification obligation.

Quality review caught the SLM-298 package's two-line growth from adding the
selection declaration. Constructing factorial cells directly from the existing
Cartesian-product tuples removes the redundant unpack/repack and restores the
package ceiling without raising its baseline. The existing five CLI/protocol
tests still cover all 24 cells and locked manifest behavior.

The compiler-trace coverage test still expected 16 samples although the existing
committed fixture contains 37 records. R33 updates that exact expectation and
removes the stale count from the probe docstring; it changes neither fixture
bytes nor coverage calculations. All four related tests passed in 1.68 seconds
with the configured OpenUI bridge. Historical measured reports retain their
original sample counts.

The three-pair NLL classifier regression expected an obsolete null-effect reason.
Current classification is correctly inconclusive: three pairs lack the declared
independent-unit evidence and do not clear the six-pair decidability floor.
R33 asserts `primary_metric_inconclusive`/`paired_inconclusive` while retaining
`positive=false`, `win=false`, and the exact three-pair count. The focused test
passed in 2.22 seconds. No classifier, significance threshold, confirmation
gate, or statistical sample count changed.

Controller/workload integration checkpoint: 66 boundary tests, 31 subsequent
cursor/CLI tests, one fresh controller-to-worker-to-scientific-subprocess test,
and 15 diagnostic provenance/startup tests passed in their recorded focused
runs. These groups are not asserted to be mutually disjoint. The main
`autoresearch.py` owner shrank from 2,091 to 2,075 lines. The remaining
composition regression must connect verification/publication to a fresh
supervisor and retained cursor across two repairs; no live-campaign acceptance
is inferred from the focused tests.

Full R32 verification identified two further failures. The resumed evaluator
reached real AgentV publication but the isolated JavaScript runtime lacked the
`yaml` dependency imported by the pinned AgentV SDK. A separate reasoning
benchmark passed algebra output through the OpenUI statement-binding admission
path and failed before training. R33 must provide a complete pinned runtime
closure and the correct grammar-aware record contract, respectively. Neither
missing publication nor rejected symbolic training is waived as a fixture issue.
The active R32 runtime and its verification evidence remain unchanged.

Independent transport review approved the final lifecycle patch against its
recorded hashes. The reasoning-benchmark failure spans both record admission
and model construction: training repeats OpenUI-only admission, and the old
benchmark additionally requests the prohibited compositional tokenizer. Its
repair must use a supported constrained grammar/tokenizer path; changing
metadata alone or restoring unrestricted output cannot close this obligation.
### Further full-verifier regressions resolved (2026-09-26)

The provider-bridge isolation fixture granted a workspace-root file, which the
current directory-mount guard correctly rejects. The fixture now places that
file at `src/module.py` and grants that exact path. All 39 provider-bridge tests
passed in 6.73 seconds with real local Bubblewrap and Unix sockets. No external
provider was called and no production grant boundary changed.

The stale-positive classification fixture omitted finite parse-rate observations
for both arms. The existing measurement-completeness predicate therefore
correctly declined reclassification. Supplying both observations repairs the
fixture; the focused regression passed, without changing classification,
statistical thresholds, or the 13,305-line file ceiling.

Focused probes of the retained R31 baseline found six failures, one pass, and
one skip in 4.80 seconds. Separate frozen-environment classifier and entrypoint
probes failed in 6.06 and 10.30 seconds respectively. These failures prevent a
narrow reconciliation-only workload repair from satisfying full-source
verification. The original campaign and its provider attempt remain untouched.
A separate source-verification grant permits 1,024 bounded invocations; this
does not confer scientific execution authority or waive any failure. The
trusted-controller upgrade and unchanged scientific workload boundary must be
resolved before dispatch. A separate controlled-defect demonstration cannot
stand in for continuation of this retained operation.
### Isolated AgentV dependency resolution repaired

The missing `yaml` import was caused by flattening `node_modules` into a
runtime mount, not by an absent installed dependency. Node's package search
could not find sibling packages through the flattened mount. R32 also selected
Node 26.5.0, outside the pinned SDK's supported engine range. An independent
read-only runtime now preserves the `node_modules` ancestry and explicitly
selects Node 22. All 267 installed packages match the lock; the 20 absent lock
entries are optional or platform-specific. Copied files share no inodes with
the original runtime, and copied symlinks stay inside the new root.

The exact isolated resumed-evaluation regression and installed-SDK publication
test both passed (2 tests, 13.86 seconds; isolated worker 37.11 seconds). A
separate real isolated AgentV case passed with zero execution errors in 1.01
seconds. These are fixture wiring results, not model-quality evidence. R32's
runtime grants and journal remain unchanged. Final R33 verification must bind
`runtime-r33-agentv-complete` and the explicit Node 22 root before these repairs
can support a release claim. The local handoff records the workload digest and
exact runtime recipe in `r33-agentv-runtime-handoff.md`.
### Completion-domain differential failure under investigation

R32 isolated verification reported a packed/reference disagreement at the empty
prefix with eight tokens remaining: the packed kernel rejected a possible false
singleton, while the reference returned a complete domain. The production
false-singleton guard remains unchanged. A local R33 run of all eleven corpus
parity cases passed in 147.78 seconds. Empty-prefix subprocess probes also agreed
for Python hash seeds 0 through 7. These results do not resolve the isolated
failure; its exact journal node is being replayed under the approved isolated
runtime. An initial reproduction selected an unapproved interpreter and another
lost parameter escaping; both stopped without executing tests and provide no
validation evidence. The subsequent invocation uses the exact node bytes from
the signed journal. No failure is waived as intermittent.

The exact failed completion-domain node passed in canonical R33 isolation:
1 test in 16.50 seconds, worker 29.57 seconds, workload digest
`7806608f97ac70ce4c36d5b344a74ae63d24a0f00d58780ccf7f7270660a4eb4`.
The root cause remains unproven; an unchanged R32 replay follows to separate
source/runtime effects from a transient result. Passing replays do not erase
the original failure.

R32 also found six SFF formal/metrics setup failures from one shared cause:
its multi-candidate campaign omitted the mandatory selection rule. The producer
now locks `best_by_primary_then_smallest`; regression coverage asserts the rule
survives manifest locking. Both focused files passed, 21 tests in 1.49 seconds;
Ruff and diff checks passed. Component
`harness.experiments.anti_e237_semantic_factor_frontier` is v12. Existing
scientific artifacts and R32 source remain unchanged.

Unchanged R32 also passed the exact parity node in isolation (1 test, 28.65
seconds; worker 66.35 seconds). This rules out treating the R33 edits as a
proven repair. The original complete shard is retained for a collection-context
reproduction; both passing isolated replays and the original failure remain
recorded.

The original 21-node R32 shard reproduced the parity failure (2 failures,
19 passes in 6.60 seconds; the second failure is the separately repaired SFF
selection declaration). The parity test executes first, so collection/import
context is the next causal boundary to inspect. This is a reproducible
integration failure, not a waived flaky test. Exact recipe and output are
retained in `r32-completion-parity-shard-reproduction.{py,json}`.

A further HTTP promotion test failed before reaching its digest assertions:
the synthetic multi-candidate campaign omitted its locked selection rule. The
fixture now declares the canonical rule; all four endpoint regressions pass in
2.00 seconds, including rejection of a forged on-disk digest and parameter-growth
charging. Production promotion and digest guards are unchanged.

### Arithmetic integration checkpoint

The canonical arithmetic adapter now has 22 passing focused tests (2.85
seconds), including actual tiny training/decode/scoring, invalid pack and symbol
rejection, terminal-witness replay, proof exhaustion, and a singleton-prefix
zero-forward assertion. Both benchmark arms use constrained lexer decoding;
bindings expand to equivalent root ASTs without restoring compositional or
unconstrained fallback. This is fixture integration evidence, not a scientific
quality result or a completed source gate.

Review still requires the model-build factory to propagate grammar identity,
resolution of one package-principle regression, and adversarial completion-domain
review. Module ceilings are restored: pack.py is 1,125 lines; twotower.py is
16,634 versus its 16,641 ceiling; the models package is 49,595 versus 49,602.
The final quality ratchet must record these decreases once the implementation
is stable. The current snapshot is not frozen or release-authorized.

### Unattended delivery preparation checkpoint

The host preparation recipe now constructs the complete strict adapter with
explicit direct Codex transport, separate mode-locked writer/reader executables,
trusted configuration digest, finite grants, and controller/source/runtime pins.
Eight preparation tests pass, plus the canonical lost-commit-response regression.
No final source identities were fabricated and no remote mutation occurred.

Parent review found a remaining executable-base issue: the retained R31/R32/R33
source directories reconstruct their published connector trees but share ambient
local Git HEAD `e9a6de0c6c52e4326252e2c6c76d1abad423d690`. The preparation
recipe still requires the fault-base checkout HEAD to equal its published source
commit. This mismatch requires an authenticated base materialization or an
existing verified-provenance path; neither local ref surgery nor changing the
expected commit can establish authority. Final adapter preparation remains open.

A lost commit response without a recoverable SHA also remains honestly pending:
the installed connector cannot pin commit timestamps, so tree/parent/message
alone cannot reconstruct the missing object identity. Existing independent
marker search and exact readback may recover it; an ambiguous search result
must not trigger a duplicate mutation.

Two Muon resume regressions pass in 3.57 seconds after removing invalid
unconstrained test overrides and reporting optimizer-fingerprint mismatch before
generic recipe mismatch. The optimizer algorithm and compatibility acceptance
conditions are unchanged; cross-optimizer resume still fails closed.
`harness.model_build.train` is v45 and now owns the extracted resume validator.

Two distillation trace regressions pass in 5.88 seconds. Fixtures use opaque
markers and constrained generation. The corruption probe now mutates an actual
in-range committed cell that was not remasked, and explicitly asserts that a
mutation occurred before checking replay rejection. Trace storage and replay
production code are unchanged. Ruff and diff checks pass. SFF source cleanup
also restores its 673-line ceiling by reusing loaded campaign/scorer data;
its 21 focused tests remain passing.

The preparation base mismatch is resolved by reusing canonical immutable source
copying and `private_git_snapshot`. A new private verifier base contains only
hash-authenticated blobs/tree/commit reconstructed from connector-observed
metadata. Its commit SHA must exactly match the published source; shared Git
metadata and all original source directories remain unchanged. This creates no
new commit and performs no publication. Eight preparation and five canonical
snapshot tests pass, including mismatched ambient HEAD and unchanged original
HEAD. Final pinned preparation and real delivery still await the completed
source candidate.

Timeout regressions still called the removed retirement helper and expected an
operational failure to exhaust a scientific hypothesis. Current production
explicitly rejects that inference. Updated coverage exercises the actual repair
and frozen-replay action producer, rejects recording a timeout as scientific
null, and verifies a persisted historical timeout entry cannot block the exact
arm. A genuine null still blocks the same data-generation identity and permits
a changed identity; reading either historical entry leaves its bytes intact.
All four focused regressions pass in 2.41 seconds. No production timeout policy,
scientific gate, or ledger was changed to satisfy obsolete expectations.

### Controller composition validation checkpoint

Six fresh security/import tests pass in 24.83 seconds, including hostile
`sitecustomize`, prior-lease launch-receipt replay, copied journals, required
split-controller pins, and the nested CLI. The fixture complexity finding is
resolved. These checks do not yet close two remaining composed runs.

The two-repair supervisor regression reached both successor activations and
wrote both evaluator rows, but its 170.20-second bounded invocation exited 124.
Its final preservation and severed-boundary assertions are unverified; this is
not passing evidence. A separate unchanged-workload controller-upgrade test woke
and executed the original driver, then failed because it expected exit 0/2
instead of the actual typed pending exit 10. The corrected assertion still
requires a successful bounded rerun. Remaining work is reducing redundant
validation overhead without caching stale authority across operation boundaries,
then rerunning both scenarios within the canonical shard allowance.

Arithmetic review found a certification defect: valid decimal input could render
as exponent notation rejected by the grammar. Numeric serialization now emits
grammar-safe fixed-point decimal, and finalization revalidates the returned
canonical text. Tiny, large, and negative-expression counterexamples preserve
evaluated values. The model factory now binds grammar identity to persisted
training-pack identity; data admission no longer imports model code.

Final arithmetic ownership is `dsl/arith_sketch.py` and `dsl/arith_completion.py`,
with the legacy pack adapter delegating to that canonical owner. ADP/SDP/SAP
counts return to 56/78/10. Parent reran all three focused files with normal
repository conftest enabled: 58 tests pass in 8.59 seconds. All ten source/test
hashes match the agent handoff. Independent final review and the complete signed
gate remain required; these results establish constrained fixture integration,
not arithmetic quality or shipment.

Independent final arithmetic re-review accepts the numeric correction. All ten
handoff hashes match before and after verification; the original tiny-decimal
counterexample and negative/large numeric round trips remain grammar-valid with
preserved values. No remaining finding was identified within this review scope.
This closes the focused P1 finding, not whole-repository release acceptance.

The cycle-stamp regression used an incomplete handoff fixture without its frozen
creation time. Repairing the fixture exposed an unsafe broad exception handler:
version provenance failures silently emitted unstamped observations. Closeout now
propagates that failure instead of publishing an incomparable result. Three
focused tests pass in 1.67 seconds, covering frozen timestamp, a real ledger
partition, and unavailable-registry rejection. Ruff passes. Separate review is
checking whether split-controller closeout attributes evaluation components to
the workload source rather than the newer controller; that boundary is not yet
claimed complete.

Split-controller provenance review confirms that closeout currently rebuilds
measurement versions from the imported controller registry, even though the
evaluator retains workload stamps and diagnostic delivery separates measurement
source from controller provenance. R31 and R33 currently share all four eval-key
component values, so no partition drift was demonstrated for these retained
measurements. The construction is still wrong: a later controller-only version
change could repartition unchanged measurements. Correction now binds closeout
to authenticated original measurement evidence and verifies comparison-component
agreement; missing workload authority must not fall back to controller versions.

Parent baseline audit rejected two upward changes introduced during concurrent
validation: the continuous driver complexity ceiling had moved from 89 to 91,
and its module ceiling from 12,184 to 12,196 lines. Both original ceilings are
restored; legitimate decreases elsewhere are retained. Current code must meet
the original limits through implementation changes. No release evidence may
rely on the raised values. The exact rejected deltas are retained in
`r33-baseline-upward-change-rejection.json`.

Controller compatibility now passes 64 focused tests in 33.73 seconds. The
corrected unchanged-workload supervisor regression passes in 77.96 seconds,
proving fresh controller execution can wake the retained operation without
rewriting its workload source. The two-repair regression still times out at
120.06 seconds and remains incomplete. Optimization is restricted to reusing a
freshly computed environment within the same validation boundary; publication
fences and later revalidation are retained, with no cross-operation cache.


### R33 retained regression closure, September 26

The train-data fixture and prompt-contract checks passed seven tests in 19.65 seconds. Preference constraint-debt checks passed five tests in 1.98 seconds. These repairs retain the constrained-output and contamination gates. The prior fixture overlapped four of eight evaluation n-grams; the replacement prompt removes that fixture contamination without relaxing admission. The retained quality report admitted two human-curated records with two unique roots, rejected none, and emitted no warnings, recommendations, or experiment candidates. The report SHA-256 is `a56f4bea07f00d511b7f1297adaa594fa09553ada84ead4e73458b1e1625b4fe`; the feedback SHA-256 is `0e642434c048fdf57a744a3e2693dc2f2ecd263950bf020f1ba826e3ad5f8268`.

The RSP-003 producer now declares the required canonical selection rule. Its fixture test passed in 8.19 seconds on CPU with one cold trial per arm and two warm trials: seven cold arms and one aggregated warm result completed, parity was 9/9, and the recommendation was `inconclusive_no_cold_gain`. This is fixture evidence with `promotion=false`, not model-quality or ship evidence. The fixture JSON SHA-256 is `6c782b3c103234638656e622afba36bdbd64ee79d339f10679c910b7ce9c4ef2`.

The VCE-009, RSP-006, and PCT-008 selection-rule repairs passed 27 tests in 33.59 seconds after normal formatting and removal of redundant budget construction. The missing-checkpoint regression fixture now uses an opaque placeholder; its exact test passed in 1.90 seconds and Ruff passed. No checkpoint guard was changed.

The controller composition regression completed two successive repairs and fresh supervisor starts in 87.47 seconds under its unchanged 120-second allowance. The unchanged-workload composition regression passed in 80.82 seconds. These are bounded integration tests, not live provider repair or completion of the retained scientific campaign. Documentation provenance checks passed 21 tests in 6.04 seconds. Missing measurement provenance remains a typed pending dependency; it does not produce an unstamped document or silently adopt the controller registry.


The parent version-registry check passed against the ambient checkout baseline (148 changed files, 20 components). This is not final release verification. A complete quality analysis initially found no regressions and four unrecorded decreases. The subsequent guarded baseline write detected concurrent Revmath module growth from 1,083 to 1,085 lines and refused to write. The baseline remains unchanged pending that owner's size-preserving fix. The parent uses one fresh analysis for both regression rejection and recording decreases; an unchecked baseline refresh is not acceptance evidence.

Final controller compatibility coverage completed in two bounded invocations: 53 tests passed in 59.23 seconds and 13 passed in 52.09 seconds. Both retained the 120-second allowance. This completes 66 focused compatibility tests after the startup consolidation, including the previously failing documentation wait boundary. The earlier combined invocation timed out and is not passing evidence. Live provider repair and the original scientific campaign remain separate, unfinished acceptance obligations.

### Quality-update guard

The canonical quality verifier now evaluates its fresh measurements before writing an existing baseline. A regression returns exit 1 without changing baseline bytes, even when another metric improves. Initial bootstrap and legitimate decreases remain supported. A changed Ruff version retains the documented complexity-only rebasing behavior; it does not excuse size or dependency regressions. The agent recorded 39 passing ratchet tests in 0.22 seconds and confirmed the prior implementation fails the new regression test. No baseline was refreshed by that implementation task.

The parent independently reran all 39 quality-ratchet tests with normal repository fixtures: 39 passed in 1.17 seconds. The Revmath missing-Lean probe now uses a temporary directory outside the candidate checkout and cleans it up. Three focused tests passed in 13.25 seconds, including frozen corpus replay and byte-identical rebuild. The production module remains at its 1,083-line ceiling. Canonical read-only isolation is being checked separately; the local result alone does not prove that boundary.

Canonical isolated Revmath validation completed successfully: all three regressions passed in 13.57 seconds with the candidate mounted read-only; the bounded worker completed in 29.46 seconds. This directly covers the original EROFS execution boundary. After the fix, the guarded quality update recorded decreases successfully. The parent audited every numeric baseline value against R32: no increases, ten decreases, and the tighter intermediate giant-test ceiling of 13,209 lines remains intact. These checks do not replace the final full merge gate.

The post-update quality check passed with no regressions. The version-stamp check passed against the ambient checkout (152 changed files, 21 components). Independent review nevertheless found unfinished integration: documentation must authenticate the delivery projection, verify complete stamps against the locked workload, and distinguish failed evaluation stages. A separate launch audit found that adding repair configuration to an original null-config request could allocate a different driver activity instead of preserving its grant. These are implementation defects under repair, not waived acceptance conditions. The retained scientific grant remains 11 attempts and 2,300.157002612 seconds; no replacement budget is authorized by this review.

A scratch-instrumented canonical replay of the original 21-node shard completed in 39.48 seconds. Grammar parity passed on this attempt; the retained R32 semantic-factor selection-rule test failed as expected from its old source. This attempt does not resolve the intermittent parity defect. Its trace is a passing comparison for further diagnosis, not evidence that the original failure can be ignored.

Additional parent static feedback passed repository policy and canonical evidence-ledger byte reconstruction. GitHub connector readback confirms both PRs remain open and unmerged at their previously recorded heads. PR #1785 now explicitly distinguishes the unpublished R33 focused results from R32 remote-source evidence and lists the remaining integration defects. No local Git publication or merge substituted for the connector.

A second fresh isolated diagnostic reproduced the grammar parity failure in 27.38 seconds. At prefix length five and remaining budget eight, both implementations reported complete/witness-pruned domains, but the packed result omitted component candidate `(45, 6)` whose reference terminal witness was `(45, 6, 8, 9, 7, 9, 7, 2)`. The earlier failure at the empty prefix is therefore not the only manifestation. The trace is retained as `parity-diagnostic-r32/r32-shard-attempt-002.json`; further instrumentation must retain the failing prefix rather than exhaust its trace limit at the initial prefix. No production guard has been relaxed.

The three documentation review findings now have an implementation and mutation coverage: the delivery projection must match its committed artifact, every complete stamp binds to the immutable original workload, and evaluation stages require a successful exit or the existing authenticated gate-rejection contract. The final focused command passed 30 tests in 29.93 seconds, with 11 unrelated tests deselected; this is not full-suite evidence. The producer uses the existing artifact reader and gate-rejection predicate. Independent review remains required before source freeze.

Independent closeout review found no remaining substantive defects in the authenticated host-configuration receipt or the revised documentation provenance boundary. The reviewer independently passed five controller authority tests in 6.98 seconds and twelve documentation mutation tests in 12.08 seconds. The parent reran the quality gate and version-stamp check after those edits; both passed, and whitespace validation was clean. A source-selection audit found that a future source-only quality-verifier change would omit its tests under `tests/test_quality`; an explicit canonical mapping and regression are being added. The current combined patch already selects the changed regression file, so this is a future source-owned coverage hole rather than evidence of a completed full gate.

The apparent retained-campaign environment mismatch was traced to probe ordering: importing `slm_training` sets `ORT_DISABLE_TELEMETRY=1`. Measuring after the same initialization reproduces the original environment digest `cf6d72c2ed3942b17808f511f81c91c782c1f3d1e6ee3902d31e63c5c03090b6`; no environment successor, drift exception, or grant reset was required. This is a configuration-identity diagnosis, not a resumed scientific cycle. The ordinary-supervisor null-to-configured repair regression passed in 57.67 seconds while retaining the original request, activity, grant, and prior charges.

The source-selection hole is closed: a quality-verifier source change selects `tests/test_quality`, and unrelated explicitly changed tests remain in the selected union. The focused selection suite passed 27 tests in 0.77 seconds; source module size stayed at 893 lines and the existing test module shrank from 507 to 505 lines.

The retained science runtime uses its original native Node 22 layout, not the flattened verifier mount that lost SDK dependencies. Its original SDK resolved and parsed YAML successfully and imported 315 exports. The new controller verification SDK remains separate; changing the original science SDK path would alter the bound workload identity and was not done. These read-only checks invoked no provider and changed no scientific source.

The next parity hypothesis follows a concrete source mismatch: the packed witness bound passes a live Lark numeric state ID into an adapter indexed by canonical state colors. The artifact builder explicitly canonicalizes because Lark IDs vary with process set iteration. Lockstep language certification does not establish numerical equality between those state spaces. This can explain process-dependent negative pruning; the owner is validating the mapping and a state-renumbering regression before changing production behavior. This paragraph records a hypothesis, not a completed fix.

### Retained RICO fixture and unknown-cause recovery regressions

The RICO fixture test passed in 5.93 seconds on September 27 at 00:00:24 UTC. Its 80 candidates produce two admitted records and 78 accounted rejections: 56 duplicates, 20 evaluation overlaps, and two verification failures. The regression now asserts complete accounting and the required low-yield/leakage warnings and recommendations instead of assuming four survivors. No synthesis, deduplication, contamination, or verification gate changed. This fixture exposes poor yield; it is not a repaired production corpus or training-readiness claim. The measured JSON retains the quality report and feedback. The rejection ledger SHA-256 is `19b686d9c0d2e87c751703a8cc60b1cab1a80edf00881f0cb01d88b8f37a58d6`.

Five recovery/park tests passed in 1.88 seconds after correcting the unknown-cause expectation: an unclassified deficit requires bounded harness diagnosis, not an invented data-rebuild cause. Original timed-out R32 shards remain incomplete evidence; the focused successful invocations are recorded separately.

The downstream delivery-consumer defect was reproduced before repair: a fresh worker had no repository publication context despite a bound receipt. The fixed consumer authenticates the live receipt and configuration bytes before passing a configuration view only to publication scope; the logical request remains unchanged. The complete post-fix 39-test compatibility suite passed in 31.11 seconds, including fresh-worker publication-context assertions and configuration/receipt/stale-lease/copied-journal rejection. Repeated lock-wait timeouts before execution were not counted as test results. The parent reserved a slot by stopping future R32 timer activations, preserved its running worker, and restored the minute timer immediately after the successful suite.

The prepublication inventory reconstructs the original R32 tree exactly and currently sees 9,931 successor source paths, 90 changed paths, 20 additions, and no deletions. This inventory is explicitly unfrozen while the grammar fix is in progress. A review against R32, rather than the ambient local HEAD, caught a missing ownership-map component history update for the quality-test selection change; that component is now v30. The broader ambient-HEAD version check had passed and was insufficient to establish this successor-specific obligation.

The parent independently confirmed the state-namespace defect by walking the live parser and the certified adapter in lockstep over the existing certification corpus. Certification reports equal behavior, but 30 positions produce different bounds when the live state number is incorrectly used as an adapter index. Before a TextContent string, live state 114 corresponds to canonical state 65: the misindexed bound is two while the canonical bound is one. This establishes a concrete implementation bug independently of whether a random process reproduces the parity assertion. The raw probe is retained in the measured JSON. Production correction and regression validation remain pending.

### State-ID correction and actual-worktree validation

The mapping repair initially existed in the dirty root checkout rather than the R33 candidate. The parent detected the mismatch by comparing source bytes, preserved the root edits, and transferred the reviewed three-file change into R33. The existing pure LALR adapter and bound calculation were extracted into `completion_lalr.py` to keep the source-size ratchet; equivalent control-flow simplifications remove the moved complexity violations. No baseline ceiling was raised. The runtime-only map comes from the exact parser reconstruction already used to verify the artifact. Frozen tensor and manifest files remain unchanged, including historical V1 consumer metadata.

The adapter and artifact suites passed 18 tests in 4.99 seconds. Restoring the old kernel lookup in a fresh test process makes the new regression fail (bound zero instead of two); the corrected implementation passes, including a deliberate state-ID offset through the kernel consumer. The exact previously failing 21-node shard then passed in canonical isolation: 21 tests in 27.78 seconds, worker 46.76 seconds. That shard ran before the subsequent equivalent complexity cleanup. Full corpus parity is being checked in bounded groups; one passing shard is not the full parity obligation.

Independent review found no substantive defect in the four-file mapping repair. The checked runtime map covers the exact parser, missing IDs disable the optional bound, and serialized artifact bytes remain unchanged. After the complexity cleanup, all 18 adapter/artifact tests passed again. The combined invocation also passed the three original staged-materialization failures, but its fourth staged legacy-byte test failed and remains under investigation; the combined invocation is not reported as wholly passing. The quality check found no regressions and three recordable decreases: artifact module 883 to 693 lines, its complexity findings six to three, and kernel 676 to 674 lines.

The existing learning-comparison CLI is now registered through the canonical `slm experiments learning-comparison` command, with explicit fixture activation retained and the experiments guide updated. Its focused registry/guide/dispatch suite passed 16 tests in 4.57 seconds. No parallel CLI implementation was introduced.

The full existing parity corpus now passes after the mapping correction: all eleven parameterized programs, every exercised prefix and configured budget, completed in three bounded groups (four tests in 76.11 seconds, four in 48.41 seconds, three in 6.77 seconds). No parity assertion or false-singleton guard was removed. The measured JSON retains each JUnit digest and exact case names. This is focused local validation, not the immutable full merge gate.

Fast static feedback passed 16 of 18 obligations. The extraction moved the adapter declaration, so the canonical ownership map and generated prose were updated to name the actual helper while retaining the existing artifact builder/loader owner; its verifier now passes. The remaining static failure requires canonical extraction of four newly added parameter tables and is being corrected without dropping cases.

The staged legacy-byte mismatch is resolved without changing the expected digest or producer: the parent invocation omitted the approved DESIGN.md bridge path. With both bridge paths matching the canonical runner, all four staged-materialization tests passed in 0.80 seconds in R33. The temporary staged-plan fixture pins generator and validator versions from the active registry; production executable-plan validation remains strict. The earlier root-checkout test result was not used as R33 proof. The missing-bridge run remains documented as invalid validation context rather than a production regression or a reason to update golden bytes.

### Final R33 focused closeout before immutable verification

All eighteen fast/static obligations passed with `release_authorized=false`, as required for local feedback. Canonical parameter-table extraction preserved case identities and function bodies; its 97 affected tests passed in 14.91 seconds. The final ordinary-supervisor two-repair regression, including the completed host-configuration consumer wiring, passed in 77.35 seconds under the unchanged 120-second allowance. The original training prefix was not repeated. The parent restored the minute verifier after the reserved test slot.

These results close the known focused failures and support publishing the successor for independent full verification. They do not establish full release authorization, live Codex repair acceptance, completed retained two-arm science, or merged delivery. Those obligations remain open.

## R33 publication and R34 verification follow-up (2026-09-27)

R33 is published on PR #1785 as `421ab751e5b8e2f6bdbd09691c89417be995cdc6`, tree `c6ad9d386ce53fdcde5ce248169787ee3e89d061`. All 9,934 source paths match that connector tree. Canonical controller materialization reconstructs the connector commit independently and reports clean provenance with source digest `eb103e74db075e923773ed44737a802aee683df68be827b95511166a2a0177b7`. This proves the controller source binding, not main membership or completed scientific acceptance.

The new bounded verifier has completed 16 of 18 isolated static obligations. Test collection and full verification remain pending. R32 journals are retained without transferring their passes to R33.

An authenticated R32 failure audit identified 26 failures and 14 skips without confirmed R33 coverage. An independent R34 workspace preserves the published R33 source while repairing these obligations. The first focused CPU regression group passed 98 tests and failed two in 8.55 seconds. Both remaining failures read the historical `champion/last.pt` location directly after bundle publication; the canonical publication resolver must be used. These failures are retained in `r34-first-repair-regressions.xml`; they are not waived. This run creates no model checkpoint and proves no capability improvement.

The corrected group passes **100 tests in 7.72 seconds**, including repair grant accounting/dispatch, champion publication, paired policy, and the locked confirm pair. Before/after byte assertions now resolve the canonical champion bundle. The earlier two failures remain recorded. This focused result does not replace complete source verification.

Independent read-only reviews found no substantive assertion loss in the fixture repairs. Historical exposure, required sidecars, canonical checkpoint bytes, paired-root independence, invalid-grammar rejection, and zero-charge rejection of an unscoped repair remain covered. Canonical extraction checks passed for all four changed test files without resource changes.

### Retained science resumed through the published controller

A real bounded ordinary-supervisor invocation using published R33 completed its preflight, reconciled the previous readiness wait, and entered cycle 4. It resumed the **same** `supervisor-1-driver-c27384e5916102e8` activity with the original request and 2,700-second/14-attempt grant. The driver yielded at a durable `driver_cycle_checkpoint` cursor after 69.8076 charged seconds. The overall invocation returned typed pending (exit 10) after 164.33 seconds, before its 170-second interrupt bound; it was not a timeout. Evidence is retained under `outputs/autonomy-integration-20260921/r33-retained-science-run-mqq4r2xj/`.

The locked recipe remains CPU, 65,826 parameters per arm, six updates per arm, and six public fixture cases. This proves genuine original-operation continuation after the controller replacement. It does not yet prove completed paired evaluation, a live repair-provider round trip, promotion, or shipment. Further cursor and artifact inspection determines the next bounded continuation; grants and observations are not reset.

### Control evaluation completed through ordinary supervision

The next retained invocation reached cycle 5 and completed the control's **six of six decoded cases**, with zero pending cases, parse rate **1.0**, and evaluation NLL **23.8947078984**. The actual scoreboard reports both `measurement_complete=true` and `publication_complete=true`; the pinned AgentV runner reports zero execution errors and retains its JSONL/spec/result bundle. Public fixture ship criteria correctly fail seven of nine assertions, including insufficient sample size. This is a completed diagnostic measurement, not a ship-qualified model.

The same original driver yielded at its next durable cursor after 87.8042 worker seconds; the supervisor returned typed pending (exit 10). Candidate and paired completion remain open. The successful control score is retained with its SHA-256 and AgentV artifact paths in the JSON companion. No training restart or grant replacement was used.

### Candidate checkpoints retained after bounded interruption

The next ordinary-supervisor invocation reached cycle 6 and created both candidate training chunks. The final summary records six updates, exact same-environment continuation from step three, and training loss 37.8932647705. The final immutable bundle is `2493d3e3b76228304a9bc66db2943683c0d001e96ac5a82e42074ccbd429c52e`; checkpoint SHA-256 is `0a125ff5e1ebcf4b456e8f0d432ca03a187eb298f045b77b8a13c81e62d27313`. The earlier three-update bundle is retained too.

The enclosing invocation hit its interrupt limit (exit 124). It is not successful end-to-end acceptance evidence. Committed checkpoint artifacts establish retained training progress only; canonical lease/cursor reconciliation must precede continuation. Candidate evaluation and paired completion remain pending. The locked recipe remains CPU scratch, 65,826 parameters, eight records, six updates in chunks of three, seed 7301, candidate learning rate 0.0006. Local no-sync is deliberate for this fixture diagnostic. No promotion or capability improvement is claimed.

Post-timeout inspection found a concrete settlement defect: the worker had yielded after both candidate training chunks committed, but the enclosing deadline interrupted settlement and the original activity became terminal `cancelled` (sequence 28). Six attempts and 696.140410079 seconds remain charged. No controller or worker survives. Arithmetic remaining budget is not runnable authorization. The successor must distinguish bounded host interruption from explicit user cancellation and recover the authorized interrupted activity through canonical events without restarting training or resetting charges. This integration repair is now active; repeated blind retries are not a remedy.

### Additional regression findings retained

The export/formal/deferred-arm/macro group passed 57 tests and failed one in 9.10 seconds. The current-contract export exposed a real ONNX consumer defect: it loads a DSL v5 sidecar through the legacy v2 tokenizer loader. Repair must load the checkpoint-bound output and context vocabularies through the canonical loader and reject contradictory metadata; reverting to an obsolete fixture would hide the defect.

The source-family group passed seven tests and failed one in 1.31 seconds. Inspection of both `quality_report.json`, `rejected.jsonl`, and `synthesis_feedback.json` proved that root `t1` was correctly rejected for n-gram overlap (0.5, size 8) with `e827_target_slots_only_v4/adversarial`. Both builds rejected the root and a layout augmentation for contamination; the capped build also rejected one child for exposure. Thus ordering alone cannot repair this fixture. The successor fixture must exercise root retention with an admitted root while retaining active decontamination and contaminated-root rejection. Generated leakage recommendations and synthetic-anchor deficit findings are retained in the JSON as diagnostic fixture evidence, not production-data improvement claims.

ONNX successor validation passes **six tests in 3.77 seconds**. Current-contract output and context vocabularies load through the canonical sidecar loader, context encoding uses the encoder vocabulary, and missing or contradictory vocabulary metadata fails closed. Constrained generation coverage remains. Existing Torch tracer warnings about Python boolean shape checks remain visible; these focused export checks do not establish arbitrary-shape ONNX equivalence or full release readiness.

The original timeout-recovery regression exposed a second-timeout edge and a cancelled-request configuration bypass. The controller now follows authenticated timeout successors while preserving the original grant, and it matches the logical request independently of host repair/delivery configuration. A configuration change cannot turn a cancelled request into a fresh activity; configuration rebinding still requires separate pinned host authority. The final parent suite passed **17 tests in 57.37 seconds**, including fresh-supervisor continuation without repeating committed training. Hooke's independent review closed the configuration-bypass finding against the frozen source before this run.

The preview runtime now exposes Playwright only through explicit, source-lock-matched Node/package and browser grants. The canonical isolated preview test passed again after closing an external-`node_modules`-symlink authority expansion: **1 passed in 45.17 seconds**, 56.505 seconds total workload, using Chromium headless-shell revision 1228. The composed runtime-root helper rejects external symlinks before resolving or granting them; Hooke independently reviewed the exact fix. Focused runtime, isolation, resumption, and resumable-workload regressions passed **50 tests; 9 were skipped by existing conditions**. The source-family mutation check exercised the actual public package consumer and failed as expected when root-first ordering was removed, selecting child `a_syn_0` instead of source root `z`. A wrong-lock fixture initially failed at an earlier package-consistency check; its corrected, internally consistent lockfile regression now passes.

These results close focused integration defects. They do not establish the full exact-source merge gate, retained candidate evaluation, real Codex repair acceptance, or release authorization. The one-minute R33 verifier timer is active again; its evidence remains bound to published R33, not the editable R34 candidate.

After extracting the shared timeout contracts from the autoresearch runtime, the complete timeout suite passed again: **17 passed in 68.45 seconds**. Independent review confirmed byte-identical contract function ASTs and preserved runtime re-exports. The Ruff 0.15.22 complexity gate now passes for the changed modules. The guarded code-quality update lowered four existing module-line ceilings; the subsequent code-quality check passed with no regressions or baseline increases.

R34's focused integration regression set passed **195 tests, with seven existing conditional skips, in 44.57 seconds**. The canonical fast/static merge gate then passed **18/18 obligations** against base `e56115002dd3f2872d2d24d1a5ef85ff36ce49cf`. This fast gate does not collect or run the full source-owned test suite. Full exact-source verification remains open.


## Cancelled predecessor environment transition (2026-09-29)

A verified repair may continue a cancelled operation after the host environment changes. Keep the predecessor environment digest unchanged in the original activity. Record the newly measured digest as the successor environment in the delivered activation artifact, then require replay to match that successor digest. Reject environment changes while the predecessor remains active, waiting, or failed. The successor consumes only the original grant remainder and adds no scientific replicate. This transition preserves lineage; it does not establish that measurements across the two environments are confirmatory.
