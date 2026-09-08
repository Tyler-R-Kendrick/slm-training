# Local release verification — 2026-09-07

> Recovery annotation: historical report restored after filesystem loss. Tests and
> referenced run artifacts have not been revalidated in the reconstructed workspace.
> This is not current release or autonomy authorization.
> Historical content SHA256: `67449452449946aad88457f7ba94c992a342106f6cd45208879f5a2edb9d7653`.


Implementation base: `e0eca9f9910244ecc20eb480d363f1852417b599`.
Audit baseline: `2c4e2df3f9b9bfb6efe13e88108a57ea2ac9f64c`.
The two existing validation entrypoints have no diff between those commits.
This is operational verification evidence; no model, promotion, or ship claim.

## Implemented obligations

| Leaf | Owner / behavior | Executed regression |
| --- | --- | --- |
| VAL-SELECT | `check_changed.select_tests`: source/resource/changed-test union; unknown changes force full scope even beside known targets; locks are load-bearing; deleted tests retain their directory owner; rename discovery includes both paths | `test_unknown_alongside_known_target_is_conservative`, `test_actual_source_resource_and_changed_test_node_union`, Hypothesis monotonicity test |
| VAL-SHARD | `merge_verification`: explicit `addopts=` and empty marker expression; real node collection; duration-weighted shards; canonical bounded child runner; interruption never passes; timeout batches split; private journal resumes compatible work | Real marked/empty/skipped/timeout child tests; `test_real_workload_resumes_verified_collection_without_duplicate_tests` |
| VAL-CACHE | `merge_verification_evidence`: full nonignored candidate tree including untracked/deleted/mode/link changes; base tree, runner/selector, effective options, executable, installed metadata and file-stat identities; external private signed journal; old schemas refused | Fixture/lock/runner identity changes; dependency edit with unchanged metadata; tampered cache and candidate-local cache refusal |
| VAL-AUTHORIZE | Canonical `verify_merge_ready` invokes the complete collected-node path. Fast checks explicitly have no release authority. Zero, missing, duplicate, skipped, xfailed, failed or unfinished node phases cannot complete. Independent scope/isolation/original-reproducer receipt remains required for autonomous release | Exact node/phase tests and `test_local_evidence_is_not_independent_release_authority` |

The historical fast selector remains intentionally narrow for feedback. It is
no longer called by the merge gate. Importing the actual complete baseline
`check_changed.py` from Git reproduced the unknown-file defect:
`[unknown/tool.ts, tests/test_dsl/test_parser.py]` selected just the known test;
the candidate selects `tests`. No manually transcribed helper was tested.

## Stable interfaces

`scripts.merge_verification_evidence` exposes:

- `source_identity(root: Path) -> str`
- `environment_identity() -> dict`
- `digest(value: object) -> str`
- `validate_workload(payload, request_digest=..., expected=...) -> list[str]`
- `authorize_release(evidence, expected_identity=..., independent_verification=...) -> bool`

`run_release_gate(steps, root=..., base_ref=..., state_dir=...,
step_seconds=..., run_step=...)` is the resumable executor used by the canonical
merge CLI. Its summary carries the binding, collected/completed/pending node
IDs, per-command/per-workload outcomes, measured elapsed seconds, and
`verification_complete`. It always labels ordinary local execution
`evidence_class=local_process`, `release_authorized=false`.

The independent verification receipt must arrive through the controller's
authenticated channel and bind `verification_identity`, `evidence_sha256`,
`scope_passed`, `original_reproducer_passed`, and `isolation_enforced`.
The authorization helper checks those bindings; it does not authenticate an
arbitrary supplied dictionary. INTEGRATOR/ISOLATION must never deserialize
worker-authored output directly as that trusted receipt.

## Commands and limits

Actual parser-checked entrypoint:

```sh
python -m scripts.verify_merge_ready --base-ref <base> --state-dir <private-controller-directory> --json
python -m scripts.verify_merge_ready --fast --json
python -m scripts.check_changed --list
```

Full mode returns 0 for complete local verification, 10 for pending bounded
work, and 1 for failure. Repeating the same full command resumes compatible
journal state. Different content or environment identities create a successor
cache namespace. `--max-step-seconds` defaults to 120 and cannot exceed the
canonical 170-second interrupt deadline; the process-group kill grace is 10.
The whole invocation also reserves finalization time inside the cap.

If `--state-dir` is omitted, the command creates a per-user/per-checkout private
temporary cache. Configure a durable controller-owned directory for restart
survival across temporary-directory cleanup. The candidate tree cannot contain
the cache. No service was installed or started.

Executed regression command (Python 3.12.3, CPU, ordinary local processes):
**62 passed, zero failures/errors/skips, 33.44 seconds**. Machine-readable
owner digests, runtime binding, and case identities are in
[`autonomy-validation-20260907.json`](autonomy-validation-20260907.json).

```sh
timeout --signal=INT --kill-after=10s 170s \
  /home/codex/repos/slm-training/.venv/bin/python -m pytest -o addopts= -q \
  tests/test_scripts/test_check_changed.py \
  tests/test_scripts/test_merge_ready_gate.py \
  tests/test_scripts/test_merge_verification.py \
  --junitxml=outputs/runs/autonomy-validation/validation-regressions.xml
```

Invocation-local `PYTHONPATH` includes this checkout's `src` and
`outputs/runs/autonomy-validation/dependencies`; that directory contains
Hypothesis 6.167.1 and sortedcontainers 2.4.0 within the existing declared
dependency bounds. The prescribed shared virtualenv lacked Hypothesis. No
dependency pins or shared environment were changed. `PYTEST_ADDOPTS` was empty;
bytecode, Ruff, and Hypothesis caches were invocation-local.

## Evidence limits and integration handoff

The child-process tests are real pytest executions. The restart test injects a
voluntary yield and a synthetic base lookup in a disposable uncommitted Git
repository; collection, source hashing, signatures, cache re-open and subsequent
pytest execution are real. It is not a hostile sandbox or real agent test.

A private directory or HMAC under one OS user is not isolation. The candidate
and test workload must not see the issuer key, journal, controller or independent
verifier when these receipts are used for automatic repair activation. Installed
dependency file-stat binding detects ordinary mutation, not privileged clock or
filesystem rollback; independently isolated verification must additionally pin
its immutable runtime. This assignment does not establish those external
authority facts with these local tests.

No full-repository release authorization is claimed. The shared candidate is
being edited by other swarms; the code-quality run reported integration-wide
regressions outside this packet. VALIDATION-owned files had no quality increases.
INTEGRATOR must bump `governance.ownership_map` for `check_changed.py`, compose
new verifier-file ownership/version coverage, and lower the quality ceilings for
the shrinking selector and its existing test module. No shared registry,
scientific threshold, frozen suite, theorem, model or decoder was edited here.

## Restored candidate verification and boundary hardening

The sections above are preserved historical evidence. The current authoring
workspace is `/home/codex/repos/slm-training/outputs/workspaces/autonomy-candidate`.
Fresh, surviving evidence is in
[`autonomy-validation-restored-20260907.json`](autonomy-validation-restored-20260907.json),
including exact domain digests, JUnit case identities, earlier failed attempts,
version stamps and the read-only integrated quality snapshot.

The complete validation-domain command below passed **86 tests in 30.87s**,
with zero skips, failures or errors. Its 11 source/test/resource digests were
unchanged across execution. This is not a full-repository release receipt:
the concurrent integrated candidate requires the entire `tests` target, and
the latest whole-tree quality follow-up still had 24 regressions. No baseline
was raised or updated by VALIDATION.

```sh
rtk proxy timeout -s INT -k 10 170 env \
  PYTHONPATH=src:outputs/runs/autonomy-validation/dependencies \
  /home/codex/repos/slm-training/.venv/bin/python -m pytest -o addopts= \
  tests/test_scripts/test_check_changed.py \
  tests/test_scripts/test_merge_ready_gate.py \
  tests/test_scripts/test_merge_verification.py \
  tests/test_scripts/test_merge_verification_isolation.py -q \
  --junitxml=outputs/runs/autonomy-validation/domain-final-3.xml
```

### Current default and migration

`verify_merge_ready` now requires the existing Bubblewrap boundary for full
verification; a missing capability returns `waiting_capability`, the failed
probe, and `isolated_verifier_available` as its unblock predicate. The explicit
`--local-feedback` option runs same-user development checks only. `--fast`
remains static feedback only. Neither mode grants autonomous release authority.
`merge_verification_binding/v2` includes isolation mode, approved runtime roots,
runtime identity and a three-attempt obligation limit, so old bindings cannot
silently acquire the new claim class. Zero, skipped, xfailed, incomplete or
quarantined workload evidence is never reusable passing proof. Source/runtime
drift preserves the charged attempt in a quarantined cache rather than losing
its cost or reusing its pass after a later restoration.

The runner and selection request are read-only mounts. The controller journal,
issuer key, credentials and host home are not mounted. Only a bounded untrusted
result file is writable; the controller validates it after process exit. The
snapshot uses the same Git-bound file universe as `source_identity`, excluding
ignored inputs. Git-dependent static checks receive a new private read-only
metadata copy, without hooks, remotes/config or alternates; existing authoring
refs are never changed. Actual Bubblewrap denial and private-Git tests passed
on this Linux host. The coding tool's inner sandbox could not create the
required network namespace; it refused rather than weakening isolation.

### Parser-checked operation

```sh
python -m scripts.verify_merge_ready --base-ref <base> --state-dir <controller-directory> --json
python -m scripts.verify_merge_ready --runtime-root <approved-python-installation> --json
python -m scripts.verify_merge_ready --local-feedback --json
python -m scripts.verify_merge_ready --fast --json
```

The step default is 120 seconds, maximum 170 plus the canonical 10-second kill
grace; snapshot/private-metadata preparation is charged to that step. The
whole invocation reserves finalization time. Reinvocation continues only
compatible pending work, and an exhausted obligation needs an actual
source/environment repair rather than another unchanged retry.

### Parent handoff

- All VAL leaf owners are reached by the canonical merge CLI; `run_workload`
  remains import-compatible through `scripts.merge_verification` after its
  execution-boundary extraction. Independent original-predicate/scope evidence
  must still arrive through the trusted controller channel, never a candidate
  JSON file or this report.
- Add `merge_verification_isolation.py`, both new verifier test modules and
  their case resource to `ci.local_merge_gate` watched paths. Keep
  `governance.ownership_map` coverage for `check_changed.py`; parent owns all
  registry composition.
- New VAL modules and tests are below 400 lines and have no complexity
  findings. Existing `check_changed.py` retains seven findings (unchanged
  ceiling) and shrank 906→885 lines; its original test shrank 514→507. Record
  those decreases only during the parent's clean ratchet composition.
- Finish integrated quality and the complete collected source/resource test
  universe before release authorization. Scientific thresholds, model/data
  artifacts, services, remote delivery and the stopped loop were not touched.

### Read-only quality follow-up (before 22:42 UTC)

The fresh canonical scan exited 1 with 24 regressions; all dimensions and
ceiling/observed values are in the JSON `quality_followup`. This is a moving-tree
diagnostic, not an immutable release check. The immediate focused recheck
already cleared `autotrain_supervisor_operations.py`; no final whole-tree
clean claim follows from that narrower observation. Storage is down to 1,826
lines with no increased complexity; choice diagnostics also cleared complexity.

| Owner | Remaining scan observations |
| --- | --- |
| INTEGRATOR | continuous complexity 105→108; engine 1796→1798 lines; schemas 1640→1647; continuous tests 13318→13430; chunk-promotion tests 487→559 |
| RECOVERY | `repair_acceptance.verify_repair`: PLR0913, 9 arguments |
| SEARCH | `search_evidence.effect_from_loss_reports`: C901, complexity 11; climb policy 1481→1501; climb-policy tests 1246→1313 |
| MEASUREMENT | denoising NLL 443→448; eval-gate tests 1411→1413; eval-resume tests 504→505 |
| LEARNING | grammar trace 488→496; TwoTower 16642→16646; learning fixture 401 (new-file ceiling 400) |
| Shared tests | harness 5792→5863; hillclimb 855→858; screening sample size 802→803 and its tests 456→459 |
| Package composition | experiments 103189→103928; models 49603→49607; autoresearch 22509 (budget 20000); SDP violations 80→82 |

Continuous HEAD already has 106 findings against ceiling 105. Candidate adds
PLR0913 in `_run_promotion_eval_chunks` (9 arguments), C901 and PLR0915 in
`_run_arm_eval_nll` (12 complexity / 60 statements), and removes one old C901
chunk-runner finding. Inherited line debt is also present in HEAD: schemas
1643, screening sample size 803, its tests 459, climb-policy tests 1299 and
continuous tests 13393. Inherited is attribution, not a waiver or permission
to raise ceilings. Parent retains registry and package-composition ownership.

All 11 validation file hashes and the final 86-test JUnit artifact were
rechecked after the follow-up and still match. The selected integrated target
remains the full `tests` universe; this domain result does not discharge it.

## 2026-09-08 completion follow-up: finite verification and exact repair dependencies

This section is additive. The recovered historical reports above retain their
original evidence class. Execution HEAD was
`e0eca9f9910244ecc20eb480d363f1852417b599`, with deliberate uncommitted candidate
work, against the directive baseline
`2c4e2df3f9b9bfb6efe13e88108a57ea2ac9f64c`. No Git mutations, hosted checks,
external-agent jobs, service activation, model training or promotion occurred.

### Implemented owner interfaces

- `merge_verification.py`: small-budget scheduling uses the actual fresh-pass
  allowance. A short tail defers work without spending a retry. Exhausted work
  cannot starve an independent shard; split children inherit ancestor attempts.
  Training/slow nodes remain explicitly selected with pytest addopts cleared.
- `merge_verification_evidence.py`: cached collection partitions, successful
  exits and all resource charges are validated. Equivalent relative/absolute
  `PYTHONPATH` entries resolve to the same execution-environment identity.
  `authorize_release` remains the separate trusted acceptance owner's surface.
- `merge_verification_summary.py`: default `merge_verification/v2` output is
  compact. It contains `phase_progress` for static, collection and tests;
  `node_counts`; `next_action`; `required_seconds`; bounded samples and recent
  steps; total retained charges; and a digest linking the complete journal.
  `_summary(state, full=True)` explicitly preserves full legacy arrays for API
  callers. Before collection, required test count is unknown, not zero success.
- `merge_verification_controller.py`: the canonical `verify-release` command
  uses existing `ActivityRuntime` and `CampaignStore` jobs. Its explicit finite
  grant bounds automatic same-activity continuation, including real retry
  timers. The child imports the controller-owned runner, not candidate control
  code, and validates the result against the exact authenticated journal.
- `autotrain_verification.py::drain_source_verification(runtime, common,
  log_event)` consumes `source_verification_requested` events and their
  content-bound `source_verification_requests` artifacts. It uses the exact
  dependency activity, base, source, cache, identity and explicit `grant`.
  No total budget or attempt limit is inferred. Each pre-cycle call executes
  at most one bounded pass; missing authority does not consume that pass or
  starve an independent dependency. Parent owns the production pre-cycle hook.

The repair dependency producer must include configured `runtime_roots`; the
interpreter prefix is the canonical default only when none were configured.
`wake_repair` rechecks current binding and authenticated complete evidence,
then calls the existing runtime's exact `WakeCondition` comparison. It never
wakes a running repair during the dependency-write/finish window, never wakes
an obsolete request, and never acknowledges or publishes a repair itself.
ISO must still independently accept the original reproducer and proposed fix.

### Status, budget and migration semantics

`pending` / `resume_verification` means another finite invocation can make
progress. `waiting_repair`, `waiting_budget`, and `waiting_prerequisite` name
different remedies; invalid evidence requests reconciliation. Scheduling does
not parse a generic human-readable failure string. `required_seconds` has units
`workload_interrupt_seconds`: it excludes controller work, kill grace and
finalization. It is a scheduling estimate, not a scientific endpoint.

The binding is `merge_verification_binding/v3`; old binding identities cannot
be reused silently. The invocation's step allowance is no longer scientific
or cache identity: changing it cannot reset evidence or exhausted retries.
Source/base/runtime/environment/selection/runner changes still reject reuse.
Every compact local receipt retains `release_authorized: false`.

The standalone finite controller reserves 35 seconds above each workload
allowance for startup/finalization and enforces the canonical outer cap. Its
CLI accepts workload allowances up to 90 seconds. Source-repair jobs instead
use the explicit supplied `ResourceGrant`, deriving the workload allowance as
`interrupt_seconds - 35`; grants too small for that boundary become a scoped
capability wait. Neither path enlarges the 170-second interrupt plus 10-second
kill-grace policy. No indefinitely restarting wrapper is installed.

### Executed evidence and honest limitations

The initial scheduling/cache suite completed **81 tests in 34.55 seconds,
exit 0**, with normal and complexity lint plus formatting checks passing.
That receipt predates the added controller and dependency-wake work; its exact
five owner hashes are preserved in the matching JSON follow-up.

The expanded six-file suite subsequently completed **105 tests in 66.71
seconds, exit 0**, including real rootless isolated processes. Its JUnit file
is `outputs/runs/autonomy-validation/validation-integrated-20260908.xml`.
Evidence classes include real pytest subprocesses, actual controller leases
and signed-cache resumes, synthetic fixture source enumeration/base selection,
explicit injected yield, and mocked missing-capability probes. None is a real
coding-agent repair or a model-quality result.

The host subsequently exhausted disk space. The added journal-to-wake test
could not complete; no cleanup was performed by this worker. After external
recovery, touched Python files, domain JSON and the successful 105-test XML
parsed correctly. A combined rerun reached the unchanged command cap with
105 tests reported passing but **exit 124**: that is incomplete evidence,
not a passing verification receipt. Its separate XML is
`outputs/runs/autonomy-validation/validation-recovered-20260908.xml`.

The isolated fixture's first bounded pass validly completed the static probe
and deferred collection. An assertion expecting one-pass completion failed
(13 passed, one failed, 22.84 seconds). The test now explicitly resumes the
same locked gate for at most three bounded invocations; no product gate,
threshold, runtime cap or missingness rule was weakened. Subsequent bounded
shard results are appended below and in the JSON rather than replacing these
failed/interrupted observations.

### Parser-checked command surface

Both `python -m scripts.autoresearch verify-release --help` and
`python -m scripts.verify_merge_ready --help` exited zero. The canonical form
below is a configuration template, not a claim that a service was started:

```text
python -m scripts.autoresearch --root CONTROLLER_EVENTS verify-release \
  --source PRIVATE_CANDIDATE --state-dir PRIVATE_CONTROLLER_CACHE \
  --job-id LOCKED_CAMPAIGN --activity-id LOCKED_ACTIVITY \
  --base-ref LOCKED_BASE --identity LOCKED_VERIFICATION_DIGEST \
  --total-seconds EXPLICIT_TOTAL --max-invocations EXPLICIT_COUNT \
  --max-step-seconds EXPLICIT_WORKLOAD_ALLOWANCE --runtime-root APPROVED_RUNTIME
```

Standalone gate exit codes are 0 for completed verification, 10 for resumable
pending work, 20 for typed waits and 1 for invalid/nonpassing evidence. The
finite command similarly returns 10 for a still-runnable/timed retry and 20
for a scoped wait. `--local-feedback` is diagnostic-only and cannot satisfy
ISO's required isolated binding. Full source/resource-aware final-tree test
collection, all required nodes, versions/ownership composition and trusted
repair acceptance remain the parent's integration obligations. The focused
counts in this report do not discharge the approximately 11k-test universe.

### Completed bounded follow-up shards

| Receipt under `outputs/runs/autonomy-validation/` | Passed | Seconds | Exit |
| --- | ---: | ---: | ---: |
| `validation-wake-resume-20260908.xml` | 14 | 42.62 | 0 |
| `validation-scheduling-final-20260908.xml` | 55 | 31.32 | 0 |
| `validation-controller-final-20260908.xml` | 52 | 86.86 | 0 |
| `validation-controller-waits-20260908.xml` | 14 | 51.97 | 0 |

The last three receipts cover **110 distinct test identities**, not 121
independent observations: the final controller rerun overlaps the preceding
shard and adds three typed-wait cases. All have zero failures, errors and skips.
The scheduling shard comprises the merge-verification, scheduling and legacy
merge-ready test files. The 52-test shard comprises isolation, controller and
dependency-wake tests; the final 14-test shard repeats the controller file
after the operational-wait refinement. These are incremental focused receipts,
not an immutable full-candidate release receipt. Handoff owner hashes and
JUnit digests are recorded separately in the matching JSON.

The completed isolated journal fixture demonstrates actual static execution,
pytest collection and test execution, signed compatible resume and exactly one
matching repair wake. Wrong locked identity is rejected before any workload
record. Its Git enumeration/base and tiny selected suite remain explicit
fixture substitutions, not a production full-source selection.

The trusted child uses `python -I` and explicit controller import paths: a real
process test verifies that candidate `sitecustomize.py` cannot execute at
interpreter startup. The controller launcher is included in the verification
binding's rule digests. Typed capability, lock/dependency and environment waits
do not require a nonexistent success journal or become fabricated completion.
Lock contention retries under the existing finite timer/grant; missing
capability remains a capability wait. Normal lint, explicit complexity checks
and formatting checks passed after the final source edits.

Existing file-duration metadata has 138 files, summing to 5,838.56 historical
seconds, of which one file accounts for 4,726.16 seconds. This sparse metadata
does not establish a full-universe duration or guarantee a sufficient grant.
The gate has one journal writer; competing processes targeting the same cache
do not provide parallel execution. Persistent controller state belongs outside
the candidate (for example in the outer repository's `outputs/runs/`), not in
a disposable `/tmp` fixture directory. Parent must drain the frozen full-node
universe and independently authorize the exact completed identity.

### Execution-context capability reconciliation

The same bounded read-only `probe_isolation()` was rechecked on September 8
in both contexts. In the normal Codex sandbox, `/usr/bin/bwrap` reports
`available: false` with `Failed to create NETLINK_ROUTE socket: Operation not
permitted`. Under explicitly authorized host execution it reports
`available: true`, backend `bubblewrap`, executable `/usr/bin/bwrap`.
These observations describe different outer execution permissions, not
contradictory results from the verifier. Earlier successful rootless-isolation
tests used the host context. The final controller may run in that authorized
host context while its candidate workloads remain inside the actual rootless
boundary. `--local-feedback` is not an acceptable substitute. This capability
probe neither authorizes release nor installs/starts a service, paid worker or
remote job; parent owns the final frozen-tree execution and acceptance.

### Bridge environment identity follow-up

Before the final source freeze, `environment_identity()` was corrected to bind
`DESIGN_MD_BRIDGE_CLI` alongside OpenUI and AgentV. All three recorded values
include the actual host path and entrypoint bytes. A regression checks both
same-path content changes and same-content path changes. The scheduling suite
passed **15 tests in 0.72 seconds, exit 0**, with all three declared overrides:

```text
OPENUI_BRIDGE_CLI=/home/codex/repos/slm-training/src/apps/openui_bridge/cli.mjs
DESIGN_MD_BRIDGE_CLI=/home/codex/repos/slm-training/src/apps/design_md_bridge/cli.mjs
AGENTV_RUNNER=/home/codex/repos/slm-training/scripts/run_agentv_eval.mjs
```

Receipt: `outputs/runs/autonomy-validation/validation-bridge-binding-final-20260908.xml`.
Normal and complexity lint passed; the evidence owner remains below 400 lines.
The two successor owner hashes are in the matching JSON, separate from the
earlier handoff. Full verification must recompute its binding after this fix.

Entry-point hashes alone do not prove JS transitive readiness. ISO's isolated
workload wiring must provide approved complete OpenUI/Design-MD dependency
trees and the AgentV SDK dependency tree. `_agentv_runtime()` derives the SDK
root from `AGENTV_RUNNER.parents[1]`; the runner then imports
`SDK_ROOT/node_modules/@agentv/core/dist/index.js`. The approved sandbox layout
must preserve that relationship and all transitive imports. A CLI-only mount,
an `@agentv/core`-only mount or mounting the entire primary checkout is not an
acceptable substitute. This follow-up fixes evidence binding; it does not claim
the ISO-owned complete bridge mount/SDK integration has been executed here.

### Finite canonical CLI and independent JS runtime closure (2026-09-08)

This follow-up supersedes the preceding untested runtime-mount limitation, not
the historical receipts. `merge_verification_runtime.py` now builds the workload
argv for the existing isolated static and pytest runners. Explicit read-only
runtime grants provide the Python environment, Node prefix, complete OpenUI and
Design-MD bridge directories, and the primary **node_modules directory only**.
The primary checkout, Git metadata, controller store, credentials, and host
sockets are not exposed. Only the source-matched AgentV runner is staged; the
960 MB SDK tree is not copied. Inside Bubblewrap, a private scratch layout keeps
the runner's expected `scripts/../node_modules` relationship. Node's explicit
`--preserve-symlinks` retains transitive package resolution through that approved
mount. Candidate-controlled CLI bytes differing from the granted bridge source
are refused. `--require-js-runtime` refuses incomplete explicit grants.

The first real-process regression produced **2 failures / 2 passes, 3.91 s**:
Node realpath resolution had discarded the `node_modules` ancestor. The fixed
adapter passed **4 tests in 15.10 s, exit 0**, including a missing-transitive-package
negative control, actual installed AgentV core import, and both installed bridge
ping protocols inside real Bubblewrap. Receipt:
`outputs/runs/autonomy-validation/validation-js-runtime-preserved-20260908.xml`.
These are SDK import/protocol observations, **not a published model evaluation**.

The canonical `scripts.autoresearch verify-release` parser/handler was exercised
through actual controller leases, subprocesses, authenticated gate cache and
automatic continuation after an injected yield. Restart replay consumed neither
another test invocation nor another resource charge. This fixture uses explicit
local-feedback mode and controlled source selection; it is not an isolated
full-repository release receipt. Combined isolation/controller tests passed
**41 tests in 44.42 s, exit 0**:
`outputs/runs/autonomy-validation/validation-production-cli-20260908.xml`.

For final authoring verification, run the following from the candidate directory
in the explicitly authorized **host controller context**, leaving actual
Bubblewrap isolation enabled. No service is installed or started. These are
finite maximum grants, not a runtime forecast or an assertion that the complete
universe has already run. Freeze source/docs first; changed source or grants
require a new controller job identity, not silent cache rebinding.

```bash
timeout -s INT -k 10 170 env \
  PATH=/home/codex/.nvm/versions/node/v22.23.1/bin:/home/codex/repos/slm-training/.venv/bin:/usr/local/bin:/usr/bin:/bin \
  PYTHONPATH=src:outputs/runs/autonomy-validation/dependencies \
  OPENUI_BRIDGE_CLI=/home/codex/repos/slm-training/src/apps/openui_bridge/cli.mjs \
  DESIGN_MD_BRIDGE_CLI=/home/codex/repos/slm-training/src/apps/design_md_bridge/cli.mjs \
  AGENTV_RUNNER=/home/codex/repos/slm-training/scripts/run_agentv_eval.mjs \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/codex/repos/slm-training/.venv/bin/python -m scripts.autoresearch \
  --root /home/codex/repos/slm-training/outputs/runs/autonomy-final-verification/events \
  verify-release \
  --source /home/codex/repos/slm-training/outputs/workspaces/autonomy-candidate \
  --state-dir /home/codex/repos/slm-training/outputs/runs/autonomy-final-verification/cache \
  --job-id autonomy-final-frozen --activity-id autonomy-final-frozen \
  --base-ref e0eca9f9910244ecc20eb480d363f1852417b599 \
  --total-seconds 14400 --max-invocations 1024 --max-step-seconds 60 \
  --require-js-runtime \
  --runtime-root /home/codex/repos/slm-training/.venv \
  --runtime-root /home/codex/.nvm/versions/node/v22.23.1 \
  --runtime-root /home/codex/repos/slm-training/src/apps/openui_bridge \
  --runtime-root /home/codex/repos/slm-training/src/apps/design_md_bridge \
  --runtime-root /home/codex/repos/slm-training/node_modules \
  --runtime-root /home/codex/repos/slm-training/outputs/workspaces/autonomy-candidate/outputs/runs/autonomy-validation/dependencies
```

Each invocation drains actual pending work until its bounded deadline, completion,
or a typed wait. Repeat the identical invocation to advance the persisted logical
grant; do not launch competing writers against one cache. Exit 0 means completed
verification; 10 means runnable/retry; 20 names a scoped wait. Compact observation
fields include `phase_progress`, `next_action`, `required_seconds`, and charged
activity seconds. Local `release_authorized` stays false; ISO's trusted acceptance
remains separate. The parent owns executing the full frozen collected universe.

One cross-owner consumer issue remains at this observation: Python's
`dsl/design_md/__init__.py` still hardcodes its bridge directory and ignores
`DESIGN_MD_BRIDGE_CLI`. Direct mounted bridge protocol tests do not establish that
this Python consumer uses the mounted override. The parent must compose that
consumer fix and its regression before claiming full Python/JS closure.

### CLI module-ceiling correction (2026-09-08)

The status/ack-action/sync argument declarations moved, unchanged, into
`autoresearch_operations.add_campaign_operations_parser`; existing command
handlers remain the callbacks and no reverse import is introduced. The entrypoint
shrinks from 2,130 to **2,109 lines**, under its 2,115 ceiling; the existing operations
owner is 186 lines. Parser regressions cover defaults, callback identity, repeated
evidence, required fields and mutually exclusive selectors. No scientific or
delivery permission changes were made. The actual quality check reports no
regressions and exits 1 solely for six improvements awaiting the parent's baseline
ratchet update, including this 2,115→2,109 reduction. No baseline was raised here.

The first expanded run finished with 51 passes and two failures in 153.97 s,
exit 1: verifier bootstrap exceeded its 15-second subprocess deadline and the
disposable supervisor failed startup. This failed receipt is retained in JSON.
Separate unchanged-deadline reruns passed the controller suite **24/24 in 50.62 s**
and operations suite **29/29 in 59.59 s**, both exit 0. This supports successful
reruns, not proof that process timing can never fail under host contention.

The canonical extractor was run with
`python -m scripts.extract_test_cases --write tests/test_scripts/test_merge_verification_controller.py`.
It generated
`src/slm_training/resources/test_cases/test_scripts/test_merge_verification_controller.json`.
All **10 extracted cases passed in 1.69 s, exit 0** (14 unrelated cases explicitly
deselected in that narrow check); the extractor check and lint exit 0. The
JSON command manifest now records candidate-relative path fields that resolve
to the original host locations; its argv parses and every declared runtime root
exists. Domain artifact-path validation reports zero errors. This representation
change does not change measured results or turn a local receipt into release
authorization. The executable absolute host command above remains the operating
reference; resolve JSON environment path fields before setting subprocess env.

After generation, the **entire controller suite passed 24/24 in 25.96 s, exit 0**,
with no deselection:
`outputs/runs/autonomy-validation/validation-controller-generated-final-20260908.xml`.
Receipt and generated case-file hashes are recorded in JSON. The final full-tree
gate must bind this generated resource along with source and documentation.

### Python consumer closure: parent fix independently exercised (2026-09-08)

This follow-up closes the earlier Design-MD override limitation. The parent
implemented import-time `DESIGN_MD_BRIDGE_CLI` pinning; this worker changed only
`tests/test_scripts/test_merge_verification_runtime.py` and these evidence notes.
The actual rootless isolation suite passed **4/4 in 15.29 s, exit 0**, using the
authorized host controller with real Bubblewrap workloads and no local-feedback
fallback. The existing installed SDK import and both bridge protocol checks
remain in the test. The fixture additionally mounts candidate `src` read-only,
imports the actual Python consumer, and verifies:

- `_CLI` and its working directory resolve to the explicitly mounted bridge.
- Public `lint` creates a live REPL; clearing the cache and enabling one-shot
  produces equal results without creating another REPL.
- A fresh Python process with a nonexistent override refuses both readiness and
  lint, despite valid dependencies elsewhere in its permitted runtime.
- The host primary checkout remains inaccessible and no dependency tree is copied.

Receipt: `outputs/runs/autonomy-validation/validation-python-js-closure-20260908.xml`.
Test, parent consumer, and receipt hashes are in the JSON follow-up. Normal lint
passes. This is actual consumer/transport evidence, not a model evaluation or full
11k-test release receipt. The documented finite verification command is ready for
the parent's final source/docs freeze and complete selected-universe execution;
the previously outstanding Python/JS integration is no longer a prerequisite.

### Prefinal mutable-source diagnostic execution (2026-09-08)

The canonical documented command was actually executed with the same six JS/
Python runtime grants and `14400/1024/60` finite budget settings, but independent
external state and job/activity identity **autonomy-prefinal-diagnostic**. The
controller ran on the authorized host; actual workloads remained in Bubblewrap.
It exited **20 after one attempt**, charging **35.10 s**. The authenticated gate
journal records `repo_policy` passing in **19.42 s**, then
`source_changed_during_verification`. Concurrent source changes correctly prevent
passing reuse. No tests were collected by that gate invocation.

That run exposed a reporting defect: `validate_observation` discarded the child
exit-1 invalidation reason. The controller now retains the explicit reason while
forcing completion/authorization false. A failed child claiming a nonfailure
outcome remains rejected. This is failure diagnosis, never success authentication.

Three additional bounded diagnostics invoked the **same canonical isolated
`run_workload` collection owner** over `tests`, recording separate artifacts and
cost events outside the candidate. Requests retained the 60-second allowance.
The first two timed out (61.88 s and 62.21 s including workload preparation and
cleanup). Their outputs are not completed collections. The first diagnostic's
printed count of 1 was the selector `[tests]`, not a measured case count; the JSON
follow-up explicitly corrects its interpretation without rewriting history.

The third collection completed **11,611 unique nodes**, no collection errors and
no deselection, in **60.22 s including setup/cleanup**, exit 0. It retained **21
training-marked and 10 slow-marked nodes**. Collection is not test execution;
none of these 11,611 cases is claimed to have passed from collecting it. Source
changed before/after the observation, so it is deliberately not a release cache.
Artifacts live at `../../../outputs/runs/autonomy-prefinal-diagnostic/` relative
to this candidate; content digests and exact elapsed charges are in JSON.

The worker no longer prints every node during collect-only execution. Its full
protocol still records node IDs/markers, and bounded original collection-error
details remain visible. A 1,200-node fixture proves compact output with complete
identity retention; an import-error canary verifies original error preservation.
The compact-output change did **not** establish a timeout cure: its first real
rerun still timed out. A delayed stack diagnostic was then added, and the next
unchanged-budget run completed without needing a stack dump. Current collection
headroom is marginal under host load, not a guaranteed performance bound.

Final quiet-source gate execution remains parent-owned. All supplemental
diagnostic costs are additional to the canonical attempt's 35.10 seconds and
are visible in their own events; no retry or logical grant was reset. No service,
training loop, remote job, promotion or release authorization was activated.

Final focused receipts after these fixes: **31 passed in 43.59 s** for controller
plus runtime/real-isolation contracts, and **27 passed in 22.72 s** for the existing
merge-verification suite; both exit 0 with addopts and markers cleared and no
deselection. Receipt hashes are recorded in JSON. The initial new invalidation
test failed from duplicate fixture-directory creation; its failure is retained
and the test now reuses its single plan. Normal lint, explicit complexity checks,
test-case extraction check and domain JSON path validation pass. New compact
collection tests reside in the existing runtime test owner, preserving the
400-line module ceilings. Parent must include the changed
`scripts/merge_test_worker.py` and controller/test hashes in final version/source
composition; the worker was already part of verification binding identity.
