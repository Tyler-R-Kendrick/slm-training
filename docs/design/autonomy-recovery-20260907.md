# Governed source-repair dispatch

> Recovery annotation: historical report restored after filesystem loss. Tests and
> referenced run artifacts have not been revalidated in the reconstructed workspace.
> This is not current release or autonomy authorization.
> Historical content SHA256: `42ba1163804476560f3c1de16dc0b15136c29746c825a5aee0cfc36b3efbab40`.


Base: `e0eca9f9910244ecc20eb480d363f1852417b599`; audit baseline:
`2c4e2df3f9b9bfb6efe13e88108a57ea2ac9f64c`.

The existing `harness_crash/v1` is deliberately diagnostic-only. Recovery now
extends the same `heal` owner with strict request/proposal contracts and an
executed Codex adapter; an agent answer cannot acknowledge its own repair.

## Integration API (RECOVERY → INTEGRATOR / ISOLATION)

`heal.repair_contracts` owns `RepairRequest`, `RepairBlocker`, `RepairGrant`,
`RepairProposal`, `RepairVerification`, and `RepairDispatchResult`. These are
new claim classes: legacy diagnostic receipts are never upgraded implicitly.
Shared action fields remain INTEGRATOR-owned. Map `blocker_code` and
`unmet_predicate` from typed producers, never invent them from a log regex.
Absent fields require diagnosis before constructing a source-repair request.

`heal.dispatch.dispatch_repair(request, executor=..., journal=CampaignStore(...),
fence_valid=...)` performs capability checking and real executor dispatch.
Call it from `_handle_hard_pending` while holding the canonical activity lease.
Use a stable campaign journal for a blocker's complete repair history, including
retries across loop cycles; pass its campaign ID in the request. RUNTIME owns
cross-campaign resource grants. Repeated identical capability waits emit no new
event. A started attempt without a result requires reconciliation rather than
blind repeated agent execution. Full reserved time including canonical kill
grace is charged conservatively even after an interrupted attempt.

`heal.agent_executor.CodexExecutor(runner, instructions=..., contract=...)`
implements actual argv/schema construction, installed-CLI checking, structured
output validation, cancellation checks and progress callbacks. The input must
contain complete local AGENTS/invariant text and owner contract, not prior chat.
`runner` is an ISOLATION/controller object, never worker-deserialized configuration:

```python
runner.capability(request) -> str | None  # exact unmet capability, None when ready
runner.run(request, argv, inputs=..., progress=..., cancelled=...) -> AgentRun
# AgentRun(outcome, returncode, seconds, final_json)
```

Required mounts for that protocol: readonly `/input/{repair-instructions.json,
proposal-schema.json}`, private source `/workspace`, writable private `/output`.
The runner owns provider-grant enforcement, private Git metadata, deadline,
process-tree cancellation, final output collection, and approved executable
mapping. No unsandboxed fallback exists. Native Codex sandbox flags supplement
the independent OS boundary. The current isolation backend may supply a mapping
adapter if it uses different private mount paths.

`heal.dispatch.accept_verification(request, proposal, verification, journal=...,
authenticated=..., fence_valid=...)` checks the independent channel, original
reproducer, restored predicate, complete checks, protected surfaces and exact
source/environment/input/manifest/grant/fence. The verifier owns `authenticated`;
checking worker-supplied hashes alone is forbidden. `verified` is eligibility for
a fenced source publication, not publication itself and never an action receipt.
Measurement-semantic changes require separately authorized successor measurement;
trust/policy changes are rejected by routine acceptance.

No runtime service was installed or started. No Git/HF/paid operation is used.

## Evidence

Installed read-only probe: `codex --version` reports `codex-cli 0.153.4`.
`codex exec --help` confirms structured output, ephemeral execution,
ignore-user-config/rules, and workspace-write sandbox support. Neither command
authenticates or executes a model. Live repair remains
`not_executed_capability_unavailable`: no explicit configured agent/provider
grant is present in this assignment. An installed CLI is not spending authority.

Contract tests use a fake sandbox/executor and are labelled as protocol evidence;
they do not prove live source repair or enforceable isolation.

Final recovery checks: 102 tests passed in 6.74 seconds, exit 0:
`python -m pytest -o addopts= --basetemp=outputs/recovery-tests-final -q
tests/test_autoresearch/test_repair_dispatch.py
tests/test_autoresearch/test_recovery_dispatch_entrypoint.py
tests/test_autoresearch/test_repair_acceptance.py
tests/test_autoresearch/test_heal_classify.py
tests/test_autoresearch/test_heal_harness_crash.py
tests/test_autoresearch/test_heal_playbooks.py`.
Every command used `timeout -s INT -k 10s 170s`, the existing Python 3.12 venv,
this clone's `src` on PYTHONPATH, and invocation-local bytecode caches.
The initial test fixture name `request` was rejected by pytest and corrected;
the initial reversed CampaignStore constructor was corrected. Neither failed
test invocation is counted as passing validation.

Two acceptance tests execute real Python child processes using the actual
independent verifier producer/consumer, with OS isolation explicitly simulated.
They prove the original failure is required to be restored even when a new test
passes. They do not prove a real agent patch or OS isolation. Actual Bubblewrap
probe: installed `/usr/bin/bwrap`, unavailable with
`Failed to create NETLINK_ROUTE socket: Operation not permitted`.

The actual baseline module loaded from its Git object classifies the historical
screening-suite prose as `data`; it lacks a `code` argument. The current typed
`screening_wall_budget` produces `code` (measurement/runtime owner),
`screening_suite_volume` produces `data`, and `screening_constraint_unknown`
stays `unknown`. Formal/delivery action authority retains precedence.

## Canonical supervisor composition

INTEGRATOR calls
`recovery_dispatch.dispatch_hard_pending(pending, RecoveryContext(...),
config=load_recovery_config(explicit_path), fence_valid=..., progress=...,
cancelled=...)`. A missing path means no grant; it emits a deduplicated durable
`agent_grant_missing` scoped wait. No manual request JSON is required.
`RecoveryConfig` is a strict `recovery_config/v1` document with grant, approved
recipes keyed by producer code, readonly runtime roots, and verifier release.
Its JSON schema is available from `RecoveryConfig.model_json_schema()`.
Each recipe pins original argv/golden output and exact observed failure exit,
stdout/stderr hashes; independent checks, exact source/test paths, input digest,
and owner-contract path. A semantic-equivalence exception additionally binds
explicit paths and differential checks from the isolation owner.

The builder embeds full AGENTS/RTK/decode instructions (bounded at 256 KiB per
instruction field), the frozen recipe, and the untrusted failure evidence.
The supervisor must pass an immutable private source release and a separate
writable campaign/output root. The source digest is the isolation owner's
`manifest_digest(tree_manifest(release))`, not a remembered Git SHA or PID.
Agent and verifier execution occupy separate invocations. Both reserve their
bounded compute in canonical events; uncertain prior starts require
reconciliation. The verifier channel calls the actual `verify_candidate` owner,
validates the original failure identities, scope, patch/tree digests, full check
manifest and current fence, and only then returns controller-consumable evidence.

## Leaf disposition and authority

| Leaf | Implemented owner and evidence | Remaining boundary |
| --- | --- | --- |
| REC-DISPATCH | `recovery_dispatch`, `dispatch`; actual canonical store requests/events and typed routing tests | Supervisor call site owned by INTEGRATOR |
| REC-AGENT | `agent_executor`, `isolated_agent`; installed CLI help/version probe, strict grant and protocol tests | Real agent not executed: no grant and namespace probe fails |
| REC-REPRO | `repair_acceptance` calls independent verifier; real subprocess original-failure regression | OS boundary tested by ISOLATION, not claimed by simulated tests |
| REC-RESUME | Verified release/predicate result binds exact request; stale receipts and scorer changes rejected | Fenced publication and original-activity resume owned by controller |
| REC-COVERAGE | Existing npm playbook retained; known code/data/formal-infra codes route distinct configured classes; frozen reproducer recipes invoke actual adapter | DATA's readiness executor and ISOLATION's policy remain their owners; formal contradiction/delivery cannot be waived |

No scientific thresholds, model size, tokenizer, decoder, model card, historical
result, or protected shared registry was changed by RECOVERY. The integrator must
bump `autoresearch.heal` from v7 and compose ownership/version checks. No source
repair worker can write action receipts, accept its own test result, or activate
a release. The checked limits are finite fault coverage, not perpetual reliability.
# Cross-lease repair continuation

The production seam now records a logical repair job keyed by immutable failure,
release, input, policy/config and verification contracts, excluding transient
lease, attempt, parent event and human log text. A later verification invocation
loads the original content-bound request/proposal through the verified event
chain. It appends a `VerificationBinding` linking that original evidence to the
new controller fence/attempt; no historical request or receipt is rewritten.
The receipt binds this fresh authority and remains controller-issued only.

`test_repair_acceptance.py` exercises two actual `dispatch_hard_pending` calls
with different lease generations, one fake-agent proposal, and real subprocess
original/check workloads under explicitly simulated isolation. Restored and
still-failing original predicates produce verified/rejected respectively.
Focused recovery selection: 39 passed, exit 0 (2026-09-07). This is protocol and
process evidence, not a live-agent or OS-isolation claim.
