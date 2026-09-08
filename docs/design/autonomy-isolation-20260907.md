# Autonomous repair isolation — 2026-09-07

Owner: ISOLATION; authoring base `e0eca9f9910244ecc20eb480d363f1852417b599`;
audit baseline `2c4e2df3f9b9bfb6efe13e88108a57ea2ac9f64c`.

## Integration interface (RECOVERY / INTEGRATOR)

The existing `autoresearch.heal` owner gains `isolation.py`,
`isolation_workspace.py`, and `repair_verifier.py`. No event/receipt authority
or scientific verdict owner moves. RECOVERY owns calls from `heal/__init__.py`.

- `IsolationSpec(workspace, writable_paths=(), runtime_roots=(), timeout_seconds=170)`:
  controller-created private snapshot; repository-relative exact writable files;
  separately approved read-only runtime roots. No network grant
  is implemented by this backend. All destinations are fixed by the backend.
- `probe_isolation() -> IsolationCapability`: real bounded Bubblewrap namespace
  probe, capability only; executable presence never grants worker authority.
- `run_isolated(spec, argv, *, on_start=None, on_heartbeat=None)` returns the
  existing `BoundedProcessResult`; raises `IsolationUnavailable` on a failed
  capability probe and `IsolationViolation` on invalid mounts/scope. No fallback.
- `private_snapshot(source, destination)` materializes a private tree, without
  shared Git metadata, caches, outputs, or host links. The caller chooses the
  frozen source release and owns the destination lifecycle.
- `verify_candidate(request, base, candidate, *, runtime_roots=())` independently
  replays the original reproducer and controller-selected checks in fresh
  isolated copies. Returned evidence is not a repair receipt or release grant;
  the controller rechecks current request/fencing identity before acceptance.

Leaf work: ISO-CAPS enforces/probes namespaces and restricted mounts;
ISO-ALLPATH uses one bounded subprocess runner for every untrusted invocation;
ISO-VERIFY protects the selected checks/result issuer and replays the original
predicate; ISO-RELEASE rejects protected paths and classifies semantic changes.
Runtime and provider capabilities remain separate authorization decisions.

`VerificationRequest` requires `failure_returncode`, `failure_stdout_sha256`,
and `failure_stderr_sha256`: the frozen original failure must reproduce exactly,
not merely exit nonzero for an unrelated missing dependency. Include these fields
in RECOVERY's locked verification-manifest digest and constructor comparison.
Every `VerificationCheck` supplies a nonempty externally observed golden stdout;
empty output plus exit zero cannot satisfy it. This is finite behavioral evidence,
not a claim to defeat deliberate overfitting or prove arbitrary program semantics.
VALIDATION still owns full source/resource/test coverage and release authorization.

Precreate each allowed new regression/output file. For the Codex mapping adapter,
grant `repair-output/proposal.json`, not the entire `repair-output` directory.
Private Git may be initialized only in isolated scratch; host/shared Git metadata
is never supplied. No worker-controlled verifier-result file is accepted.

## Initial host observation

`bwrap --version`: `bubblewrap 0.9.0`. The namespace probe inside the tool
sandbox failed with `NETLINK_ROUTE ... Operation not permitted`; the same
read-only probe with approved host execution exited 0. This establishes host
namespace capability only, not an authorized provider or live agent repair.
No service was installed or started; no training or model evaluation ran.

## Measured acceptance and integration handoff

Final bounded host invocation (Python 3.12.3, ARM64 WSL2, Bubblewrap 0.9.0):

```sh
timeout -s INT -k 10 170 env \
  PYTHONPATH=/tmp/slm-autonomy-20260907/src PYTHONDONTWRITEBYTECODE=1 \
  SLM_REQUIRE_ISOLATION=1 /home/codex/repos/slm-training/.venv/bin/python \
  -m pytest -o addopts='' -q \
  tests/test_autoresearch/test_heal_isolation.py \
  tests/test_autoresearch/test_repair_verifier.py \
  --basetemp=/tmp/slm-autonomy-20260907/outputs/isolation-acceptance-3 \
  --junitxml=/tmp/slm-autonomy-20260907/outputs/autoresearch/isolation/acceptance-3.xml
```

Exit 0: **33 passed, 0 skipped, 0 failed**, 2.721 seconds. The required-capability
flag makes an unavailable namespace fail the live test, never silently skip.
Ruff E/F/I and C901/PLR0911/12/13/15 also passed on all six code/test files.
All modules are below 400 lines. Machine receipt:
[`autonomy-isolation-20260907.json`](autonomy-isolation-20260907.json).

| Leaf | Implemented observation | Evidence class |
| --- | --- | --- |
| ISO-CAPS | Rootless namespaces, hidden host home/environment/socket, hard file limit, bounded tmpfs, exact writable files | Actual host processes/filesystems |
| ISO-ALLPATH | Shared bounded runner, timeout/descendant cleanup, finally-path scope checks, private Git-free snapshots, byte/mode/link detection | Actual host processes and imported-owner tests |
| ISO-VERIFY | Original failure identity reproduced; original and independent predicates restored; empty zero-exit output refused; controller socket/result file inaccessible | Actual isolated Python processes, no agent |
| ISO-RELEASE | Wrong identity, forged receipt, existing-test edits and skip/xfail refused; wiring equivalence compared on both releases; changed score refused | Imported contracts and isolated differential fixture |

Audit reproduction: importing unchanged `heal._scope_violation` with before/after
both `('tests/protected.py',)` and allowed `('node_modules/',)` returns `()`.
The same module has no source diff between the audit baseline and authoring base.
This reproduces the already-dirty-file gap using the actual owner. New
`scope_changes` checks content/mode/link manifests; the real mount denies the
protected write before it reaches the snapshot.

The trusted controller calls the verifier from its pinned installation and
receives evidence in memory. A workload never receives its verifier, request
ledger, receipt directory, credentials, host socket or shared Git metadata.
The isolated system-Python bootstrap closes inherited stdin before candidate
imports and applies hard file-size/core/FD limits. Hashes supplement OS isolation.

Evaluator wiring is not blanket-forbidden: lock exact `semantics_preserving_paths`
and nonempty `equivalence_checks` in `VerificationRequest`; those golden responses
must pass on both releases, and the original failed predicate must be restored.
Scorers, eval/ship policy, DSL/formal/model authority, resources, controller and
existing tests stay protected. A changed score requires separate semantic release
authority and remeasurement. RECOVERY must lock both added fields in its request.

Earlier development failures are retained: Bubblewrap required explicit
`--unshare-user`; the first socket fixture exceeded Linux's socket pathname
length and now uses a host directory-FD address; one concurrent run timed out a
five-second original probe and correctly refused acceptance (exact rerun passed
in 0.43s). Failed/interrupted trials are not passing evidence.

Remaining integration: RECOVERY owns dispatch/current-fence authentication and
the exact output-file mount; INTEGRATOR owns the `autoresearch.heal` bump (v7 at
authoring), ownership/coverage composition and supervisor wiring. VALIDATION owns
full affected-test merge authorization. Source snapshots must be pinned/quiescent
before copying. Scope failures preserve private attempts for quarantine.

No provider egress/credential relay, directory quota mount, or multi-host/NFS
guarantee is implemented. Directory write grants fail closed; npm repair needing
mutable directories needs a separately bounded environment backend. No real
coding agent was invoked: `not_executed_capability_unavailable` for an explicit
provider grant, with provider egress additionally unsupported here. The tests are
finite operational evidence, not arbitrary semantic equivalence, perpetual
reliability, model quality, or fully hands-off live repair. No scientific policy,
historical result, checkpoint, service, human worktree or Git ref was changed.

## Restored host-boundary qualification — 2026-09-08

Both current probes used the same bounded command and current candidate imports:
`timeout -s INT -k 10 170 python -c '...probe_isolation()...'`. Normal outer Codex
execution returned `available=false`, `/usr/bin/bwrap`, with
`bwrap: loopback: Failed to create NETLINK_ROUTE socket: Operation not permitted`.
Approved host execution outside that outer sandbox returned `available=true`
from the same binary. Neither result conveys provider authority.

All current `SLM_REQUIRE_ISOLATION=1` boundary/verification receipts refer to an
approved **host controller with real rootless Bubblewrap workloads**. They must
not be represented as passing inside the normal Codex sandbox or as local-feedback
verification. The configured provider grant remains absent; no live agent job
was run. The current finite repair fixture and its narrower evidence scope are
documented in [autonomy-adversary-20260907.md](autonomy-adversary-20260907.md#restored-candidate-successor-v3--2026-09-08).

### VAL JS closure independently exercised — 2026-09-08

ISO independently ran the current VAL-owned
`tests/test_scripts/test_merge_verification_runtime.py`: **4 passed, 0 skipped,
exit 0, 6.41 seconds**, with `-o addopts= -m '' -p no:cacheprovider -q --tb=short`,
`SLM_REQUIRE_ISOLATION=1`, and the canonical 170-second interrupt / 10-second
kill grace. The approved host controller launched real Bubblewrap workloads;
no local-feedback fallback, network access, dependency install, or agent was used.
The installed Node 22.23.1, complete explicitly granted dependency trees, real
AgentV SDK import and both OpenUI/Design-MD ping protocols passed. A deliberately
missing transitive import failed as required, and the shared primary checkout
was inaccessible inside the workload. This is readiness/protocol evidence, not
an AgentV evaluation publication or source-release authorization.

Parent/VAL coordination receipt:
`outputs/runs/autonomy-validation/iso-js-closure-20260908.xml`, SHA-256
`4c5775255b5669f1f54dbf741475bde282e9159d09e43f01de67cdbd7b52a3a4`.
Tested `scripts/merge_verification_runtime.py` SHA-256:
`012f4c18c68427bc55d03f854cfe4e0f997fbf3f3f7b6dedbc5f30079bd4ac72`;
test SHA-256: `a01bc21a83886e75dc445a419c116fa03ac7fa81e363e96df8682ab99af0e632`.
No VAL-owned source or tests were edited by ISO. Full verification must still use
the same explicit runtime grants and its own current authenticated binding.

After VAL extended that test selection, ISO reran its current source: **6 passed,
0 skipped, exit 0, 12.54 seconds**, using the same explicit host/rootless context
and cleared pytest options. Receipt
`outputs/runs/autonomy-validation/iso-js-closure-final-20260908.xml`, SHA-256
`a4f6f472d5b5631874da6ffd58fe738f384608147ab23aaea8c8c14d55f0a850`.
The runtime owner digest is unchanged; the final test digest is
`c4f0db00441ca7846c77a831f47f483579d4b7a7f66589fdb2e1653b48f36c38`.
This supersedes the four-test selection for the current-test claim, without
overwriting its historical receipt or extending it into a release verdict.
