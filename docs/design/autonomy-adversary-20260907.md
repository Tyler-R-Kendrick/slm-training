# Finite autonomy adversary acceptance — 2026-09-07

Historical restoration note: the sections below describe the vanished `/tmp`
workspace and its original observations; underlying old run payloads were not
recovered. Fresh restored-candidate execution and current payload references are
recorded in [autonomy-operations-20260907.md](autonomy-operations-20260907.md).

Base: `e0eca9f9910244ecc20eb480d363f1852417b599`; evidence baseline:
`2c4e2df3f9b9bfb6efe13e88108a57ea2ac9f64c`. This is operational fixture
acceptance, not neural capability, scientific improvement, or live agent repair.
The ISOLATION implementation and 33 actual boundary checks are recorded separately
in [autonomy-isolation-20260907.md](autonomy-isolation-20260907.md).

## Preregistered finite fault plan

This plan was written before the acceptance campaign below. A separate two-attempt
development plumbing run under `adversary-development` is not acceptance evidence.
`scripts.verify_autonomy` locks the exact source hashes, environment identity,
100 ordered workload attempts, injected faults and expected results in the existing
CampaignStore append-only history before dispatch. No separate scheduler or verdict
store is introduced. Each controller subprocess invokes the actual ActivityRuntime,
bounded process executor, artifact validation and replay owners.

Five groups of 20 attempts each have: ordinary code crash, zero-exit missing output,
malformed JSON, real interrupt-ignoring hang, stale publication after cancellation,
controller loss before commit, controller loss after durable output verification,
and 13 healthy calculations. Expected final state: 70 succeeded, five cancelled,
25 scoped repair waits, plus one external-capability wait with no workload attempt.
Healthy jobs must progress despite that independent wait. Verified-output recovery
must not repeat work. All 70 committed manifests must still validate; duplicate
acceptance, stale publication and scientific promotion events must be absent.

Resource checks distinguish observed workload seconds, conservative lost-owner
reservation charges, and actual controller seconds inclusive of its child. They
are not added together as though inclusive measurements were independent costs.
Each v2 controller subprocess has 15 seconds plus canonical ten-second kill grace;
the declared aggregate controller ceiling is 2,500 seconds. Fixture workers have
smaller explicit caps. Each batch voluntarily reserves 30 seconds for finalization
and the command remains bounded by the canonical 170+10 seconds.

Run five separately bounded batches against the same root (repeat only pending
work; final successful replay runs zero new attempts):

```sh
timeout -s INT -k 10 170 env PYTHONPATH=/tmp/slm-autonomy-20260907/src \
  PYTHONDONTWRITEBYTECODE=1 /home/codex/repos/slm-training/.venv/bin/python \
  -m scripts.verify_autonomy --root outputs/autoresearch/adversary-successor-v2 --batch-size 20
```

Exit 10 is incomplete/pending, not failure or scientific loss. Exit 0 requires all
100 workload observations and the locked invariants. Source changes reject old-plan
reuse; retain the old evidence and register a versioned successor after correction.
No service is installed or activated. Artifacts are fixture calculations, not
hand-authored metric scoreboards or model evaluations.

## Independent checks and limitations

The adversary tests use Hypothesis rules against the actual reducer and compare
accounting with a small independent reference, and remove a real output-validation
guard to ensure the acceptance assertion detects it. This finite sampling and
targeted mutation are not an exhaustive proof of perpetual operation. Real model,
data readiness, optimizer continuation and configured agent repair require their
own producer-to-consumer evidence and cannot be discharged by this fixture.

Hypothesis was absent from the supplied interpreter. The repository-declared
`hypothesis>=6.100,<7` dependency was installed into invocation-local
`outputs/adversary-deps` (6.167.1; sortedcontainers 2.4.0); the shared virtualenv
was not modified. Tests clear default pytest marker exclusions with `-o addopts=''`.
Results and first counterexamples will be appended below after execution.

## First counterexample, retained and qualified

Plan v1 under `outputs/autoresearch/adversary-final` stopped at actual attempt 3
(real hang) with `StaleLease` from `attempt_dir`; exit 1. The child was correctly
killed and never accepted. Claim-to-expiry was only 0.4 seconds: 0.2 execution,
0.1 kill grace, 0.1 finalization. Durable state preparation, subprocess setup,
observation and repeated projection reads consumed this real allowance. The child
itself used 0.178933 seconds. This establishes an insufficient fixture grant and
a finalization-expiry route needing explicit controller handling, not permission
to bypass fencing or a demonstrated model-quality defect. RUNTIME/INTEGRATOR
received the exact failure; their files are not edited by ADVERSARY.

Versioned successor v2 preserves the seven fault types, order and expected states;
it allocates a measured-cost-aware five-second finalization reserve, one-second
hang workload cap and 15-second controller cap, all below the canonical limit.
These are explicit operational resource changes, not a rewritten v1 result.
Its source/plan is locked before execution in `adversary-successor-v2`.

## Restored-candidate successor v3 — 2026-09-08

The v2 observations above remain historical. V3 changes the fault contract, not
its historical results. `_plan(..., version=2)` retains the old 100/70 contract
for characterization; current source still refuses reuse of an older locked
source identity. The default canonical CLI now prepares `autonomy_adversary_plan/v3`.

Each of five groups retains the seven v2 fault positions. Positions 7 and 8 now
dispatch a deterministic fake repair adapter through the existing `dispatch_repair`
owner and replay the original blocked activity after real `verify_candidate`
checks in Bubblewrap. The first child actually exits 1 with the frozen failure;
the repaired child runs the same argv/input against the checked fixture candidate.
The replay uses a new lease/attempt, not a new independent scientific sample.
The fake adapter is controller-owned test code, **not** a sandboxed live coding
agent. Its independent verifier workloads are genuinely isolated.

Positions 9 and 10 inject `OSError(ENOSPC)` at the actual CampaignStore event
boundary, respectively after output verification and before output verification,
then exit the controller process. This simulates storage errors without filling
or damaging the host filesystem. Restart must reconcile the verified artifact
once; an unverified artifact stays a scoped repair wait and retains its lost-owner
charge. This is not coverage of every disk/fsync failure or physical state loss.

The locked v3 expectation is **100 actual ActivityRuntime.run attempts, 65
completed artifacts, five cancellations, 25 repair waits, one independent external
capability wait, five fake-agent predicate repairs, five original-activity replays,
and ten injected storage interruptions**. Nested independent-check subprocesses
are included in the repair activity's measured duration, not added again as
independent activity attempts. Predicate proof is revalidated before both wake
and summary credit. No source release, authoritative repair receipt, champion,
model trial, AgentV evaluation, or production promotion is issued by this fixture.
Full source-aware release authorization remains a separate obligation.

Ordinary v3 controller invocations have 30-second interrupt budgets and repair
controller invocations 65 seconds, each with canonical ten-second kill grace.
The explicit aggregate controller ceiling is 4,175 seconds. Workload-specific
deadlines remain smaller (including the one-second interrupt-ignoring hang).
Every top-level command retains the canonical 170+10 cap; a batch yields before
starting work that cannot fit its remaining allowance. V2's 15-second controller
allowance remains unchanged. V3 development encountered actual 15-second startup
timeouts under host contention; its larger controller allowance was declared
before the final v3 plan was locked, not retrospectively applied to old results.

Run from the candidate with the provided Python/dependency environment, on an
authorized host controller where the real Bubblewrap probe succeeds:

```sh
timeout -s INT -k 10 170 python -m scripts.verify_autonomy \
  --root outputs/autoresearch/autonomy-adversary-v3 --batch-size 20
```

Repeat separately bounded invocations until exit 0. Exit 10 remains incomplete.
Missing rootless isolation produces a durable capability wait before any workload,
not local-feedback execution. The final source/versions must be frozen first;
source drift is rechecked before each batch evidence summary.

### Executed development verification (not the final canonical 100-run)

Machine record: [autonomy-adversary-v3-20260908.json](autonomy-adversary-v3-20260908.json).
All commands used `timeout -s INT -k 10 170`, the existing Python 3.12 environment,
`PYTHONDONTWRITEBYTECODE=1`, `-o addopts= -m '' -p no:cacheprovider`, and no paid
agent or network execution.

| Check | Actual result | Evidence scope |
| --- | --- | --- |
| `tests/test_autoresearch/test_autonomy_adversary.py` | 8 passed, exit 0, 35.04 s | Property/mutation/contract tests plus 11 selected real controller invocations, real isolated predicate checks, both storage boundaries |
| `tests/test_autoresearch/test_activity_runtime_demonstration.py` | 2 passed, exit 0, 23.42 s | Separate 100-child runtime fixture; 75 artifacts; actual code crashes; content identity covers runtime owners and the fixture test itself; mocked missing-isolation preflight |
| Ruff and explicit C901/PLR0911/12/13/15 | Passed | Four edited source/test files; no threshold changes |

The first check used approved host execution outside the outer Codex sandbox with
`SLM_REQUIRE_ISOLATION=1`; candidate checks still ran inside real Bubblewrap.
The second check ran in the normal Codex sandbox and makes no verifier-boundary
claim. Dedicated temporary namespaces were `/tmp/av3.VA7BlQ/pytest-final-focused`
and `/tmp/av3.VA7BlQ/pytest-runtime-v3` via `--basetemp` (disposable, not canonical
evidence roots). Source hashes and version stamp are recorded in the JSON.

The runtime fixture's old literal HEAD and healthy-child relabeling were removed
in its own `runtime_fault_fixture/v3` successor. Ten code-failure children now
actually raise `RuntimeError`; a healthy exit cannot satisfy that branch. Historical
v2 receipts are not rewritten. This supplemental 100-child test is not a substitute
for the coordinator's final current-source canonical v3 campaign.

Development failures remain qualified: a host-only virtualenv executable path was
not visible inside the verifier, so exact original-failure reproduction correctly
failed. The fixture now uses the already exposed system Python 3.12.3 for both
original and replay argv; no host path was exposed and no isolation fallback added.
Failed verifier observations are retained even when the fixture refuses completion.
The canonical v3 100-attempt campaign is **not yet claimed executed by this record**.
## Canonical successor execution follow-up — 2026-09-08

**Final pinned v5 result: complete, 100 actual attempts.** Six separately bounded
CLI invocations reached cumulative counts 20, 39, 59, 79, 99 and 100, with exit
codes 10, 10, 10, 10, 10 and 0. The locked expectations all held: **65 verified
operational artifacts, five actual original code-failure exits, five fake-agent
patches independently verified in real rootless workloads, five successful
original-activity replays, ten injected storage interruptions, five cancellations
and 25 scoped repair waits**. An independent unavailable-provider activity
remained a capability wait without starving healthy work. No source release,
live-agent execution, model trial or scientific promotion occurred.

The event chain has 790 events and 112 controller contexts; all 100 attempt IDs
are distinct. Canonical validation checked the resource charge partition,
verified artifact content and nonduplication, stale-fence refusal and original
replay predicates. Charged workload time is **185.4603s**, observed workload time
**35.1563s**, and observed controller time including workloads **495.7994s**;
these overlap and must not be summed as independent costs. Lost-owner attempts
retain conservative reservations. Controller time stayed within the declared
5,725-second logical ceiling. All work used the approved host controller and
real rootless verifier; fake adapter and injected ENOSPC labels remain explicit.

Invocation (from the pinned source directory; same installed Python and declared
runtime environment as this report):

```text
timeout -s INT -k 10 170 python -m scripts.verify_autonomy --root <candidate>/outputs/autoresearch/autonomy-adversary-v5-20260908-pinned-iso --batch-size 20
```

Final content-addressed evidence:
`outputs/autoresearch/autonomy-adversary-v5-20260908-pinned-iso/autonomy-adversary-finite-100/artifacts/autonomy_adversary_evidence/79cc26651584b3eb473263e170e0e3848f58e2a302cb5d38a24239f7f0455087.json`.
Its byte SHA-256 is `95936844addad7fd1df118fa2a041f3ea6a560356590458fcc326ad750c2fa94`;
event-chain file SHA-256 is
`782dc00488a2380cd4f2ddff10a521fa5afc7797ba390df758435cb595921128`.
The full pinned payload digest was unchanged at completion. All executable
owner hashes match the moving candidate at closeout; only its version registry
differs. This is pinned execution evidence, not source-release authorization of
the moving candidate.

Current-candidate closeout regression: **9 passed, 46.73s, exit 0, zero skips**
under the required real rootless boundary. Selection:
`tests/test_autoresearch/test_autonomy_adversary.py` plus
`test_activity_runtime_demonstration.py::test_successor_resource_contract_preserves_historical_timeout_plan`,
with `-o addopts= -m '' -p no:cacheprovider -q --tb=short`.
Receipt `outputs/runs/autonomy-validation/iso-v5-final-tests-20260908.xml`, SHA-256
`770c179d0b7b68395381541a4add8990b3299dcf2c3a27491cbfb331a68af34f`.
Full source-aware release verification remains independently owned by VAL/the
integrator. The 97 earlier failed or source-drift-interrupted attempts listed
below are retained separately, never pooled to satisfy this 100-attempt result.

The `-r2` v5 run similarly refused a subsequent version-registry edit before
index 32, after 32 actual attempts. All executable owner hashes still matched.
Rather than requiring other integration workers to stop, the final run uses the
existing `private_snapshot` owner to copy only `src/` and `scripts/` into
`outputs/workspaces/autonomy-finite-iso-wltduzgo` relative to the primary checkout.
The copy is 466 MB and excludes Git metadata, environments, installed dependencies
and run outputs. Locked owner hashes matched before and after copying. The
complete pinned payload has 2,990 entries and tree-manifest digest
`d18e0e9351f08ccd0af4dde7f7f0d773c16a7f8b08fcabc487181a390217cc7b`.
Final campaign artifacts remain under the candidate's canonical output root,
`outputs/autoresearch/autonomy-adversary-v5-20260908-pinned-iso`.

Provenance qualification: without private Git metadata, canonical stamp discovery
reports the enclosing primary checkout commit `2c4e2df...`, not candidate origin
`e0eca9f...`. The origin and actual locked content/tree hashes explicitly identify
executed code; the enclosing Git SHA alone is not a candidate-release identity.

The first v5 root, `outputs/autoresearch/autonomy-adversary-v5-20260908-final-iso`,
retains 30 actual attempts and two predicate repair/replays, but is incomplete:
the next controller refused source drift before executing index 30. Only
`src/slm_training/resources/versions.json` changed, from SHA-256
`da4ac1c67bea2884b9d7c917b74a17b482190923a0c7db7ee789f1ad5e66b26c`
to `fe5ae09c09d71673b2f70fb605fa6e7061e6e855186d5bca447dfa7e9dd5fbf4`.
The failed invocation exited 1; no old/new identities or partial attempt counts
are pooled. A fresh v5 root ending `-r2` binds the new version identity.

The v4 run at `outputs/autoresearch/autonomy-adversary-v4-20260908-final-iso`
also remains failed/incomplete: eight workload attempts, repair index 7 exited 1
with `isolation_preparation_budget_exhausted` during the independently isolated
check's two-second allowance. No predicate receipt or release was accepted.
Successor v5 explicitly declares ten seconds per isolated check and a 60-second
repair workload ceiling (90-second repair controller cap). Ordinary v4 budgets,
fault order and expected outcomes are unchanged. Its aggregate controller
ceiling is 5,725 seconds, including kill grace; no individual command exceeds
170+10. The v3/v4 budget-preservation and plan-lock regressions passed together
(**2 passed, 0.27s, exit 0**). First failures are retained, not washed out by the
successor results.

The first canonical v3 execution at
`outputs/autoresearch/autonomy-adversary-v3-20260908-final-iso` is preserved as
**failed/incomplete evidence**, not a passing 100-attempt demonstration. Its
locked plan digest is
`7b2bf056065b6c4079fa38d76d2f2b395eee0d3df2025c68b6ccc477d2d55aa3`.
The first bounded invocation completed 13 actual attempts (exit 10/pending).
The second exited 1 after observing attempt index 26: its nominally healthy
`verified_before_crash` workload was killed (`returncode=-9`, timed out, observed
3.7160 seconds) under the declared two-second workload limit, leaving no result.
The real output owner correctly rejected its empty manifest. There are 27
observed workload attempts, not 100; no timeout is relabelled as success.

The fresh v4 successor retains the identical 100-attempt fault sequence and
expected outcomes, but explicitly grants ordinary workloads ten seconds and
ten seconds of finalization reserve, with a 31-second logical activity allowance
covering the declared original/replay pair. The injected hang remains one second.
Controller caps are 45 seconds ordinary / 65 seconds repair plus the canonical
ten-second kill grace: aggregate controller ceiling 5,600 seconds. Each CLI
invocation still has the canonical 170-second interrupt / ten-second kill grace.
Historical v2/v3 resource contracts are characterized unchanged by a regression;
the failed locked evidence is neither overwritten nor resumed under new rules.

Pre-execution successor verification: **8 adversary tests passed in 103.31s**
(host controller, required real rootless isolation), and **3 runtime tests passed
in 70.51s** (normal Codex sandbox, no isolation claim), both exit 0 and zero skips.
Both selections cleared addopts and markers. Receipts under
`outputs/runs/autonomy-validation/`: `iso-v4-fixture-tests-20260908.xml` SHA-256
`9e66da493e9d88b603367b3c59aa25b2d683cfa6ed8dde66e4982f758a204750`,
`iso-v4-runtime-tests-20260908.xml` SHA-256
`6f697fabd8430044f60e6fd27e9c3274e3b6314a026a6046610d249e298906f5`.
