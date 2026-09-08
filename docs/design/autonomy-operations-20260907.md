# Local autonomy operations and fresh fault evidence

Current authoring source: `outputs/workspaces/autonomy-candidate`, base
`e0eca9f9910244ecc20eb480d363f1852417b599`; audit baseline
`2c4e2df3f9b9bfb6efe13e88108a57ea2ac9f64c`. Exact source/lock hashes,
test receipts, campaign event tip and artifact identities are in the
[machine manifest](autonomy-operations-20260907.json). This is operational
fixture evidence, not neural capability or scientific improvement.

## Fresh execution

| Check | Actual result |
| --- | --- |
| Current pinned `heal v8` finite campaign | 100 real workloads across five bounded CLI invocations; exits 10,10,10,10,0 |
| Final state | 70 verified artifacts; five cancellations; 25 repair waits; one separate capability wait |
| Workload accounting | 17.235835 measured seconds; 36.528827 charged seconds including lost-owner reservations |
| Controller accounting | 229.832699 seconds inclusive of workloads across five invocations; do not add nested workload time |
| Operations, state-machine, mutations, cross-owner and supervisor replay | Current candidate: 54 passed in 28.24 seconds; same pinned snapshot: 54 passed in 35.67 seconds; zero skipped/errors/failures |
| Launcher provenance correction | 29 operations tests passed in 18.66 seconds after adding Git discovery ceiling |
| Separate runtime regression fixture | 100 real processes, 75 completed artifacts; one test passed in 21.95 seconds; some outcome classifications injected |
| Approved-host bubblewrap and independent-verifier suite | 33 passed, zero skipped/errors/failures, 3.25 seconds |
| Live source-repair provider | `not_executed_capability_unavailable`: `agent_grant_missing` |

All 70 committed output manifests are revalidated by the finite runner. Five
interrupt-ignoring hangs are terminated; missing/malformed outputs do not succeed;
stale owners cannot publish; verified-before-terminal crashes reconcile without
repeating work. A capability wait does not starve independent fixture families.
The plan is locked before dispatch and includes the current version registry.
A separate whole-batch flock prevents competing CLI invocations from dispatching
the same pending attempts. Activity leases remain the canonical execution owner.

A moving-source development run was refused after 60 attempts when source changed.
That evidence remains under `outputs/autoresearch/ops-finite-current`; it is not
the final campaign. A separate older pinned snapshot completed 100 attempts under
`heal v7`, retained under `ops-finite-pinned`; its metadata remains historical to
that snapshot. `ops-finite-v8` remains prior complete evidence. A subsequent
`ops-final-current` run correctly refused a version-registry change after 64
attempts. The final evidence above is `ops-current2-pinned`, source release digest
`377fad5178643c6a76c1c226beebd4240582956c3fd60d3153b13773aa290bff`, plan
`9454365c9d8483f1b7db259d60af67309eba4074910485d655fabdb1dee76f98`.
Its locked runtime files match the candidate at handoff. Parent's subsequent
package relocation requires fresh successor evidence; this plan remains unchanged.
Recovered `/tmp` reports are not substituted for any of these current payloads.

The separate runtime regression's original short operational grant exhausted its
launch allowance under host load. Its failed evidence is retained, and explicit
fixture-plan v2 grants five-second work, 0.1-second kill grace, three-second
finalization reserve and 20-second total per activity. No scientific recipe or
global run cap changed; synthetic failure classifications are not live repairs.

## Commands accepted by the current parser

The final finite run executed from the prepared sibling
`../autonomy-ops-current2-execution`, using its `outputs` link into candidate
`outputs/autoresearch/ops-current2-pinned`. Every invocation uses the canonical cap:

```sh
timeout -s INT -k 10 170 env PYTHONPATH=src \
  GIT_CEILING_DIRECTORIES=/home/codex/repos/slm-training/outputs/workspaces \
  /home/codex/repos/slm-training/.venv/bin/python -m scripts.autoresearch \
  --root outputs verify-autonomy --batch-size 20
```

Repeat the same finite command for pending work. Exit 10 means pending, not a
negative model result. Exit 0 requires the entire locked plan and artifact checks.
Changed source requires a distinct successor root; never rewrite a locked plan.
The first snapshot invocation lacked the Git ceiling and inherited enclosing
checkout metadata. Its receipt remains unchanged. Later invocations truthfully
stamp `UNKNOWN` for a Git-free release; the full source digest identifies code.
The start/resume launcher now sets this ceiling automatically, with a real-process
regression test. Operational commands below run from the authoring checkout.

```text
python -m scripts.autoresearch --root ROOT doctor --source SOURCE
python -m scripts.autoresearch --root ROOT status --loop-id LOOP
python -m scripts.autoresearch prepare-release --source SOURCE --release RELEASE --execution EXECUTION --outputs OUTPUTS
python -m scripts.autoresearch --root ROOT start --config CONFIG.json
python -m scripts.autoresearch --root ROOT resume --config CONFIG.json
python -m scripts.autoresearch --root ROOT stop --loop-id LOOP
python -m scripts.autoresearch --root ROOT storage-health --minimum-free-bytes 268435456
python -m scripts.autoresearch --root ROOT service-template --config CONFIG.json
```

`start`/`resume` require an explicit local grant and a prepared execution release:

```json
{
  "schema_version": "local_supervisor_start/v1",
  "execution": "/absolute/approved/execution",
  "loop_id": "approved-loop",
  "train_version": "approved-immutable-data-version",
  "steps": 1,
  "max_cycles": 1,
  "local_execution_authorized": true,
  "recovery_config": null
}
```

This example is a configuration shape, not permission to train or a recommended
scientific recipe. A recovery config additionally requires its preapproved grant,
recipes and independent verification contract. No grant is inferred from a CLI
being installed. `service-template` prints a lifecycle recipe; it does not install
or activate a service. No persistent service was installed or started for this
packet. The start/stop test uses a disposable controller fixture with real process,
lease and signal handling, not the training driver. A separate stop test terminates
a real owned worker and leaves an unrelated session alive.

## Readiness, identity and storage

Doctor executes real Python imports, AgentV's actual SDK import, and both JS bridge
stdio ping protocols. A `node_modules` directory alone cannot pass. On this host,
select the already installed Node 22.23.1 on PATH; the default Node version is
outside the repository's `>=20 <23` requirement. The successful local probe used:

```text
AGENTV_RUNNER=/home/codex/repos/slm-training/scripts/run_agentv_eval.mjs
OPENUI_BRIDGE_CLI=/home/codex/repos/slm-training/src/apps/openui_bridge/cli.mjs
DESIGN_MD_BRIDGE_CLI=/home/codex/repos/slm-training/src/apps/design_md_bridge/cli.mjs
```

These are explicit shared dependency locations for this checkout, not portable
defaults. A fresh environment uses its own installed pinned dependencies. Tool
sandbox bubblewrap is denied, while the separately authorized host probe and all
33 host-boundary tests pass. Provider access remains untested without a grant.
Doctor emits stderr hashes instead of provider stderr. Disk/memory pressure causes
backpressure; it never deletes checkpoints, evidence or human workspaces.

Prepared releases exclude Git metadata, caches, environment files and local agent
settings. Outputs may live in the explicitly excluded `SOURCE/outputs` namespace;
release and execution trees remain disjoint. Source identity now rejects *all*
execution-source mutations, including README/model-card changes. Reports and learned
mutable state must use explicit output destinations; parent driver integration owns
those writer arguments. Read-only release modes prevent accidental mutation, not a
malicious same-user process. Repair workers use the separately tested OS boundary.

Status derives liveness from process identity, held lock and heartbeat; artifact
advances are not model wins. Unknown paired counts and shipment remain explicitly
unestablished rather than fabricated. Offline `compare-signs` output delivery is
idempotent: after a document failure, retry validates and reuses the recorded result
instead of recomputing a new timestamped artifact or resetting evidence history.
Fresh status for `autonomy-ops-not-started` reports `active=false`,
`no_controller_identity`, and no established paired comparisons. This is a scoped
status observation, not a claim about unrelated host processes.

## Verification scope

Tests run with `-o addopts=` so default training/slow exclusions cannot hide selected
nodes. Hypothesis 6.167.1 and sortedcontainers 2.4.0 were installed only into
`outputs/runs/autonomy-ops-verification/dependencies`; the shared environment was
not modified. Domain Ruff syntax/import/complexity checks and test-case extraction
pass. Full repository quality, ownership, version composition and release authority
remain integration checks; these domain results do not authorize a merge.
The fresh full code-quality check reports 14 integration regressions, including
the autoresearch package ceiling. The parent owns the planned runtime package
boundary and shared baseline updates; no baseline was raised to hide debt.

This finite campaign does not prove perpetual reliability, network-filesystem or
multi-host leases, real autonomous source repair, model quality, or shipment.
