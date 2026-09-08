# Runtime package relocation: operational successor evidence

The canonical local activity/service owner is now
`src/slm_training/autoresearch/runtime/`. Ten `activity_*`/`operations_*` helpers
moved without compatibility shims; `__init__.py` contains only the package
description. CampaignStore, campaign history, repair authority and scientific
verdict ownership did not move. Public `scripts.autoresearch` commands are unchanged.

Absolute, relative, child-process and test monkeypatch imports were updated,
including search, repair, checkpoint and supervisor consumers. The ownership-map
change is only the activity contract's owner path. No serialized activity schema,
ledger history, scientific threshold, resource cap or promotion rule changed.
The finite plan now locks the new package initializer as well as its actual owners.

## Identities and measured outcomes

The [machine receipt](autonomy-runtime-package-20260907.json) binds both runs,
changed paths, test receipts, final state, source drift and controller observation.

| Evidence | Source digest / result |
| --- | --- |
| Preserved pre-move release | `377fad5178643c6a76c1c226beebd4240582956c3fd60d3153b13773aa290bff` |
| Post-move release | `313a97a43f12f5587ce10a68674cfb01d02f188e8b6941011ff8a9ab4a20a2d4` |
| Post-move locked plan | `42f5e9be433c206945d0646aae88f7f07823bcbf98ec434c46704cb9da1433db` |
| New-path core tests | 80 passed, 26.71 seconds |
| Repair/checkpoint/search/supervisor consumer tests | 74 passed, 13.27 seconds; two multiprocessing fork warnings |
| Same pinned source core tests | 80 passed, 20.96 seconds |
| Actual finite attempts | 100 across five bounded invocations; exits 10,10,10,10,0 |
| Terminal/parked partition | 70 verified artifacts; five cancellations; 25 repair waits; one separate capability wait |
| Resource accounting | 13.416744 measured workload seconds; 33.694189 charged seconds including lost-owner reservations |
| Controller accounting | 100.585803 seconds across invocations, inclusive of workloads; do not add nested workload time |
| Scientific claims | Zero real model trials, live agent repairs, promotions or shipment claims |

The original [operations report](autonomy-operations-20260907.md), its JSON,
source hashes and raw artifacts remain unchanged. Its manifest SHA-256 remains
`3ff9a5c88269edb9136a9ae80ec887952d84802ffe9f4b3b559c748d9eae7d13`.
The successor is not pooled with it as independent scientific evidence.
The version registry changed concurrently after snapshot preparation; the receipt
records both hashes. Other locked runtime owners still matched at final observation.

Initial new-path checks retained two stale test-call failures: the publication
test omitted required `cwd`, and a supervisor test double lacked process-result
fields. The publication test now supplies the repository execution directory;
the parent updated its test to the actual bounded-process result contract.
Original failed receipts remain alongside successful successors; no production
guard was weakened to pass them.

## Finite command and service status

From the prepared sibling `../autonomy-runtime-package-execution`, each invocation
used the existing CLI and its output link into candidate
`outputs/autoresearch/runtime-package-successor`:

```sh
timeout -s INT -k 10 170 env \
  GIT_CEILING_DIRECTORIES=/home/codex/repos/slm-training/outputs/workspaces \
  PYTHONPATH=src /home/codex/repos/slm-training/.venv/bin/python \
  -m scripts.autoresearch --root outputs verify-autonomy --batch-size 20
```

Five invocations completed the locked plan. A changed source needs another
successor namespace, never a rewritten lock. The Git-free execution correctly
stamps `UNKNOWN`; its complete release digest identifies the tested source.

**This packet fixed and verified the harness; it did not resume continuous
autotraining.** No persistent service was installed or started, no old loop was
restarted, and no unrelated process was stopped. All owned finite/test workloads
have completed. The final campaign controller observation is `alive=false` and
`lock_owned=false`, with zero service-start events. Disposable lifecycle tests
exercise real start/stop mechanics and close their own controller; they are not
long-lived model runs. The explicit owned-workload list is in the machine receipt.
Read-only process inspection was limited to the tool PID namespace, not claimed
as a global host process inventory.

Fresh canonical `doctor --source .` passes Python imports, Node 22, actual AgentV
import and both bridge calls using the previously documented explicit shared
dependency paths. Local storage is ready. Live repair remains
`not_executed_capability_unavailable: agent_grant_missing`; no provider was called.
Tool-nested bubblewrap remains unavailable; the prior approved-host isolation
tests are historical evidence, not claimed as re-executed by this relocation.

## Quality and handoff limits

Ownership validation, focused Ruff syntax/import/complexity checks, old-import
search and test-case extraction passed. Measured component sizes are 19,953 lines
for autoresearch and 1,937 for runtime: the 20,000-line package violation is removed.
The full ratchet still reported eight integration regressions, including the
runtime/heal dependency cycle and SDP edges. No baseline was raised. Further
dependency changes are parent-owned integration, not silently waived here.

The full repository version check still needs the parent-owned README/model-card
component update; repository policy flags existing absolute paths in prior design
records, which were preserved as requested. A whitespace check also reported
three parent-owned test EOF lines. These results do not authorize repository
release, prove perpetual reliability, or establish neural capability.
