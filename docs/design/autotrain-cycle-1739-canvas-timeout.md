# Autotrain c1739: runtime repair improves progress but replay is incomplete

**Verdict:** the exact frozen c1737 replay again finalized both three-record
smoke suites, but each arm timed out on `smoke_hero_01`. The repaired harness
now classifies the cycle as non-positive because the authoritative measurement
is incomplete. Canvas is neither promoted nor rejected.

## Result matrix

| Arm | Params | Complete | Timeout | Parse | Meaning | Structure | Binder F1 | p50 complete | p50 incl. incomplete | Init | Forwards | States | Witnesses | Forks | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 8,860.24 ms | 20,108.53 ms | 145.596 ms | 118 | 310,372 | 22,908 | 349,615 | incomplete |
| compact canvas | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 9,001.83 ms | 19,685.63 ms | 142.754 ms | 119 | 313,492 | 23,165 | 352,977 | incomplete |

Quality rates and completed-document p50 exclude the timed-out record. The
inclusive p50 treats the incomplete record as its observed timeout duration;
neither latency column is a complete-suite comparison. Both AgentV bundles
fail the runtime timeout criterion, so ship gates remain blocked.

## Signals and next run

- The accepted cache, parser bridge, cold-initialization, and alarm-lifecycle
  repairs increase exact progress versus c1738 and preserve typed evidence,
  but do not finish the shared blocker.
- `sdlc_delivery.json` now correctly reports `positive=false`,
  `measurement_complete=false`, and a typed `measurement_incomplete` reason
  for each arm. Partial quality equality is not a model result.
- Reusing the verified final branch state and reusing all intermediate verified
  branch states both changed exact candidate/witness ordering in parity tests;
  both optimizations were reverted.
- The next repair must preserve exact search order and proof budgets. The
  current priority is mechanical parser-state copy/allocation cost, followed by
  the identical frozen replay.
- Lean/formal status is not applicable to this incomplete screening result,
  but the repair delivery remains gated by `verify_formal_contracts`.

Machine-readable evidence is in
[`autotrain-cycle-1739-canvas-timeout.json`](autotrain-cycle-1739-canvas-timeout.json).
