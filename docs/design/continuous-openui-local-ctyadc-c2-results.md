# Continuous autotrain: 2026-08-03 (session ctyadc, scheduled) cycle 2 — measurement incomplete (operator-caused)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c2`

**Verdict:** no attribution claim this cycle. The `component-plan` candidate
completed (`structural_similarity=.3828`, matching prior independent
reproductions of this hypothesis), but the matched `control` arm never ran.

| Arm | Params | Parse | Struct | MPR | Recall | Binder F1 | Fidelity | p50 ms | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | — | — | — | — | — | — | — | — | not executed |
| component-plan | 1,755,764 | 1.000 | .3828 | 0 | .1667 | 0 | 0 | 23961.35 | gate reject |

## Root cause: operator git mutation mid-cycle, not a harness bug

The cycle-2 driver locked `integration_commit=14a748ca` (the cycle-1 docs
commit) at start. Before the `control` arm ran, this session amended that
same commit — `git commit --amend --no-edit -S` — to attach a required SSH
signature after stop-hook feedback flagged the tip commit as unsigned. That
rewrote `HEAD` to `de14a9d0` while the supervised cycle was still in flight.
`scripts/autoresearch.py`'s `_validate_continuous_commits` correctly rejected
the `control` run because `HEAD` no longer matched the campaign's locked
`integration_commit`:

```
ValueError: integration_commit must be the current checked-out HEAD
```

This is **operator process error**, not a canonical-harness defect — no
`HarnessSignalV1` and no `improve-openui-harnesses` routing applies. The fix
is procedural: never mutate git `HEAD` (amend, rebase, reset) while a
supervised autotrain cycle is running. No further mid-cycle git mutation will
occur for the remainder of this loop.

## SDLC Phase A

**Non-positive** (`measurement_incomplete`). Per `sdlc`
autotrain-iteration-delivery, no new stack layer opens. The driver's own
handoff queues a typed `retry_measurement` action
(`frozen_manifest_sha256=1c312e8c...df9bb0f`): cycle 3 replays the identical
frozen `c2` control + `component-plan` arms before any new hypothesis is
attempted.

## Next priorities

1. Replay the exact frozen `c2` arms in cycle 3 (rank 1, confidence 0.95) to
   complete the attribution the git-mutation interruption blocked.

Machine evidence:
[`continuous-openui-local-ctyadc-c2-results.json`](continuous-openui-local-ctyadc-c2-results.json).
