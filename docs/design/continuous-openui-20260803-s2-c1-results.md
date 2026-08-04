# Continuous autotrain: 2026-08-03 session 2, cycle 1 (non-positive)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c1`
**Base commit:** `9dcfa7e6` (current `main` tip; already includes the earlier
same-day session's c1/c2 via merged
[#1369](https://github.com/Tyler-R-Kendrick/slm-training/pull/1369))

| Arm | Params | parse | MPR | structural_similarity | binder F1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | 1.0 | 0.0 | .05750 | .6333 | 1250.52 |
| bounds | 1,608,962 | 1.0 | 0.0 | .05750 | .6333 | 1271.20 |

**Verdict: non-positive.** This is a second `continuous-openui-local`
session started fresh today; the driver's campaign-id is deterministic on
`loop_id` + date and its local lineage lives only in the (gitignored)
`outputs/` tree, so a brand-new container re-derives the same campaign id
(`...-c1`) and re-selects the same `bounds` hypothesis the earlier session
already screened. Result reproduces that screen almost exactly (structural
tie, latency noise only) — `bounds` remains exhausted. Named `s2-c1` (not
`c1`) in this doc's filename to avoid overwriting the already-merged
[`continuous-openui-20260803-c1-results.json`](continuous-openui-20260803-c1-results.json).

Per `sdlc` autotrain-iteration-delivery: **no stacked PR** for this cycle's
model result — docs-only local commit.

## Harness signal found and worked around (not re-fixed in code)

Evaluation initially crashed for every arm:

```
RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER
```

and, once `npm ci` supplied `node_modules/@agentv/core`, the pinned Node
runner then refused to start at all because the container's ambient
`NODE_OPTIONS` (`"--import tsx" --max-old-space-size=8192`) is malformed for
a plain `node` invocation.

**This is not a new defect.** [PR #1351](https://github.com/Tyler-R-Kendrick/slm-training/pull/1351)
(open since 2026-08-02, still unmerged) already carries the canonical fix —
sanitizing the AgentV subprocess environment in
`src/slm_training/evals/agentv.py`, mirroring the existing `_sanitized_env()`
pattern already used by the GraphQL-JS bridge — plus a regression test and an
`evals.agentv` version bump. Rather than duplicate that diff on a second
branch, this session ran `npm ci` and invoked the driver with `NODE_OPTIONS`
unset (session-local workaround only, no source change) to get a real
measurement. **#1351 should be merged** so future fresh-container sessions
stop rediscovering this same blocker from scratch.

## Next priorities (ranked by the driver)

1. `component-plan` fresh-seed confirmation vs matched control — the actual
   objective of this session's loop (rank 1, confidence 0.90).
2. Keep the matched control as the baseline every cycle.
3. Merge #1351 so the AgentV NODE_OPTIONS/missing-SDK defect stops recurring.

Machine evidence:
[`continuous-openui-20260803-s2-c1-results.json`](continuous-openui-20260803-s2-c1-results.json).
