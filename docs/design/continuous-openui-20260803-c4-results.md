# Continuous autotrain: 2026-08-03 cycle 4 — control + component-plan decode timeout (incomplete measurement)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `7a63cce4` (docs commit from cycle c3, on top of `main` tip `318492c5`)

**Verdict:** measurement incomplete — AgentV smoke decode timed out on all 3
documents for **both** arms. Training itself completed fast (control 4.09s,
component-plan 8.15s wall, 22 steps, matched 1,755,764 params); the timeout
is isolated to the eval/decode stage. No primary-metric comparison is
available. Non-positive — no stacked PR.

| Arm | Params | Seed | Train wall (s) | Smoke decode timeouts | structural_similarity |
| --- | ---: | ---: | ---: | ---: | --- |
| control (`grammar levers off`) | 1,755,764 | 100002 | 4.09 | 3/3 | unmeasured |
| component-plan | 1,755,764 | 100002 | 8.15 | 3/3 | unmeasured |

## Why this is not treated as a new harness bug

This campaign's per-experiment wall budget is `max_wall_minutes≈1.1667`
(~70s), most of which is consumed by AgentV/Node bridge startup, leaving
little headroom for decode. The control hypothesis for this cycle
specifically was **"both grammar levers off"** — an unconstrained-decode
diagnostic arm. Per `AGENTS.md`'s decode invariants, unconstrained arms are
diagnostic controls only, never defaults, and are expected to be slower
(potentially much slower) than constrained decoding. That the *control* arm
(not just the candidate) times out here is consistent with that expectation
rather than a new regression. The same "AgentV decode timeout under a tight
wall budget" pattern already recurs across many prior cycles (e.g.
`autotrain-cycle-1821`, `-1796`, `-1770`, `-1766`, `-1751` in
`docs/MODEL_CARD.md`), so no speculative code fix is applied here without a
reproduced code-level defect.

## Open handoff actions (not yet closed)

The driver's auto-generated `cycle_handoff.json` names:

1. `repair_harness` (owner `improve-openui-harnesses`, family `model_build`) —
   needs a dedicated harness-focused cycle to either fix a genuine bug or
   document that unconstrained-grammar control arms require a larger
   `decode_timeout_seconds` / `max_wall_minutes` allocation.
2. `retry_measurement` — replay the identical frozen `control`/`component-plan`
   pair (seed 100002) once (1) is resolved.

Per autotrain continuous-mode ordering, no new model hypothesis is selected
ahead of the `repair_harness` prerequisite.

## SDLC Phase A

**Non-positive** (`primary_metric_unavailable` + `fixture_insufficient_n_alone`).
Local commit only, no stacked PR.

Full JSON: [`continuous-openui-20260803-c4-results.json`](continuous-openui-20260803-c4-results.json).
