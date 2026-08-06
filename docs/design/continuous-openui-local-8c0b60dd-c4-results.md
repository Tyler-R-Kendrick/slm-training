# Continuous autotrain: 2026-08-05 (session 8c0b60dd, scheduled run) cycle 4 — batch1 runtime diagnostic is a null result (screening, session wrap)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c4`
**Integration commit:** `05491a8a` (cycle 3's docs commit, `origin/main` tip unchanged)

**Verdict:** the `generate_batch_size` (`batch1`) runtime-diagnostic arm ties
its control on every smoke quality metric — a null result, with a latency
cost and no quality gain.

| Arm | Seed | structural_similarity | binder_reference_f1 | reward_score | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 100004 | .41667 | .95238 | .9360 | 24067.22 |
| batch1 | 100004 | .41667 | .95238 | .9360 | 25549.59 |

Every quality metric (`structural_similarity`, `meaningful_program_rate`,
`binder_reference_f1`, `placeholder_fidelity`, `reward_score`) is
byte-identical between arms; only `latency_ms_p50` differs, and `batch1` is
*slower*, not faster — not a latency win either. Ship gates fail as expected
on fixture `n=3`.

## Session summary: the registered lever bank is exhausted at this recipe

Across this session's four cycles (`8c0b60dd-c1`..`c4`), every currently
registered knob at the `wf_smoke_v2` / 20-step / CPU screening recipe has now
been independently reproduced as exhausted:

1. `bounds` (knob rotation) — null delta, third reproduction across sessions
   ([cycle 1](continuous-openui-local-8c0b60dd-c1-results.md)).
2. `component-plan` — screening win, but **rejected on fresh-seed
   confirmation**, matching every prior reproduction of this fingerprint
   ([cycle 2](continuous-openui-local-8c0b60dd-c2-results.md),
   [cycle 3](continuous-openui-local-8c0b60dd-c3-results.md)).
3. `generate_batch_size` (`batch1`) — null on quality, latency-negative
   (this cycle).

No new stack layer opened this session (no positive-with-tracked-delta
result). Per the driver's own cycle-3 rank-5 priority, the next useful step
is a **new preregistered structural/meaningful-quality objective** — harness
research work routed through `openui-autoresearch` / `improve-openui-harnesses`
— rather than a fifth cycle recycling the same exhausted lever bank.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`). No stack layer. Session
wraps here: docs land locally, branch pushes, and a single PR bundles the
four docs-only cycles per the repo's established
`continuous-openui-local` delivery pattern.

## Next priorities

1. Do not recycle `component-plan` or `bounds` again this session — both
   exhausted per cycles 1–3.
2. Preregister a new structural/meaningful-quality objective before further
   screening cycles add value.
3. Push accumulated docs commits and open a PR (this session has no
   positive stack layer to deliver).

Machine evidence:
[`continuous-openui-local-8c0b60dd-c4-results.json`](continuous-openui-local-8c0b60dd-c4-results.json).
