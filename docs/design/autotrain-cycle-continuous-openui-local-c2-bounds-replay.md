# Autotrain `continuous-openui-local` c2: frozen bounds replay, quality-neutral reject

**Verdict:** fixture measurement **complete** (first full scoreboard this loop
gets since the [c1 AgentV-SDK infra
finding](autotrain-cycle-continuous-openui-local-c1-agentv-bootstrap.md)),
quality-null reject. `control` and `bounds` are identical on every smoke
quality metric; `bounds` is slower.

## Result matrix

| Arm | n | parse | meaningful | structure | component recall | AST béq | canonical béq | placeholder fidelity | reward | p50 ms | Ship gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 3 | 1.0 | 0.0 | 0.0575 | 0.0 | 0.0 | 0.0 | 0.5278 | 0.0 | 3748.48 | fail (fixture n, quality) |
| bounds | 3 | 1.0 | 0.0 | 0.0575 | 0.0 | 0.0 | 0.0 | 0.5278 | 0.0 | 3897.55 | fail (fixture n, quality) |

Both arms fail the honest ship gates identically: `smoke:insufficient_n`
(n=3, need ≥20) plus every quality-threshold assertion
(`meaningful_program_rate`, `structural_similarity`, `component_type_recall`,
`ast_beq_rate`, `canonical_beq_rate`, `reward_score`), and all four non-smoke
suites (`held_out`/`adversarial`/`ood`/`rico_held`) are missing. This is a
smoke-scale fixture screen, not a ship claim.

## SDLC Phase A classification

`SDLC_PHASE_A NON_POSITIVE` — `fixture_insufficient_n` on both arms plus
`primary_metric_null_or_worse` (`smoke.structural_similarity` control =
candidate = 0.0575, improvement 0.0). Per the autotrain-iteration-delivery
positive-result gate, fixture `insufficient_n` / null deltas alone are never
positive. **No stack layer opened for this cycle** — local commit + docs only.

This closes out the `retry_measurement` action queued by c1's `repair_harness`
handoff: the identical frozen `control`/`bounds` pair
(`replay_of_manifest_sha256 c4c7bc4837d5a4f076eaa574ad119c9a9709376e9923d8becf7360215f813528`)
now completes end to end with a usable scoreboard, so the c1 harness incompleteness
is confirmed resolved rather than merely re-attempted.

## Next-run priorities (ranked, from the typed handoff)

1. **model** — test the distinct size-matched `component-plan` quality
   hypothesis next (the bounds/control pair is now exhausted for this loop
   without a new preregistered hypothesis).
2. **evaluation** — keep the matched control every cycle to avoid false
   positives from recipe drift.
3. **infrastructure** — soft ship-gate fails on fixture `n` never stop the
   continuous loop (this cycle is exactly that case).

No checkpoint was created or promoted, so no `MODEL_CARD.md` / README update is
required. Machine-readable values are in
[`autotrain-cycle-continuous-openui-local-c2-bounds-replay.json`](autotrain-cycle-continuous-openui-local-c2-bounds-replay.json).
