# Continuous autotrain: 2026-08-03 cycle 2 — component-plan structural win (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `1cb9ef56` (docs commit from cycle 1, on top of `main`
tip `e1f5e4f0`)

**Verdict:** `component-plan` beats its size-matched control on the declared
primary at this seed and queues for fresh-seed confirmation. Fixture
screening only — not a ship or promotion claim.

| Arm | Params | Seed | structural_similarity | component_type_recall | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 1,755,764 | 100002 | .32667 | 0 | 9416.83 |
| component-plan | 1,755,764 | 100002 | .38280 | .16667 | 8369.16 |

Primary improvement `+.05613`; component-type recall improves by `.1667`;
p50 latency improves 11.1%. Both arms parse all 3 documents and use the same
1,755,764 trainable parameters. `meaningful_program_rate`,
`binder_reference_f1`, `placeholder_fidelity`, `ast_beq_rate`, and
`canonical_beq_rate` are **0 on both arms** — the win is confined to raw
structural similarity and component-type recall, not full program
correctness, so this does **not** establish a meaningful-program win.

Ship gates fail as expected: `insufficient_n` (n=3, need 20), and
`held_out`/`adversarial`/`ood`/`rico_held` suites were not run.

## Relation to prior component-plan evidence

This is the same hypothesis and the same trainable-param count (1,755,764)
flagged in
[`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md)
as needing fresh-seed confirmation. This cycle is an **independent first
measurement** under a brand-new campaign (not a replay of that frozen pair),
so it does not by itself satisfy that outstanding confirmation requirement —
it adds one more positive data point for the same structural direction.

## SDLC Phase A

**Positive** (`primary_metric_win`). Per `sdlc` autotrain-iteration-delivery,
documenting this result creates the reviewable delta required to open a
stacked layer for this cycle.

## Next priorities (ranked by the driver)

1. Fresh-seed confirmation of `component-plan` vs control with the identical
   size-matched recipe before any promotion or Lean preflight
   (confidence 0.95).
2. Keep promotion formal preflight locked until fresh confirmation
   establishes a champion (confidence 1.0, lean assumption).
3. Do not treat this as ship-ready: `meaningful_program_rate`,
   `binder_reference_f1`, and `placeholder_fidelity` are all still 0.

Machine evidence:
[`continuous-openui-20260803-c2-results.json`](continuous-openui-20260803-c2-results.json).
