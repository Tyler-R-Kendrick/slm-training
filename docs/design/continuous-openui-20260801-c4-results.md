# Continuous autotrain cycle 4 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260801-c4` (predecessor: `continuous-loop-20260801-c3`) |
| Role / intent | promotion |
| Source | `1bdfb14ebcf2393976a7c969e7bdd449fc5ada39` |
| Device | CPU |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` (`smoke` + `held_out` suites) |
| Primary endpoint (locked) | `held_out.structural_similarity`, direction increase, min effect 0.01 |
| Wall cap | 3 minutes |

Hypothesis under test: *doubling steps without other levers only raises cost
and does not improve unit decode latency.*

## Run matrix

| Arm | smoke mpr | smoke struct_sim | smoke p50 (ms) | held_out struct_sim | held_out p50 (ms) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| c4-control | 0.333 | 0.417 | 19694.76 | 0.38248 | 19385.10 | eval completed; ship gates fail (insufficient n) |
| c4-steps | 0.333 | 0.510 | 10244.61 | 0.37006 | 10256.33 | eval completed; ship gates fail (same) |

Primary delta (steps − control) on the **locked** `held_out.structural_similarity`:
**-0.0124** (regression, below the +0.01 minimum-effect floor).

## Diagnostics

1. The `steps` arm nearly halved p50 latency on both suites (~19.4-19.7s →
   ~10.2-10.3s) and even improved `smoke.structural_similarity` (0.417 →
   0.51) — an attractive-looking smoke-only result.
2. The campaign's **locked primary endpoint is `held_out.structural_similarity`**,
   not the smoke metric. On that endpoint the candidate regressed
   (0.38248 → 0.37006), so SDLC Phase A correctly classifies this cycle
   **non-positive** regardless of the smoke-suite improvement and the large
   latency win.
3. This is a concrete instance of why the held-out primary is locked before
   execution: a smoke-only read of this cycle would have looked like a clear
   win.
4. Ship gates still fail on `insufficient_n` for both arms (fixture suites
   are far below the `n>=20` gate) — expected, not a regression signal.

## SDLC Phase A

`classification=NON_POSITIVE stack_layer=False` — no stacked PR opened for
this cycle. Docs-only, local-commit delivery per
`autotrain-iteration-delivery`.

## Next-run priorities

1. Do not promote the steps-only variant; the locked held-out primary
   regressed.
2. Check whether the smoke-suite gain is fixture overfitting given `n=3` per
   suite before trusting it as a real signal.
3. Re-run the steps lever against a larger held-out fixture before drawing
   further conclusions.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c4/`
- Runs: `.../runs/c20260801-c4-control/`, `.../runs/c20260801-c4-steps/`
- JSON twin: `continuous-openui-20260801-c4-results.json`
- Predecessor cycle: `continuous-openui-20260801-c3-results.md`
