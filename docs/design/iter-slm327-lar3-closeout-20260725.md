# SLM-327 / LAR3-05: Recurrence-line closeout (slm327_lar3_closeout)

Matrix set: `slm327-recurrent-core-activation` · Version: `slm327-v1` · Status: **closeout**
Decision: **recursive_core_negative** — the recurrence line closes for this repository state; infrastructure is preserved (default-off).

## Why the activation experiment cannot run

LAR3-01..04 are all closed `not_authorized` (same LAR3 entry-gate evidence), so no candidate configurations exist to advance into a powered run:

- [SLM-319 closeout](iter-slm319-lar3-gate-closeout-20260725.md) — LAR3 entry gates unmet
- [SLM-321 closeout](iter-slm321-lar3-gate-closeout-20260725.md)
- [SLM-324 closeout](iter-slm324-lar3-gate-closeout-20260725.md)
- [SLM-326 closeout](iter-slm326-lar3-gate-closeout-20260725.md)

## Independent evidence against the line

| Source | Verdict |
| --- | --- |
| SLM-233 matched recursive-depth campaign (`RecursiveCoreGateV2`) | `architecture_not_identifiable`; `rsc3`/`rsc4` in blocked claims |
| SLM-282 preregistered contraction audit | `recursive_core_negative` (1/2 required seeds) |
| SLM-230/231/232 observability/dynamics/z-use | `stagnant` / `expansive_unstable` / `unstable` |
| SLM-317 repair advancement screen | `inconclusive` (value gate failed, Wilson [0.0, 0.194] vs 0.05) |
| SLM-139 stochastic width | already closed `no_supported_probabilistic_regime` |

## Consequences

- **SLM-139:** remains closed; nothing here reopens it.
- **LAR3:** recurrence line closed for this repository state; the recursive
  denoiser, telemetry, and fixtures are preserved default-off.
- **LAR4:** remains blocked — it opens only on a powered semantic
  improvement that does not exist.
- **Production defaults, checkpoints, gates:** unchanged.

## Reopening conditions

All of: `floor_escaped` on the semantic floor gate, a recurrence-health
audit returning `recursive_core_positive` (≥2 seeds), and a passing
valid-state repair advancement screen.
