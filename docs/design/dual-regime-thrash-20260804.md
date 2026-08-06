# Dual-regime thrash (isolate + climb + timeout→decode residual)

**Honesty:** production harness design. Not a ship-quality model claim.
Fixture screening still uses smoke-scale steps and published smoke suites.

## Decision

Continuous autotrain screening now has three first-class thrash regimes:

| Regime | Control baseline | Treatment | When |
| --- | --- | --- | --- |
| `isolate` | Zeroed / precursor package (causal OFAT) | Single bank arm extras | Default when no sticky champion |
| `climb` | Sticky champion knobs (`confirmed` / `climb_accepted` / `promoted`) | Champion + residual arm extras | Sticky recipe available; screening only |
| `timeout_decode_residual` | Climb baseline if sticky else isolate | Prefers decode-cost residual arm | Prior cycle incomplete with `timeout_dominant_phase=compiler_ms` |

Confirm and promote paths still **freeze** champion lever sets and do not thrash the bank.

## Decode residual set

Registered production thrash arms only (never unconstrained / `compiler_decode_mode=off`):

- `bounds` — `grammar_completion_bounds=True`
- `canvas` — `compact_active_canvas=True`
- `both` — bounds + canvas
- `cached-compiler-decision-margin` — `grammar_equivalence_cache=True`

Canonical list: `DECODE_RESIDUAL_SLUGS` in
`src/slm_training/autoresearch/thrash_regime.py` and
`measurement.thrash_regimes.decode_residual_slugs` in
`resources/experiments/autotrain_climb/policy.v1.json` (policy version **v6**).

## Artifacts

- `matrix-proposal.json`: `thrash_regime`, `decode_residual_slugs`, regime-tagged
  `selection_rationale`
- `cycle_handoff.json` (`AutotrainCycleHandoffV1.thrash_regime`): same label when the
  screening matrix carried one (confirm/promote leave it null)
- `thrash_timing.json` / ledger: optional `thrash_regime` echo
- `sdlc_delivery.json`: `thrash_regime` when matrix carried one
- Driver logs: `THRASH_REGIME …` and `THRASH_ROTATE … regime=…`

## Climb no-op residual rule

When the sticky champion already includes a residual arm’s knobs (e.g. bounds
champion thrashing `bounds` again), the matrix **skips** that arm after
comparing *materialized* `knobs()` signatures (not sparse residual overlays).
Recommendation retargets to the next distinct treatment.

## Invariants preserved

- Size-matched dual-arm attribution
- Grammar-constrained decode; no unconstrained residual default
- Climb baseline does not silently promote incomplete timeout cycles as quality wins
- Isolate multi-seed bank close unchanged when no climb baseline and no timeout signal

## Implementation pointers

- Pure helpers: `src/slm_training/autoresearch/thrash_regime.py`
- Driver wiring: `scripts/run_autotrain_continuous.py` (`_matrix`,
  `_select_cycle_slug`, `_screening_regime_decision`,
  `_predecessor_compiler_ms_timeout`)
- Tests: `tests/test_autoresearch/test_thrash_regime.py`,
  regime cases in `tests/test_scripts/test_run_autotrain_continuous.py`

## Non-goals (still deferred)

Full PBT / MO-PBT / BOHB populations; L3 table-driven forest consumption;
default-on speculative n-gram or prefill schedule.
