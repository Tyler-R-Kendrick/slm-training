# Autotrain c1864: `grammar_completion_bounds` latency win, quality null

**Verdict:** real latency delta, does not qualify as a positive result.

Scheduled continuous loop `continuous-openui-scheduled-72bc0d`, campaign
`continuous-loop-20260804-continuous-openui-schedu-6699f447-c1`, cycle 1. The
rotated thrash arm `bounds` toggled `grammar_completion_bounds=true` against
the matched fixture control (both size-matched at 1,608,962 params, CPU
device, `wf_smoke_v2`/21 steps, smoke suite `n=3`).

`parse_rate` held at `1.0` and `latency_ms_p50` dropped **10,815.8ms →
5,639.31ms (≈47.9% faster)** at zero decode timeouts either arm. Every quality
metric was unchanged: `meaningful_program_rate=0.0`, `structural_similarity`,
`binder_reference_f1`, `placeholder_fidelity`, `reward_score`, and exact
AST/canonical rates were identical between control and candidate.

| Arm | Params | Parse | Struct | MPR | Recall | Binder F1 | Fidelity | Exact AST / canonical | p50 ms | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 1,608,962 | 1.000 | .0575 | .000 | .000 | .633 | .528 | 0 / 0 | 10,815.80 | gate reject |
| bounds | 1,608,962 | 1.000 | .0575 | .000 | .000 | .633 | .528 | 0 / 0 | 5,639.31 | gate reject |

Per the quality-aware tradeoff gate (`autotrain-iteration-delivery.md`), a
latency-only win requires held parse/mpr with mpr ≥ ~1/3. `meaningful_program_rate`
is `0.0` on both arms here, so the loop classified this
`SDLC_PHASE_A NON_POSITIVE` (`primary_metric_null_or_worse`,
`fixture_insufficient_n_alone` — smoke `n=3` on both arms, held-out/adversarial/
OOD/RICO suites absent). No stack layer was opened for this cycle; local
commit only.

The 48% latency cut is a genuine, reproducible signal worth keeping as a
diagnostic — `grammar_completion_bounds` is cheap and harmless at this scale —
but it is not evidence of higher-quality OpenUI generation and must not be
promoted on its own. The next cycle should test the ranked priority-1
successor, the size-matched `component-plan` quality hypothesis, per
`cycle_handoff.json`.

Machine evidence: [`autotrain-cycle-1864-bounds-lever-latency-win-quality-null.json`](autotrain-cycle-1864-bounds-lever-latency-win-quality-null.json).
