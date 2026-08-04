# Autotrain c1854: slot-component exposure cap

**Verdict:** real fixture signal, queued for fresh-seed confirmation; not a
ship or promotion result.

Targeted exposure plus the implemented slot-component owner improved the
fixture's structural and meaningful-program signals, but at a substantial
decode-cost and capacity cost. The result is useful enough to confirm once,
not strong enough to claim generalization.

| Arm | Params | Loss | Struct | MPR | Recall | Binder F1 | Fidelity | Reward | p50 ms | Tokens | Forwards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| capacity-aware control | 1,608,962 | 16.2126 | .0575 | 0 | 0 | 0 | 0 | 0 | 917 | 21 | 4 |
| slot-component exposure cap | 1,613,477 | 12.6578 | .1353 | .333 | .167 | .633 | .528 | .765 | 1005 | 36 | 7 |

The candidate wins the primary fixture metric by `.0778` and raises MPR and
recall, but p50 latency rises `9.63%`, tokens `71%`, forwards `75%`, and exact
AST/canonical rates remain zero. Smoke `n=3` and absent production suites are
the binding blockers. Run the exact fresh-seed confirmation; keep formal Lean
promotion preflight locked and do not sync or serve this checkpoint.

Machine evidence:
[`autotrain-cycle-1854-slot-component-exposure-cap-positive.json`](autotrain-cycle-1854-slot-component-exposure-cap-positive.json).
