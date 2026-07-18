# E460 E396 semantic-slot full-RICO refresh — 2026-07-18

E460 refreshes E454's repaired 1,500-row RICO evidence after E459's semantic
slot-density change. The impact audit finds 49 explicit Switch/Slider rows in
shards 0–14. Those shards are rerun; shard 15 is provably unaffected and may
be reused from E454. Every process is externally capped at 290 seconds with a
ten-second forced kill. Interrupted or timed-out shards are excluded.

Recipe: unchanged E396 checkpoint, CPU, local HF context, 320-token grammar
LTR, automatic content floor, component-plan weight 2, slot-component weight
8, visible prompt-component constrained decode, honest constrained slot
contracts, eight generation steps, three attempts, and no fallback.

| Shard | Rows | Affected rows | n | Meaningful | Fidelity | Structure | Recall | Reward | Status |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0–96 | 1 | 96 | 1.0 | 1.0 | 0.8852 | 1.0 | 0.9955 | complete |
| 1 | 96–192 | 4 | 96 | 1.0 | 1.0 | 0.8652 | 1.0 | 0.9936 | complete r2 |
| 2 | 192–288 | 2 | 96 | 1.0 | 1.0 | 0.8541 | 1.0 | 0.9943 | complete |
| 3 | 288–384 | 3 | 96 | 1.0 | 1.0 | 0.8770 | 1.0 | 0.9928 | complete |
| 4 | 384–480 | 6 | 96 | 1.0 | 1.0 | 0.8803 | 1.0 | 0.9925 | complete |
| 5 | 480–576 | 3 | 96 | 1.0 | 1.0 | 0.8761 | 1.0 | 0.9935 | complete |
| 6 | 576–672 | 2 | 96 | 1.0 | 1.0 | 0.8836 | 1.0 | 0.9922 | complete |
| 7 | 672–768 | 6 | 96 | 1.0 | 1.0 | 0.8766 | 1.0 | 0.9932 | complete |
| 8 | 768–864 | 4 | 96 | 1.0 | 1.0 | 0.8639 | 1.0 | 0.9946 | complete |

Shard 0 completes normally in about 187 seconds with zero failures, fallback,
or decode timeouts. Its affected row improves structure 0.9069→1.0 while
preserving fidelity, recall, and reward 1.0. Diagnostic AgentV is 0/5 with
zero execution errors because the complete RICO and four bounded suites are
absent. Shard 1 completes normally in about 216 seconds with zero failures,
fallback, or decode timeouts. Its first execution is excluded because the
style scrubber removed a valid SwitchItem identifier `"m"` before reward
evaluation. The corrected `r2` preserves identical predictions, restores the
affected exact-quality row's reward from 0 to 0.961, and is the only shard-1
artifact admitted to E460.
Shard 2 completes normally in about 206 seconds with zero failures, fallback,
or decode timeouts. Relative to E454, structure improves 0.8521→0.8541 while
reward decreases 0.9948→0.9943; meaningful rate, fidelity, and recall remain
1.0.
Shard 3 completes normally in about 216 seconds with zero failures, fallback,
or decode timeouts. Structure improves 0.8687→0.8770 and recall
0.9948→1.0; reward remains 0.9928.
Shard 4 completes normally in about 193 seconds with zero failures, fallback,
or decode timeouts. Structure improves 0.8641→0.8803 and recall
0.9844→1.0; reward changes 0.9927→0.9925.
Shard 5 completes normally in about 217 seconds with zero failures, fallback,
or decode timeouts. Structure improves 0.8747→0.8761 while recall remains
1.0 and reward remains 0.9935.
Shard 6 completes normally in about 203 seconds with zero failures, fallback,
or decode timeouts. Structure improves 0.8768→0.8836 and recall
0.9948→1.0; reward changes 0.9926→0.9922.
Shard 7 completes normally in about 220 seconds with zero failures, fallback,
or decode timeouts. Structure improves 0.8637→0.8766 and recall
0.9896→1.0; reward changes 0.9934→0.9932.
Shard 8 completes normally in about 220 seconds with zero failures, fallback,
or decode timeouts. Structure improves 0.8631→0.8639 while recall remains
1.0 and reward remains 0.9946. This is the first Slider-bearing shard.

**Current verdict:** 864/1500 refreshed rows are complete. This is a partial
diagnostic, not a full-RICO, ship-gate, champion, or production HF claim.
