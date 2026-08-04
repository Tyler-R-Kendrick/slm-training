# Autotrain c1731: component-plan replication and efficiency-floor repair

**Verdict:** a new-seed, exactly size-matched replication again produced no
quality difference between component-plan supervision and its zero-loss
control. The candidate's p50 advantage collapsed to 22.51 ms (0.65%), while
every quality and deterministic-work counter remained identical. The arm is
rejected as a quality null and its latency delta is below the new policy-owned
5% minimum efficiency effect.

## Result matrix

| Arm | Records | Trainable params | Parse | Binder F1 | Meaningful | Structure | Component recall | AST node / edge F1 | p50 | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control, component-plan head prebuilt | 3/3 | 1,755,764 | 1.0 | .6333 | .3333 | .4197 | .1667 | .4833 / 0.0 | 3,453.06 ms | Complete fixture control; ship gates fail |
| component-plan loss `1.0` | 3/3 | 1,755,764 | 1.0 | .6333 | .3333 | .4197 | .1667 | .4833 / 0.0 | 3,430.55 ms | Quality null; 0.65% latency delta rejected as noise-scale |

Both arms also match placeholder fidelity (.5278), reward (.7953), 38 neural
forwards, 60,749 unique completion states, 2,113 witness expansions, and 63,017
parser forks. The candidate's training loss is worse (17.7472 vs 14.0445) and
its training wall is 2.13 times the control (8.232 s vs 3.870 s).

## Replication and harness signal

| Signal | c1730 screen | c1731 replication | Interpretation |
| --- | ---: | ---: | --- |
| structural delta | 0.0 | 0.0 | Component-plan supervision has not improved decoded quality |
| meaningful delta | 0.0 | 0.0 | No semantic-density gain |
| p50 reduction | 13.23% | 0.65% | Initial fixture latency signal did not replicate |
| efficiency gain (`mpr/ms`) | 15.25% | 0.66% | c1731 is below the preregistered 5% minimum effect |
| parameters | matched | matched | Attribution is capacity-clean |

The v3 Phase A tradeoff helper treated any positive floating-point efficiency
delta as a win, overriding the climb policy's 0.01 structural minimum effect.
That made c1731 `positive` and retained the already-executed component-plan arm
despite only a 0.66% `mpr/ms` change. Policy v4 now owns
`minimum_efficiency_gain_fraction=0.05`; the helper emits
`efficiency_win_rejected_min_effect` below that bar. The exact c1731 metrics are
covered by regression tests and reclassify as non-positive without changing
the frozen historical outcome.

This is a canonical `autoresearch` harness repair. It does not change the model,
evaluation cases, grammar authority, ship gates, or Lean bounds. Screening still
has no bound LeverProof band; confirmation/promotion remains fail-closed on the
locked expectations digest, proved preflight, and v2 certificate replay.

## Honest gate state and next priorities

AgentV completed with zero execution errors. Honest gates fail smoke `n=3`,
meaningful-program rate, component recall, AST BEq, and canonical BEq; the other
required suites were not run. Structure, parse, placeholder fidelity, and reward
clear their smoke thresholds, but partial fixture clears are not ship evidence.

1. Mark component-plan supervision exhausted for this data/eval identity after
   two exact quality nulls; rotate to the size-matched component-inventory arm.
2. Require at least 5% `mpr/ms` improvement before an efficiency-only screening
   result becomes positive; confirm any such result at a new seed before promotion.
3. Keep structural similarity primary and retain parameter equality, parse,
   meaning, binder F1, component recall, AST node/edge F1, and completed latency
   in the result matrix.
4. Render screening formal status explicitly as `not_applicable`; never let that
   presentation change weaken missing-certificate handling on promotion.

Machine-readable evidence is in
[`autotrain-cycle-1731-component-plan-replication.json`](autotrain-cycle-1731-component-plan-replication.json).
