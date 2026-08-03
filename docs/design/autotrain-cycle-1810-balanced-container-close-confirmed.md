# Autotrain c1810: balanced container-close confirms on a fresh seed

**Verdict:** champion confirmed for the promotion suite; not promoted or
shippable. Both 20-step CPU scratch arms completed at 1,608,962 parameters and
fresh seed 101810. The candidate repeats the c1809 direction across structure,
meaningful-program rate, binder F1, component recall, fidelity, and reward.

| Arm | Loss | Structure | Binder F1 | Recall | MPR | Fidelity | Reward | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weight 0 control | 19.39207 | .0964 | 0 | .08333 | 0 | 0 | 0 | 906.83 |
| balance .25 + close 1 | 25.94068 | .174167 | .63333 | .25 | .33333 | .52778 | .76533 | 983.80 |

Decode remains bounded: both arms emit 24 tokens with four neural forwards.
Candidate compiler time is `2202.69` versus `2172.36` ms, completion states are
`31,118` versus `31,111`, and parser forks are `32,126` versus `32,115`.
This is a 76.97 ms p50 cost, not a continuation explosion.

The v300 telemetry repair is verified in-run. On the final candidate batch,
component CE is `22.3189` over eight tokens, `STRUCT` CE is `11.4891` over 42
tokens, the typed-family auxiliary is `4.2260`, and close-alignment loss is
`1.36e-6` over five rows with zero margin violations. Training signals from
both objectives are now present together.

The local explicit-no-sync checkpoint SHA-256 values are `274e7d34...a1d`
(control) and `f5d05181...cb00` (candidate). Neither is reusable, promoted,
synced, or ship evidence. AgentV bundles are complete, fixture gates fail, and
Lean is `not_applicable:confirmation`. The next cycle must run the exact matched
recipes under the promotion suite and Lean preflight; only that result can
advance or reject the confirmed champion.

Machine evidence:
[`autotrain-cycle-1810-balanced-container-close-confirmed.json`](autotrain-cycle-1810-balanced-container-close-confirmed.json).
