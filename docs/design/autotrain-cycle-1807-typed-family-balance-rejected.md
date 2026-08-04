# Autotrain c1807: typed-family balance improves quality but explodes decode

**Verdict:** reject. Both 21-step CPU scratch arms completed at 1,608,962
parameters and seed 101807. Count-normalized component/`STRUCT` supervision
improves smoke structure `.0575→.104467`, binder F1 `.6333→1.0`, component
recall `0→.0833`, and fidelity `.5278→1.0`; it still produces no meaningful
program and raises p50 latency `1084.71→5868.74` ms.

| Arm | Loss | Structure | Binder F1 | Recall | MPR | Fidelity | Reward | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weight 0 control | 12.12485 | .0575 | .63333 | 0 | 0 | .52778 | .76533 | 1,084.71 |
| family balance .25 | 15.43139 | .104467 | 1.0 | .08333 | 0 | 1.0 | .93700 | 5,868.74 |

The objective is active and balanced: on the final matched batch, component CE
falls `23.6554→18.5659` and `STRUCT` CE is stable `7.8651→7.8352`; the added
family-mean auxiliary is `3.3001`. The decode trace exposes the cost: emitted
tokens rise `27→201`, neural forwards `5→51`, compiler time `2.27→10.18` s,
completion states `31,499→115,693`, and parser forks `32,891→123,198`.

The local explicit-no-sync checkpoint SHA-256 values are `3886a403...e0c79`
(control) and `1850d25f...ff06e` (candidate). Neither is reusable, promotable,
synced, or ship evidence. AgentV bundles are complete, fixture gates fail, and
Lean is `not_applicable:screening`.

The next distinct hypothesis uses the existing grammar-derived compiler
alignment objective only at gold container-close decisions where `)` or `]`
competes with a legal comma continuation. This targets the observed runaway
continuation without a decode heuristic, parameter growth, or any weakening of
the certified grammar domain.

Machine evidence:
[`autotrain-cycle-1807-typed-family-balance-rejected.json`](autotrain-cycle-1807-typed-family-balance-rejected.json).
