# E309 component-plan weight scaling — 2026-07-17

E309 repeats E308 on the same E307 v4 corpus, seed, CPU scratch architecture,
20k target-token budget, and frozen E305 evaluation policy. The only changed
training knob is component-plan loss weight 1→4; decode weight stays 1.

The run stopped at 420 steps / 20,001 target tokens in 145.29 seconds.
Checkpoint SHA:
`18da6dc916bc2a5e4b84e0e96b648eb2108c9437451e3da76662b13df6d59075`.
It is a local scratch artifact with explicit `--no-sync-checkpoints`.

| Measure | E308 weight 1 | E309 weight 4 |
| --- | ---: | ---: |
| Weighted NLL | **4.8836** | 4.8847 |
| Broad NLL | 4.9812 | **4.9732** |
| Final-20 plan loss | **2.3286** | 2.3361 |
| Root accuracy | 0.8500 | 0.8500 |
| Bound top-k recall | 0.4104 | 0.4104 |
| Bound-count MAE | **0.3440** | 0.3466 |

Loss-suite AgentV passes 1/1. The honest board's selected-metric JSON hashes
are identical (`6c5b60a3…`): all five suites exactly match E308, seven
thresholds fail, and AgentV is 2/5.

**Verdict:** reject E309 and stop plan-weight scaling. The existing head is
not limited by scalar loss weight; the next improvement must change the
supervision target or representation.
