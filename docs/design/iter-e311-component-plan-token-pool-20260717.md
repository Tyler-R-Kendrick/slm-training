# E311 component-plan token pooling — 2026-07-17

E311 repeats E308 on the same E307 v4 corpus, seed, CPU scratch architecture,
20k target-token budget, and frozen E305 evaluation policy. The sole model
delta applies the existing component-plan classifier to every prompt token,
then uses component-specific log-mean-exp pooling. It adds no parameters and
directly tests whether global pooling erases component-bearing words.

The run stopped at 420 steps / 20,001 target tokens in 130.06 seconds.
Checkpoint SHA:
`e0e8a2951c227a928167f73c038d7897896f3812405071cb552371c9fafaae32`.
It is a local scratch artifact with explicit `--no-sync-checkpoints`.

| Measure | E308 mean pool | E311 token pool |
| --- | ---: | ---: |
| Weighted NLL | 4.8836 | **4.8819** |
| Broad NLL | 4.9812 | **4.9806** |
| Final-20 plan loss | 2.3286 | **2.3283** |
| Root accuracy | 0.8500 | 0.8500 |
| Bound top-k recall | 0.4104 | 0.4104 |
| Bound-count MAE | 0.3440 | **0.3418** |

Loss-suite AgentV passes 1/1. The honest board's selected-metric JSON hash is
identical to E308–E310 (`6c5b60a3…`): all five suites exactly match, seven
thresholds fail, and AgentV is 2/5.

Decoder telemetry shows the plan bias was applied 35 times but changed only
one legal choice: 0/3 smoke, 0/5 held-out, 0/4 adversarial, 0/4 OOD, and 1/19
limited RICO.

**Verdict:** reject token pooling at decode weight 1. Global pooling was not
the limiting factor. Before changing the training target again, test whether
the learned plan signal can affect decoding at a stronger inference-only
weight.
