# E310 component-plan attention pooling — 2026-07-17

E310 repeats E308 on the same E307 v4 corpus, seed, CPU scratch architecture,
20k target-token budget, and frozen E305 evaluation policy. The only changed
model knob replaces mean prompt pooling with a learned attention pool for the
component-plan head and its decode bias. The base denoiser path is unchanged.

The run stopped at 420 steps / 20,001 target tokens in 156.58 seconds.
Checkpoint SHA:
`7d8888e06ebad0a6cff4e41814bd2da4e13d86977e85e52b402fc8f2b6f2f7b3`.
It is a local scratch artifact with explicit `--no-sync-checkpoints`.

| Measure | E308 mean pool | E310 attention pool |
| --- | ---: | ---: |
| Weighted NLL | **4.8836** | 4.8842 |
| Broad NLL | **4.9812** | 4.9814 |
| Final-20 plan loss | 2.3286 | **2.3282** |
| Root accuracy | 0.8500 | 0.8500 |
| Bound top-k recall | 0.4104 | 0.4104 |
| Bound-count MAE | 0.3440 | **0.3439** |

Loss-suite AgentV passes 1/1. The honest board's selected-metric JSON hash is
identical to E308/E309 (`6c5b60a3…`): all five suites exactly match, seven
thresholds fail, and AgentV is 2/5.

**Verdict:** reject attention pooling. The plan target still behaves as a
global inventory classifier, and changing only how prompt states are pooled
does not improve component selection. The next lever must change the
supervision target or connect prompt tokens to component decisions directly.
