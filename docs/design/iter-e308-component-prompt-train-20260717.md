# E308 component-prompt matched train — 2026-07-17

E308 tests whether E307's component-aligned natural prompts improve the trained
choice-native component plan. It matches E304's CPU scratch recipe: d64/h2,
one context and two denoiser layers, choice tokenizer, diffusion corruption,
batch 2, seed 0, no DESIGN context, plan loss/decode weight 1, and a 20k target
token budget. The only intended intervention is the 592-row E307 v4 corpus.
Its larger prompt vocabulary adds 2,432 context-embedding parameters (0.61%);
all architecture and explicit recipe knobs are otherwise fixed.

Training stopped at 420 steps / 20,001 target tokens after 146.99 seconds.
The local-only checkpoint SHA is
`f56089052dbc804754fb0d201bd7a4d6cbd356b6d72b42959147fce9233e2b55`.
It was intentionally run with `--no-sync-checkpoints`.

## Loss result

| Frozen suite | E304 | E308 | Delta |
| --- | ---: | ---: | ---: |
| Weighted NLL | 5.1647 | **4.8836** | -0.2811 |
| Binding | **5.5514** | 6.7068 | +1.1554 |
| Structural | 4.1387 | **3.3550** | -0.7837 |
| Repair | 5.6202 | **4.1703** | -1.4499 |
| Schema OOD | 5.3263 | **4.6710** | -0.6553 |
| Broad | 5.4165 | **4.9812** | -0.4353 |

Loss AgentV passes 1/1. Final-20 plan averages are loss 2.3286, root accuracy
0.85, bound top-k recall 0.4104, and bound-count MAE 0.3440. Binding and
bound-recall regress despite lower aggregate NLL.

## Honest ship board

The unchanged E305 policy uses tree decode, `decode_min_content=-1`, plan
weight 1, visible-slot constraints, no DESIGN context, and no unconstrained
fallback.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.5278 | 0.4642 | 0.1667 | 0.2497 |
| held_out | 5 | 1.0000 | 0.0000 | 0.2800 | 0.3369 | 0.0000 | 0.0000 |
| adversarial | 4 | 1.0000 | 0.5000 | 0.5417 | 0.4744 | 0.3750 | 0.4245 |
| ood | 4 | 1.0000 | 0.0000 | 0.2583 | 0.3750 | 0.0000 | 0.0000 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.5417 | 0.3333 | 0.3333 | 0.5567 |

Smoke, held, adversarial, and OOD are exactly E305. Limited RICO regresses
from meaningful 1.0 / recall 0.5556 / reward 0.8515. Seven thresholds fail and
AgentV remains 2/5.

**Verdict:** reject E308. Component-name prose improves aggregate denoising
loss but does not improve frozen component decisions. Do not scale this corpus
or claim NLL as quality; the next lever must strengthen prompt-conditioned
component-plan supervision rather than add more equivalent prompt rows.
