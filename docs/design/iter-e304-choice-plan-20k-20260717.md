# E304 20k-token choice component plan (2026-07-17)

## Matched duration arm

E304 extends the E293 no-DESIGN choice-plan recipe from 5k to 20k target
tokens without changing model capacity or loss weights: CPU scratch, d64/h2,
one context and two denoiser layers, batch 2, seed 0, diffusion corruption,
component-plan loss/decode weight 1, and local-only checkpoints
(`--no-sync-checkpoints`). Training stops on budget at 418 steps / 20,003
target tokens in 124.08 seconds. Checkpoint SHA-256:
`2081378f2a3f11530a2193e79a0b98d4f487c2631c3f814018117bbd2677d420`.

Frozen weighted NLL improves monotonically:

| Step | Target tokens | Weighted NLL |
| ---: | ---: | ---: |
| 100 | 4,688 | 7.8280 |
| 200 | 9,627 | 5.9382 |
| 300 | 14,179 | 5.4201 |
| 400 | 19,082 | 5.3445 |
| 418 | 20,003 | **5.1647** |

Final category NLL is binding 5.5514, structural 4.1387, repair 5.6202,
schema OOD 5.3263, and broad 5.4165. All loss suites are complete and their
AgentV result passes 1/1. Across the final 20 minibatches, plan loss averages
2.3686, root accuracy 0.8750, bound top-k recall 0.4592, and bound-count MAE
0.3448.

## Honest ship board

The authoritative `e304-choice-plan-20k-honest-r1` evaluation uses E301
concise connected topology, prompt-visible inventory, no DESIGN context, plan
decode weight 1, and no unconstrained fallback.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 0.6667 | 0.3333 | 0.3889 | 0.3458 | 0.1667 | 0.2830 |
| held_out | 5 | 1.0000 | 0.0000 | 0.5600 | 0.3369 | 0.0000 | 0.0000 |
| adversarial | 4 | 0.7500 | 0.2500 | 0.5833 | 0.2244 | 0.1250 | 0.2123 |
| ood | 4 | 1.0000 | 0.0000 | 0.5167 | 0.3750 | 0.0000 | 0.0000 |
| rico_held | 3 | 1.0000 | **1.0000** | 0.2500 | 0.3460 | 0.5556 | 0.7640 |

The longer train improves limited RICO substantially, but global quality
regresses from E301's seven failures / AgentV 2/5 to ten failures / AgentV
1/5. One smoke and one adversarial request produce no root. Held-out and OOD
meaningful/recall remain zero.

## Verdict

Lower denoising NLL and improved training-head averages do not imply better
ship quality. Reject E304 for promotion and stop the duration arm. The new
checkpoint is scratch/local-only and intentionally not synced. The next
decoder repair should fail closed on content component productions that cannot
complete inside the remaining canvas; only after parse is restored should a
new component-target training recipe be attempted.

Artifacts:

- `outputs/runs/e304-choice-plan-20k-r1/`
- `outputs/runs/e304-choice-plan-20k-honest-r1/`
- [machine-readable result](choice-plan-20k-results-iter-e304-20260717.json)
