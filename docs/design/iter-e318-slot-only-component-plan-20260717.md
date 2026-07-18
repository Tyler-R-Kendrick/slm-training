# E318 slot-only component plan — 2026-07-17

E318 isolates whether E317's pooled whole-prompt vector overwhelms local slot
semantics. The sole valid-arm change is
`slot_component_prompt_context=False`: the same learned head classifies the
same direct component owners from the compositional slot text alone. All other
data, seed, architecture, diffusion objective, global plan, token budget, and
honest decode settings match E317.

## Setup correction

The first run, `e318-slot-only-component-20k-r1`, accidentally omitted
`--mask-pattern diffusion`. It therefore used random masking, omitted the
diffusion length head, and had 405,197 rather than 405,717 trainable parameters.
Its checkpoint SHA is
`a16a00f27649b31dc1a2125ea9f15bbf4fb83ad372c685237f52ab1832a9e205`.
The 446-step run and loss AgentV 1/1 are retained as invalid setup evidence,
but it was not quality-evaluated and supports no causal claim.

The corrected `e318-slot-only-component-20k-r2` checkpoint has identical state
keys, tensor shapes, and 449,301 saved parameters to E317. Comparing saved
configs shows exactly one difference:
`slot_component_prompt_context: null/legacy-true → false`.

## Matched training result

The corrected CPU scratch run stopped at 446 steps / 20,044 target tokens in
127.00 seconds. Checkpoint SHA:
`b4e5a87b158e9c2b184f3d850d45948c76ac613f6d2034c92e5787f126f534d9`.
It used explicit `--no-sync-checkpoints`.

| Measure | E317 prompt + slot | E318 slot only |
| --- | ---: | ---: |
| Weighted NLL | 5.4483 | **5.4271** |
| Broad NLL | 5.5233 | **5.5002** |
| Final-20 global plan loss | **1.9176** | 1.9233 |
| Global root accuracy | 0.9500 | 0.9500 |
| Global bound top-k recall | 0.4621 | 0.4621 |
| Global bound-count MAE | 0.2829 | **0.2791** |
| Slot-component loss | 1.2056 | **1.2022** |
| Slot-component accuracy | 0.7008 | 0.7008 |

Loss-suite AgentV passes 1/1. Removing prompt context slightly improves NLL but
does not change in-sample slot accuracy.

## Honest five-suite result

The intended weight-1 head was evaluated under the frozen honest E315 policy.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Component recall | Reward | Slot changes | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | 3 | 1.0 | 1.0000 | 0.5464 | 0.6667 | 0.3333 | 0.6407 | 2 | Fail: recall needs 0.35 |
| held_out | 5 | 1.0 | 1.0000 | 0.4431 | 0.4000 | 0.2000 | 0.3916 | 5 | Fail: recall needs 0.30 |
| adversarial | 4 | 1.0 | 1.0000 | 0.5970 | 0.5000 | 0.3750 | 0.4805 | 4 | Pass |
| ood | 4 | 1.0 | 1.0000 | 0.4304 | 0.5000 | 0.2500 | 0.4992 | 4 | Pass |
| limited `rico_held` | 3 | 1.0 | 0.4167 | 0.2468 | 1.0000 | 0.5556 | 0.7910 | 9 | Pass |

AgentV remains 3/5 with the same two metric failures as E316. Relative to
E317, slot-only context restores held-out meaningful/recall from 0.20/0.10 to
0.40/0.20. It still regresses E316 OOD meaningful/recall from 1.0/0.5417 to
0.50/0.25, and limited-RICO fidelity falls from 1.0 to 0.4167.

The RICO failure reveals a structural mismatch: decode scores a candidate from
only the next unconsumed slot even though legal composite candidates may consume
multiple slots. A locally plausible component can therefore bind or duplicate
later slots incorrectly.

**Verdict:** reject E318 as a checkpoint and do not promote or claim ship.
Retain the slot-only representation as the less harmful E317 correction, but do
not enable its decode head until candidate scoring accounts for every slot the
candidate would consume. E316 remains the strongest scratch checkpoint.
