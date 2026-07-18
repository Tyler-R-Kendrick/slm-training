# E295 deterministic DESIGN-context dropout (2026-07-17)

## Question

E292's all-DESIGN choice arm has the best matched complete NLL but transfers no
meaningful programs to prompt-only inference. E294's no-DESIGN control also
produces none. E295 tests the smallest interpolation: retain the same recipe,
but deterministically omit DESIGN.md for half of training records.

## Harness and recipe

`--design-md-dropout` is a generalized probability in `[0, 1]`. A stable
SHA-256 draw from `(seed, record key)` selects each record once, keeping context
caches and token accounting reproducible. Dropped DESIGN is also excluded from
slot-contract derivation. Evaluation never applies this train-only transform.

`e295-choice-design-dropout-r1` uses CPU scratch context, choice codec, d64/h2,
seed 0, batch 2, diffusion corruption, 107 steps / 5,022 target tokens, and no
checkpoint sync. All 480 records contain DESIGN.md; seed 0 assigns exactly 240
to dropped context and 240 to retained context. Seen prompt tokens are 35,135,
between E294 no-DESIGN (12,739) and E292 all-DESIGN (54,476).

| Loss suite | E292 all-DESIGN | E295 50% dropout | E294 no-DESIGN |
| --- | ---: | ---: | ---: |
| Weighted NLL | **7.2265** | 7.3785 | 7.4977 |
| Binding | **8.0201** | 8.0963 | 8.0988 |
| Structural | **5.6419** | 5.7866 | 5.8927 |
| Repair | **7.6943** | 8.0118 | 8.3503 |
| Schema OOD | **7.0693** | 7.1997 | 7.3295 |
| Broad | **8.1075** | 8.2060 | 8.2535 |

The loss-suite AgentV record passes 1/1 with zero execution errors. Checkpoint
SHA-256:
`5b4c50467454f7a9dddbc28da2e115c31a8eba8071587e95eda096729a16fb50`.

## Frozen prompt-only evaluation

`e295-choice-design-dropout-honest-r1` uses the unchanged ship gates, scratch
context, prompt-derived honest slot contracts, no DESIGN context, and no
unconstrained fallback.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Component recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.0 | 0.3333 | 0.3500 | 0.0 | 0.0 |
| held_out | 5 | 1.0 | 0.0 | 0.0000 | 0.2514 | 0.0 | 0.0 |
| adversarial | 4 | 1.0 | **0.25** | 0.2500 | 0.2697 | **0.25** | **0.2343** |
| ood | 4 | 1.0 | 0.0 | 0.0000 | 0.2369 | 0.0 | 0.0 |
| rico_held | 3 | 1.0 | 0.0 | 0.0000 | 0.0901 | 0.0 | 0.0 |

E295 improves adversarial meaningful rate from 0.0 in both controls to 0.25,
passes AgentV 1/5 instead of 0/5, and reduces frozen failures from E292's 15 and
E294's 17 to 14. The other four suite scoreboards are exactly E294's, so this is
a one-example context-robustness signal, not broad transfer.

## Verdict

Keep deterministic DESIGN dropout as a reproducible training lever. The matched
NLL interpolation and one adversarial success justify replication, but four
suites remain at meaningful 0.0 and the checkpoint fails 14 frozen thresholds.
It is neither promotable nor ship-ready. A replicated dropout-rate/seed check is
required before combining this lever with component-plan training.

Artifacts:

- `outputs/runs/e295-choice-design-dropout-r1/`
- `outputs/runs/e295-choice-design-dropout-honest-r1/`
- [machine-readable result](choice-design-dropout-results-iter-e295-20260717.json)
