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

## Frozen prompt-only evaluation (superseded)

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

This original evaluation reported adversarial meaningful 0.25 and AgentV 1/5.
E298 later showed that the single output was pathological over-generation
(72 lexical symbols versus 14 gold) and tightened the shared evaluator. The
authoritative corrected result is meaningful/component recall/reward 0.0 on
all suites, AgentV 0/5, and 16 failed thresholds. See
[E297–E298](iter-e297-e298-dropout-replication-metric-guard-20260717.md).

## Verdict

Keep deterministic DESIGN dropout as a reproducible mechanism, but E297 fails
cross-seed replication and E298 invalidates the only apparent semantic gain.
Stop this sweep at the current budget; it is neither promotable nor ship-ready.

Artifacts:

- `outputs/runs/e295-choice-design-dropout-r1/`
- `outputs/runs/e295-choice-design-dropout-honest-r1/`
- [machine-readable result](choice-design-dropout-results-iter-e295-20260717.json)
