# E297–E298 dropout replication and metric guard (2026-07-17)

## E297 cross-seed replication

E297 repeats E295's 50% deterministic DESIGN dropout with seed 1. The CPU
scratch choice recipe is otherwise unchanged: d64/h2, batch 2, diffusion
corruption, 106 steps / 5,061 target tokens, and no checkpoint sync. The stable
partition drops 237/480 DESIGN records (49.38%). Complete weighted NLL is
7.5864 versus seed-0 E295's 7.3785.

| Category | E295 seed 0 | E297 seed 1 |
| --- | ---: | ---: |
| Weighted | **7.3785** | 7.5864 |
| Binding | **8.0963** | 8.7964 |
| Structural | **5.7866** | 6.0681 |
| Repair | 8.0118 | **7.8760** |
| Schema OOD | 7.1997 | **7.1090** |
| Broad | 8.2060 | **7.8895** |

Frozen prompt-only evaluation has parse 1.0 but meaningful/component
recall/reward 0.0 on all five suites, AgentV 0/5 with zero execution errors,
and 17 failed thresholds. Checkpoint SHA-256:
`a78193f91ee12d07791cab008a75267e3f6e19cfd223fbc726b3896dd98d14ee`.

## E298 evaluator repair

Failure-level audit showed E295's only reported meaningful output repeated the
fallback placeholder through a 72-symbol program against a 14-symbol gold
program. It passed component recall because the required `TextContent` appeared
somewhere, inflating reward despite pathological over-generation.

The shared meaningful-program verifier now rejects validated, serialized
outputs above 4× the gold lexical-token count. Failed meaningful programs also
receive zero component-recall and reward credit; structural and placeholder
metrics remain available as diagnostics. The new failure bucket is
`pathological_overgeneration`.

E298 r1 was an intermediate check before failed-program recall/reward were
zeroed. The final `e298-choice-dropout50-metric-corrected-r2` reevaluation uses
the unchanged E295 checkpoint and frozen policy. It finds one pathological
adversarial output plus 18 trivial layouts: meaningful/component recall/reward
0.0 throughout, AgentV 0/5, and 16 failed thresholds.

## Verdict

E295's apparent adversarial gain is invalidated, and E297 independently fails
to reproduce it. Retain the generalized dropout mechanism but stop sweeping or
combining it at this budget. The next model lever must address empty-root
collapse; future evaluation is protected from repetition-based false positives.
Neither checkpoint is promotable or ship-ready.

Artifacts:

- `outputs/runs/e297-choice-design-dropout50-seed1-r1/`
- `outputs/runs/e297-choice-design-dropout50-seed1-honest-r1/`
- `outputs/runs/e298-choice-dropout50-metric-corrected-r1/` (superseded intermediate)
- `outputs/runs/e298-choice-dropout50-metric-corrected-r2/` (authoritative)
- [machine-readable result](choice-dropout-results-iter-e297-e298-20260717.json)
