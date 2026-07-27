# Mix loadable (exposure12 + programspec-doc) under hard wall — NOT SHIP

**Honesty:** fixture/scratch smoke n=3. **Not ship.**

## Hypothesis
A larger train-loadable mix (`lever_exposure12_v1` + filtered programspec-doc, **n=566**)
raises `meaningful_program_rate` vs exposure12-only hard-wall baseline under
s16 · lr1e-3 · bs2 · sb1.5 · seed47 · ASAP · t30 hard wall · multi-rep.

## Corpus filter
`assert_no_template_semantic_labels` + `assert_canonical_template_markers` +
`assert_symbol_only_output` + `assert_role_safe_output` (matches `TwoTower.from_records`).

## Results (hard wall, `--seed 47`)

| arm | n_train | reps | meanful median | meanful vals | parse mean | empty mean | max_lat mean | wall_ok |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| baseline exposure12 | 107 | 3 | 0.333 | [0.3333333333333333, 0.3333333333333333, 0.3333333333333333] | 0.889 | 0.33 | 30984 | True |
| mix loadable | 566 | 3 | 0.333 | [0.3333333333333333, 0.3333333333333333, 0.3333333333333333] | 1.000 | 0.00 | 30043 | True |

train last_loss mix=15.150311470031738

## Decision
**ACCEPT** — mix iso meanful with better parse 0.889→1.000

Captured: 2026-07-27T18:45:10.060158+00:00
