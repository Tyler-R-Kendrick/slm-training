# E266 — decision-kind-stratified safe set FTPO

Date: 2026-07-17
Status: **completed; no safe update exists in the tested FTPO direction; parent restored**

E266 closes the aggregate-safety hole exposed by E265. It keeps the same E228
parent, committed E261 corpus, 30 `ftpo_set` proposals, learning rate `5e-5`,
and backtracking scales `1, 1/2, 1/4, 1/8, 1/16`. A proposal must now satisfy
the four aggregate held-out guards and preserve those same metrics separately
for every grammar/AST-derived `decision_kind`. The implementation batches
same-length exact states and caches frozen context encodings; it does not add
literal case rules.

Immediately before training, the branch fetched and rebased latest
`origin/main`, was clean, and proved `0 behind / 1 ahead` at harness commit
`841fe7a281f4a3e78bd482298456909a698cddff`. Training trace:
`c88f2dc3e725eb1a8512491b86c73a19`. Evaluation trace:
`611952705524d69e01c85b4a1da77e06`.

## Stratified guard result

All 30 proposed updates were rejected after 150 candidate scales. The final
held-out delta is exactly zero. The serialized checkpoint SHA is
`518d4736571df2f3842ffd338801cfcc4a855d50358c87bd7563facb191935ba`;
all 374 tensors and the model config are bit-identical to E228.

The most frequent stratum blockers across candidate scales were:

| Decision kind | Metric | Regressing trials |
| --- | --- | ---: |
| `lit` | mean margin | 130 |
| `bind_reference_bound_children` | loss | 126 |
| `grammar_comma` | good probability mass | 120 |
| `bind_reference_bound_children` | mean margin | 120 |
| `component_root` | bad probability mass | 118 |
| `grammar_rsqb_bound_empty` | bad probability mass | 114 |
| `grammar_rpar` | good probability mass | 114 |
| `component_root` | good probability mass | 110 |
| `sym` | good probability mass | 110 |

This confirms E265's diagnosis: aggregate gains came from trading away specific
semantic decision families. Under the tested update direction and scales, the
strict per-kind Pareto cone is empty.

## Batched validation result

E266 evaluated the same logical 5,850 held-out event checks as the worst-case
30-step recipe, but grouped them into 150 same-length batches and reused frozen
context encodings. The local preference stage took 79.77 seconds versus E265's
3,009.05 seconds, a 37.7× speedup. The stronger guard therefore costs far less
wall time than E265's weaker unbatched implementation and is suitable for
continued harness use.

## Full evaluation and matched control

Because E266 restored the parent, the unchanged E228 checkpoint was evaluated
again with the same current code. The current E248 control exactly reproduces
every E266 suite metric and all five gate failures; its evaluation trace is
`3113fc873d6fce55e77dcca5191859e1`. No score below is a training effect.

| Suite | n | Syntax | Meaningful | Fidelity | Structure | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.7222 | 0.5100 | 0.8777 |
| held_out | 5 | 1.0000 | 0 | 0.5600 | 0.3943 | 0.8290 |
| adversarial | 4 | 1.0000 | 0.5000 | 0.8333 | 0.4654 | 0.9110 |
| ood | 4 | 1.0000 | 0 | 0.5167 | 0.4081 | 0.8160 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.2500 | 0.2355 | 0.7360 |

Five ship thresholds fail and AgentV passes 2/5 with zero execution errors.
Syntax remains 1.0 with zero fallback and timeout counts, continuing to confirm
the deterministic compiler-tree path rather than learned structure.

## Decision

Retain batched validation and the decision-kind-stratified guard. Reject and do
not sync or promote the parent-equivalent E266 artifact. Do not continue the
same global FTPO direction with a longer run or smaller scalar learning rate:
all tested scales violate at least one grammar/AST stratum. The next hypothesis
should use block-coordinate proposals derived from `decision_kind`, allowing a
category-specific gradient to be accepted only when every other category is
preserved.

Machine-readable evidence:
[`quality-matrix-v10-e266-results.json`](quality-matrix-v10-e266-results.json)
and matched
[`quality-matrix-v10-e266-current-parent-control-results.json`](quality-matrix-v10-e266-current-parent-control-results.json).
