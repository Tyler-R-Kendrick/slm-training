# E868-E878: hard-tail selection and warm-start vocabulary ownership

This iteration fixed two harness defects without assigning template-marker
conversion to the model.

E868 cloned E852 at zero training steps and emitted per-record loss evidence.
All 131 tensors were unchanged; the raw checkpoint hash differs only because
run metadata differs. The initial E869 build exposed a derive-path defect:
refreshing canonical language-contract bases silently discarded five derived
hard cases. The harness now refreshes base ids while preserving derivatives.

The difficulty lever previously only stamped `curation_score` metadata that no
trainer consumed. The canonical lever now defines a 20% easy-tail selection;
`--difficulty-from` writes those removals to `rejected.jsonl` as selection,
not producer failure. E872 is the accepted committed snapshot: 351 scored, 281
retained, 70 selected out, zero verifier/quality/dedup failures, zero synthesis
recommendations, and all retained completions pass the symbol-only contract.
Semantic namespace augmentation and its config, catalog, dedup, feedback, and
mixture paths were deleted rather than hidden.

E873 ended without a summary or checkpoint and is invalid. E874 failed before
training because warm-start constructed the filtered corpus vocabulary before
loading E852's larger context embedding. E875 and E876 serialized checkpoints
after only 37/300 and 43/600 steps at the wall budget; policy excludes both
from evidence.

The first compatibility patch preserved only the filtered corpus tokenizer and
remapped shared rows. Follow-up E879 found that this still discarded 78 parent
tokens. The superseding fix and matched evidence are recorded in
[`iter-e879-e885-vocab-union-matched-eval-20260722.md`](iter-e879-e885-vocab-union-matched-eval-20260722.md).
E877 exercised the real E852 → E872 path and completed all 20 requested local
CPU steps in 52.71 seconds. E878 then ran compiler-tree smoke `n=3`, but omitted
the structural AST-plan weights used by the retained baseline.

| Run | Result |
| --- | --- |
| E865 retained baseline | parse / meaning-v1 / strict-v2 / normalized fidelity 1.0000; structure 0.6589; recall 0.7500; reward 0.9490; AgentV 0/1 |
| E878 E877 warm-start | **invalid unmatched recipe** (AST-plan weights 0/0): parse 1.0000; meaning-v1 0.6667; strict-v2 0.0000; normalized fidelity 0.9167; marker validity 0.9500; structure 0.4625; recall 0.3333; reward 0.9360; AgentV 0/1 |

The E872 data-harness corrections are retained. E878 cannot accept or reject
E877 because it is not matched evidence. The later union-vocabulary fix
supersedes the initial warm-start patch. All checkpoints are local scratch
artifacts with explicit no-sync policy; no remote workflow ran.

Canonical evidence:
[`iter-e868-e878-hard-tail-warmstart-20260722.json`](iter-e868-e878-hard-tail-warmstart-20260722.json).
