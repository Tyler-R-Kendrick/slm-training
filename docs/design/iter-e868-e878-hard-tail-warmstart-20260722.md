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

Warm-start now preserves the current corpus tokenizer and remaps shared context
embedding rows by token identity before loading weights. The regression test
uses deliberately disjoint vocabularies and verifies shared special-token rows.
E877 exercised the real E852 → E872 path and completed all 20 requested local
CPU steps in 52.71 seconds. E878 then ran strict compiler-tree smoke `n=3`.

| Run | Result |
| --- | --- |
| E865 retained baseline | parse / meaning-v1 / strict-v2 / normalized fidelity 1.0000; structure 0.6589; recall 0.7500; reward 0.9490; AgentV 0/1 |
| E878 E877 warm-start | parse 1.0000; meaning-v1 0.6667; strict-v2 0.0000; normalized fidelity 0.9167; marker validity 0.9500; structure 0.4625; recall 0.3333; reward 0.9360; AgentV 0/1 |

The warm-start compatibility fix and E872 data-harness corrections are retained.
The E877 checkpoint is rejected: its short hard-tail continuation regressed the
matched smoke metrics and makes no ship or promotion claim. All checkpoints are
local scratch artifacts with explicit no-sync policy; no remote workflow ran.

Canonical evidence:
[`iter-e868-e878-hard-tail-warmstart-20260722.json`](iter-e868-e878-hard-tail-warmstart-20260722.json).
