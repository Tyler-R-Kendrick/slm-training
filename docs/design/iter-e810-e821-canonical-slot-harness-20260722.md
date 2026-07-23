# E810-E821: canonical-slot harness and output contract v4

## Outcome

Template-marker conversion now belongs entirely to the train/test data
harnesses. Before persistence, every caller-authored marker is replaced by a
contiguous opaque `:slot_<ordinal>` identity across prompts, targets, accepted
outputs, and nested metadata. Shared model-build loaders reject any corpus that
retains a user-defined marker name. Output contract v4 invalidates all earlier
checkpoints because their data provenance cannot prove this boundary.

The canonical CLI defaults are also fail-closed: training defaults to the
lexer-only output tokenizer, and evaluation applies a 12-second per-record
decode deadline inside the 95-second cumulative harness budget. E816 is invalid
because the pre-fix evaluator reached the outer command interrupt. E817 and
E821 finalized their AgentEvals and AgentV artifacts before that interrupt.

## Data and synthesis feedback

E818 rebuilt the frozen E230 roots locally under strict admission. It collected
674 candidates from 126 roots and admitted 350. All admitted records parse;
placeholder-contract violations, judge-contract violations, sanitization
fallbacks, and source errors are zero. Sanitization changed 643 candidates and
templatized 223 free-form target literals across 148 records. Two independently
rejected edit/patch candidates remain quarantined at G11; no gate was weakened.

The final content fingerprint is
`57e702de72e3b816f94cc74797cbe06063ce76001b7683a3d5939a6126dc2736`.
The synthesis feedback retains its source-leakage and redundant-expansion
candidates; notably prompt paraphrase/template has duplicate share 0.8125.
E819 rebuilt all five disjoint eval suites: smoke 3, held_out 5, adversarial 4,
ood 4, and rico_held 7. Strict loaders accepted all 23 records.

## Training

E814 failed before step 1 because the old CLI default selected a forbidden
free-form-capable tokenizer. The default was corrected centrally; this launch
is invalid evidence. E815 verified the fix, then E820 produced a checkpoint
from E818. A later E822 artifact audit found metadata-only slots in its declared
completion inventory, so E820 is invalidated by E826-E829.

E820 ran locally on CPU with scratch context, lexer output, batch size 4, AdamW,
and no checkpoint sync. It completed all 120 requested steps in 13.64 seconds,
ending at loss 6.0485 after seeing 134,858 prompt and 26,617 target tokens. The
checkpoint uses output contract v4 and has SHA-256
`0dc1f81d2cbf59acb8e027b3df06354643abd8072843cd259cdbd7e1e433523e`.

## Held-out evaluation

E821 evaluated the five-record held-out diagnostic locally. It emitted the
canonical AgentEvals JSONL and AgentV SDK bundle.

| n | Parse | Meaning-v1 | Strict-v2 | Fidelity | Validity | Structure | Recall | Reward | p50 / p95 | Fallback / timeout | AgentV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0044 | 0.0286 | 0.0000 | 12001.79 / 27224.56 ms | 1 / 4 | 0/1 |

## Decision

Retain output contract v4, atomic data canonicalization, strict persisted-data
validation, symbol-only defaults, and bounded eval finalization. Reject E820
for promotion, serving, bucket sync, or ship claims. No remote compute ran.
