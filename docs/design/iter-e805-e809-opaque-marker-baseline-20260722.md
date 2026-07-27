# E805-E809: invalid marker-name v3 baseline

## Outcome

This entire sequence is invalid evidence. The corpus persisted caller-authored
marker names, so conversion was incorrectly left on a model-visible path.
Output contract v4 supersedes it and makes canonicalization a train/test harness
responsibility. E808 must not be loaded, resumed, promoted, or deployed.

## Data and synthesis feedback

The local E805 build admitted 372 of 674 candidates from 126 roots with mean
quality 0.9903. Enforced sanitization changed 643 records and templatized 223
free-form literals across 148 records with zero fallbacks. It rejected 302
candidates, including all 15 detected eval n-gram overlaps. The canonical
train loader then accepted all 372 records and found zero `Semantic roles:` or
`Template roles:` lines.

The 13 emitted experiment candidates remain filed in
`synthesis_feedback.json`. They identify eval-adjacent producer inputs and high
duplicate shares for corruption repair (0.7031) and prompt paraphrase/template
(0.8750). Follow-up must filter those inputs and reduce redundant expansion;
the admission gates stay unchanged.

## Training

E806 r1 requested 120 steps but reached the command interrupt at step 27 and
is invalid evidence. The harness was corrected to reserve 15 seconds for
finalization before the 110-second command interrupt. E806 r2 then completed,
but a subsequent audit found that training-time context formatting and scratch
tokenizer construction still consumed raw marker spellings. E806 and its E807
evaluation are invalid evidence and must never be used, synced, or promoted.

E808 projects every marker spelling to an ordinal before both scratch-vocabulary
construction and training context encoding. Renaming `:hero.title` to `:x` now
produces identical target token IDs and identical scratch context vocabularies.
It stopped gracefully after 33 steps and 100.17 elapsed seconds.

The E808 run used local CPU, scratch context, lexer-native output, compiler-tree
decode, batch size 4, and component-plan, component-edge, and slot-component
loss weights of 1. It saw 35,341 prompt and 7,751 target tokens. The checkpoint
SHA-256 is `3b089229163357803ac4a3159a596a15137e325136f8add11c9875babd5fbae5`.
Its output contract is v3, `symbol_anonymization=true`, and every prohibited
marker-semantic lever is false or zero. This was an explicit no-sync scratch
run.

## Held-out evaluation

E809 evaluated the five-record held-out diagnostic locally under the strict
compiler-tree policy. AgentEvals JSONL and the AgentV SDK bundle were emitted.

| n | Parse | Meaning-v1 | Strict-v2 | Fidelity | Validity | Structure | Recall | Reward | p50 / p95 | Fallback / timeout | AgentV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.2000 | 0.0000 | 0.0000 | 0.0333 | 0.1000 | 0.0293 | 0.0667 | 0.1314 | 12002.03 / 13905.16 ms | 1 / 4 | 0/1 |

## Superseding decision

Retain only the graceful-finalization lesson. Reject the E805 corpus and E808
checkpoint for all current use. E810-E821 rebuild train/test artifacts with
harness-owned opaque ordinals and retrain under output contract v4.
