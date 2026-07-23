# E822-E829: target-only slot inventory

## Outcome

The canonicalization harness now separates sanitized metadata values from the
model completion contract. Target markers are numbered first, and only markers
actually present in `openui` or accepted targets are persisted in
`placeholders`. Metadata-only values are still rewritten to opaque ordinals but
never become completion obligations. Model constructors and request APIs reject
named marker identities; they do not rename them.

E822 is invalid because the shared editable virtualenv resolved the clean main
checkout instead of this worktree (`code_commit=5658fe7`, train harness v9).
E826 reran locally with the worktree source explicitly selected and is the
evidence-bearing strict build. It admitted 350 of 674 candidates, with zero
parse, placeholder-contract, judge-contract, sanitization-fallback, or source
errors. Its fingerprint is
`451f3a094362f6ebba24590e8d47b5aeb82334b0339b9f3b254b97c67e26982e`.
An audit of all E826 and E827 rows found zero mismatches between declared
completion slots and target-used slots.

E827 contains 23 disjoint eval records (smoke 3, held-out 5, adversarial 4,
OOD 4, Rico held-out 7) and rejected 33 leakage candidates.

## Training and evaluation

E828 trained locally on CPU from E826 with scratch context, lexer output,
batch size 4, AdamW, and 120 steps. It completed in 14.46 seconds with final
loss 7.9968, 123,909 prompt tokens, and 26,395 target tokens. Checkpoint SHA-256
is `84f35247f9962d591150f4c379c6a70eab6cd4d44f6430cdf116f38a6df38e36`.
This is an explicit no-sync scratch diagnostic.

E829 evaluated the five held-out records locally and finalized AgentEvals plus
the pinned AgentV bundle.

| n | Parse | Meaning-v1 | Strict-v2 | Fidelity | Validity | Structure | Recall | Reward | p50 / p95 | Fallback / timeout | AgentV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0108 | 0.0667 | 0.0000 | 12001.97 / 28525.38 ms | 1 / 4 | 0/1 |

## Decision

Invalidate E820 because its completion inventories included metadata-only
slots. Retain the E826/E827 harness boundary and the v4 symbol-only contract.
Reject E828 for promotion, serving, bucket sync, or ship claims because quality
remains zero on parse, meaning, fidelity, and reward. No remote compute ran.
