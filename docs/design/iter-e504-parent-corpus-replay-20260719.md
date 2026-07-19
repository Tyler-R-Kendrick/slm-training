# E504 — provenance-preserving parent-corpus replay

E504 tests whether exact parent-data replay can preserve E396 hierarchy while
continuing on the E500 expression corpus. The canonical training loop now
supports a deterministic replay corpus, namespaces replay IDs, binds both corpus
fingerprints and the requested fraction into full-state resume provenance, and
records requested versus effective example exposure.

## Durable replay source

The historical 998-row `e357_card_hierarchy_v1` corpus was reconstructed from
training-source commit `27721a801c1b17d2e808f0e1b9b8ac2ad5699349` and uploaded
as an immutable eight-file snapshot:

`hf://buckets/TKendrick/OpenUI/data/train/e357_card_hierarchy_v1/`

Its semantic manifest SHA is
`a4f212a3444d0f219fe1b3604f70929fe1a1b91d4fdc11a73167cb74c55b6a51`;
the records SHA is
`b1b2c3d0c1965bd9829edfc6ae34b5dce916a68c33bb17497a6392c80d7ea6ef`.
After upload, a dry sync reported all eight files identical, and an independent
download reproduced the semantic, records, and manifest-file hashes.

## Matched recipe

All arms use the same E396 bucket checkpoint and committed 260-row E500 primary
corpus, CPU, frozen local SmolLM2-135M context, choice output,
d128/h4/c2/dn4, batch 2, LR `3e-4`, seed 0, and the E396 slot/component recipe.
Each stops at approximately 5,000 target tokens. Every train summary records
`max_wall_minutes=3.0`; every process had an external 170-second cap.

Evaluation is the same honest diagnostic smoke `n=3`: prompt-derived slot
contracts, constrained LTR decode, no fallback, four generation steps, one
attempt, and a 96-token cap. Every evaluation emitted AgentEvals plus a pinned
AgentV bundle without execution errors.

| Replay | Effective examples | RMS drift | Last loss | Structure | Recall | AST node F1 | Meaningful / fidelity / reward | AgentV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0% control | 0 / 198 | 0.003123 | 12.8937 | 0.0927 | **0.1667** | 0.1972 | 0 / 0 / 0 | 0/1 |
| 12.5% | 25 / 200 | 0.003219 | 23.7526 | 0.1558 | 0.0 | 0.1333 | 0 / 0 / 0 | 0/1 |
| 25% | 50 / 200 | 0.003098 | 8.8749 | 0.0964 | 0.0833 | 0.0952 | 0 / 0 / 0 | 0/1 |
| 50% | 102 / 202 | **0.002796** | 9.8487 | **0.2469** | 0.0833 | **0.3148** | 0 / 0 / 0 | 0/1 |
| 50% + 1% retention | 102 / 202 | **0.001775** | 9.5478 | 0.0634 | 0.0 | 0.0667 | 0 / 0 / 0 | 0/1 |

The zero-replay v5 control reproduces E503 structure and recall. Fifty-percent
replay cuts RMS drift by 10.46% and raises structure by 0.1542 to 0.2469, above
the frozen E396 parent's 0.2117 smoke result, but it halves component recall and
does not move any semantic gate. The adaptive 1% retention follow-up cuts drift
another 36.51% yet loses 0.1835 structure, showing a negative interaction rather
than complementary regularization.

## Decision

Keep the replay lever, exact corpus provenance, and effective-exposure
telemetry. Reject all five checkpoints for promotion or checkpoint sync. No
synthesis producer or acceptance-gate change is justified: both input corpora
are clean and immutable, while the failure points to an objective/output-codec
conflict. Measure primary-versus-replay loss or gradient conflict before adding
another replay schedule or regularizer.

Exact hashes and metrics:
[machine-readable record](iter-e504-parent-corpus-replay-20260719.json).
