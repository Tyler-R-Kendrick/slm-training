# E500 — documentized expression corpus

E500 tests whether language-contract expression coverage can be converted into
high-quality, choice-compatible full documents without weakening data gates.
The repaired projection succeeds as data infrastructure but does not improve
the bounded learned-model result.

## Data intervention

The shared train-data harness now supports two explicit, provenance-preserving
operations:

- `--documentize-expressions` converts an expression task into a complete
  `Stack` document containing the requested expression. Components without a
  content placeholder receive one context-label child so the existing quality
  contract remains meaningful.
- `--target-kinds` retains declared output kinds for a codec-specific snapshot.
  Excluded rows remain in `rejected.jsonl` at the `selection` stage and are not
  misreported as producer-yield failures.

Both matched builds derive from the committed `remediated_roots` snapshot, use
the strict profile, layout-only synthesis, and document-only selection.

| Corpus | Rows | Root parents | Program families | Structural families | Choice preflight | Feedback |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Document-only control r1 | 96 | 31 | 16 | 78 | 96/96 | 0 warnings, 0 recommendations |
| Singleton projection r1 | 123 | 54 | 39 | 105 | 123/123 | **invalid:** 15 placeholder-contract violations |
| Complete-layout projection r2 | 260 | 87 | 72 | 241 | 260/260 | 0 warnings, 0 recommendations |

The clean r2 snapshot is published at
`src/slm_training/resources/data/train/e500_documentized_expression_candidate_r2_20260718/`
with fingerprint `bc256915…463bc62`. The redundant control stays derivable and
unpublished.

## Matched bounded training

Every process used `timeout --signal=INT --kill-after=10s 170s`; every train
summary records `max_wall_minutes=3.0`. Both arms use CPU, frozen local
SmolLM2-135M context, choice output, d64/h2/c1/dn2, seed 0, batch 4, lr
`3e-4`, no DESIGN context, and explicit `--no-sync-checkpoints`.

| Budget / arm | Steps / target tokens | Last loss | Elapsed |
| --- | ---: | ---: | ---: |
| 1k control | 9 / 1,028 | 30.3844 | 7.00s |
| 1k projected | 11 / 1,039 | 27.6250 | 8.95s |
| 5k control | 43 / 5,040 | 10.5529 | 10.14s |
| 5k projected | 50 / 5,062 | 12.6778 | 13.95s |

The candidate’s 1k loss advantage reverses at 5k. Loss alone is not a quality
metric, so both pairs received the same honest constrained smoke evaluation:
`n=1`, prompt-derived slot contract, four generation steps, 96-token LTR cap,
one attempt, and no unconstrained fallback.

| Budget / arm | Syntax | Meaningful | Fidelity | Structure | Recall | Reward | Prediction | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1k control | 1.0 | 0.0 | 0.0 | 0.0375 | 0.0 | 0.0 | `ImageGallery($s1)` | 0/1 |
| 1k projected | 1.0 | 0.0 | 0.0 | 0.0375 | 0.0 | 0.0 | `ImageGallery($s1)` | 0/1 |
| 5k control | 1.0 | 0.0 | 0.0 | 0.0375 | 0.0 | 0.0 | `Input("xs")` | 0/1 |
| 5k projected | 1.0 | 0.0 | 0.0 | 0.0375 | 0.0 | 0.0 | `Input("xs")` | 0/1 |

All four final evaluations emitted AgentEvals JSONL and pinned AgentV bundles
with zero execution errors. A first control evaluation accidentally omitted
`--honest-slot-contract`; it was overwritten by the explicit honest run and is
invalid evidence.

## Decision

Keep the generalized projection and the clean published dataset: they resolve
E499’s fragment/codec boundary while increasing independent roots and
structural coverage without red feedback. Do not claim model improvement,
promote a checkpoint, or sync these rejected diagnostics to the HF bucket.
The exact metrics and all four checkpoint hashes are in the
[machine-readable record](iter-e500-documentized-expression-corpus-20260718.json).
