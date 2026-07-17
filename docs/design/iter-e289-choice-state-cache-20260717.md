# E289 — exact choice-state cache (2026-07-17)

## Hypothesis and recipe

Repeated production states should have identical legal token sets. `ChoiceTokenizer`
now caches those exact sets by immutable production-state signature and remaining
positions. The cache never approximates legality.

The matched CPU scratch run used the B3 choice recipe: width 64, depth 2, seed 0,
5,000-token arm budget, 200-step ceiling, batch size 2, and configurable
`max_wall_minutes=5`. It stopped at 107 steps / 5,022 target tokens after 29.74
seconds. The checkpoint SHA-256 is identical to E288, so this is a decoder-runtime
comparison rather than evidence of a newly learned model.

## Results

Standalone all-suite evaluation emitted AgentEvals JSONL and the pinned AgentV
bundle (`0/5`, zero execution errors).

| suite | n | parse | meaningful / fidelity / reward | cache hit rate | p50 ms | E288 p50 | speedup | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.0 | 57.6% | 2,337 | 6,196 | 2.65× | 8,664 |
| held_out | 5 | 1.0 | 0.0 | 76.4% | 1,196 | 6,007 | 5.02× | 5,914 |
| adversarial | 4 | 1.0 | 0.0 | 71.8% | 1,418 | 6,152 | 4.34× | 5,900 |
| ood | 4 | 1.0 | 0.0 | 68.2% | 1,169 | 6,001 | 5.13× | 6,167 |
| rico_held | 3 | 1.0 | 0.0 | 65.5% | 1,037 | 6,074 | 5.86× | 5,996 |

Every suite retained parse 1.0 with zero decoder dead ends. The exact cache cuts
median latency substantially, while cold/unseen states still leave p95 near six
seconds (and smoke above eight seconds).

## Verdict and feedback

The generalized state cache works and preserves deterministic structure, but the
checkpoint still has meaningful parse, fidelity, reward, and AgentV pass rate of
zero. It is neither promotable nor ship-ready.

The next performance lever should target cache misses: derive candidate IDs
directly from pushdown-frame categories, then compare every result bit-for-bit
with the exhaustive legal-token oracle. That addresses cold p95 without adding
prompt/component literals or weakening the symbolic layer.

Machine-readable evidence:
[`iter-e289-choice-state-cache-20260717.json`](iter-e289-choice-state-cache-20260717.json).
