# SLM-267 (VSD2-02) uniform-policy first reading — inconclusive

Status: local smoke evidence; non-promotable; no checkpoint, training, or ship
claim.

Full narrative and follow-up scope:
[`compiler-inverted-program-data.md`](compiler-inverted-program-data.md).
Machine-readable evidence:
[`iter-slm267-uniform-saturation-20260725.json`](iter-slm267-uniform-saturation-20260725.json).

| Shard | Fresh start | Calls | New accepted | Cumulative accepted | Disposition |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | yes | 334 | 334 | 334 | `wall_clock_stopped` |
| 2 (resume) | no | 324 | 0 | 334 | `wall_clock_stopped` |

- Config: all 54 pinned components, `max_depth=3`, `max_width=3`,
  `seed=0`, `--max-wall-minutes 2.5` per shard (under the repo's
  `MAX_RUN_MINUTES` cap).
- Candidate grid size for this config: **1,781** — verified near-constant
  across `max_depth`/`max_width` up to (10, 8) → 1,793, confirming the grid
  is dominated by pairwise/prop-target coverage, not depth/width.
- Zero verifier rejections (334/334 accepted as Silver).
- **Resume-throughput blocker**: shard 2 spent its entire capped budget
  replaying shard 1's calls and landed 0 net-new records — the current MVP
  resume mechanism does not scale across many small shards at measured
  verifier cost (~0.45–0.46s/call).

Disposition: `inconclusive` for VSD-H7a. Neither 10k reached nor the current
1,781-candidate grid exhausted. Two concrete blockers are filed for the next
increment: a real candidate-cursor resume (not replay) and a wider
`GeneratorConfig` candidate space.
