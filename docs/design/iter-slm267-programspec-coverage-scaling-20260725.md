# SLM-267 (VSD2-02): ProgramSpec coverage-scaling wiring evidence

**Claim class:** wiring / fixture only

**Status:** `inconclusive`

## Recipe

```json
{
  "target_count": 80,
  "seed": 0,
  "shards": 2,
  "components": [
    "TextContent",
    "Button",
    "Separator"
  ],
  "max_depth": 3,
  "max_width": 3
}
```

## Arms

| policy | shard | seed | proposed | accepted | duplicate_roots | unique_roots | exhausted_at | cells_covered/total | programs_to_full_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 0 | 705206174 | 37 | 36 | 8 | 28 | 36 | 58/64 | 34 |
| uniform | 1 | 3027814441 | 37 | 36 | 8 | 28 | 36 | 58/64 | 35 |
| coverage_targeted | 0 | 705206174 | 37 | 36 | 8 | 28 | 36 | 58/64 | 19 |
| coverage_targeted | 1 | 3027814441 | 37 | 36 | 8 | 28 | 36 | 58/64 | 19 |

## Comparison

- uniform programs to full coverage: 34
- coverage-targeted programs to full coverage: 19
- coverage-targeted reaches full coverage no slower than uniform: True

## Limitations

- Fixture-scale only: target_count and the candidate grid are far below the 10k/100k/1M rungs VSD-H7a-c require; no claim is made about those hypotheses at this scale.
- The 'coverage_targeted' arm reuses the generator's existing greedy CoverageTracker.score() bias toward uncovered cells; it does not yet consume an external CoverageGapManifestV1 (SLM-265) gap manifest — that mapping is explicitly future work.
- Canonical-root dedup here is in-memory only; a disk-backed exact index for 100k/1M-scale generation is not implemented.
- No training/model comparison is run and no corpus is published to the DataStore; this measures the generator mechanism only.
