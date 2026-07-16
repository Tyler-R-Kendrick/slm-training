# Retrieval-conditioned root-target diagnostic — 2026-07-15

This run enabled the existing `retrieval_k=1` skeleton-conditioning path on
the source-controlled `remediated_roots` corpus, with diffusion masking and
the matched 128-step TwoTower scratch recipe.

- 108 records / 94 unique targets
- 60,846 target tokens; 78,932 prompt tokens after retrieval conditioning
- Telemetry total: 34.98 seconds
- Best held-out weighted NLL: 4.846048
- Broad mean NLL: 5.393851

Constrained smoke (`n=3`, LTR repair, 64-token cap, one attempt) remained 0.0
parse, 0.0 structural similarity, and 0.0 reward. There were no timeouts and
p50 latency was 1,778 ms. Reject the candidate: retrieval conditioning does
not address the structural generation collapse and increases prompt work.
