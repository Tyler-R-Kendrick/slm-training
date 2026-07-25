# Semantic failure census

`SemanticFailureTaxonomyV1` is a deterministic grouping layer over the
existing binding-aware meaningfulness report and ordered G0--G12 verifier
results. It is not an evaluator, a decoder policy, or a human-rating gate.
Native reason codes, source evidence, and every gate outcome remain in each
`SemanticFailureTraceV1`; an unrecognized reason is surfaced as `unknown`.

Use the canonical replay command for immutable envelopes:

```bash
python -m scripts.audit_semantic_failures \
  --replay-bundle src/slm_training/resources/evals/meaningful_v2_frontier_replay.json \
  --cache-mode read_write \
  --cache-root outputs/runs/slm263-semantic-failure-census/cache \
  --out docs/design/iter-slm263-semantic-failure-census-20260724.json \
  --run-dir outputs/runs/slm263-semantic-failure-census
```

The command records `NOT_APPLICABLE` for absent exact regret-oracle evidence.
Human and AgentV labels are optional descriptive metadata only: neither can
make a trace pass, fail, nor qualify a result. Cache keys include prediction
and source-record hashes plus taxonomy version; cache corruption is recomputed
by the shared fail-closed evaluation cache.

## Measured local replay — 2026-07-24

The versioned [result JSON](iter-slm263-semantic-failure-census-20260724.json)
replayed the committed `meaningful_v2_frontier_replay.json` bundle on CPU with
the deterministic local scorer (no training, checkpoint creation, or remote
replay). It emitted an AgentV bundle with **1/1** envelope checks passing and
no execution errors. The cache-read replay and a two-shard recombination each
reproduced all **33/33** trace objects byte-for-byte.

| Measure | Result |
| --- | --- |
| Complete immutable generation rows | 33 / 33 |
| Rejected rows | 0 |
| Deterministic-failure reason codes mapped to a non-unknown family | 100% |
| First failure: component/role selection | 19 |
| First failure: trivial/empty shell | 13 |
| First failure: prompt-contract inventory | 1 |
| Verified checkpoint hashes represented | 8 |
| Principal-coverage requirements | **Blocked**: 33 < 100 rows; 8 < 10 hashes; no third declared architecture/factorization family |

This is diagnostic replay evidence only. It does not select a targeted
intervention, tune a protected suite, claim a semantic improvement, or relax
ship gates. A matched targeted/untargeted/placebo intervention remains blocked
until a training/validation-only census has the declared coverage and a third
provenanced family; protected suites may be evaluated only once after that
preregistration.
