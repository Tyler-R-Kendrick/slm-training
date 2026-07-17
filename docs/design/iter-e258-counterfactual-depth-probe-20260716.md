# E258 — depth-stratified semantic counterfactual probe

Date: 2026-07-16
Status: **completed; sampler hypothesis confirmed; corpus not admitted for training**

E258 reran the E256 semantic counterfactual probe after replacing chronological
state truncation with deterministic stratification over compiler-derived
decision kinds and relative trajectory-depth quartiles. It used the unchanged
E228 checkpoint, all 65 document records from the E230 judged source, strict
compiler-tree decoding, seed 258, four states per record, and up to four legal
candidates per state. Every candidate was independently judged before labels
were derived from the verified Pareto frontier.

The run started from merged commit
`cc63a9c2bede69969417dc05c00099c03c2ffa94` after a fresh fetch/rebase and a
clean `0 behind / 0 ahead` proof. Trace ID:
`52e3121d0317eefc3fcc29ff8879d0a3`.

## Measured result

| Measure | Result |
| --- | ---: |
| Accepted document traces | 65 / 65 |
| Exact states replayed | 260 |
| Grammar-legal candidates | 761 |
| Independent-judge pass | 199 / 761 |
| Fully verified candidates | 50 / 761 |
| Qualified events | 18 |
| Qualified decision kinds | 6 |
| Qualified prompt groups | 8 |
| Train / held-out events | 17 / 1 |
| Train / held-out groups | 7 / 1 |
| Set-valued events | 12 |

The 260 probes cover 13 compiler-derived decision kinds and all four relative
trajectory quartiles. Qualified events are no longer root-only: they include
`bind_reference_root_children`, `component_bound`,
`grammar_rsqb_root_populated`, and `sym`, in addition to root binding and root
component choices. This confirms that the generalized sampler removed E256's
chronological root-state bias.

## Decision

Do not admit the E258 event export for training yet. Although semantic decision
depth improved, the qualified data still spans only eight prompt groups and one
stable held-out group. That cannot test broad group generalization and fails the
prerequisite recorded after E252. The next probe may expand uniformly to eight
states per record; offline selection analysis shows the additional strata cover
all 65 source groups, including 11 groups assigned to held-out by the unchanged
group hash. Any resulting corpus must deduplicate exact states and persist only
judge-qualified events with their complete probes.

No checkpoint was written. Machine-readable evidence:
[`quality-matrix-v10-e258-depth-probe-results.json`](quality-matrix-v10-e258-depth-probe-results.json).
