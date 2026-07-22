# E792-E802 — schema array contract

**Date:** 2026-07-22  
**Decision:** retain model v225 schema-array enforcement; reject global marker-closure treatment and checkpoint promotion  
**Evidence:** [`iter-e792-e802-schema-array-contract-20260722.json`](iter-e792-e802-schema-array-contract-20260722.json)

E792 started from E789's missing-marker failure and tested a hard global marker
inventory closure. That treatment was wrong: it made the first selected
component absorb unrelated markers, starved later prompt-required families,
and either timed out or fell back. E792-E799 are retained as negative evidence;
none is accepted as a quality result.

The traces exposed a separate compiler defect. Once an array-valued positional
property opened, only component arrays were schema-constrained. A primitive
array such as `Slider.defaultValue: number[]` could admit components, arbitrary
expressions, and unbounded comma-separated values. Model v225 derives both item
starts and post-item separators from the official item schema. It also keeps
the array's comma available despite the enclosing component already reaching
maximum positional arity. The rule is schema-general and contains no fixture
strings, prompt phrases, or template-specific cases.

E800 targeted the settings record locally. It restored the accepted E788
behavior: parse and meaningful-v1 1.0, fidelity 0.8, structure 0.4133,
component recall 0.5, and zero fallback/timeout. E802 is the final local CPU
held-out n=5 replay stamped with model v225:

| n | Parse | Meaning-v1 | Strict-v2 | Fidelity | Validity | Structure | Recall | Reward | p50 / p95 | Fallback / timeout | AgentV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 1.0000 | 0.6000 | 0.0000 | 0.7076 | 0.8246 | 0.5622 | 0.5857 | 0.8613 | 2796.95 / 3165.11 ms | 0 / 0 | 0/1 |

The headline quality is unchanged from E789, so this is a compiler-correctness
retention, not a claimed model-quality gain. Required-inventory coverage was
0.8 on the final replay and AgentV remained 0/1. The global marker treatment is
therefore rejected; the next coverage intervention must allocate marker
ownership by schema role and planned family before imposing closure.

All runs used local CPU and completed under the two-minute command cap. Every
eval emitted AgentEvals JSONL and an AgentV SDK bundle. No remote experiment ran,
and no checkpoint was created, synced, or promoted. Predictions contain only
grammar/AST symbols, schema enum literals, and request-declared template
markers; no free-form completion string was introduced.
