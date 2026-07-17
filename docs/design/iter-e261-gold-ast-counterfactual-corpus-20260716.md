# E261 — full grammar/AST-aligned counterfactual corpus

Date: 2026-07-16
Status: **completed; source corpus admitted for future semantic preference training**

E261 applies the E260 grammar/AST state miner to all 65 E230 document records.
Each exact branch state comes from `gold_compiler_decisions()`, not a literal
case matcher or a poor policy trajectory. Gold remains outside model context:
it identifies the offline selected branch, while the unchanged E228 policy
completes verifier-legal alternatives. The independent judge and
meaningful-program/Pareto verifier must approve a same-state comparison before
it becomes training data.

The valid run used four fixed source shards (17/16/16/16 records), three CPU
threads per process, strict compiler-tree decoding, four stratified states per
record, up to four candidates per state, and seed 260. It started from merged
commit `75779711032c88cdd33124bde494d17deb0a3140` after a clean synchronized
checkout. Valid trace IDs:

- `eefdac58c5888c293759876194597ed7`
- `1c578ecc8d011a9eed8f4a14d9d2f25f`
- `bd225dfe343adaca16626516a07dce96`
- `f05413aa2fd8c3582ed35af115ea364e`

## Measured result

| Measure | Result |
| --- | ---: |
| Accepted document traces | 65 / 65 |
| Exact states replayed | 260 |
| Grammar-legal candidates | 736 |
| Independent-judge pass | 455 / 736 |
| Fully verified candidates | 359 / 736 |
| Qualified events | 239 |
| Qualified decision kinds | 14 |
| Qualified prompt groups | 64 |
| Train / held-out events | 200 / 39 |
| Train / held-out groups | 53 / 11 |
| Set-valued events | 108 |

The retained evidence spans all four relative trajectory-depth buckets
(110/53/44/32). Every one of the 239 probes has
`state_source=gold_ast`, `same_state_verified=true`, a selected gold-AST
completion, and at least one policy-completed alternative. All events share one
policy checkpoint, tokenizer, and decode identity. The accepted record
`program_6bf891b163fb6489` is the only source record without a retained event:
its four states honestly failed with `no_verified_frontier`.

## Durable training data

The admitted immutable corpus is committed at
`src/slm_training/resources/data/preference/e261_gold_ast_counterfactual_v1/`:

- `events.jsonl`: 239 exact-state semantic preference events;
- `evidence.jsonl`: the matching 239 independent judge probes;
- `manifest.json`: identities, split counts, source traces, and content
  fingerprints.

The dashboard's Training data page discovers committed preference manifests,
so this corpus appears there as semantic preference training data. Future local
preference training must point at this committed `events.jsonl`; the historical
step outputs are not the training source of truth.

## Execution correction and decision

Two incomplete attempts are excluded from every headline number and corpus
fingerprint. A serial run was stopped after two records because it could not
finish efficiently. The first four-process attempt was also stopped after CPU
oversubscription; only one shard had written three records. The complete
thread-capped four-shard rerun above is the sole admitted evidence.

The full-corpus prerequisite identified by E260 is satisfied. E261 is admitted
for a subsequent set-valued semantic preference experiment, but it is not a
model-quality or ship result. The next training run must use this committed
corpus and then execute the full honest quality gates; syntax success from the
deterministic constrained layer cannot substitute for semantic fidelity.

Machine-readable evidence:
[`quality-matrix-v10-e261-gold-ast-corpus-results.json`](quality-matrix-v10-e261-gold-ast-corpus-results.json).
