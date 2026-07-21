# E714 symbol-only scratch baseline

E714 is the first checkpoint trained from scratch under output contract v2. It
establishes a clean local baseline after intentionally invalidating every
free-form-capable checkpoint. Training and evaluation used local CPU compute;
no remote workflow, Hugging Face Job, checkpoint upload, or promotion was used.

## Recipe

- Run: `e714-symbol-only-scratch600-r1`
- Dataset: strict symbol-only integrated smoke snapshot, 141 admitted records
  and 207 primary plus accepted targets
- Dataset manifest SHA:
  `78096191b5272658a1f6c48b6cdceaba5492f363630c0bf7d1b19d2f465b2b45`
- Context/output: scratch context, lexer output tokenizer, grammar/AST symbols
  and template placeholders only
- Train: CPU, 600 steps, batch 4, 48.72 seconds,
  `max_wall_minutes=2`, honest slot contract, constrained slot decode,
  grammar-LTR primary, fast-train
- Capacity: 1,620,226 trainable parameters; 63,841 target tokens and 307,003
  prompt tokens
- Loss: final 3.1286; first-20 example proxy 49.4538 to last-20 2.9101
- Serving checkpoint SHA:
  `71ef1d25efb5186b885e8ee9e1370002f419bf42ea53985111734c49cfd2b49e`
- Persistence: local `outputs/` only by explicit `--no-sync-checkpoints`; this
  scratch diagnostic is neither promotable nor a ship checkpoint

The strict synthesis audit found zero output-contract violations. Quarantined
rows remained rejected: Awwwards and Awwwards-plus-design contrastive sources
had zero yield, two human-curated targets and one design contrastive target used
reserved structure. Gates were not weakened.

## Bounded evaluation

Each suite used grammar-constrained LTR decoding, a 160-symbol canvas, one
attempt, an eight-second per-record decode timeout, no unconstrained fallback,
and the honest slot contract. These are diagnostic subsets, not full ship
suites.

| Suite | n | Parse | Strict v2 | Fidelity | Structure | Recall | AST node / edge F1 | Timeouts | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| smoke | 3 | 0.3333 | 0.0 | 1.0 | 0.0880 | 0.4167 | 0.2667 / 0.1538 | 0 | 0/1 |
| held_out | 4 | 0.5 | 0.0 | 0.5 | 0.1638 | 0.0833 | 0.3848 / 0.1 | 2 | 0/1 |
| adversarial | 4 | 0.5 | 0.0 | 1.0 | 0.1221 | 0.4583 | 0.2847 / 0.0 | 0 | 0/1 |
| ood | 4 | 0.5 | 0.0 | 0.5 | 0.0483 | 0.1458 | 0.2426 / 0.0667 | 2 | 0/1 |
| rico_held | 3 | 0.0 | 0.0 | 0.5 | 0.2248 | 0.2778 | unavailable | 1 | 0/1 |

All evaluated predictions pass the symbol-only contract. The failure moved to
the intended problem: learned LTR decoding emits long invalid schema
constructions, omits required components or slots, and times out. Aggregate
AgentV is 0/5 with zero execution errors and mean score 0.5. Reward and
meaningful-program v1/v2 are 0 on every suite.

An earlier multi-suite invocation produced only a smoke artifact before the
process ended and is excluded from evidence. The five rows above come from
separate terminal, exit-zero local runs with AgentV bundles.

## Decision

Keep E714 as the first compatible local baseline and provenance record. Do not
upload, promote, serve as a quality checkpoint, or claim ship readiness. The
next experiment should change constrained decoding on the same checkpoint and
single smoke subset before spending another training cycle.

Machine-readable evidence:
[iter-e714-symbol-only-baseline-20260721.json](iter-e714-symbol-only-baseline-20260721.json).
