# E249 — E1 bits-per-semantic-decision: how much grammar must a model learn? (2026-07-16)

Measurement, not a train/ship run. Evidence:
[iter-e249-semantic-bits-20260716.json](iter-e249-semantic-bits-20260716.json).
Code: [`src/slm_training/evals/semantic_bits.py`](../../src/slm_training/evals/semantic_bits.py),
CLI [`scripts/measure_semantic_bits.py`](../../scripts/measure_semantic_bits.py). Linear SLM-32.

## Question

Externalizing the grammar (template markers + deterministic decoder) is supposed
to shrink what a tiny model has to learn. No paper quantifies "how small can the
model be if the grammar is externalized" (confirmed gap; TinyStories is the
nearest empirical anchor). E1 gives that question a tokenizer-independent metric.

## Metric

Over the same programs, two token streams:

- **production** — the `ProductionCodec` stream (`dsl/production_codec.encode_openui`),
  where non-lexical punctuation/structure is externalized, so every token is a
  grammar decision (production / slot / reference);
- **surface** — the compiler lexeme stream (`dsl/parser.lexical_tokens`), which
  still carries structural symbols.

For a stream, the corpus unigram description length is
`total_bits = sum_tokens -log2 p_hat(token)` (= `N·H`). `bits_per_decision` is the
empirical entropy `H`; `params_per_bit(n)` divides a model's trainable parameter
count by the corpus's total choice bits (the capacity spent per bit of grammar —
the number the B3 capacity ladder compares across representations).

## Result

Recipe: committed eval corpus `resources/data/eval/remediated/suites` (all five
suites, n=19 documents), CPU, no model loaded (pure corpus information measure).

| Stream | bits/decision (H) | relative total bits |
| --- | ---: | ---: |
| production (grammar choice points) | **4.32** | 1.00× |
| surface (compiler lexemes) | 5.40 | 1.88× |

- **surface→production total-bit ratio = 1.88×** — externalizing the grammar
  nearly halves the total choice bits the model must reproduce.
- **decision reduction = 1.51×** — the production stream has 1.51× fewer tokens
  than the surface stream for the same programs.
- `params_per_bit` is null here (no model loaded); it is populated when the CLI
  is given `--params <trainable_count>`, and is the headline the capacity ladder
  (B3, SLM-23) reports per representation.

## Reading

On this corpus, moving from surface lexemes to grammar choice points removes
~47% of the total bits a model must learn while preserving the program set — the
quantitative form of the "externalized grammar shrinks the target" hypothesis.
This is a *representation* property (corpus information), not a model result; it
predicts, but does not prove, that a smaller model suffices. B3 will test the
prediction by measuring quality-vs-`d_model` for each representation and reporting
`params_per_bit` at matched quality.

## Honesty

Corpus information measurement only; no checkpoint, no ship claim, no quality
number. n=19 is the committed eval fixture; a production-scale claim would report
the full corpus.
