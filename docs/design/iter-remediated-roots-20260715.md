# Root-target corpus diagnostic — 2026-07-15

This diagnostic publishes the source-controlled training corpus assembled from
canonical repository inputs without synthesis or derivative expansion. It is
intended to remain available for future training and the Training Data view;
it is not promoted as a production-quality checkpoint.

## Corpus

- Version: `remediated_roots`
- Records: 108
- Unique OpenUI targets: 94
- Content fingerprint: `f8d714f122ac7f091236fd4e562935758de330534cac146abf30af13d0ac98ce`
- Recipe: `build_train_data.py --source all --synthesizer none --max-children 0`
- Device: CPU; no checkpoint sync (scratch diagnostic)
- Sources: language contract 61, fixture 17, generated ProgramSpec 14,
  corruption repair 14, renderer visual 1, web distilled 1

The builder collected 129 candidates and emitted 108 records. Twelve inputs
were rejected by governance and four by quality checks. RICO/Awwwards roots
were unavailable or quarantined, so this corpus is coverage-diagnostic rather
than a replacement for the published remediated corpus.

## Train/eval result

The 64-step TwoTower scratch run
`iter-remediated-roots-64step-ltr2-20260715` reached held-out weighted NLL
7.270489 at step 64 (broad mean NLL 6.437990).

The corrected constrained smoke evaluation used the checkpoint's compositional
tokenizer with grammar-constrained decoding and LTR repair. It evaluated 3
examples: parse rate 0.0, structural similarity 0.0, reward 0.0, and p50
latency 2895 ms. There were no decode timeouts. The checkpoint is rejected;
the corpus remains published for future data-driven iterations.
