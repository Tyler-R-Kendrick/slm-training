# VAR3-03 (SLM-433): TurnDispositionHead trained on a real corpus

**Disposition: `trained_head_improves_on_derived_only`**

Fixture-scale (real, freshly-enumerated symbolic corpus, not natural-language data). **Not a ship claim.**

## Preregistered

- Hypothesis: A TurnDispositionHead trained on real, corpus-derived clarify-margin examples reduces composite_penalized_error_rate on a held-out split relative to derived_only (free rules, clarify never fires).
- Stop rule: If disposition_trained shows no held-out improvement over derived_only, that is a legitimate negative result; do not add further trained-head complexity to this surface.
- Leakage safety: split is trace-level; all four rows of one trace share one split

## Corpus

- `n_per_tier`=40, `train_fraction`=0.7, tier margins=[0.0, 0.2, 0.4, 0.6, 1.0]
- Total traces: 200, total rows: 800
- Rows by kind: {'answer': 200, 'forced_emit': 200, 'out_of_scope': 200, 'scored': 200}
- Rows by split: {'heldout': 240, 'train': 560}
- `out_of_scope_never_from_nonempty_set`: True
- Leakage-safe split (disjoint trace ids): True
- Scored rows: 140 train / 60 heldout

## Three-arm comparison (identical held-out split, matched budget)

| arm | case_count | wrong_op_rate | abstention_rate | composite_penalized_error_rate |
| --- | ---: | ---: | ---: | ---: |
| disposition_off | 240 | 0.5583 | 0.0000 | 0.4188 |
| derived_only | 240 | 0.0583 | 0.0000 | 0.0437 |
| disposition_trained | 240 | 0.0333 | 0.0500 | 0.0375 |

## Honesty

fixture_or_scratch, capability class -- not a ship claim. The corpus is a real, freshly-enumerated symbolic fixture (5 real declared-cost tiers x n_per_tier traces), not natural-language data; the classifier is a tiny 3-feature MLP. Held-out split is trace-level leakage-safe. No promotion, ship-gate, or production-readiness claim is made regardless of outcome.
