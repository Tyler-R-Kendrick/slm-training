# E42 root-surface and constrained-LTR probe — 2026-07-15

The E42 judged-corpus checkpoint was used for a decode-only probe after fixing
three harness defects:

- lexer binder slot zero is reserved for root;
- native slot zero decodes to surface text root;
- grammar-state advancement uses decoded surface text rather than raw
  native binder vocabulary text.

The constrained picker also rejects a non-root first significant token.

| path | parse | fallback rate | p50 latency |
| --- | ---: | ---: | ---: |
| parallel strict, before surface-cache fix | 0/3 | 1.000 | 9,621 ms |
| constrained LTR-primary strict | 0/3 | 0.000 | 4,832 ms |

Focused grammar, lexer-smoke, and inference tests passed: 37 passed, 1
deselected. Direct LTR output now begins root =, proving the representation
and DFA-prefix fixes are active, but the complete predictions remain malformed
and AgentV remains 0/5.

Decision: keep the fixes, reject the checkpoint as a quality candidate, and
retrain with the corrected root/surface contract. Parse remains the correct
structural gate; constrained decoding is now exposing the remaining model
coverage failure instead of masking it.

Evidence:

- outputs/runs/iter-e42-judged-factorized-20260715/ltr_surface/e42-ltr-surface/eval_smoke.json
- outputs/runs/iter-e42-judged-factorized-20260715/root_reserved/e42-root-reserved/eval_smoke.json

This is a decode-only scratch result and is not a ship claim.
