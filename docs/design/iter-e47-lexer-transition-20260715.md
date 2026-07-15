# E47 lexer transition alignment — 2026-07-15

The constrained target-transition probe previously rejected the first byte of
the framed literal channel. The alignment fixes now admit every token in a
37-token Silver+ target, including root, BIND, NL, BYTE, and LIT_STR/LIT_END
tokens.

The corrected picker was then applied to the existing E47 checkpoint with
strict constrained LTR:

| path | parse | fallback rate | p50 |
| --- | ---: | ---: | ---: |
| E47 strict LTR after transition fix | 0/3 | 0.000 | 8,503 ms |

AgentV recorded 0/5 passed with no execution errors. The target language is
now fully admissible; the existing checkpoint still fails complete generation,
so it must be retrained after the corrected surface contract rather than
judged as evidence against the repaired decoder.

Focused grammar, lexer-smoke, and inference tests passed: 40 passed, 1
deselected.

Decision: keep the transition fixes and retrain; stop adding decode-only
heuristics or loss-weight-only variants until a checkpoint has learned the
corrected lexer surface.

Evidence:

- outputs/runs/iter-e47-silver-256-20260715/strict_ltr_surface/e47-silver-256-strict-ltr-surface/eval_smoke.json
- target transition probe over outputs/train_data/remediated_roots_silver/records.jsonl

This is a scratch, smoke-only result and is not a ship claim.
