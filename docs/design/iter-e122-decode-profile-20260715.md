# E122 decode profile — 2026-07-15

E122 profiles the E121 judged-corpus scratch checkpoint after the bounded E121
smoke evaluation timed out. It is a diagnostic, not a ship evaluation.

| Variant | Seconds / generation | Denoiser ms | Pick ms | Dead ends |
| --- | ---: | ---: | ---: | ---: |
| Incremental + chosen-token verification | 19.85 | 14,773 | 4,679 | 1 |
| Incremental, no chosen-token verification | 27.09 | 22,097 | 4,208 | 1 |
| No incremental state + chosen-token verification | 29.85 | 23,019 | 6,122 | 1 |

The constrained path emitted a valid sample after recording a dead-end at
position 24. The trace had zero legal candidates. The singleton fast path is
already implemented in `models/grammar.py`: when the DFA produces exactly one
legal token it returns it without candidate probing. That optimization cannot
apply to an empty legal set without emitting an illegal token, so no legality
weakening was made.

The next lever is the CPU scratch denoiser and bounded dead-end recovery, not
additional stream-probe bypasses. Incremental grammar state remains enabled.

Recipe: E121 checkpoint, CPU, one prompt, one generation per variant,
`grammar_verify_chosen_only`/incremental toggles as shown above. Result:
diagnostic-only; no ship-gate claim.
