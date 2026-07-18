# E411–E412 23k continuation boundary — 2026-07-18

E410 places the first observed smoke collapse between E396 at 22,044 target
tokens and E409 at 25,036. E411 narrows that interval with the same power-zero
continuation stopped at 23,000 target tokens. The resume checkpoint, data,
optimizer, model policy, and decode policy are unchanged.

E411 runs only 19 additional optimizer steps and reaches 446 cumulative steps /
23,019 target tokens in 20.9 seconds. Its local-only checkpoint SHA is
`2fb103ca184bc6de999374c77ba27ca48e5566dfee2c37cc6918570c686a334e`.
It inherits best weighted NLL 5.8091 without a fresh loss evaluation and is
not promoted.

E412 already fails smoke meaningful rate and type recall at 0.3333 / 0.1667.
Held recall remains 0.4833, equal to E396, but smoke semantic density is lost.
This places the first observed regression within the first 19 resumed steps,
between 22,044 and 23,019 target tokens.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.3333 | 1.0 | 0.4975 | 0.1667 | 0.3203 |
| held_out | 5 | 1.0 | 0.8 | 1.0 | 0.5524 | 0.4833 | 0.7784 |
| adversarial | 4 | 1.0 | 0.5 | 1.0 | 0.5970 | 0.3750 | 0.4805 |
| ood | 4 | 1.0 | 0.5 | 1.0 | 0.3910 | 0.4375 | 0.4932 |

Every command used an external 290-second interrupt plus a forced kill ten
seconds later. Training also used the internal 4.5-minute wall limit. E411
stopped normally on token budget; E412 completed in 24.9 seconds and returned
the expected gate-failure exit 8. No timed-out process contributes evidence.

**Verdict:** reject E411 and retain E396. Quality can regress within 19 resumed
steps, so any continuation from E396 requires an immediate bounded gate. Skip
RICO and make no promotion or ship claim.
