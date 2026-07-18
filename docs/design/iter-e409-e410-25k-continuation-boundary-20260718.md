# E409–E410 25k continuation boundary — 2026-07-18

E408 proves that continuing E396 from 22,044 to 29,066 target tokens causes
the smoke regression even with component-plan balance disabled. E409 narrows
the failure boundary with a power-zero continuation to 25,000 target tokens.
It uses the same E396 full state, unchanged 998-record E357 corpus, optimizer,
model policy, and decode policy as the 29k control.

E409 reaches 485 cumulative steps / 25,036 target tokens in 51.4 seconds and
stops normally on token budget. Its local-only checkpoint SHA is
`cb3ae2163ad2d633eb2c6dd51ee0333136bffd0f6f6aef34d35c7107b064dc99`.
It inherits best weighted NLL 5.8091 without a fresh loss evaluation and is
not promoted.

E410's complete bounded suites show that smoke has already collapsed by 25k:
meaningful rate 0.3333 and type recall 0.1667 fail their 0.66 / 0.35 gates.
Unlike the 29k control, OOD remains parse- and fidelity-perfect with meaningful
rate 0.75 and reward 0.7335. This places the first observed regression between
E396's 22,044 tokens and E409's 25,036 tokens.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.3333 | 1.0 | 0.4242 | 0.1667 | 0.3243 |
| held_out | 5 | 1.0 | 0.8 | 1.0 | 0.5488 | 0.4333 | 0.7808 |
| adversarial | 4 | 1.0 | 0.5 | 1.0 | 0.5970 | 0.3750 | 0.4805 |
| ood | 4 | 1.0 | 0.75 | 1.0 | 0.5680 | 0.6042 | 0.7335 |

Every command used an external 290-second interrupt plus a forced kill ten
seconds later. Training also used the internal 4.5-minute wall limit. E409
stopped normally on token budget; E410 completed in 22.2 seconds and returned
the expected gate-failure exit 8. No timed-out process contributes evidence.

**Verdict:** reject E409 and retain E396. The earliest observed collapse is now
25,036 tokens; any additional continuation must be evaluated before that
point. Skip RICO and make no promotion or ship claim.
