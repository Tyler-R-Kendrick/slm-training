# E407–E408 continuation control — 2026-07-18

E400 and E404 both continued E396 from 22,044 to 29,066 target tokens while
adding component-plan class balance. Their shared smoke collapse could
therefore come from either the weighting lever or the extra optimization.
E407 supplies the missing matched control: it resumes the same E396 full state
on the unchanged 998-record E357 corpus, keeps component-plan balance power at
zero, and changes no other model or decoder policy.

E407 reaches 567 cumulative steps / 29,066 target tokens in 103.7 seconds on
CPU and stops normally on its token budget. Its local-only checkpoint SHA is
`6373436bfe5504f48615c093ff7d9c3bd28056d7f134af94fe39ceaa75346f82`.
It inherits best weighted NLL 5.8091 without a fresh loss evaluation and is
not promoted.

E408 rejects the control on the complete bounded suites. Crucially, smoke
exactly matches both balanced continuations at meaningful rate 0.3333,
structure 0.5114, type recall 0.1667, and reward 0.3163. Held and adversarial
also match E404's aggregate metrics. The unbalanced control is worse on OOD:
meaningful rate 0.25, structure 0.2846, type recall 0.25, and reward 0.2432.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.3333 | 1.0 | 0.5114 | 0.1667 | 0.3163 |
| held_out | 5 | 1.0 | 0.8 | 1.0 | 0.5161 | 0.4333 | 0.7814 |
| adversarial | 4 | 1.0 | 0.75 | 1.0 | 0.6304 | 0.6250 | 0.7238 |
| ood | 4 | 0.75 | 0.25 | 0.75 | 0.2846 | 0.2500 | 0.2432 |

Every command used an external 290-second interrupt plus a forced kill ten
seconds later. Training also used the internal 4.5-minute wall limit. E407
stopped normally on token budget; E408 completed in 65.9 seconds and returned
the expected gate-failure exit 8. No timed-out process contributes evidence.

**Verdict:** the 29k continuation length, not component-plan class weighting,
causes the smoke regression. Mild balance helps OOD relative to the matched
unbalanced control, but neither continuation is acceptable. Reject E407,
retain E396 at 22,044 tokens as the bounded candidate, and require bounded
evaluation before extending it further. Skip RICO and make no promotion or
ship claim.
