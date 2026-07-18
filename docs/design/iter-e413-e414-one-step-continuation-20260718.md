# E413–E414 one-step continuation — 2026-07-18

E412 shows smoke collapse after 19 resumed optimizer steps. E413 tests the
smallest possible continuation from E396: one power-zero optimizer step on the
same full state, data, model policy, and decoder policy.

E413 reaches step 428 / 22,074 target tokens in 7.5 seconds and stops normally
on its 22,050-token budget. Its local-only checkpoint SHA is
`b3cca00c06337d25fd908a3168b295a05b0b1f4f602dcd034f847866a8da66cb`.
It inherits best weighted NLL 5.8091 without a fresh loss evaluation and is
not promoted.

E414 passes every complete bounded suite and AgentV is 4/4. Smoke remains
fully meaningful with type recall 0.5. Held metrics exactly retain E396's
meaningful rate 0.6 and recall 0.4833. The only global gate failure is the
intentionally absent full RICO suite.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5600 | 0.5000 | 0.9770 |
| held_out | 5 | 1.0 | 0.6 | 1.0 | 0.5933 | 0.4833 | 0.5916 |
| adversarial | 4 | 1.0 | 0.75 | 1.0 | 0.6304 | 0.6250 | 0.7238 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.5511 | 0.7292 | 0.9827 |

Every command used an external 290-second interrupt plus a forced kill ten
seconds later. Training also used the internal 4.5-minute wall limit. E413
stopped normally on token budget; E414 completed in 30.0 seconds and returned
exit 8 only because full RICO is absent. No timed-out process contributes
evidence.

**Verdict:** one resumed step is bounded-safe but does not establish a new
champion or ship checkpoint. Retain E396 for selection and place the smoke
failure boundary between 2 and 19 additional steps. Do not run full RICO until
the boundary and selection benefit are established.
