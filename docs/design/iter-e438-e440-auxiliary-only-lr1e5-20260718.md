# E438–E440 auxiliary-only LR `1e-5` — 2026-07-18

E438 lowers only the controlled E435 auxiliary-head learning rate from `3e-5`
to `1e-5`. It resumes E396 for the same 19 batches, stops normally at step 446
/ 23,019 target tokens in 21.2 seconds, and changes the same four plan/slot
auxiliary tensors while changing zero base tensors. The saved optimizer uses
`lr=1e-5`. Checkpoint SHA is
`eebb6d31645bf49643c5887085c735b810357b06eea49dad244ef90a5f622bff`.
It is local-only, inherits best weighted NLL 5.8091, and is not promoted.

E439 passes bounded AgentV 4/4 with no execution errors; exit 8 reflects only
absent full RICO:

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5600 | 0.5000 | 0.9770 |
| held_out | 5 | 1.0 | 0.6 | 1.0 | 0.5933 | 0.4833 | 0.5916 |
| adversarial | 4 | 1.0 | 0.75 | 1.0 | 0.6762 | 0.7500 | 0.7268 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.5511 | 0.7292 | 0.9828 |

E440 completes matched RICO rows 336–384 with parse, fidelity, and meaningful
rate 1.0, structure 0.6386, recall 0.8889, reward 0.9991, and no recorded
failures. E396 remains higher at structure 0.6401 and recall 0.8993. The
diagnostic AgentV envelope is 0/5 because four required suites are absent and
RICO is only 48/1500; it has no execution errors.

All 16 bounded predictions and all 48 RICO predictions are byte-identical
between E435 (`lr=3e-5`) and E438 (`lr=1e-5`). The lower rate therefore remains
in the same discrete decode regime and does not remove the single-row RICO
regression.

Every command used an external 290-second interrupt and ten-second forced
kill; training also used the internal 4.5-minute wall limit. E438–E440
completed normally, and no timed-out process contributes evidence.

**Verdict:** reject E438 and close the auxiliary learning-rate sweep. It
duplicates E435 predictions and remains below E396 on matched RICO structure
and recall. Do not run full RICO, sync, promote, or make a ship claim.
