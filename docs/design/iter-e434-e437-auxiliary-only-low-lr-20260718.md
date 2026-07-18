# E434–E437 auxiliary-only low-LR continuation — 2026-07-18

E434 asks whether E431's auxiliary-head overfit is learning-rate sensitive.
It resumes E396 for the same 19 batches with auxiliary-only training and
`lr=3e-5`, but the initial invocation inherited CLI defaults that enabled slot
prompt context and disabled the slot lexeme prior. Those settings differ from
E396/E431. E434 stops normally at step 446 / 23,019 target tokens in 23.3
seconds, but is protocol-invalid and is not evaluated. Its local-only
checkpoint SHA is
`4fc7ed3205f1730d780ecdcda7e4bb7039208b571151b4c2c15fbe54062c6508`.

E435 repeats the run with slot prompt context explicitly disabled, lexeme prior
weight 1, and the inherited slot decode weight 4. Only LR differs from E431.
It stops normally at step 446 / 23,019 target tokens in 21.4 seconds. The
optimizer contains `lr=3e-5`; four plan/slot auxiliary tensors change and zero
base tensors change. Checkpoint SHA is
`d912bceeb5aeea1af53dafeea7de626d1be092a8e02952d4b7500101c3372c55`.
The checkpoint is local-only, inherits best weighted NLL 5.8091, and is not
promoted.

E436 uses the matched 320-token honest LTR protocol:

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5600 | 0.5000 | 0.9770 |
| held_out | 5 | 1.0 | 0.6 | 1.0 | 0.5933 | 0.4833 | 0.5916 |
| adversarial | 4 | 1.0 | 0.75 | 1.0 | 0.6762 | 0.7500 | 0.7268 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.5511 | 0.7292 | 0.9828 |

AgentV passes 4/4 with no execution errors. Exit 8 reflects only absent full
RICO. Relative to high-rate auxiliary-only E431/E432, the only aggregate
change is improved adversarial structure and recall.

E437 evaluates the exact RICO rows 336–384 used by the preceding controls. It
completes all 48 rows with parse, fidelity, and meaningful rate 1.0, structure
0.6386, type recall 0.8889, reward 0.9991, and no recorded failures. E396 on
the identical rows has structure 0.6401 and recall 0.8993 at the same
meaningful rate and reward. The diagnostic AgentV envelope is 0/5 because four
required suites are absent and RICO is only 48/1500; it has no execution
errors.

Every command used an external 290-second interrupt and ten-second forced
kill; training also used the internal 4.5-minute wall limit. E434–E437
completed normally, and no timed-out process contributes evidence.

**Verdict:** reject E435 as an E396 replacement. Low-rate auxiliary-only
training almost preserves the matched RICO slice and improves the bounded
adversarial suite, but it does not improve the authoritative matched RICO
structure or recall. Do not run full RICO, sync, promote, or make a ship claim.
