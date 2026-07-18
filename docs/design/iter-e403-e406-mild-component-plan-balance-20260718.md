# E403–E406 mild component-plan balance — 2026-07-18

E400 showed that component-plan inverse-frequency power 0.5 improved one rare
held-out type but collapsed smoke quality. The persisted weights spanned
0.3310–4.1411 across observed types, a 12.5× ratio. E403 therefore tested the
single milder value 0.25 from E396's full state on the unchanged 998-record
E357 corpus.

Recipe validation after E403 caught an unintended difference:
`slot_component_prompt_context` was true because the CLI default overrode the
resumed checkpoint, while E396 and E400 used false. E403 reached 29,066 target
tokens in 107.8 seconds and wrote checkpoint SHA
`bf0ca0f8fd14d6d6fad301d4fe82063c358f79eec749901276ae338204f5344e`,
but it is protocol-invalid for the comparison and was not evaluated.

E404 repeats the run from E396 with explicit
`--no-slot-component-prompt-context`. All other inputs match: CPU, frozen
SmolLM2 HF context, choice tokenizer, component-plan loss 4, slot-owner balance
power 0.5, 29,000-token budget, and no DESIGN.md context. It reaches 567
cumulative steps / 29,066 target tokens in 108.7 seconds. Checkpoint SHA is
`fee1fdfe3f83c711161ee259951b4a7917ce3e968463235ce03f380577ea6219`.
The power-0.25 weights span 0.6300–2.2273, a 3.54× ratio. The checkpoint is
local-only, inherits best weighted NLL 5.8091 without a fresh loss evaluation,
and is not promoted.

E405's complete held suite retains the settings fix but does not improve the
aggregate: meaningful rate is 0.8, structure 0.5161, and type recall 0.4333
versus E396's 0.6 / 0.5933 / 0.4833. Tabs remains the only non-meaningful row.

E406 rejects E404 on the complete bounded suite set. Smoke exactly reproduces
E400's meaningful/recall collapse at 0.3333/0.1667. OOD again has one
parse/fidelity failure. Adversarial quality improves relative to E400, but
AgentV remains 3/4 and the checkpoint cannot pass the smoke gates.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.3333 | 1.0 | 0.5114 | 0.1667 | 0.3163 |
| held_out | 5 | 1.0 | 0.8 | 1.0 | 0.5161 | 0.4333 | 0.7814 |
| adversarial | 4 | 1.0 | 0.75 | 1.0 | 0.6304 | 0.6250 | 0.7238 |
| ood | 4 | 0.75 | 0.5 | 0.75 | 0.4652 | 0.4167 | 0.4835 |

Every command used an external 290-second interrupt plus a forced kill ten
seconds later. Both trains additionally used the internal 4.5-minute wall
limit and stopped normally on token budget. E405 and E406 completed normally;
E406's exit 8 is the expected ship-gate rejection, not a timeout. No timed-out
process contributes evidence.

**Verdict at E406:** reject E403 as protocol-invalid and reject E404 on smoke
gates. E396 remains the bounded candidate; skip RICO and make no promotion or
ship claim.

**Later control:** E407–E408 repeats the 29k continuation with balance power
zero and reproduces the smoke collapse exactly. That
[matched control](iter-e407-e408-continuation-control-20260718.md) supersedes
the attribution to class weighting: continuation length is causal for the
smoke regression, while mild weighting improves OOD relative to the matched
control.
