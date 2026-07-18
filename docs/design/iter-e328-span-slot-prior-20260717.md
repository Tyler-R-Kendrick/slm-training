# E328 ordered multi-slot owner prior — 2026-07-17

E328 fixes the choice tokenizer/scorer boundary first: the scorer previously
queried a tokenizer method that did not exist and silently treated every
component as one-slot. The tokenizer now derives contiguous placeholder
content arity from schema (`CardHeader=2`, `Callout=2`, `Input=2`).

The matched arm adds ordered two-slot owner priors derived only from E316
training records. The persisted `(title, subtitle)` prior has CardHeader score
2.266 at weight 2, and checkpoint reload confirms it is active.

The 446-step / 20,044-token CPU run took 139.44s. Checkpoint SHA:
`888cf8ac495f0349a8347132c3121c08561d0759567844984b40d19b9ccc4ebe`.
Weighted/broad NLL reproduce E326 at 5.4084/5.4961; loss AgentV passes 1/1.
Final-20 slot accuracy is 0.8026 versus majority baseline 0.6184.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Recall | Reward | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | 3 | 1.0 | 1.0 | 0.5464 | 0.6667 | 0.3333 | 0.6407 | Fail recall |
| held_out | 5 | 1.0 | 1.0 | 0.4758 | 0.6000 | 0.3000 | 0.5862 | Pass at floor |
| adversarial | 4 | 1.0 | 1.0 | 0.6304 | 0.7500 | 0.6250 | 0.7238 | Pass |
| ood | 4 | 1.0 | 1.0 | 0.5229 | 1.0000 | 0.5417 | 0.9857 | Pass |
| limited `rico_held` | 3 | 1.0 | 1.0 | 0.4826 | 1.0000 | 0.5556 | 1.0000 | Pass |

The smoke outputs are unchanged, while held-out recall regresses
0.40→0.30 and structure 0.5458→0.4758.

**Verdict:** reject the E328 checkpoint and retain E326 as strongest scratch.
Keep the tokenizer content-arity correction as a generalized bug fix. The
remaining smoke failure is not solved by additive local priors.
