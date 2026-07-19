# E502 — checkpoint-prior retention during E396→E500 continuation

E502 follows E501's finding that short continuation preserves structure while
5k-token continuation forgets it. The first hypothesis was optimizer update
magnitude; the matched sweep instead exposed a silent initialization confound:
`initialize_from` restored tensors and tokenizers but kept slot-component
lexeme/span priors rebuilt from the new E500 corpus.

## Harness intervention

TwoTower initialization now restores those corpus-derived serving priors from
the checkpoint. `train_summary.json` records `initialized_prior_fields` and
the slot-head loss/decode/prior recipe, so future corpus comparisons cannot
mistake static prior drift for learned adaptation. Bit-exact resume guards,
optimizer reset, data isolation, and checkpoint compatibility remain
unchanged. The train harness is stamped `v3`.

## Matched recipe

All successful arms use CPU, frozen local SmolLM2-135M context, choice output,
d128/h4/c2/dn4, batch 2, seed 0, uniform sampling of the committed 260-row E500
corpus, the E396 slot/component auxiliary recipe, no DESIGN context, and the
same E396 parent SHA. Every train summary records `max_wall_minutes=3.0`;
every process used an external 170-second cap.

Evaluation is the same honest smoke `n=3`: prompt-derived slot contract,
constrained LTR decode, no unconstrained fallback, four generation steps, one
attempt, and a 96-token cap. Every evaluation emitted AgentEvals plus a pinned
AgentV bundle without execution errors.

| Arm | LR | Tokens | Priors restored | Last loss | Structure | Recall | Meaningful / fidelity / reward | AgentV |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| E501 uniform 1k reference | 3e-4 | 1,039 | No | 26.0208 | 0.2317 | 0.0 | 0 / 0 / 0 | 0/1 |
| Lower LR | 1e-4 | 1,039 | No | 28.8950 | 0.1133 | 0.1667 | 0 / 0 / 0 | 0/1 |
| Lower LR | 3e-5 | 1,039 | No | 29.5542 | 0.1167 | 0.0833 | 0 / 0 / 0 | 0/1 |
| Prior retained | 3e-4 | 1,039 | Lexeme + span | 25.6905 | **0.3169** | 0.0833 | 0 / 0 / 0 | 0/1 |
| Prior retained stress | 3e-4 | 5,019 | Lexeme + span | 12.8937 | 0.0927 | 0.1667 | 0 / 0 / 0 | 0/1 |

The lower-LR arms collapse to nearly the same structure despite a 3.3× LR
difference, falsifying optimizer magnitude as the sole cause. Restoring priors
raises 1k structure by `+0.0853` over E501 and `+0.1053` over the frozen parent,
while adding nonzero recall. At 5k tokens the gain disappears; duplicate
subtree spam returns and structure matches the earlier uniform 5k failure.

One initial preflight failed closed before training because the reconstructed
command omitted the checkpoint's slot-component head flags. It produced no
usable checkpoint. This motivated the new complete slot-head recipe telemetry.

## Decision

Keep the prior-preserving load behavior and complete recipe telemetry. Reject
all four E502 checkpoints for promotion and bucket sync: every semantic gate
and AgentV remains red, and this is smoke-only evidence. The next continuation
lever should regularize trainable weights toward the initialized checkpoint
or interleave parent replay; prior preservation alone is insufficient beyond
1k target tokens.

Exact hashes and metrics:
[machine-readable record](iter-e502-initialization-prior-retention-20260719.json).
