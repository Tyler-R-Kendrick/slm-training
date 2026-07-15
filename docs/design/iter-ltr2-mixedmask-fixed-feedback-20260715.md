# Mixed-mask compositional feedback — 2026-07-15

The published `remediated` corpus is the source-controlled training input for this run: 585 records, manifest fingerprint `928ec8d4921954c7736d2386fe7abf88bbef75523a7cfe404792f45ddcd5d4ba`. The run used `--train-version remediated`, scratch context, compositional output tokens, 64 steps, batch size 8, seed 0, LTR loss weight 2.0, fidelity loss weight 0.5, and `mask_pattern=mixed`.

## Harness correction

The first mixed-mask candidate was not a valid mixed-mask experiment: statement-span masking only ran for the lexer tokenizer, so compositional training silently used the random-mask path. The corrected implementation derives newline-delimited spans for the compositional tokenizer. It does not use gold output as context or weaken evaluation.

## Results

| Run | Train records | Weighted NLL (step 64) | Smoke n | Parse | Structural | Component recall | Placeholder validity | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `iter-published-remediated-64step-ltr2-mixedmask-fixed-20260715` | 585 | 7.2320 | 3 | 0.0000 | 0.1986 | 0.2500 | 0.1111 | Reject |

The AgentEvals bundle recorded one failed ship-gate evaluation. All three smoke predictions failed to parse; reward was 0.0. The corrected masking policy therefore did not recover useful generation quality at this budget. Keep the published `remediated` corpus as the current training control and do not promote this candidate.

Artifacts remain in the local run directory under `outputs/runs/iter-published-remediated-64step-ltr2-mixedmask-fixed-20260715` and its full-smoke evaluation directory. This note records the durable decision; no historical per-step archive is required.
