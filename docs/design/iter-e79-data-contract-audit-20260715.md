# E79 curriculum prompt/contract audit — 2026-07-15

The current curriculum contains 1,165 records. 1,123 records have declared
placeholders but no explicit placeholder inventory in the prompt; this is
common across `prompt_paraphrase`, `layout_augment`, `rico_real`, and
`language_contract` records. Only 42 records contain visible `:name`-style
tokens in their prompts.

This explains the current exact-fidelity failure without changing the gate:
production-like prompts such as “Build a hero card...” do not expose whether
the required contract is `:hero.*` or a namespaced contract such as
`:smoke.hero.*`. The decoder cannot legitimately infer that namespace from
the prompt alone.

Decision: do not weaken exact fidelity or inject gold contracts at evaluation.
The next data intervention should make the contract visible in synthesized
training prompts (and validate that the same contract is available in the
serving request), then rerun the full feedback loop.

This is a data audit, not a model or ship result.
