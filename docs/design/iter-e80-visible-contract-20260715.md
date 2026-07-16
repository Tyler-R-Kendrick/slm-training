# E80 visible prompt contracts — 2026-07-15

E80 retrained the V7 cache recipe on `v2_prompt_contract`, a derived corpus
whose synthesized prompts explicitly append their declared placeholder
inventory. The build produced 1,417 records with content fingerprint
`c3ae5b41bb41010ee2d78b5b0c9bd257295234853ce6f9645c127fdca29e8dd3`.

Current gold-free smoke evaluation (n=3) scored parse 1.0, raw syntax 1.0,
structural similarity 0.6111, and reward 0.669. Exact placeholder fidelity
remained 0.0; normalized namespace-stripped overlap was 0.3889.

Decision: retain the prompt-contract dataset as a valid data-pipeline
intervention but reject E80 for promotion. Training-visible contracts alone
do not teach a model to infer namespaces when serving prompts omit them. The
next experiment must either expose the contract in the serving request or
measure an explicit namespace policy instead of hiding it in the metric.

This is scratch smoke evidence, not a ship claim.
