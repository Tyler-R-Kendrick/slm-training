# E335 HF Jobs credit preflight — 2026-07-17

E335 prepared a reproducible full-HF job from source commit
`7a10b96a59637636eef9e40493252633077af226`, pinned
`HuggingFaceTB/SmolLM2-135M` at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, mounted
`hf://buckets/TKendrick/OpenUI`, and carried E333's exact choice-tokenizer,
component-plan, lexical-prior, and no-DESIGN recipe.

The authenticated `TKendrick` submission was rejected before scheduling:
`402 Payment Required: Pre-paid credit balance is insufficient`. No GPU or
other compute started, no checkpoint or AgentV record was created, and cost
was $0. The proposed A10G-large rate was $1.50/hour.

**Verdict:** this is a zero-compute infrastructure preflight, not model
evidence. Retry only after credits are available. Per the subsequently imposed
run policy, any retry must use a hard five-minute timeout.

