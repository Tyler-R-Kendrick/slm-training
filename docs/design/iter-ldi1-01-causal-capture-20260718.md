# LDI1-01 — exact causal decision-state capture

Date: 2026-07-18
Status: **capture core, DecisionEventV2 emission, TraceStore persistence with a
fail-closed loader, and the plug-in traced-decode + forced-action replay landed with
tests and a fixture-grade evidence artifact. No FTPO trainer, no judge, no adapter
matrix, no training, and no model-quality or ship claim.**

## Why this exists

The causal OpenUI plug-in (`models/causal_lm_openui.py`) decodes under a hard grammar
constraint via `model.generate(..., prefix_allowed_tokens_fn=...)`, which hides the
per-step raw logits and the raw-vs-constrained selection. LDI1-01 needs every supervised
causal decision recoverable from **exact prefix token ids and model logits** — never from
decoded strings or later retokenization — and emitted as evidence compatible with the
shared preference harness's `DecisionEventV2` (LDI0-02, SLM-116). This iteration adds that
capture path; it deliberately trains nothing and runs no judge (those are LDI1-02/1-03).

## What landed

### Torch-free capture core — `models/causal_trace.py`

The capture is split into a torch-free core driven by two injected callables so the whole
algorithm is deterministic and unit-testable without torch, a real tokenizer, or the
grammar:

- `forward_logits(prefix_ids) -> Sequence[float]` — abstracts `model(input_ids).logits[:, -1, :]`;
- `allowed_ids(prefix_ids) -> Sequence[int]` — abstracts the plug-in's `_allowed_ids`.

`capture_raw_steps(...)` runs a greedy constrained decode and, at each step, computes the
raw (pre-mask) argmax and the constrained selection (greedy over the legal set) from the
**same** logits, so a `constraint_shadow` — a raw winner outside the legal set overridden
by a legal selection — is exact. It returns a `CaptureResult` with the policy-retained
`RawStepObservation`s, the full emitted `generated_token_ids`, and an honest `stop_reason`
(`eos` / `no_legal_continuation` / `max_new_tokens`). Bounded selection policies
(`every`, `constraint_shadow_only`, `margin_threshold`, `sampled_positions`, `named_roles`)
choose which decisions are retained but never change which token is emitted. Forced
(single-legal) steps are recorded as deductions, not decisions, so `decision_index` counts
only real choices.

### DecisionEventV2 emission

`emit_causal_decision(obs, identity)` materializes a causal `DecisionStateV2` whose
`context_ids` are the exact integer prefix and whose `grammar_state_hash` is the
content-addressed legal set. Because `DecisionStateV2` carries a single
`policy_checkpoint_sha` with no dedicated adapter field, `fold_policy_identity(base,
adapter)` folds both the base checkpoint and the active adapter into it — so an
adapter-enabled and adapter-disabled capture over the same prefix receive **different**
state identities. A `constraint_shadow` step additionally yields a legality-only
`ActionOutcomeV2` (empty reward/verifier vectors) and a **non-trainable** view via
`materialize_constraint_shadow`; `admit_semantic_corpus` refuses it, so a legality
diagnostic can never supervise a semantic objective.

### Persistence + fail-closed loader

`CausalTraceWriter` appends each decision to the shared `TraceStore` (no second trace
format) with a distinct `kind="causal_decision"`, lifting identity hashes to the row top
level, and tracks a reproducible manifest (identities, state/shadow counts, unique legal
sets, duplicate-set reuse, bytes/state). `load_causal_decision_states(store, *,
expected_checkpoint_sha, expected_tokenizer_sha)` fails closed before returning any state
when the checkpoint or tokenizer does not match — mirroring `local_train._validate_identity`
— and `DecisionStateV2.from_dict` re-verifies each state id (tamper check).

### Plug-in wiring — `models/causal_lm_openui.py`

`generate_constrained_traced(...)` drives the per-step loop, storing `context_ids` as the
full prefix (prompt + generated suffix) so a consumer can replay
`model(context_ids).logits[:, -1, :]` exactly; `replay_causal_action(state,
forced_action_id, continuation_seed, ...)` applies the forced action to the exact stored
prefix, continues under the deterministic constrained policy, and returns a **pre-judge**
`GeneratedOutcome` for the shared counterfactual owner (it runs no judge itself).
`generate_constrained` (trace-off) is unchanged. An injectable `allowed_ids_fn` documents
the grammar seam and keeps the torch loop testable.

## Tests and evidence

- `tests/test_models/test_causal_trace.py` (torch-free): shadow fires only on illegal
  raw + legal selection; shadow view non-trainable and admission-refused; forced-vs-decision
  indexing; EOS only after the prefix validates; integer prefixes are state authority;
  bounded policies; content-addressed legal sets; store round-trip; fail-closed load;
  pre-judge candidate shape.
- `tests/test_models/test_causal_trace_plugin.py` (torch): the real torch forward through
  the capture loop; **stored logits replay within tolerance**; tracing does not change
  emitted tokens; reproducible forced-action replay; adapter-identity folding.
- `tests/test_models/test_causal_trace_fixture.py` + `tests/test_models/fixtures/ldi1/`:
  a committed, deterministic, torch-free trace showing exact prefix replay, raw/legal
  telemetry, one non-admittable constraint shadow, and one forced legal counterfactual
  replay outcome (canonicalization deferred to the strict validator — the fixture does
  not assert catalog-validity, which is environment/pack-dependent) — no semantic label,
  no model-quality claim.

Regenerate the fixture with `capture_raw_steps` over the logit/legal map in the fixture
`README.md`, a fixed `CausalTraceIdentity`, and a `TraceStore` opened with explicit
`run_id`/`trace_id`/`span_id` (drop the store's timestamped `manifest.json`; keep the
reproducible `causal_trace_manifest.json`). `ruff` and `python -m scripts.repo_policy` clean.

## Honest remaining scope

- Efficiency: `forward_logits` recomputes a full forward per step (no KV cache); capture
  correctness, not decode throughput, is the goal here.
- The semantic action evidence (running `replay_causal_action` outcomes through the
  counterfactual owner's judge and Pareto partition) is exercised by the owner, not this
  module — LDI1-02 (causal PEFT FTPO) and LDI1-03 (the matched matrix) are the follow-ons.
- This iteration adds no token/component special cases, runs no training, and makes no
  model-quality claim.
