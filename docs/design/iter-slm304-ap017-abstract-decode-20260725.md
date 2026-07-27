# SLM-304 / AP-017 — codebook-only abstract trace decoding with hard cap and forced end

Iter doc for Linear **SLM-304** (AP-017). Evidence class: **wiring/fixture
only** — no training run, no quality or ship claim. The feature is
**default-off**; every legacy generation path is byte-identical when it is off.

## Design

`slm_training.models.abstract_decode` implements a three-phase decode on top of
the AP-016 `AbstractPlanV1` codebook (`slm_training.dsl.abstract_plan`, SLM-302):

- **prompt** — the caller-supplied prefix. No tokens are generated in this
  phase; the prefix is validated for well-formed abstract delimiters before any
  decoding (a dangling `<endabstract>` or unclosed/nested block is a hard
  error).
- **abstract** — opens with a deterministic `<beginabstract>` commit (a
  singleton: no neural forward, per the forced-bypass invariant), then each
  forward's logits are hard-masked (`mask_logits_to_abstract_codebook`,
  `-inf` outside) to the M codebook slot ids plus `<endabstract>` — no
  out-of-codebook token can be selected, and a logit row too short to score the
  codebook fails closed instead of falling back to the full vocabulary. The
  phase ends when `<endabstract>` is **sampled** (`abstract_termination =
  "sampled_end"`) or when `m_max` slot tokens have been emitted, in which case
  the end delimiter is **forced** — appended without a forward — and recorded
  (`forced_end = True`, `abstract_termination = "forced_cap"`). `m_max`
  defaults to the plan's `max_slot_count` and is validated against it.
- **answer** — control hands to the existing constrained program-decode
  discipline (the same singleton-bypass + legal-set-masked selection used by
  `generate_constrained_traced`, over the same injected grammar legality seam).
  If the answer phase ever emits an abstract codebook/delimiter token, the
  trace is rejected with `AbstractDecodeError` — malformed/nested delimiters
  are hard errors, never repaired.

Sampling: abstract and answer phases carry **independent**
`AbstractSamplingParams(temperature, seed)`; `temperature == 0.0` is greedy
(deterministic), positive temperature samples from the masked distribution
under a per-phase `random.Random(seed)`.

Telemetry: `AbstractTraceCapture` records per-phase `token_ids`, `token_count`,
and **per-token log probabilities** (full-vocabulary log-softmax of the
selected token). A token committed without a forward — the opening
`<beginabstract>`, a forced cap end, or a singleton bypass — records `None`.
The plan's `compatibility_fingerprint` is stamped on every capture.

## Wiring (default-off)

`CausalLMOpenUIConfig.abstract: AbstractDecodeConfig | None = None` opts in.
`CausalLMOpenUIPlugin.generate_abstract_traced(...)` raises unless the config
is set; `generate_constrained`, `generate_constrained_traced`, and
`replay_causal_action` are untouched, and off-mode parity is asserted in tests.
Only the answer-phase tokens are certified as program text (same
validate-or-honest-fallback discipline as the traced path); the abstract span
is trace evidence, not output.

`grammar.py` was **not** modified: masking needs only the plan's token-id
helpers (`slot_token_ids`, `delimiter_token_ids`) from AP-016.

## Verification

```bash
PYTHONPATH=src python -m pytest tests/test_models/test_abstract_decode.py -q
PYTHONPATH=src python -m pytest tests/test_models/test_causal_trace.py \
  tests/test_models/test_causal_trace_plugin.py \
  tests/test_models/test_causal_trace_fixture.py -q
python -m scripts.verify_version_stamps --check
python -m scripts.repo_policy
```

Covered: early sampled end, exact `m_max` cap with recorded forced
termination, empty abstract span, malformed delimiter rejection (end without
begin, nested begin, non-codebook inside a block, unclosed block), malformed
prompt prefix rejection, answer-phase nested-delimiter rejection, no
out-of-codebook token even when the raw argmax is outside the codebook,
deterministic seeded sampling (same seed ⇒ identical capture; independent
abstract/answer seeds), per-phase counts/logprobs, answer stop reasons, config
validation, plug-in decode through a tiny deterministic torch model,
default-off refusal, and off-mode parity of the legacy paths.

## Honest scope

Wiring/fixture only. The abstract span is non-interpretable codebook identity
(AP-016 base variant); nothing here claims the span carries semantic content,
improves any metric, or is shippable. Enabling the path by default, assigning
slot roles, or training on abstract traces are separate, later experiments.
