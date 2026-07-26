# AP-020 (SLM-309): on-policy abstract self-distillation

Status: harness wiring only. **No training run or benchmark was executed as
part of this change** -- this document registers a collector and its
regression tests, not an experiment result. Do not read anything below as a
ship claim; honest-ship-eval gates are unaffected because no metric was
produced.

## Decision

Close the warm-up train-inference gap identified for the Abstract-CoT causal
pilot (AP2) by sampling prompt-only abstract traces on-policy and training on
`[prompt; abstract; verified target]` under ordinary causal attention, per
arXiv:2604.22709. The generating policy is never the sole semantic judge: a
candidate answer is only ever eligible for training after an independent
verifier accepts it.

## What this adds

`src/slm_training/harnesses/distill/self_distill_collect.py`:

- `collect_on_policy_traces(...)` -- the on-policy collector. It calls an
  injected `generate: prompt -> AbstractTracedGeneration` (the AP-017
  `capture_abstract_trace` path, wired as
  `CausalLMOpenUIPlugin.generate_abstract_traced`) once per prompt, verifies
  the answer text through an injected `VerifierCascade` (defaults to
  `slm_training.evals.verifier_cascade.default_openui_cascade`), and appends
  every outcome -- accepted or rejected -- to an existing
  `harnesses.distill.trace_store.TraceStore`.
- `SelfDistillCollectConfig` -- `enabled` (default-on; `False` is a pure
  no-op, so a bottleneck-only training run stays reproducible whether or not
  this collector runs), `max_records`, `shard_index`/`num_shards`
  (deterministic sharding: `prompts[shard_index::num_shards]`),
  `reject_forced_end`, `min_abstract_tokens` (configurable acceptance
  filters beyond the verifier's own pass/fail).
- `CollectionSummary` -- processed/accepted/rejected/duplicate/skipped-resumed
  counts and a trace-length (abstract token count) histogram, so collection
  yield and trace-length distribution are always reported.
- `segment_ids_for_capture` / `training_example_from_capture` -- adapters
  from a collected `AbstractTraceCapture` to the `input_ids`/`segment_ids`
  pair AP-019's `CausalLMOpenUIPlugin.forward_with_segments` already expects.
  No privileged-plan segment is used (self-distillation traces carry no
  separate plan channel), so this degenerates to AP-019's bottleneck mask
  with the middle segment empty: plain causal attention, loss restricted to
  the abstract + target positions. No new trainer was written.

## Reused, not reinvented

- Trace storage/shape: `harnesses/distill/trace_store.TraceStore` (append,
  `iter_traces`), matching the `meta`/`labels`/`final` shape
  `harnesses/distill/select.py` already filters (`corpus_label`,
  `filter_traces`) -- selection and SFT need no new plumbing.
- Provenance fingerprints: `harness_core.lineage.records.content_sha`
  (canonical-JSON SHA256), the same primitive `evals/verifier_cascade.py`'s
  cache keys use.
- Verifier: `evals/verifier_cascade.VerifierCascade` /
  `default_openui_cascade` (the existing G0-G12 gate stack), not a bespoke
  checker.
- Decode: `models/abstract_decode.capture_abstract_trace` (AP-017 / SLM-304),
  unmodified.
- Training-loss wiring: `models/block_attention` (AP-019 / SLM-307),
  unmodified.

## Determinism, resume, and dedup

Two content-hashed fingerprints, both scoped to `(policy_checkpoint_sha,
decode_config_hash)`:

- **prompt fingerprint** (`+ prompt`) -- resume support. Before calling
  `generate`, the collector checks whether this exact policy/config/prompt
  combination already has a row in the store (scanned once at the start of a
  run) and skips regenerating it if so. A restarted or re-invoked collection
  run never re-samples a prompt it already collected.
- **answer fingerprint** (`+ generated answer text`, independent of the
  prompt) -- duplicate detection. Catches the same generated output recurring
  for a *different* prompt (e.g. mode collapse onto a generic fallback)
  without ever flagging the (already resume-guarded) same-prompt case twice.

Rejected candidates are written with `labels.accepted = False` and a
`reject_reason` string -- nothing is discarded silently; the "rejected"
corpus already recognized by `select.py`'s `corpus_label` covers them.

## Acceptance criteria mapping

- "Collection is deterministic, resumable, and duplicate-free." -- ordered
  iteration + shard split; prompt-fingerprint resume check; answer-fingerprint
  duplicate check.
- "Every record has complete policy and verifier provenance." -- `meta`
  carries `policy_checkpoint_sha`, `tokenizer_sha`, `decode_config_hash`,
  `plan_fingerprint`; the full `VerifierCascadeResult.to_dict()` is stored
  under `verifier`.
- "Bottleneck-only control remains reproducible when self-distillation is
  disabled." -- `SelfDistillCollectConfig(enabled=False)` is a no-op; neither
  `abstract_decode.py` nor `block_attention.py`/`causal_lm_openui.py` is
  modified by this change.
- "Failed/rejected samples remain auditable." -- every candidate is appended
  regardless of verdict.

## Tests

`tests/test_harnesses/distill/test_self_distill_collect.py` covers: accept
plus auditable-rejection wiring, resume (a second run against the same store
never re-invokes `generate` for already-collected prompts and appends no
duplicate rows), cross-prompt answer-duplicate detection, the disabled/no-op
path, deterministic shard partitioning (disjoint, reproducible, and covers
every prompt across shards), the configurable acceptance filters
(`reject_forced_end`, `min_abstract_tokens`), and the
`training_example_from_capture` / `segment_ids_for_capture` adapters against
AP-019's `loss_position_mask`.

Tests use a hermetic fixture `VerifierCascade` rather than
`default_openui_cascade()`, because the real G2 schema gate depends on the
`openui_bridge` Node toolchain (`npm ci`), which is not assumed to be
available in a unit-test environment. Production callers should still default
to `default_openui_cascade()`.
