# On-policy abstract self-distillation collector (AP-020 / SLM-309)

`slm_training.harnesses.distill.abstract_self_distill` closes the warm-up
train-inference gap: it samples prompt-only abstract traces from a live
policy via AP-017's `CausalLMOpenUIPlugin.generate_abstract_traced` (SLM-304,
`models/abstract_decode.py`), joins them with AP-020's own policy/verifier
provenance, and persists deterministic, resumable, deduplicated, sharded
records. Nothing here is called by any existing generation or training
path, so bottleneck-only (self-distillation disabled) behavior is
unaffected by this module's existence — it is a pure side channel, same
posture as AP-019's `forward_with_segments`.

## Why the verifier re-derives from the raw decode

`generate_abstract_traced` always reports `valid=True`: on a parse failure
it silently substitutes `_certified_fallback()` (a fixed trivial program,
`"root = Separator()"`) rather than raising. Its own `valid` flag therefore
cannot be trusted as an acceptance signal — the ticket's "the generating
policy cannot be the sole semantic judge" requirement is not satisfiable by
reading `generation.valid`. `build_self_distill_record` instead re-decodes
`capture.answer.token_ids` itself and independently calls
`dsl.parser.validate` on that raw text, so acceptance reflects whether the
policy's own raw output parsed — not whether the generation path's decode-
repair fallback kicked in.

## Provenance joined per record (`SelfDistillRecordV1`)

- **Policy identity**: `policy_checkpoint_sha` (`compatibility_fingerprint()`),
  `codebook_version` and `plan_fingerprint` (`AbstractPlanV1`).
- **Sampling settings**: `abstract_temperature`/`abstract_seed`,
  `answer_temperature`/`answer_seed` (from the plugin's
  `AbstractDecodeConfig`).
- **Decode-side trace** (from AP-017's `AbstractTraceCapture`):
  `forced_end`, `abstract_termination`, `stop_reason`, `trace_logprob` (sum
  of every recorded per-token logprob; `None` entries are forward-free
  commits and are excluded).
- **Verifier evidence**: `verifier_accepted`/`verifier_reason`, and the raw
  `raw_answer_text` itself — kept on *every* record, accepted or rejected,
  so failed samples remain auditable rather than dropped.

## Determinism, resume, dedup, sharding

- **Dedup key**: `content_hash` is a sha256 over
  `(policy_checkpoint_sha, prompt, abstract_token_ids, answer_token_ids)`,
  mirroring `data.leakage`'s `fingerprint_pair` convention.
- **Resume**: `load_seen_hashes(path)` reads a prior JSONL manifest's
  `content_hash` column into a set (mirrors
  `data.leakage.load_train_fingerprints`'s membership-set pattern);
  `collect_on_policy_traces(..., seen_content_hashes=seen)` skips matching
  records as duplicates without re-invoking the policy.
- **Sharding**: `shard = int(content_hash, 16) % num_shards`, the same
  sha256-mod-modulus pattern `harnesses/train_data/split_policy.py` already
  uses for deterministic root-family splits.
- **Persistence**: `append_records(path, records)` is append-only JSONL,
  mirroring `harnesses/distill/trace_store.py`'s "never rewrite existing
  rows" convention.

## Training loss on abstract plus target tokens

`self_distill_training_batch` builds `[prompt; abstract; target]` segment
ids and calls AP-019's `CausalLMOpenUIPlugin.forward_with_segments`
directly — no new loss math, since `forward_with_segments`'s masked SFT
loss already restricts credit to abstract+target positions. There is no
privileged-plan segment at on-policy generation time (unlike AP-018's
static warm-up records), so the segment sequence is simply prompt then
abstract then target; the bottleneck mask's one added restriction (target
never attends `PRIVILEGED_PLAN`) is vacuous here since no such segment is
present, and every other rule falls out of plain causal order as usual.
Only accepted (verifier-passed) records may build a training batch —
`self_distill_training_batch` raises on a rejected record.

## Reproduction

```bash
pytest -q tests/test_harnesses/distill/test_abstract_self_distill.py
```

Plain unit tests (one case uses the tiny
`hf-internal-testing/tiny-random-LlamaForCausalLM` fixture already used
elsewhere in this repo for the `forward_with_segments` wiring check);
completes in seconds, well inside the repository's hard run cap
(AGENTS.md § "Hard run cap").
