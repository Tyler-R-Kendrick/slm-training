# Meta-model trace capture (G5, SLM-37)

Training data for the eventual DSL-generating meta-model: what the decoder
did, what the harnesses decided, and how the matrices scored it — persisted
with enough identity to be replayable and never mixed unlabeled.

Owner: [`harnesses/distill/trace_store.py`](../../src/slm_training/harnesses/distill/trace_store.py)
(extends the existing V2 decode-trajectory store — schema + retention, no new
infrastructure). Fixture evidence: `tests/test_harnesses/distill/test_meta_traces.py`.

## Trace kinds (one append-only store, `kind` discriminated)

| Kind | Producer | Payload |
| --- | --- | --- |
| `decode` (default) | `DecodeTraceRecorder` via `model.trace_recorder` | per-step canvases, commits (`t`, `id`, log-prob, constrained/forced flags), remasks with reasons, NFE counters, final text, reward vector, labels |
| `harness_decision` | `record_harness_decision` | harness name, bounded decision, typed inputs/outcome |
| `matrix_outcome` | `record_matrix_outcome` | experiment id, matrix set, pass/fail, failures, per-suite scoreboards |

Every row carries the shared identity envelope: `trajectory_id` (index +
content hash), `run_id` / `trace_id` / `span_id` (from active runtime
telemetry), and — for decode rows — policy checkpoint SHA, decode-config
hash (`decode_config_hash`), tokenizer/grammar versions, and seed. Rollouts
from different checkpoints are never mixed unlabeled.

## Replayability

`replay_violations(trace)` certifies the decode stream is self-consistent:
each recorded step canvas must reflect that step's commits except where the
same step's remasks removed them (or EOS truncation padded the tail), and a
trace with steps must carry a final canvas. Empty list = replayable; the
fixture test proves both directions (a clean fixture decode passes, a
corrupted canvas is caught).

## Retention and bucket layout

Local stores live where run evidence already lives
(`outputs/runs/<run-id>/traces/` or a campaign's
`outputs/autoresearch/<campaign>/` bundle). Durable mirroring reuses the
existing checkpoint/evidence bucket:

```
hf://buckets/TKendrick/OpenUI/traces/<run_id>/   ← sync_traces(root, run_id)
```

`sync_traces` mirrors `autoresearch/persistence.sync_campaign` exactly:
`hf buckets sync … --no-delete`, command-plan by default, `push=True` to
execute. The store is append-only (existing rows are never rewritten), so
`--no-delete` retention is safe by construction.

## Honesty

Fixture-grade wiring only: schema, replay invariant, and bucket plan are
tested; no meta-model exists, no trace corpus is claimed as sufficient
training data, and nothing is pushed to the bucket by default.
