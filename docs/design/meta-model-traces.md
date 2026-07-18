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

### Solver-transition events (VSS1-04 / SLM-64)

When the verified solver runs during decode (`verified_solver_decode`, default
off), each decision emits typed events into the same decode row's `events`
list — `solver_state`, `support_result`, `certified_deduction`, `decision`,
`backtrack`, `nogood`, `solver_terminal` — built by
[`dsl/solver/replay.py`](../../src/slm_training/dsl/solver/replay.py). Exact-closure
decode emits the `solver_state` / `support_result` / `certified_deduction` /
`solver_terminal` subset; the reversible search controller additionally emits
`decision` / `backtrack` / `nogood`. The row also carries a bounded `solver`
sidecar — `{schema_version, certificate_mode, certificates, counters}` — where
`certificate_mode` (`none`/`summary`/`full`) gates certificate detail: `none`
keeps only counters + honest status, `summary` compact descriptors, `full` the
replay material (each certificate's `to_dict()`, whose recomputed digest must
equal its id). Events and certificates carry only token/path ids and SHA-256
digests — never raw region/user text. The schema bumps the decode trace to
`version = 3`; v1/v2 rows load and replay unchanged.

## Replayability

`replay_violations(trace)` certifies the decode stream is self-consistent:
each recorded step canvas must reflect that step's commits except where the
same step's remasks removed them (or EOS truncation padded the tail), and a
trace with steps must carry a final canvas. Empty list = replayable; the
fixture test proves both directions (a clean fixture decode passes, a
corrupted canvas is caught).

For solver-transition events it additionally invokes
`solver_replay_violations` ([`dsl/solver/replay.py`](../../src/slm_training/dsl/solver/replay.py)),
which checks the ten VSS1-04 invariants: fingerprint lineage (each transition's
`before_fingerprint` matches the active replay state), certified deductions
remove only live values and — in `full` mode — cite a present, digest-consistent
certificate (tamper detection), `unknown` support never removes, decisions select
exactly one live value and record the rest, backtracks restore a recorded
state/level, a `nogood` is never a certified deduction, a `solved` terminal
carries a verifier report, `certified_unsat` is impossible once any
`unknown`/budget/truncation appears, event counts match the sidecar counters, and
a truncated snapshot is reported non-replayable rather than accepted as an
exhaustive proof. Violations are human-readable strings, never assertions.

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

## Grammar-state decision traces (CAP1-02 / SLM-82)

A fourth trace kind, `grammar_decision`, captures the legal-action distribution,
selected/gold action, margin, posterior entropy, and optional sensitivity at
individual grammar-constrained decode decisions. Owner:
[`harnesses/distill/grammar_trace.py`](../../src/slm_training/harnesses/distill/grammar_trace.py);
store helper: `record_grammar_decisions` in
[`harnesses/distill/trace_store.py`](../../src/slm_training/harnesses/distill/trace_store.py).

### Schema (`trace_schema_version: cap1-02.v1`)

Each record carries: `run_id`, `checkpoint_id`, `dataset_id`, `example_id`,
`seed`, `state_fingerprint`, `state_signature_version`, `decision_index`,
`diffusion_timestep`, `legal_action_ids`, `legal_action_kinds`,
`compiler_coverage` (`complete`/`partial`/`none`), `selected_action_id`,
`target_action_ids`, `target_semantics`, `logits_or_energies`,
`normalized_probs`, `top1_margin`, `posterior_entropy_bits`,
`scope_signature`, `expected_type`, `template_signature`,
`completion_support_size_exact`, `model_feature_ref`, `sensitivity`, and
`verification_outcome`.

Legal membership comes only from the compiler/DFA owner (`allowed_id_set`,
`build_completion_forest`). The model supplies scores; the trace recorder
computes softmax, margin, and entropy over the legal set. Sensitivity adapters
are disabled by default and must run in eval mode without mutating parameters.

### Collection

`scripts/collect_trajectories.py` gains:

* `--capture-grammar-decisions`
* `--capture-logits`
* `--arity-profile`
* `--capture-sensitivity {fisher_group,gradient_squared,jacobian_norm,quantization_probe}`
* `--max-sensitivity-records N`
* `--state-stratified`

When enabled, a `GrammarTraceRecorder` is attached to the model alongside the
existing `DecodeTraceRecorder`; decisions are emitted and appended as
`grammar_decision` rows sharing the same run/trace/span identity envelope.

### Replay and coverage

`grammar_trace_replay_violations` checks that the selected action is in the
legal set, entropy lies in `[0, log2(|legal|)]`, normalized probabilities sum
to 1, and coverage is one of the allowed values.
`grammar_trace_coverage_report` produces an estimated-evidence summary: unique
state fingerprints, state-action pairs, scope/template coverage, branching
statistics, margin/entropy histograms, forced-decision fraction, and provenance.

### Honesty caveats

* The current TwoTower hook is wired into the LTR repair path. MaskGIT
  non-repair commits and compiler-tree ranked-path decisions are not yet
  instrumented; they should reuse the same `GrammarTraceRecorder` when added.
* Real checkpoint diagnostics and a full join to the CAP1-01 exact state graph
  require a scheduled GPU run and are documented as estimated evidence only
  until that run completes.
* Sensitivity adapters are stubbed at the API level; concrete Fisher/gradient/
  Jacobian/quantization probes are added per later CAP issue.
