# G5 — meta-trace capture fixture (2026-07-17)

Fixture-grade wiring for Track G5 (Linear SLM-37). Schema documentation:
the "Meta-trace corpus" section of [telemetry.md](telemetry.md). Code:
[`src/slm_training/harnesses/distill/meta_trace.py`](../../src/slm_training/harnesses/distill/meta_trace.py).

## What was built

Schema + retention over artifacts the stack already writes (no new
infrastructure, per the issue):

- **`MetaTraceRecord`** — strict pydantic record joining, per example: the
  request, the decode spec (model kind, checkpoint sha, config, seed), the
  emitted program, the per-example eval metrics, and the run's honest-gate
  verdicts, with W3C trace ids and source-artifact provenance. First-class
  `dsl_id` (the gap the recon identified in the existing distill
  TraceStore).
- **Collector** — `harvest_run_dir` joins `eval_<suite>.json` details,
  gate verdicts, the training recipe, and `trace.json` ids; absent
  artifacts degrade gracefully.
- **Retention** — append-only `traces.jsonl` + sha256-per-line manifest
  under a `campaign.json`-gated tree (campaign-store conventions;
  `sync_campaign`-mirrorable, dry by default, local authoritative).
- **Replay** — `replay_trace` re-decodes from spec, fail-closed on
  checkpoint-sha mismatch; bit-exact only for the deterministic tree-edit
  decoder (stated boundary: MaskGIT paths are replay-from-spec).

## Fixture evidence

- **Real-artifact harvest**: 76 records from four of this session's run
  dirs (E259, E260, E261, X22-reeval) into corpus
  `outputs/experiments/meta_traces/g5_fixture_20260717/` — trace-id joins,
  suite verdicts, and predictions all populated from the artifacts as they
  already exist on disk.
- **Replay proof** (`test_replay_reproduces_deterministic_decode`): a
  trace recorded from a live tree-edit decode replays to the **identical**
  output through `from_checkpoint`; tampered checkpoint shas and
  non-deterministic model kinds are rejected.

## Verification

- `tests/test_harnesses/distill/test_meta_trace.py`: strict-schema
  round-trip (extra fields rejected), harvest joins (eval details + gates +
  trace ids + recipe seed), append-only retention with checksums and
  campaign gate, and the replay proof with both fail-closed paths.
- `repo_policy`, `check-changed`, `ruff`, `git diff --check` clean.

## Honesty and limits

- Wiring evidence only; the corpus is a fixture harvest, not a curated
  meta-model training set. Per-step trajectories (`trajectory` slot) are
  populated only when distill `TraceStore` rows exist — the standalone
  `collect_trajectories` path produces them; matrix eval does not (yet).
- Bit-exact replay is limited to deterministic decoders by design; wiring
  a seed-pinned MaskGIT replay is follow-up if the meta-model needs it.
- The grammar matrix (`run_grammar_matrix.py`) does not emit `trace.json`
  bundles today — its harvested records carry no trace ids (observed on
  the X22 re-eval rows); adding `run_trace` there is a small follow-up.
