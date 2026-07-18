---
name: synthesis-feedback
description: Close the data-quality loop after any training-data synthesis or build. Use whenever build_train_data (or the build_train_data job) runs, when a quality_report.json shows warnings, when planning changes to producers/synthesizers, or when deciding what data experiment to run next — the build's own quality_report.json, rejected.jsonl, and synthesis_feedback.json are the evidence that drives synthesis-harness improvements. Never weaken gates to make numbers pass.
---

# Synthesis feedback loop

Every train-data build measures itself. The loop is: **build → read the
feedback → improve the synthesis harness (or its inputs) → rebuild → compare.**
Gates and thresholds are never loosened to make a build look better; the
producer or synthesizer is fixed instead.

## Artifacts every build emits (strict profile, the default)

Beside `records.jsonl` in `outputs/data/train/<version>/`:

| Artifact | What it holds |
| --- | --- |
| `quality_report.json` | Constraint fitness (parse/judge/placeholder), garbage, redundancy per dedup layer, decontamination, per-family stats, `warnings[]` |
| `rejected.jsonl` | Every dropped candidate with stage + reason; full payloads for normalize/verification/quality stages |
| `synthesis_feedback.json` | Per-family / per-synthesizer yields, dominant rejection reasons, rule-based `recommendations[]`, autoresearch-shaped `experiment_candidates[]` |

Also served live: `GET /api/data/train/{version}/quality` (report + feedback)
and `GET /api/data/train/{version}/rejected`.

## The loop, step by step

1. **Build** with the strict profile (default): `python -m scripts.build_train_data --version <v>`.
2. **Read** `synthesis_feedback.json` first — `recommendations[]` names the
   family or synthesizer to fix (`redundant_expansion`, `low_yield`,
   `eval_leakage_source`) with the evidence numbers. Cross-check
   `quality_report.json` `warnings[]`.
3. **Act on the harness, not the gate**:
   - `redundant_expansion` → reduce expansion counts, diversify templates or
     namespaces in `harnesses/train_data/synth.py` (via
     `improve-openui-harnesses`).
   - `low_yield` → fix the producer input or synthesizer for the named family;
     the top rejection reason says what breaks.
   - `eval_leakage_source` → audit that producer's inputs for eval-adjacent
     material; the decontamination gate stays as-is.
4. **Experiment, don't guess**: `experiment_candidates[]` entries carry
   hypothesis / expected_effect / falsification_criteria / knobs in the
   autoresearch shape — file them as experiments (`openui-autoresearch`,
   `running-experiment-matrices`) rather than tweaking blind.
5. **Rebuild under a new version** and compare the two quality reports; the
   feedback numbers are the success metric (yield up, dup_share down,
   admitted count held). Document per `documenting-experiment-results`.
6. **Escalate scope**: overlap ACROSS committed snapshots is measured by
   `python -m scripts.audit_data_corpora` (durable results in
   `docs/design/data-corpus-audit.*`); exclude already-covered pairs from new
   builds with `--dedup-against`.

## Derivative data from the same evidence

- **Curate**: `build_train_data --source existing --derive-from … --profile strict`
  (also launchable from a run's dashboard page).
- **Difficulty-aware curation**: train with `--emit-record-nll`, then rebuild
  with `--difficulty-from outputs/runs/<id>/record_nll.jsonl` — the
  trivially-easy NLL tail is discounted in `curation_score` (Superfiltering).
- **Preference negatives**: `python -m scripts.mine_rejected_preferences
  --dataset <version>` pairs quality/quarantine rejects against their best
  admitted twins (`pair_corpus=rejected_ledger`).

## Invariants

- Strict is the default; `--profile permissive` is a diagnostic escape hatch,
  never a fix. Threshold changes go through `honest-ship-eval`.
- Nothing is dropped silently — if a gate fires, it must be visible in
  `rejected.jsonl` and the report.
- Feedback flows into harness changes and experiments; the loop never edits
  its own acceptance thresholds (`improve-openui-harnesses` contract).
