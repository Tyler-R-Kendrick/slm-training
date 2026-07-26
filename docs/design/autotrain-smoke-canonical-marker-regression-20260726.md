# Autotrain smoke pipeline regression: non-canonical persisted template markers

**Found by:** scheduled autotrain loop continuation (2026-07-26), attempting
`autotrain_wf_smoke_20260726` (iter51 of the fixture smoke loop started in
[`autotrain-loop-ledger-20260725.md`](autotrain-loop-ledger-20260725.md)).

**Status:** fix implemented and verified locally (see Verification); **not
yet committed**. `.githooks/pre-commit` runs `scripts/check_changed.py
--staged --changed-tests-only`, which for a change touching
`src/slm_training/harnesses/train_data/pipeline.py` +
`src/slm_training/resources/versions.json` selects the full
`tests/test_harnesses/train_data` + `tests/test_versioning` suites and
requires all of them green — no pre-existing-failure carve-out. That suite
already has 9 failures on `HEAD` (`7cddae6`, before this session touched
anything), confirmed via `git stash` bisection: identical failing-test list
with or without this fix applied. Repository policy prohibits bypassing
hooks (`--no-verify`) without explicit user authorization, and none was
available in this scheduled/unattended run, so the verified fix below is
left uncommitted in the working tree rather than force-committed. See
Pre-existing hook-blocking debt.

No ship claim either way (`fixture_or_scratch` evidence only).

## Symptom

`slm sft train --train-dir outputs/data/train/wf_v0 --model twotower
--context-backend scratch --output-tokenizer choice --steps 8 --fast-train
--no-sync-checkpoints` against freshly built `slm data build-train --source
fixture --version wf_v0 --synthesizer quality --programspec-seed 0
--no-publish` output failed at model construction:

```
File "src/slm_training/models/twotower.py", line 13944, in from_records
    assert_canonical_template_markers(record)
File "src/slm_training/data/contract.py", line 195, in assert_canonical_template_marker_inventory
    raise ValueError(
ValueError: persisted template markers must use opaque :slot_<ordinal> identities
```

This blocked every phase downstream of `data_train` in the smoke recipe (`sft`,
`eval`, `closeout`, `verify`), i.e. the entire autotrain loop this scheduled
task exists to continue.

## Root cause

PR #952 (`7cddae6`, SLM-315 / AP-023, merged 2026-07-25) added
`assert_canonical_template_markers(record)` to both
`TwoTowerModel.from_records` and `TwoTowerModel.training_loss`
(`src/slm_training/models/twotower.py:2812,13944`) as a fail-closed guard: any
record fed to the model must carry opaque `:slot_<ordinal>` placeholder
identities, never human-named markers like `:auth.title`.

`src/slm_training/harnesses/test_data/pipeline.py` already called
`canonicalize_example_template_markers` on every built record before
returning it (pre-existing, unrelated to #952). `src/slm_training/harnesses
/train_data/pipeline.py::build_train_data` never did — the fixture /
programspec synthesizers persist human-named markers
(`:auth.title`, `:auth.email.placeholder`, ...) straight to
`outputs/data/train/<version>/records.jsonl`. Before #952 nothing asserted
canonical markers at train time, so this gap was latent. After #952, every
`slm data build-train` output — not just this smoke fixture — fails the
moment it reaches `TwoTowerModel.from_records`.

## Fix

`src/slm_training/harnesses/train_data/pipeline.py::build_train_data`: after
the optional prompt-component/slot/semantic-role contract enrichment (which
still needs the human-readable marker names to render "Components:" /
"Semantic roles:" prompt lines) and before the leakage fingerprint block,
apply `canonicalize_example_template_markers` to every record and assert
`assert_canonical_template_markers` on the result — mirroring
`test_data/pipeline.py`'s existing pattern. Fingerprints are computed after
canonicalization (as the existing comment at that call site requires:
"Fingerprint final records after every train-only transformation so the
leakage manifest describes the exact bytes written to records.jsonl"), so
train/test fingerprinting is now on the same (canonical) representation on
both sides.

Component bump: `harness.train_data` v20 → v21 (see
`src/slm_training/resources/versions.json`).

A second, narrower fix in the same change: `build_train_data`'s
`--dedup-against` cross-corpus check fingerprints the *in-progress* build's
records before the new canonicalization step runs (it has to — it runs
earlier in the pipeline, ahead of the prompt-contract enrichment that still
needs human-readable names), but compares them against an on-disk corpus that
*is* already canonicalized by this same fix. Left alone, that skews every
`--dedup-against` fingerprint comparison to never match. Fixed by
canonicalizing each in-progress record only for that comparison (not
mutating the record used later), so both sides of the fingerprint check use
the same representation.

## Verification

- Rebuilt `outputs/data/train/wf_v0` and `outputs/data/eval/wf_v0` (fixture,
  seed 0); records now persist `:slot_0`, `:slot_1`, ... markers.
- `slm sft train` (twotower/choice/scratch/cpu, 8 steps, `--fast-train
  --no-sync-checkpoints`) and `slm eval model` (`--eval-limit 3 --suites
  smoke`) both complete — see
  [`autotrain-wf-smoke-20260726-measured-results.md`](autotrain-wf-smoke-20260726-measured-results.md)
  and iter52–iter55 in
  [`autotrain-loop-ledger-20260726.md`](autotrain-loop-ledger-20260726.md).
- `uv run --extra torch python -m scripts.verify_version_stamps --check` →
  `ok`.
- `tests/test_harnesses/train_data/test_quality_report.py::test_dedup_against_excludes_pairs_already_in_other_corpora`
  fails with `assert 0 > 0` (cross-corpus duplicates silently stop matching)
  with the marker-canonicalization fix alone; passes once the dedup-against
  comparison is also canonicalized.
- Full targeted suite —
  `tests/test_harnesses/train_data/`,
  `tests/test_scripts/test_build_train_data_cli.py`,
  `tests/test_data/test_language_contract.py`,
  `tests/test_versioning` — is **identical** with and without this change:
  127 passed / 9 failed either way (verified via `git stash`). The 9 failures
  (`test_pipeline.py::test_prompt_contracts_expose_component_counts_and_slots`,
  `test_pipeline.py::test_semantic_role_contract_uses_only_visible_slots_and_types`,
  `test_scope_corpus.py::test_identity_rows_normalize_to_canonical_opaque_slots`,
  `test_source_families.py::test_pipeline_manifest_source_families`,
  `test_staged_materialization.py::test_no_plan_legacy_fixture_bytes_remain_pinned`,
  `test_staged_materialization.py::test_staged_graph_uses_canonical_pipeline_and_rebuilds_deterministically`,
  `test_staged_materialization.py::test_invalid_staged_target_fails_closed_with_retained_evidence`,
  `test_staged_materialization.py::test_qa_without_canonical_preference_still_materializes_answers`,
  `test_build_train_data_cli.py::test_programspec_natural_prompts_opt_in`)
  pre-exist on `HEAD` and are unrelated to persisted-marker canonicalization
  — see Pre-existing hook-blocking debt below for root causes. Not fixed
  here to keep this change scoped to the loop-blocking regression.

## Pre-existing hook-blocking debt (why this fix isn't committed yet)

`.githooks/pre-commit` → `scripts/check_changed.py --staged
--changed-tests-only` requires the full selected suite to pass, with no
allowance for pre-existing failures. Any commit that touches
`src/slm_training/harnesses/train_data/` or
`src/slm_training/resources/versions.json` currently selects
`tests/test_harnesses/train_data` + `tests/test_versioning`, and that suite
already fails 9 tests on `HEAD` (`7cddae6`) before this session's diff is
applied at all — i.e. **no commit touching either path can currently pass
this repo's local pre-commit hook**, independent of this fix. Root causes,
by test:

- `test_pipeline.py::test_prompt_contracts_expose_component_counts_and_slots`,
  `test_pipeline.py::test_semantic_role_contract_uses_only_visible_slots_and_types`
  — `template_fill.py::normalize_placeholders` (a decode-time helper whose
  docstring states markers must already be canonical) is called by
  `build_train_data`'s `--prompt-slot-contract` /
  `--prompt-semantic-role-contract` enrichment on **pre-canonicalization**
  human-named markers, raising the same `assert_canonical_template_marker_inventory`
  error this fix works around elsewhere. Fixing this safely means deciding
  whether prompt-facing enrichment should use a separate, non-canonical-only
  helper or whether canonicalization should move earlier in the pipeline —
  an architecture call for `template_fill.py` (decode-critical, see
  `docs/design/decode-invariants.md`) that is out of scope for an unattended
  run to make alone.
- `test_scope_corpus.py::test_identity_rows_normalize_to_canonical_opaque_slots`
  — idempotency assertion failure on the `scope_identity_statement` family;
  not investigated (unrelated to persisted-marker canonicalization).
- `test_source_families.py::test_pipeline_manifest_source_families` —
  `assert not hasattr(config, "namespace_augment")` fails; `TrainDataConfig`
  still carries that field. Not investigated.
- `test_staged_materialization.py` (4 tests) — the committed
  `src/slm_training/resources/synthesis_plans/dsh0_cap0_fixture.json` fixture
  pins `pack.corpus_generator`/`pack.oracle` at `v18` and `gates.ship` at
  `openui_ship_gates_v2`, both several versions stale against the active
  registry (`v21` / `openui_ship_gates_v4` after this fix; already stale at
  `v20` / `v4` before it). Bumping those pins in isolation (tried, then
  reverted here to avoid widening this change) clears the version-mismatch
  failure but then surfaces `record_count == 0` from the staged
  artifact-graph path — a second, deeper, undiagnosed issue.
- `test_build_train_data_cli.py::test_programspec_natural_prompts_opt_in` —
  asserts `--programspec-natural-prompts` is a valid CLI flag; no such flag,
  config field, or behavior exists anywhere in
  `scripts/build_train_data.py` or `TrainDataConfig`. Looks like a test
  written ahead of an unimplemented feature, not a regression.

None of these reproduce differently with or without this change (confirmed
via `git stash`). Recommendation: either pay down this test debt in a
dedicated follow-up (it spans at least 4 unrelated root causes and touches a
decode-critical helper, so it deserves its own scoped session) or have a
human explicitly authorize a `--no-verify` commit for this specific,
independently-verified fix.

## Honesty

`fixture_or_scratch` wiring only — 8-step scratch CPU smoke runs, not a ship
claim. No `--ship-gates` run performed.
