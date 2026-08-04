# Autotrain a6r56e c2: `allowed_id_set` fingerprint hot-path fix

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.** The fix is
**verified but not committed to source in this PR** — see
["Why this fix is not applied here"](#why-this-fix-is-not-applied-here)
below. This doc records the diagnosis, the fix, and the replay evidence for
a follow-up PR to apply once the blocking pre-existing failures are
triaged.

## What happened

Cycle 2 of loop `continuous-openui-a6r56e` trained a matched control/canvas
pair on `wf_smoke_v2` (`--steps 20`, CPU, `twotower`) — both arms completed
training normally. Evaluation against the published `e938_role_safe_all_targets_v2`
smoke suite hit the repository's `MAX_RUN_MINUTES=3` wall on **both** arms and
was killed mid-flight with a `KeyboardInterrupt`, before scoring a single
record:

```text
terminal_witness -> outgoing -> _build_openui_completion_forest_direct
  -> _decision_kind -> _grammar_terminal_kind -> allowed_id_set
  -> hash(tuple(sorted(tokenizer.token_to_id.items())))
```

## Root cause

`allowed_id_set` (`src/slm_training/dsl/grammar/fastpath/token_map.py`)
unconditionally computed a fingerprint of the entire tokenizer vocabulary —
`hash(tuple(sorted(tokenizer.token_to_id.items())))` — on **every** call, even
when `use_cache=False`, purely to build a cache key that was then discarded
because the cache path was skipped. Every terminal-witness call site in
`compiler_draft.py` (`_grammar_terminal_kind`, the completion-forest builders,
etc.) calls `allowed_id_set` without `use_cache=True` (the default), so this
was pure waste on the hot decode path: an O(vocab·log vocab) sort-and-hash
computed and thrown away for each of the many terminal checks per candidate
branch in the recursive completion-forest traversal.

## Fix

`model.twotower` v271 → v272
(`src/slm_training/dsl/grammar/fastpath/token_map.py`, `allowed_id_set`):
compute the fingerprint and cache key **only** inside the `use_cache=True`
branch; the `use_cache=False` path now calls `_allowed_id_set_dsl` directly
with no cache-key overhead. No caller-visible behavior change — same return
value for the same inputs, just without computing and discarding a fingerprint
nobody uses on the non-cached path.

Microbenchmark (569-token DSL-native vocab, 20,000 calls, this container):

| | per-call cost |
| --- | --- |
| old fingerprint computation alone | 77.23 µs |
| new full `allowed_id_set(use_cache=False)` call | 3.06 µs |

## Replay evidence (identical frozen arms)

Same checkpoints, same eval manifest (`frozen_manifest_sha256`
`7e3757a0…5b59d6d`), same command, only the harness fix applied:

| Arm | Before fix | After fix |
| --- | --- | --- |
| control | interrupted at >180s wall, zero scoreboard | completed in 81.8s; full `gates.json`; `insufficient_n=3`, `decode_timeout_count=3` (all 3 records individually exceeded the unrelated `--decode-timeout-seconds 24.0` budget, so quality metrics are `None`) |
| canvas (`grammar_completion_bounds=True`) | interrupted at >180s wall, zero scoreboard | completed in 81.5s; full `gates.json`; `insufficient_n=3`, `decode_timeout_count=2`; 1/3 records scored (`structural_similarity=0.175`, `meaningful_program_rate=0.0`) |

Both arms went from an **unrecoverable process interrupt** to a **completed
run with a real AgentEvals scoreboard** — the SDLC Phase A "executable
unblocking" gate (a fix removes a prior hard blocker and the identical arm
then completes with a usable scoreboard, replay-proven). Ship gates still
fail on this cycle — expected at fixture scale (`n=3` vs the suite's `n>=20`
requirement) and because 2-3 of 3 records still individually exceed the
per-record 24s decode-timeout budget. **No quality or promotion claim is made
here.**

## Regression coverage

- `tests/test_models/test_compiler_decode.py`: 223 passed, 0 failed.
- `tests/test_dsl/`: 1289 passed, 6 skipped, 11 failed. All 11 failures were
  independently reproduced against unpatched `09ab77c` (`git stash` +
  rerun) — pre-existing fixture/contract-version drift (`output_contract_version`
  mismatch, non-canonical template markers, etc.), unrelated to this change.
- `python -m scripts.verify_version_stamps --check`: ok, 1 component touched
  (`model.twotower` v271 → v272).

## Why this fix is not applied here

`token_map.py` is owned by the `model.twotower` component. Bumping that
component's version (required by this repo's `versions.json` contract for
any change to an owned file) makes `.githooks/pre-commit` →
`scripts.check_changed --staged --changed-tests-only` run the **full**
`tests/test_dsl/`, `tests/test_harnesses/model_build/`, and
`tests/test_versioning/` suites (≈2,800+ nodes) before allowing the commit.
On `main` HEAD `09ab77cb7221188b0d16838f4d25f9cd732ed42c`, that combined
surface already has pre-existing failures that have nothing to do with this
change. Independently reproduced with the fix backed out (`git stash` +
rerun on unpatched HEAD):

- `tests/test_dsl/test_language_contract.py::test_to_dict_round_trips_fields`
  — asserts `OUTPUT_CONTRACT_VERSION == 4`; the source constant is `2`
  (`src/slm_training/dsl/language_contract.py:36`, deliberately documented
  as "v2 is intentionally checkpoint-incompatible"). Looks like a stale test
  expectation, not a source bug — but that is a call for whoever owns
  `language_contract.py`, not this cycle.
- `tests/test_dsl/test_language_contract.py::test_symbolic_surface_policy_preserves_closed_terms_and_declared_markers`
- `tests/test_dsl/test_language_contract.py::test_symbolic_surface_evidence_hashes_match_committed_sources`
- `tests/test_dsl/test_latent_pack.py::test_synthesized_pack_honest_none_slots_fail_closed`
- `tests/test_dsl/test_packs.py::test_pack_fixture_loop_generate_train_eval`
- `tests/test_dsl/test_production_codec.py::test_generation_request_from_record`
- `tests/test_dsl/test_production_codec.py::test_normalize_switchitem_and_slider_signatures`
- `tests/test_dsl/test_production_codec.py::test_fixture_settings_schema_consistency`
- `tests/test_dsl/test_production_codec.py::test_normalize_full_slider_signature_from_generated_schema`
- `tests/test_dsl/test_speculative_rank.py::test_committed_table_matches_its_builder`
- `tests/test_dsl/test_speculative_rank.py::test_committed_table_ranks_real_branch_points_confidently`
- `tests/test_harnesses/model_build/test_full_state_resume.py::test_initialize_from_unions_context_vocab_for_filtered_corpus`
- `tests/test_harnesses/model_build/test_train_model_cli_decode_weights.py::test_train_model_cli_threads_decode_weights_into_model_build_config`

None of these touch `token_map.py`, `compiler_draft.py`, `allowed_id_set`,
or anything on the decode path this cycle changed. Per this repo's rules,
`--no-verify` is not an option, and fixing ~13 unrelated pre-existing
failures spanning contract versioning, production codec, speculative-rank
committed tables, and vocab-union resume is out of scope for one decode-perf
cycle — it needs its own dedicated triage, ideally per-file since each looks
like a different root cause.

The verified fix is included below as a diff for a follow-up PR/session to
apply directly once the target test surface is green (or once the
`check_changed` component→test mapping is scoped more precisely than whole
directories):

```diff
--- a/src/slm_training/dsl/grammar/fastpath/token_map.py
+++ b/src/slm_training/dsl/grammar/fastpath/token_map.py
@@ -97,13 +97,16 @@ def allowed_id_set(
     if not terminals:
         return None
     if _is_dsl_native(tokenizer):
-        fingerprint = hash(tuple(sorted(tokenizer.token_to_id.items())))
-        key = (fingerprint, int(getattr(tokenizer, "version", 0)), terminals)
-        cached = _DSL_ALLOWED_CACHE.get(key) if use_cache else None
-        if cached is None:
-            result = _allowed_id_set_dsl(tokenizer, terminals)
-            if result is not None and use_cache:
-                _DSL_ALLOWED_CACHE[key] = frozenset(result)
-        else:
-            result = set(cached)
+        if use_cache:
+            fingerprint = hash(tuple(sorted(tokenizer.token_to_id.items())))
+            key = (fingerprint, int(getattr(tokenizer, "version", 0)), terminals)
+            cached = _DSL_ALLOWED_CACHE.get(key)
+            if cached is None:
+                result = _allowed_id_set_dsl(tokenizer, terminals)
+                if result is not None:
+                    _DSL_ALLOWED_CACHE[key] = frozenset(result)
+            else:
+                result = set(cached)
+        else:
+            result = _allowed_id_set_dsl(tokenizer, terminals)
         if result is not None and active_dynamic_ids is not None:
             from slm_training.models.dsl_tokenizer import TokenKind
```

Plus the matching `model.twotower` `v271` → `v272` bump in
`src/slm_training/resources/versions.json`'s history list (component version
number + a dated changelog entry — same shape as every prior entry in that
file's `history` array).

## Next steps

The completion-forest crash this cycle fixed is now resolved. A separate,
still-open issue remains: individual records on this fixture still exceed the
24s per-record decode-timeout budget (2/3 and 3/3 records respectively).
That is a distinct diagnostic — likely remaining decode-path cost beyond the
fingerprint bug — and should be profiled independently before re-attempting
the bounds-on vs bounds-off model comparison this cycle originally set out to
run.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-continuous-openui-a6r56e-05c9cc71-c2/`
- Runs: `.../runs/c20260801-continuous-openui-a6r56e-05c9cc71-c2-control/`,
  `.../runs/c20260801-continuous-openui-a6r56e-05c9cc71-c2-canvas/`
- JSON twin: [`autotrain-cycle-a6r56e-c2-allowed-id-set-fingerprint-fix.json`](autotrain-cycle-a6r56e-c2-allowed-id-set-fingerprint-fix.json)
