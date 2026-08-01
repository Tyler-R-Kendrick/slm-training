# Autotrain continuous-loop blockers (2026-08-01)

Loop: `continuous-openui-local`, cycles `continuous-loop-20260801-c1` /
`-c2`. Both cycles failed at the `scripts.evaluate_model --ship-gates` step
before any scoreboard was produced — this is an infra diagnosis, not a
training result. No metric claims below.

## 1. Main CI was fully red (fixed, shipped)

HEAD of `main` (`51227b0`, #1252) failed CI on every job: `python-static`
(`ruff check .`) and all 8 `python(N)` regression-test shards
(`python -m scripts.check_changed`, GitHub Actions run `30693586285`). Root
cause: a single `ruff` E741 violation (bare `l` loop variable) in
`tests/test_scripts/test_repair_formal_timeout_history.py`, added by #1252.
`check_changed`'s ruff gate runs *before* pytest in every shard, so this one
lint error blocked pytest from running anywhere on `main` — confirmed by
diffing against the prior green run `30693383313` (head `18e8796`, all
green).

Fix: rename `l` → `line`. Shipped: PR
[#1254](https://github.com/Tyler-R-Kendrick/slm-training/pull/1254).

## 2. AgentV eval spawn fails under inherited `NODE_OPTIONS` (fix ready, not yet landed)

`publish_agentv_evaluation` (`src/slm_training/evals/agentv.py`) spawns
`node scripts/run_agentv_eval.mjs` via `subprocess.run` without sanitizing
the environment. Any `NODE_OPTIONS` inherited from the calling shell that
carries an `--import` flag unrelated to the runner (e.g. `--import tsx` from
unrelated dev tooling in this execution environment) makes plain `node`
refuse to start: `node: --import tsx is not allowed in NODE_OPTIONS`. Every
`evaluate_model --ship-gates` run then hard-fails at the eval-publish step,
unrelated to the run's actual recipe/knobs. Reproduced on both continuous
cycles above.

Verified fix: strip `NODE_OPTIONS` from the subprocess env before spawning
`node`. Confirmed `node scripts/run_agentv_eval.mjs --help` starts cleanly
once `NODE_OPTIONS` is cleared, and 2 of 4 previously-failing
`tests/test_evals/test_agentv.py` cases pass with the fix (the other 2 are
pre-existing/unrelated, see §3).

**Not yet committed** — see §3 for why, and the patch below for the exact
diff to reapply once unblocked.

```diff
diff --git a/src/slm_training/evals/agentv.py b/src/slm_training/evals/agentv.py
index 79ab337..5871ed0 100644
--- a/src/slm_training/evals/agentv.py
+++ b/src/slm_training/evals/agentv.py
@@ -116,12 +116,15 @@ def publish_agentv_evaluation(
     ]
     if trace_id is not None:
         command.extend(("--trace-id", trace_id, "--run-id", Path(run_dir).name))
+    env = dict(os.environ)
+    env.pop("NODE_OPTIONS", None)
     completed = subprocess.run(
         command,
         cwd=runtime_root,
         check=False,
         capture_output=True,
         text=True,
+        env=env,
     )
     if completed.returncode:
         detail = (completed.stderr or completed.stdout).strip()
```

Plus a `no-bump:` history entry on `components["evals.agentv"]` (still `v6`)
in `src/slm_training/resources/versions.json` — subprocess env plumbing
only, no change to published artifact shape or criteria semantics.

## 3. Blocker on landing §2: pre-existing DSL/grammar-backend serialization gap (HarnessSignalV1, family unclear — likely core `dsl`, not one of the nine harness families)

`agentv.py` lives under `src/slm_training/evals/`, which `scripts.check_changed`
maps to the full `tests/test_evals` + `tests/test_harnesses/model_build`
directories (by design — eval-harness changes are considered training-affecting).
Running those directories on this commit surfaces **9 pre-existing failures**,
confirmed unrelated to the NODE_OPTIONS fix and to local environment
(reproduced with the CI-pinned `torch==2.5.1+cpu` wheel, not just the
CUDA build `pip install -e .` pulls by default):

- `tests/test_evals/test_ambiguous_operator_followups.py::test_corpus_covers_all_seven_named_relations_at_least_once`
- `tests/test_evals/test_cap2_operator.py::test_missing_prediction_fails_closed_with_confidence_bounds`
- `tests/test_evals/test_denoising_nll.py::test_loss_suites_clear_stale_runtime_symbol_features`
- `tests/test_evals/test_meaningful_program.py::test_binding_aware_v2_gaming_corpus`
- `tests/test_evals/test_metric_gaming.py::test_retry_first_selected_oracle_metrics_differ`
- `tests/test_evals/test_operator_systems_benchmark.py::test_partial_stratum_never_forces_regardless_of_bypass_toggle`
- `tests/test_evals/test_oracle_scoring_replay.py::test_score_prediction_keys_and_production_path`
- `tests/test_evals/test_semantic_fidelity.py::test_ast_beq_true_for_style_normalized_match`
- `tests/test_harnesses/model_build/test_train_model_cli_decode_weights.py::test_train_model_cli_threads_decode_weights_into_model_build_config`

Root-caused at least one of these directly: `ast_beq` (`src/slm_training/evals/semantic_fidelity.py`)
calls `slm_training.dsl.parser.validate()`, whose `LarkBackend.serialize()`
(`src/slm_training/dsl/grammar/backends/lark_backend.py:419-420`) is a
pass-through (`program.serialized or program.source`) — it does **not**
normalize whitespace around punctuation. Two style-equivalent programs that
differ only in a space after a comma
(`Stack([t], "column")` vs `Stack([t],"column")`) parse to the same
structure but serialize back to their original, non-identical source text,
so `ast_beq` — which expects style-normalized equality after
serialize/strip — returns `False` where the test expects `True`.

Why this doc doesn't fix it: `dsl`/grammar backend selection and
serialization is core, shared infrastructure, not one of the nine harness
families this loop is scoped to repair
(`autoresearch`, `annotations`, `distill`, `experiments`, `model_build`,
`preference`, `quality`, `rl`, `test_data`, `train_data`) per
`autotrain-iteration-delivery.md`'s "never mix harness and model changes in
one attribution arm." It also predates this session — CI's per-commit
`--base-ref` diffing means the full `test_evals` + `model_build` directories
only get exercised when a commit's diff happens to touch a file under
those prefixes, so this has likely been latent since whichever commit
introduced the serialization gap, uncaught until this diagnosis.

**Action for a future cycle:** treat as a `HarnessSignalV1` on the `dsl`
grammar-backend surface (not `evals.agentv`), reproduce standalone (repro
above), and repair `LarkBackend.serialize()` (or whichever backend is
canonical) to round-trip through a real normalizing serializer rather than
passing `program.source` through unchanged. Once that lands, reapply the
§2 patch and land it through the normal `test_evals` + `model_build` gate.
