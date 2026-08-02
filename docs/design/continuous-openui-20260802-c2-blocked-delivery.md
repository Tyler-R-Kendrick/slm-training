# Continuous autotrain cycle 2 + delivery block (2026-08-02)

**Honesty:** diagnostic / infrastructure finding. **Not a ship claim.**

## Summary

Cycle 2 (`continuous-loop-20260802-continuous-openui-local-8c0b60dd-c2`) consumed
cycle 1's pending `retry_measurement` action and replayed the frozen
control/bounds arms at evaluation only (checkpoints reused, no retraining).
It hit a second, distinct infrastructure blocker in the same
`publish_model_evaluation` call path: `node: --import tsx is not allowed in
NODE_OPTIONS`, because `publish_agentv_evaluation` (`src/slm_training/evals/agentv.py`)
spawns `node scripts/run_agentv_eval.mjs` via `subprocess.run` without
controlling the child environment, so it inherits whatever `NODE_OPTIONS` the
calling shell has set. In this container `NODE_OPTIONS="--import tsx"
--max-old-space-size=8192` is set for unrelated tooling, and Node hard-rejects
`--import` when it arrives via `NODE_OPTIONS` instead of argv.

**A fix was written, verified, and is ready to apply — but this session could
not land it as a commit.** See "Why the fix is undelivered" below.

## The fix (verified, uncommitted)

`src/slm_training/evals/agentv.py`, in `publish_agentv_evaluation`, right
before the `subprocess.run(command, ...)` call:

```diff
     if trace_id is not None:
         command.extend(("--trace-id", trace_id, "--run-id", Path(run_dir).name))
+    # Node rejects certain flags (e.g. --import) when they arrive via
+    # NODE_OPTIONS instead of argv, so an ambient NODE_OPTIONS inherited from
+    # the caller's shell (set for unrelated tooling) can hard-fail every
+    # invocation of the pinned runner. This subprocess needs no NODE_OPTIONS.
+    child_env = dict(os.environ)
+    child_env.pop("NODE_OPTIONS", None)
     completed = subprocess.run(
         command,
         cwd=runtime_root,
         check=False,
         capture_output=True,
         text=True,
+        env=child_env,
     )
```

Verified: with this change, `tests/test_evals/test_agentv.py::test_publish_agentv_evaluation_uses_sdk_and_jsonl`
and `::test_agentv_contract_checks_fail_even_when_pass_flag_is_true` (both of
which exercise the real subprocess, no mocking) go from failing with exactly
this `RuntimeError` to passing. `evals.agentv` would bump `v6 -> v7` per the
version-stamp contract; the bump text is drafted in the working tree's
`src/slm_training/resources/versions.json` diff alongside the code change.

## Why the fix is undelivered

`src/slm_training/evals/` changes conservatively select the **entire**
`tests/test_evals/` + `tests/test_harnesses/model_build/` directories in both
the local pre-commit hook (`scripts/check_changed.py`) and CI's
`--changed-tests-only` selection. Running that full selection in this
container (`uv sync --extra dev` + root `npm ci`, matching CI's own install
steps) produces:

```
69 failed, 698 passed, 12 skipped, 12 deselected, 5 errors in 26.63s
```

None of these 69 failures are caused by the AgentV fix above — the identical
set (modulo the AgentV-crash-cascaded ones, which the fix resolves) reproduces
on a clean checkout of `e8ad8f0d` (this branch's pre-cycle-1 HEAD) and on
`30639ac1` (current `origin/main` tip) with **no changes applied**. The
repository's pre-commit hook (`.githooks/pre-commit`) requires the full
selected suite to pass before allowing a commit, and hooks are not bypassed
without explicit user authorization, so this session could not commit the
`agentv.py` fix.

**This is not an environment artifact.** GitHub Actions CI shows green on both
commits (`list_workflow_runs` for `Tyler-R-Kendrick/slm-training`), but CI's
`check_changed.py --changed-tests-only --base-ref <base>` selection is scoped
to each PR's own diff. Recent merges (e.g. `#1314`, `30639ac`) touched
`src/slm_training/dsl/grammar/fastpath/*` and `models/twotower.py` but not
`tests/test_evals/` broadly, so CI never re-ran the full `test_evals` +
`model_build` directories and this breakage has been sitting undetected. The
first PR whose diff touches `src/slm_training/evals/` or
`src/slm_training/dsl/` broadly (this one included) is the one that surfaces
it, in CI exactly as it did locally.

### Root-cause breakdown of the 69 failures (grouped by traceback site)

| Count | Site | Error |
| ---: | --- | --- |
| 24 | `src/slm_training/dsl/operators/registry.py:128` | `OperatorAuthorityError: pack static/schema oracle rejected source` |
| 16 | `src/slm_training/data/contract.py:195` | `ValueError: persisted template markers must use opaque :slot_<ordinal> identities` |
| 3 | `src/slm_training/dsl/lang_core.py:172` | `RuntimeError: Install bridge deps: cd src/apps/openui_bridge && npm ci` (missing node_modules in this checkout; low-effort env fix, not a code bug) |
| ~26 | scattered across `test_meaningful_program.py`, `test_metric_gaming.py`, `test_semantic_fidelity.py`, `test_semantic_failure.py`, `test_judge_resolution.py`, `test_cap2_operator.py`, `test_agentv.py` (2 pre-existing, unrelated to the fix above) | distinct assertion failures, no shared traceback site |

The top two clusters (24 + 16 = 40 of 69, ~58%) share a plausible common
thread: both reject **human-readable placeholder markers** (e.g.
`:hero.title`, used pervasively as illustrative fixture literals across
`tests/test_evals/`) against a validator that now requires opaque
`:slot_<ordinal>` identities
(`assert_canonical_template_marker_inventory` in `contract.py`, introduced by
`c9dbfce` / PR #1234). `registry.py`'s `oracle()` call for the `openui` pack
rejects the same style of source (confirmed directly: `'root =
TextContent(":hero.title")'` fails `validate_with_pack_authority`). This
reads as **test-fixture migration debt**: production/persisted-record paths
were tightened to require opaque slot markers, but the large body of
`tests/test_evals/` fixtures that construct sources inline with readable
names for developer legibility were never migrated. Confirming and fixing
this is a real, scoped harness task, but it spans many files and multiple
test modules (`dsl.operators` + `data.contract` + downstream `evals` and
`model_build` consumers) — deliberately **not** attempted in this cycle to
avoid mixing it with the single-family AgentV fix above, per the "one
harness family per attribution arm" rule.

Attempting `npm ci` in `src/apps/openui_bridge/` (to clear the 3-count
`lang_core.py` cluster) actually **increased** the local failure count to 167
in this container — installing that bridge's node_modules flips some tests
from an implicit skip/short-circuit path to an attempted-and-differently-failing
path. Not investigated further; recorded here so a future session doesn't
re-try it expecting a clean win.

## Next-run priorities

1. **infrastructure (this fix):** apply the `agentv.py` NODE_OPTIONS diff
   above once the suite is green enough to commit, or once a human authorizes
   landing it despite the pre-existing failures (e.g. via a hook exception or
   by fixing the marker-migration debt first). Re-run
   `tests/test_evals/test_agentv.py` to confirm both subprocess-backed tests
   still pass.
2. **harness (data/dsl, separate arm):** audit
   `assert_canonical_template_marker_inventory` vs. `registry.py`'s
   `oracle()` for the `openui` pack — determine whether human-readable
   fixture markers should be accepted pre-canonicalization, or whether
   ~40 test fixtures need migrating to `:slot_<ordinal>` literals. This is
   the highest-leverage single fix (58% of current `test_evals` +
   `model_build` failures).
3. **process:** `scripts/check_changed.py`'s CI selection is diff-scoped per
   PR, so a broad pre-existing break like this can sit undetected until an
   unrelated PR's diff happens to touch the same directory. Consider a
   periodic (not diff-gated) full-suite CI job so this class of debt surfaces
   on a schedule rather than by accident.
4. **retry_measurement:** cycle 2's frozen replay (arms
   `c20260802-continuous-openui-local-8c0b60dd-c2-{control,bounds}`) is still
   pending on the AgentV fix landing; re-run the continuous driver once (1)
   is resolved to get a real, complete scoreboard for the frozen arms instead
   of another `measurement_incomplete`.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c2/`
- Prior cycle: `docs/design/continuous-openui-20260802-c1-results.{json,md}`
- Verified-but-uncommitted fix: `src/slm_training/evals/agentv.py` (working
  tree diff at end of session; reproduced above for durability)
