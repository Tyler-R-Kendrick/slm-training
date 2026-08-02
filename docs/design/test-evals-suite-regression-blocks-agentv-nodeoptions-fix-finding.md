# Finding: `tests/test_evals` has a pre-existing 64-test regression that blocks the local commit hook for any AgentV NODE_OPTIONS fix

**Honesty:** `fixture_or_scratch`. Diagnostic finding surfaced while repairing an
autotrain continuous-loop cycle. **Not ship. Not a fix — a diagnostic
finding**, same status as
[`decode-compiler-tree-lexer-rebuild-cost-finding.md`](decode-compiler-tree-lexer-rebuild-cost-finding.md).
The candidate fix described below was **reverted, not committed**, because it
cannot pass the repo's own pre-commit gate.

## Context

Continuous loop `continuous-openui-local`, cycle
`continuous-loop-20260802-continuous-openui-local-8c0b60dd-c1` (see
[`agentv-sdk-wiring-results.json`](agentv-sdk-wiring-results.json)'s
`npm_ci_repair_verification` block), had the AgentV SDK availability gap
repaired via `npm ci`. Retrying the frozen-replay-at-eval path for the
successor cycle (`...-8c0b60dd-c2`) then hit a second, distinct blocker:

```
RuntimeError: AgentV SDK evaluation failed: node: --import tsx is not allowed in NODE_OPTIONS
```

## Root cause

`publish_agentv_evaluation` (`src/slm_training/evals/agentv.py:119`) spawns
`node <runner.mjs> ...` via `subprocess.run` without an explicit `env=`, so it
inherits the parent process's full environment, including this sandbox's
ambient `NODE_OPTIONS='"--import tsx" --max-old-space-size=8192'` (set for
unrelated TypeScript tooling, not for this bare-`node` eval runner script).
Node rejects the `--import` flag when it arrives via `NODE_OPTIONS` for a
plain script invocation, so every AgentV ship-gate evaluation fails in any
environment carrying that ambient variable — independent of model quality,
independent of the AgentV SDK install itself.

## Candidate fix (verified, then reverted — see below)

```python
# NODE_OPTIONS is process-wide ambient config (e.g. a `--import tsx` loader
# set for unrelated TS tooling); Node rejects some flags there for a bare
# `node <script>` invocation, so this subprocess must not inherit it.
env = dict(os.environ)
env.pop("NODE_OPTIONS", None)
completed = subprocess.run(
    command,
    cwd=runtime_root,
    check=False,
    capture_output=True,
    text=True,
    env=env,
)
```

Verification performed before reverting:
- `python -m py_compile src/slm_training/evals/agentv.py` — clean.
- `ruff check src/slm_training/evals/agentv.py` — all checks passed.
- `pytest tests/test_evals/test_agentv.py` — 7/7 passed (both with and
  without the fix — this file does not exercise the real `node` subprocess
  path, so it does not independently confirm the fix, only that it causes no
  regression there).
- Would have bumped `evals.agentv` to `v7` in
  `src/slm_training/resources/versions.json` per the version-stamp contract
  (`python -m scripts.verify_version_stamps --check` passed with the bump
  staged).

## Why it was reverted instead of committed

This repo's pre-commit hook (`.githooks/pre-commit` ->
`check-changed --staged --changed-tests-only`) maps any change under
`src/slm_training/evals/` to the **entire** `tests/test_evals` directory
(`scripts/check_changed.py` line ~124-131), not just the touched file's own
test module. Running that directory at the current commit
(`f95f85e47e6c9e9d5eb30a001df168854694766a`, and its parent
`c9144577938ff407e1bbe401649f580ad3cec2c3`) fails **independently of this
fix**:

```
$ pytest tests/test_evals -q
64 failed, 333 passed, 5 errors in 15.80s
```

Confirmed via `git stash` / `git stash pop` around the candidate diff: the
same 64 failures + 5 errors reproduce byte-identically with the fix present
or absent. Example (one of several distinct root causes observed, not an
exhaustive list):

```
tests/test_evals/test_semantic_fidelity.py::test_ast_beq_true_for_style_normalized_match
AssertionError: assert False is True
 +  where False = ast_beq('root = Stack([t], "column")\nt = TextContent(":x")',
                           'root = Stack([t],"column")\nt = TextContent(":x")')
```

```
tests/test_evals/test_operator_systems_benchmark.py (multiple)
slm_training.dsl.operators.registry.OperatorAuthorityError: pack static/schema oracle rejected source
```

These are genuine, reproducible, pre-existing failures in this checkout —
not environment-setup gaps like the AgentV `npm ci` issue, and not flaky
(each reproduces standalone, not just as part of the full-directory run).
They are unrelated to the NODE_OPTIONS subprocess-env change. A cheap check
for "is this a very recent regression from the Lark lexer/scanner caching
work" (`20b658f` "cache Lark lexer/scanner per grammar", part of the ~50
commits currently ahead of this sandbox's `origin/main` mirror) was
inconclusive in the time available — `tests/test_evals/test_semantic_fidelity.py`
did not exist yet at `20b658f^`, so a direct before/after diff on that file
isn't possible; a real bisection needs its own dedicated session.

Per this repo's own delivery rule (`sdlc` / `autotrain-iteration-delivery.md`
A1: "commit as soon as a coherent unit is green locally") and the
"never weaken a gate to green CI" law, the correct action was **not** to
force the commit through with `--no-verify` (not authorized in this
unattended run) and **not** to spend this cycle fixing 64 unrelated
pre-existing failures. The candidate diff was reverted (`git checkout --
src/slm_training/evals/agentv.py src/slm_training/resources/versions.json`)
to keep the continuous-loop worktree clean, and is recorded here in full so
it can be reapplied trivially once `tests/test_evals` health is restored.

## Why this is a finding, not a patch

`tests/test_evals` is the required (directory-wide) test target for **any**
change under `src/slm_training/evals/` per `scripts/check_changed.py`. Until
someone fixes this pre-existing failure surface, the local pre-commit hook
structurally blocks all future commits that touch AgentV/eval-runner code —
including this NODE_OPTIONS fix, and including any other legitimate eval
harness work. This is a `HarnessSignalV1`-shaped blocker for the `model_build`
/ eval family and should route through `improve-openui-harnesses`, not be
patched blind inside an unrelated cycle.

## Proposed next steps (not applied this session)

1. Bisect the ~50 commits ahead of the sandbox's `origin/main` mirror
   (`git log --oneline origin/main..HEAD`) — the Lark lexer/scanner caching
   cluster (`20b658f` and neighbors) is the most likely suspect for the
   `ast_beq` AST-equivalence regressions, since it touches the same
   incremental-parse path documented in
   [`decode-compiler-tree-lexer-rebuild-cost-finding.md`](decode-compiler-tree-lexer-rebuild-cost-finding.md).
   The `OperatorAuthorityError` failures in
   `test_operator_systems_benchmark.py` may be a separate, second root cause
   — do not assume one fix repairs all 64.
2. Once `tests/test_evals` is green (or the remaining failures are
   individually triaged and explicitly marked/ticketed as accepted, not
   silently skipped), reapply the NODE_OPTIONS fix in `## Candidate fix`
   above verbatim and let it land through the normal hook.
3. Re-run the continuous-loop supervised cycle for `continuous-openui-local`
   — the AgentV SDK gap is already repaired (`npm ci` committed evidence via
   `f95f85e`), so once NODE_OPTIONS is also fixed, the frozen-replay-at-eval
   path should reach a real scoreboard instead of failing pre-scoring.

## Cleanup note

No new files created outside `docs/design/`; the reverted diff is not present
in the working tree (verified clean via `git status --short` after revert).

Captured: 2026-08-02T21:50:00Z
