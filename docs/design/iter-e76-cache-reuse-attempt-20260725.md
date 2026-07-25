# E76 (V7 cache-reuse) attempt — bugfix landed, matrix result blocked (2026-07-25)

JSON: [iter-e76-cache-reuse-attempt-20260725.json](iter-e76-cache-reuse-attempt-20260725.json).

## What was picked and why

`E76` (`qx_e76_cache_reuse`, "V7 champion without trust/entropy remask gates to
measure successor reuse") in `scripts/run_quality_matrix.py`'s `--matrix v7`
set. It, and its siblings `E77/E78/E80/E82/E84/E88`, are wired into the script
(`--matrix v7 --list` returns them) but never appear in this doc's V7 table
(stops at `E75`) or in `quality-matrix-results.json`'s `results` array —
unclaimed by any prior row or in-flight ticket in recent `git log`.

## Bug found and fixed (real, verified)

Every quality-matrix `Experiment` row — including the plain `E0` baseline —
currently raised:

```
ValueError: allow_unconstrained_fallback=True is unsafe for OpenUI generation
```

immediately on `ModelBuildConfig` construction, before any training step ran.
Root cause: commit `d9a526f` (SLM-291, 2026-07-24) added
`Experiment.allow_unconstrained_fallback: bool = True` plus a matching
`getattr(exp, "allow_unconstrained_fallback", True)` fallback in `_train_cfg` /
`_eval_cfg`. Because the dataclass attribute always exists, the `getattr`
default was moot: every row silently requested the decode-invariant-I6
unconstrained-fallback override, which `ModelBuildConfig.__post_init__`
fail-closes on by design. This is a regression against
[decode-invariants.md](decode-invariants.md) — the harness was requesting a
bypass no row ever intended, not the model.

**Fix** (`scripts/run_quality_matrix.py`): flipped the field default and both
`getattr` fallbacks to `False`, matching `ModelBuildConfig`'s own default. No
existing row set this field explicitly, so no row's *intended* behavior
changes — every row simply stops requesting an override it never meant to
request. Verified by reproducing the identical `ValueError` on `E0` and `E76`
before the fix, and confirming both progress past config construction into
real training after it.

`matrix.quality` bumped `v7 -> v8` in `versions.json` (behavior-changing fix,
not a no-bump).

## E76 itself: blocked, not a result

After the fix, E76 was attempted twice. **Both attempts were killed by the
repo's `MAX_RUN_MINUTES=3` bounded run cap during the `evaluate_suites()`
ship-gate stage** (generation + grammar-bridge verification + AgentV grading),
not during training:

| Attempt | Command tail | Outcome |
| --- | --- | --- |
| 1 | `--rico-limit 4` (5 suites: smoke, held_out, adversarial, ood, rico_held) | training completed (checkpoints + `loss_suites_step_40.json` written); `evaluate_suites()` never wrote a result before `timeout 175` killed it |
| 2 | `--rico-limit 2 --suites smoke,held_out` (2 suites only) | training completed again in 10.3s; the *narrowed* `evaluate_suites()` still did not finish before `timeout 170` killed it |

**Per the iron law of honest run reporting: a killed/timed-out run is not
evidence.** No parse / structural_similarity / placeholder_fidelity / reward
numbers for E76 are reported here, and none were added to
`quality-matrix-results.json`. The only real numbers available are from the
training loop's own internal loss-suite probe (not the ship-gate scoreboard):
40 steps over 310 records, 10.3s wall time, `last_loss=1848.01`,
`final_loss_eval.weighted_nll=8.87`. These characterize that training ran and
converged to *something*, not whether E76 clears any gate.

## Environment/data prep encountered along the way

None of the following are E76-specific; a fresh checkout hits every one of
them for *any* quality- or grammar-matrix run:

- `src/apps/openui_bridge` needs `NODE_OPTIONS= npm ci` — the G2/G8 verifier
  gates shell out to it; without it every generated `ProgramSpec` is
  quarantined at G2 (`ValueError: generated ProgramSpec failed F2 at G2`).
- Repo-root `NODE_OPTIONS= npm ci` is needed for the AgentV SDK
  (`node_modules/@agentv/core`, `scripts/run_agentv_eval.mjs`) — the
  unconditional final loss-suite AgentV publish in `train()` raises
  `RuntimeError: AgentV SDK is unavailable` otherwise.
- No committed venv; base image is Python 3.11 but the project needs 3.12
  (`uv venv .venv --python 3.12 && uv pip install -e '.[dev,grammar,rico]'`).
- A fresh `build_train_data --source all --curriculum` run (this doc's own
  canonical build command) defaults `include_scope_corpus=True` and pulls in
  language-contract fragment records (both from `d9a526f` too), which lack a
  root binding and break V5+ arms' `output_tokenizer=lexer` +
  `use_symbol_table` parser. Filtered with `--no-scope-corpus
  --no-language-contract` plus a post-hoc `root =` regex filter.
- The committed `e937_role_safe_all_targets_v2` corpus (grammar X0's own
  train dir) has the same root-binding contamination on 173/524 records, plus
  ~13% of the clean remainder whose prompt doesn't enumerate every declared
  `:slot_N` contiguously once the shared boilerplate DESIGN.md text is folded
  into the honest-slot-contract heuristic (`inventory_from_prompt`) — the
  DESIGN.md's own "Do's and Don'ts" example text says `:slot_2`, which
  pollutes the extracted inventory for any record declaring fewer than 3
  slots. Worked around locally (`outputs/data/train/e937_root_only`,
  `outputs/data/eval/e938_dm_stripped`, design_md stripped, root-bound only,
  310/524 kept) — this is a session-local scratch workaround, not a fix to the
  shared corpus or the `inventory_from_prompt` heuristic.

## Honest takeaway

Fixture-scale, not ship: the one real, verified, complete deliverable this
session is the `allow_unconstrained_fallback` fail-closed fix — a genuine
correctness bug independent of any specific matrix ID. E76's actual
quality-matrix numbers remain unmeasured; a follow-up session should budget
the eval stage specifically (it is the expensive part, not training) or
narrow `--suites` further before attempting E76/E77/E78/E80/E82/E84/E88 again.
