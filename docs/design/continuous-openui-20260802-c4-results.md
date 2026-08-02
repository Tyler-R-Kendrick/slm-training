# Continuous autotrain cycle 4 results (2026-08-02, `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c4` |
| Source | `ca8785a5bbc74d07e56efa96a283ffe41450b01a` |
| Cycle role / intent | `promotion` / `promotion` (first promotion-role cycle in this lineage) |
| Train | `wf_smoke_v2`, 21 steps / batch 2 / seed 100001 |
| Eval suites | `smoke`, `held_out` (promotion adds `held_out` over screening's `smoke`-only) |

## What happened

Both `c4-control` and `c4-component-plan` trained quickly (3.24s and 8.42s)
but **every** smoke (3/3) and held_out (5/5) record hit
`decode_timeout_count` during evaluation — a full, not partial, timeout.

## Root cause (traced, not just observed)

- `stage_wall_minutes_for_role()` (`src/slm_training/autoresearch/climb_policy.py`)
  clamps **both** `screening` and `promotion` roles to the same
  `MAX_RUN_MINUTES=3` ceiling — there is no larger allowance for promotion.
- `eval_suites_for_role()` gives promotion cycles `("smoke", "held_out")` —
  8 total records here — versus screening's `("smoke",)` alone (3 records).
- The per-experiment wall budget (`campaign.budget.max_wall_minutes ≈
  0.778 min`, ~46.7 s) is split between the train and eval subprocess calls
  by `execute_commands()` (`src/slm_training/autoresearch/engine.py`) purely
  as "whatever's left of the deadline" — it does not scale the eval share by
  how many suites/records that role needs.
- Because training was fast, eval got ~26–31 s either way (screening or
  promotion). `_effective_record_decode_timeout()`
  (`src/slm_training/harnesses/model_build/eval_runner.py`) then fair-shares
  that wall time across all remaining records: ~10 s/record for cycle 3's 3
  records, but only ~2.7–3.4 s/record for cycle 4's 8 records — too tight
  for this CPU sandbox's real decode latency, especially with
  `component-plan`'s extra head.

This mirrors several historical incidents in this loop family
(`autotrain-cycle-1715/1716/1717/1737` in `docs/MODEL_CARD.md`) where CPU
decode timing conflicts with the fixed wall cap.

## Why this cycle does not include a code fix

A real fix means re-balancing `execute_commands()`'s train/eval wall-time
split by `cycle_role`/`eval_suites_for_role()` count. That function has
exact-value timing tests
(`test_execute_shares_one_wall_budget_across_stages`,
`test_execute_passes_inner_wall_to_evaluation` in
`tests/test_autoresearch/test_harness.py`) and this exact problem class has
had roughly ten dedicated PRs land very recently (`fix(autotrain): ...
finalize bounded autotrain evaluations`, `... reserve autotrain evaluation
finalization tail`, etc.) — it is an actively-worked area with tightly tuned
semantics. Patching it speculatively in one pass, without the context those
PRs were developed under, risks contradicting ongoing work more than it
helps. Deferred to a dedicated `improve-openui-harnesses` pass rather than
risked here.

## Next-run priorities

1. **infrastructure (model_build):** re-balance `execute_commands()`'s
   per-stage wall split so promotion cycles get a suite-count-aware eval
   floor instead of pure leftover-after-training.
2. Until that lands, promotion-role cycles will keep hitting
   `decode_timeout` on this CPU sandbox; frozen replay of
   `c4-control`/`c4-component-plan` is the only reuse path.
3. Do not promote or ship either checkpoint.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c4/`
- JSON twin: `continuous-openui-20260802-c4-results.json`
