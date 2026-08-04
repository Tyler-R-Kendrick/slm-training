# Continuous autotrain: 2026-08-04 (session ixpohr) cycle 3 — frozen replay confirms the fix is real but can't apply retroactively

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c3`
**Integration commit:** `e113f5b9` (this session's `generate_batch_size` harness fix, on top of `main` tip `eba6db30`)
**Cycle intent:** `retry_measurement` — frozen replay of c2's locked manifest,
per the `repair_harness` → `retry_measurement` handoff chain from
[`continuous-openui-local-ixpohr-c2-results.md`](continuous-openui-local-ixpohr-c2-results.md).

Two distinct manifest identities appear below — do not conflate them:

- `recipe.source_frozen_manifest_sha256` = `5b2ed5b9...` — c2's original
  locked manifest, the one this cycle replays byte-for-byte.
- `harness_signal.replay_manifest_sha256` = `9cf48c80...513daf4e` — **this
  cycle's own** (c3's) locked manifest identity, produced by executing the
  source manifest under `retry_measurement`. It is not the source; it is
  what a future `repair_harness` action against c3 itself would cite.

**Verdict:** measurement incomplete again — `decode_timeout_count=3/3` on both
arms, identical to c2. This is **not a fix failure**. It confirms a
structural property of frozen replay: the c2 manifest was locked *before*
the `generate_batch_size` fix
([`e113f5b9`](https://github.com/tyler-r-kendrick/slm-training/commit/e113f5b987cc41e4be0eb2955691d695d325ad23))
existed, so it does not — and by the frozen-replay-fidelity contract,
structurally cannot — request `generate_batch_size=1`. `scoreboard.json`
confirms: `decode_batch_size_configured=16` for both arms this cycle, i.e.
the fix's screening-role default never had a chance to apply, because
frozen replay reuses the exact locked knobs byte-for-byte rather than
regenerating a fresh recipe from the current `_matrix()`/`knobs()` builder.

## What this does and doesn't mean

- The fix itself is real, landed, and unit-tested
  (`test_config_generate_batch_size_overrides_plugin_default`,
  `harness.model_build.eval` v84→v85). It takes effect for **every new**
  screening campaign generated from this commit forward.
- It **cannot** unblock this specific c2/c3 frozen-manifest chain. Frozen
  replay's entire purpose is byte-identical reproduction of a prior
  experiment's knobs for a valid before/after comparison — a new opt-in
  knob added after the manifest was locked is, by design, invisible to that
  replay.
- `measurement.max_consecutive_frozen_replays=1` (from the climb policy) is
  now exhausted for this arm pair.

## Recommendation: stop retrying this exact frozen arm

The underlying recipe's compiler cost (`component-plan`/`control` at
1,755,764 matched params) is inherent to its **locked** configuration and
cannot complete under `decode_timeout_seconds=8.0` without the batch-size
fix — which this frozen arm structurally cannot receive. Continuing to
acknowledge `repair_harness` receipts against this exact
`replay_manifest_sha256` (`9cf48c80...513daf4e`) would not converge; it
would just consume another `max_consecutive_frozen_replays` budget for a
predictable identical failure.

The correct next step is a **fresh** screening hypothesis (a new campaign,
not a frozen replay), which will pick up `generate_batch_size=1`
automatically from the current `_matrix()`/`knobs()` builder.

## SDLC Phase A

**Not positive** (`measurement_incomplete`, `harness_failure` on both arms,
`primary_metric_unavailable`). No stack layer for this cycle.

## Next priorities

1. Do not open another `repair_harness` cycle against this exact frozen
   manifest — it cannot succeed regardless of further repairs.
2. Start a fresh screening hypothesis so the current builder's
   `generate_batch_size=1` applies from the start.
3. If a *fresh* smoke eval still times out despite `generate_batch_size=1`,
   that is new information warranting a distinct harness investigation
   (e.g. `decode_timeout_seconds=8.0` itself too tight for compiler-heavy
   candidates even at per-record chunking).

Machine evidence:
[`continuous-openui-local-ixpohr-c3-results.json`](continuous-openui-local-ixpohr-c3-results.json).
