# Hybrid unmask scheduler (HX4) + shared repair session (HX5) — 2026-08-04

Wiring/fixture claim only. No quality claim, no ship claim, both levers
default-off. Delivered by a parallel subagent swarm (three isolated lanes,
integration, adversarial review). Companions:
[`residual-honesty-block-diffusion-20260804.md`](residual-honesty-block-diffusion-20260804.md),
[`precompiled-grammar-admissibility-20260804.md`](precompiled-grammar-admissibility-20260804.md).

## Why

The `block_diffusion_decode` lever (L-D) was **binary**: it replaced the whole
selection policy, pitting block-parallel denoising *against* positionwise
MaskGIT rather than combining them. The measured decode facts argue for a
split: MaskGIT commits are effectively left-contiguous (picks left of holes
mostly fail — a genuine LTR frontier exists), while template seeding and
mid-decode fragmentation create span-shaped interior mask runs that are
exactly what block-parallel denoising is for.

## Preregistered criteria (locked before measurement)

(i) default-path (`unmask_mode="positions"`, levers off) outputs
**byte-identical** — reject outright on failure; (ii) hybrid smoke generates
grammar-valid output with `hybrid_span_commits > 0` **AND**
`hybrid_frontier_commits > 0`; (iii) W2 E1 parity green + deterministic
rebuild; (iv) W3 worst-shard weight bounded with fallback equivalence proven.

## Results

### (i) Byte-parity — PASS

`profile_generate --maskgit --rounds 2` (checkpoint `outputs/runs/s1_d64`,
local/gitignored — the committed demo predates the output contract):
outputs byte-identical to `origin/main` on both the default and
`--no-incremental` configs.

### HX5 speedup — NOT DEMONSTRATED (honest negative)

An early comparison against baselines recorded *while the swarm lanes were
saturating the CPU* showed a 40% wall reduction. A fair A/B re-measured on an
idle machine against a pristine `origin/main` worktree shows the truth:

| | sec/generate | finalize_ms mean |
| --- | --- | --- |
| origin/main (idle) | 4.2747 | 3308.9 |
| this branch (idle) | 4.3599 | 3363.8 |
| delta | **+2.0%** | **+1.7%** |

Within the declared ±10% noise band, i.e. **neutral, marginally negative**.
The earlier 40% figure was pure load artifact and is retracted here rather
than reported. HX5's mechanism (one repair session across
`_ensure_valid_openui` attempts + deterministic-attempt early exit) only
engages when attempts > 1; this fixture resolves on the first attempt, so the
path never activates. HX5 remains justified as a *correctness/telemetry*
simplification (one engine-stat fold per certify instead of one per attempt),
not as a measured speedup. A repair-heavy fixture is required to test it.

### (ii) Hybrid lane engagement — FAILED, then FIXED

The criterion caught a real design bug. As first implemented, every
contiguous run was classified wholesale, so on a fresh (unfragmented) canvas
— one single all-mask run — **everything** became a span and the positionwise
lane never engaged. Measured at three thresholds:

| span threshold | span commits | frontier commits |
| --- | --- | --- |
| `hybrid_span_min_run=3` | 64 | **0** |
| `hybrid_span_min_run=8` | 64 | **0** |
| `hybrid_span_min_run=99` | **0** | 105 |

That is a threshold-flipped **binary switch** — precisely the failure the
hybrid was meant to remove. Left-contiguous commits keep the remaining mask
region as one run, so the degeneracy is structural, not incidental.

**Fix**: a run touching committed context (or the BOS edge) hands its leading
`hybrid_frontier_head` positions (new lever, default 2) to the positionwise
lane; the interior remainder stays span-eligible. That edge is where left
grammar context is richest and where a block-parallel commit is most likely
to be rejected. After the fix, on the same fixture:

| arm | wall s | span | frontier | span reverts | finalize_ms | output |
| --- | --- | --- | --- | --- | --- | --- |
| positions (default) | 3.99 | 0 | 0 | 0 | 2612 | valid |
| hybrid, head=2 | 4.97 | 132 | 16 | **2** | 1616 | valid |
| hybrid, head=4 | 5.08 | 472 | 32 | **8** | 2529 | valid |

Criterion (ii) passes: both lanes live, output grammar-valid. Note the
**region-scoped revert fired** (2 and 8 times) — the L-C exact multi-region
checker doing real work on parallel span commits, reverting span-lane
commits while keeping the sequentially-admitted frontier commits.

**Hybrid is slower than the default on this fixture** (4.97 vs 3.99 s/gen).
Parallel span denoising does not pay for its joint-validation cost at this
model/canvas scale. Reported as an observation, not a regression to fix: the
lever is default-off and carries no performance claim.

### (iii) W2 artifact + certified adapter — PASS

`rule_origins` / `rule_lengths` I64 tensors plus `state_min_terminals`
(control-only **lower** bound, negative-direction pruning only, max 3, 54/119
states nonzero, validated against a brute-force BFS oracle over 600 reachable
stack configurations). `StaticLalrAdapter` executes shift/reduce over the
arrays and is certified in lockstep against the live Lark `InteractiveParser`
over the E1 corpus (accepts-set equality at every terminal prefix: 8
programs, 56 steps, 14 states). `require_certified_static_lalr()` is
fail-closed and cached per artifact digest.

**Consumed by nothing**: `completion_kernel.py` and `engine.py` are
untouched; no default decode path changed. The artifact was regenerated
(53824 → 56224 bytes) so its sha256 changed — checkpoints declaring the old
`completion_artifact` identity must be re-stamped (fail-closed contract, not
a regression).

### (iv) W3 duration-aware sharding — PASS, with a caveat that matters

138 test files measured (1541.5s of weight) into
`src/slm_training/resources/ci/test_durations_v1.json`; `_shard_test_nodes`
now LPT-packs by measured file weight, degenerating exactly to the previous
count-balanced packing when the table is absent (proven by test).

The measurement **re-pointed the premise**: the heavy files are
`test_topology_apply.py` (429s, itself a partial measurement) and
`test_run_dsh5_03_bulk_operator_crossover.py` (301s) — *not*
`test_run_autotrain_continuous.py` (~2.9s), which the earlier CI-timeout
diagnosis had blamed. **Two files individually exceed the 3-minute per-job
wall**; no sharding scheme fixes that, and they need their own remediation
(marker, split, or opt-out). LPT also trades a lower median shard for a
higher max (the max becomes the single heaviest file, which count-balancing
implicitly split). Follow-up filed.

### Pre-existing bug fixed en route

`tests/test_dsl/test_arity_analysis.py::test_analysis_package_is_torch_free`
**fails on pristine `origin/main`** (verified in a clean worktree):
`dsl/analysis/arity/decision_difficulty.py` imported
`harnesses.distill.grammar_trace` at runtime, pulling torch into a package
whose torch-freedom is an asserted invariant. Fixed with a `TYPE_CHECKING`
guard (the module already has `from __future__ import annotations`, so the
annotation still resolves).

## Verification

243 targeted tests green across all lanes (hybrid, grammar-diffusion,
decode-stats, factory overrides, completion artifact, static LALR adapter,
E1 static-control-domain, residual support, engine direct-feed, grammar
fastpath, check_changed, arity analysis), plus 67 re-run after the
frontier-head fix and a new regression test pinning the degenerate-lane bug.

## Honesty

Fixture-scale evidence, one local checkpoint, one prompt per screening arm,
single-run timings on a shared box (±10% wall is noise). No quality suite was
run; no ship claim; both new levers default off. The HX5 40% figure is
retracted as load artifact. Hybrid costs wall time on this fixture. The W2
adapter and bound are certified but unconsumed. Any run using
`unmask_mode="hybrid"` for numbers that matter needs
`documenting-experiment-results` and its own preregistered campaign.

## Follow-ups

- Repair-heavy fixture to actually test HX5 (attempts > 1).
- Kernel consumption of the certified adapter + `state_min_terminals` under
  the campaign's E2 cold-speed gate.
- Remediate the two >3-minute test files; consider per-node weighting if the
  goal shifts from heavy-file isolation to minimizing max shard wall.
- Hybrid quality screening (preregistered) before any default change; the
  cluster-mode (V7) comparison remains open — hybrid is *structural* region
  typing, cluster is *attention-derived*, and they are mutually exclusive by
  guard today.
