# Autotrain performance and reliability audit — 2026-08-03

## Decision

The loop is learning weights, but c1846 is not evidence of a useful production
capability. It is a three-document scratch evaluation with 21 optimizer steps:
the exposure arm lowered training loss (`20.8952 -> 20.3384`) and improved
meaningful-program rate (`0.333 -> 0.667`), binder F1 (`0.633 -> 1.000`), and
placeholder fidelity (`0.528 -> 1.000`), while structural similarity fell
(`0.464 -> 0.412`) and exact AST/canonical matches remained zero. The result is
therefore a mixed, fixture-only signal, not a promotion or ship result.

The measured failure is not a hidden Lean or orchestration stall:

| surface | c1846 evidence | interpretation | disposition |
|---|---:|---|---|
| training | 12.46 s control / 14.43 s candidate; forward + backward = 93% of telemetry | ordinary CPU model compute dominates; sampler overhead is not the current wall blocker | keep size/exposure controls matched; optimize only with a measured model-side benchmark |
| exact decode | 7.11 s control / 17.02 s candidate compiler time | exact compiler enumeration is expensive but constrained and fail-closed | retain authority; profile cache/key reuse before any algorithmic change |
| proof | LeverProof build/test passed; warm rerun ~1.2 s | Lean is not preventing this cycle from learning | keep fresh proof preflight for promotion; use the existing incremental build cache |
| evaluation | `n=3`, missing held-out/adversarial/OOD/RICO suites, AST/canonical = 0 | quality evidence is underpowered and the model is not yet at the ship rung | do not promote; advance the ladder and run full suites for claims |
| caching | preloaded-model evaluations had no checkpoint identity | a read/write suite cache could replay another live model's metrics | fixed fail-closed in `harness.model_build.eval` v79 |

## One-shot work packets

These are the bounded packets used for the cross-harness audit. Each packet has
one owner, one canonical surface, and an acceptance test; none may relax I2/I6,
the Lean gate, the size budget, or the evidence contract.

1. **Lean/proof — `improve-lean-optimums`**
   - Confirm `make -C src/leverproof_lean test` is incremental and portable.
   - Reject stale proof sidecars by source digest; never reuse a proof across a
     changed theorem bundle.
   - Acceptance: full Lean test, formal-contract audit, and a stale-digest
     regression all pass within the repository run cap.
   - Status: complete on the current branch/origin lineage; warm replay is fast
     and the audit is fail-closed.

2. **Supervisor/e2e — `improve-openui-harnesses`**
   - Preserve matched arms, dynamic formal/eval reservations, typed handoffs,
     and get-latest reconciliation between cycles.
   - Classify timeout/incomplete evidence separately from model failures.
   - Acceptance: supervisor and continuous-loop tests pass; c1846 handoff keeps
     its exact recipes, checkpoints, and next-run priorities.
   - Status: complete in the merged continuous-supervisor hardening lineage;
     c1846 correctly remained a rejected screening result and left the loop
     `IDLE` for the next successor.

3. **Training/eval — `improve-openui-harnesses`**
   - Keep exposure targeting deterministic and size-matched; measure train
     telemetry and exact generated metrics separately.
   - Prefer production batch APIs when available; do not turn a tiny scratch
     suite into a ship claim.
   - Acceptance: model-build/eval tests, version-stamp checks, and a matrix row
     with recipe, suite `n`, and honest pass/fail.
   - Status: current telemetry shows no sampler bottleneck; the next useful
     experiment is a distinct semantic/compiler objective, not another capacity
     or exposure replay.

4. **Cache/reliability — `improve-openui-harnesses`**
   - A cache key must contain a checkpoint or immutable model-state identity.
   - Preloaded mutable models without that identity must recompute and emit a
     typed reason rather than replay stale quality.
   - Acceptance: `test_preloaded_model_never_replays_checkpointless_eval_cache`
     passes and `cache_bypass_reason` appears in the result payload.
   - Status: implemented in `eval_runner.py`, version `harness.model_build.eval`
     v79.

5. **Parallelism — `autotrain`/`sdlc`**
   - Do not run matched CPU training arms concurrently by default: the current
     bottleneck is model compute, so concurrency would contend for the same
     cores and corrupt the wall comparison.
   - Parallelize only independent, read-only discovery or already isolated
     evaluation shards after a benchmark proves a wall win and the aggregate
     cap remains respected.
   - Acceptance: a preregistered benchmark records wall, CPU, memory, and metric
     parity; no production decode authority is moved to an unconstrained path.
   - Status: intentionally not enabled; no evidence currently justifies the
     risk.

## Next-run steering

The next screening arm is the preregistered
`exposure-targeted-semantic-exhaustive-compiler-decision-margin` successor. It
is the first useful discriminator because c1846 improved binding/meaningful
signals but regressed structure and cost. It must retain the matched control,
report trainable parameters, expose compiler/decode counters, and remain
fixture-only until the full held-out ladder is available. If the successor is
null again, route the five Lean diagnosis lanes rather than recycling the
quality-arm bank.

