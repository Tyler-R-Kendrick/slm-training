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

## c1847 result update

c1847 supplied that successor and falsified the hoped-for quality/cost tradeoff
on the screening fixture. Semantic-exhaustive supervision raised structural
similarity (`.2383 -> .3225`) and cut emitted tokens (`130 -> 60`), forwards
(`28 -> 11`), and p50 latency (`6886 -> 2811 ms`), but binder F1 (`.9524 ->
.8222`), fidelity (`.9167 -> .7222`), reward (`.9360 -> .8657`), and training
loss (`22.6143 -> 23.4322`) regressed. Meaningful-program rate and exact
AST/canonical matches stayed at zero. Both arms are therefore rejected fixture
screening results; this is an objective tradeoff, not evidence that Lean,
parallelism, or caching prevented learning.

The next cycle must use a distinct preregistered quality objective, retain the
matched control, and expand beyond `n=3` before any capability claim. The
cacheless-preloaded-model fix is now covered by `harness.model_build.eval` v79;
no matched-CPU parallelism is justified until a wall/CPU/memory parity
benchmark demonstrates a win.

## c1848 harness-repair update

The first c1848 attempt made the new binder-slot-ownership training signal
visible to the campaign compiler, but its decode weight was rejected by the
model capability gate because the arm omitted `compiler_decode_mode=tree`.
That is a genuine harness wiring gap, not evidence against the model: the
matched control completed, while the candidate never reached training. Commit
`d876037b5` adds the required tree capability and regression coverage; the
supervisor has acknowledged the repair and must replay the frozen c1848 arm
before interpreting quality metrics.

c1849 reached the model constructor and exposed the next fail-closed boundary:
the reserved binder-slot ownership fields have no runtime owner. The control
still completed (`structural_similarity=.0575`, binder F1 `.8222`, fidelity
`.7222`, `n=3`), but the candidate produced no model artifact. Commit
`fb9093314` routes the next distinct objective to the implemented
`slot-component-coverage` owner; c1849 remains infrastructure evidence only.

## c1851 learning-status update

c1851 is the first executable measurement of that repaired owner. The
slot-component treatment improved meaningful-program rate (`0 -> .667`),
component recall (`0 -> .3333`), and held binder F1/fidelity at `1.0`, while
structural similarity rose only `.1425 -> .1767`. It also raised training loss
(`20.9509 -> 22.5437`), parameters (`+4,520`), tokens (`+16.3%`), forwards
(`+12.8%`), compiler time (`19.5 -> 23.4 s`), and p50 latency
(`7400 -> 8771 ms`). With smoke `n=3`, exact AST/canonical agreement at zero,
and production suites absent, this is a narrow fixture learning signal, not a
quality or ship result. The c1851 matrix therefore classifies the candidate as
`NON_POSITIVE` and routes a distinct, size-matched objective; it must not be
replayed as a positive arm or used to justify capacity growth.
