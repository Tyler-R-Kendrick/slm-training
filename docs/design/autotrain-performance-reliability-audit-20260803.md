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

## c1852 learning-status update

c1852 executed the distinct slot-component/fidelity coupling with a matched
control. The treatment was a complete null on every guarded quality metric:
structure `.1742/.1742`, MPR `.333/.333`, recall `.25/.25`, binder F1
`.633/.633`, fidelity `.528/.528`, reward `.765/.765`, and exact AST/canonical
rates `0/0`. It added `4,515` parameters, doubled training loss
(`12.00 -> 24.16`), increased tokens `21 -> 30`, forwards `4 -> 5`, compiler
time `2361 -> 2440 ms`, and p50 latency `910 -> 966 ms`. This confirms that
the c1851 fixture gain does not transfer to this coupled objective; current
prevention is weak/overfit supervision plus an underpowered `n=3` evaluation,
not Lean or a training crash. The loop remains fail-closed and routes another
distinct, size-matched hypothesis rather than promoting capacity.

## c1853 learning-status update

c1853 tested a different hierarchical objective coupling slot ownership to the
component inventory head. It produced a complete null: both arms had structure
`.115`, MPR/recall/binder/fidelity/reward `0`, and exact AST/canonical rates
`0`. The candidate used `77,916` additional parameters (`+4.84%`) and raised
p50 latency `856 -> 902 ms` with no token/forward reduction. This is the
strongest current evidence that the blocker is not failure to update weights,
but inadequate target/data coverage and auxiliary objectives that add capacity
without transferring capability; the fixture remains only `n=3`. Capacity
growth is therefore blocked by the parameter-efficiency law and the loop keeps
the matched control while seeking a data/target-oriented, size-matched arm.

## c1854 learning-status update

c1854 is the first follow-on that shows a useful fixture signal from targeted
exposure plus the implemented slot-component owner: structural similarity
improved `.0575 -> .1353`, MPR `0 -> .333`, and recall `0 -> .167`. However,
binder/fidelity remain low, exact AST/canonical rates are zero, p50 latency
rises `917 -> 1005 ms`, tokens `21 -> 36`, forwards `4 -> 7`, and the candidate
adds `4,515` parameters. With smoke `n=3` and no held-out or production suites,
this is learning on a narrow fixture, not high-quality OpenUI generalization.
The loop correctly queues a fresh-seed confirmation, keeps Lean promotion gates
locked, and blocks capacity promotion until parameter-efficiency and full-suite
evidence exist.

## c1855 learning-status update

The fresh-seed exposure-cap confirmation raises structural similarity again
(`.230 -> .354`) and reaches the structure threshold, but it does not re-establish
the complete quality contract: MPR is only `.333`, component recall `.333`,
exact AST/canonical rates are zero, p50 rises `2433 -> 2574 ms`, and forwards
`7 -> 12`. Training loss moves in the opposite direction (`10.20 -> 18.46`),
confirming that loss is not a promotion proxy. The campaign correctly rejects
the confirmation and exhausts the fingerprint. The model is learning a narrow
structural fixture pattern, but high-quality OpenUI production learning is
prevented by weak meaning/recall targets, zero exact agreement, small `n=3`
evaluation, and cost regressions—not by Lean or a runtime training failure.

## c1858-c1859 and current optimization closeout

c1858 (slot-contract context) and c1859 (constraint-graph conditioning) both
completed their frozen smoke measurements with size-matched parameters. Neither
changed guarded quality: c1858 held structure `.1742`, MPR `.333`, recall `.25`,
binder F1 `.633`, and fidelity `.528`, while c1859 held structure `.3058`, MPR
`.333`, recall `.25`, binder F1 `.952`, and fidelity `.917`. Exact AST and
canonical agreement remained zero in both; c1858's `7.66%` fixture latency
improvement and c1859's `3.44%` regression are efficiency diagnostics only.
The registered arm bank was therefore exhausted and the loop stopped at a
typed `repair_harness` handoff instead of retrying a rejected approach.

The reliability/performance repair is now implemented:

- `evaluate_suites()` constructs a checkpoint-backed model once and reuses that
  immutable instance across suites. The exact checkpoint digest is bound into
  the suite cache key; mutable preloaded models without a digest still bypass
  cache fail-closed. This is `harness.model_build.eval` v80.
- Formal claims reuse only a successful, complete-project Lean check within one
  process, keyed by the full proof-project digest and runner identity. Failed,
  timed-out, non-total, or axiom-tainted checks are never cached. This is
  `harness.autoresearch.formal` v9.
- Matched CPU arms remain serialized by policy. Parallelizing them would
  contend for the same cores and corrupt the wall comparison; only read-only
  discovery or explicitly isolated shards may be parallelized after a measured
  wall/CPU/memory parity benchmark.
- A new preregistered `semantic-contrast-compiler-margin` arm combines two
  distinct learning signals (hard-valid semantic contrast and grammar-oracle
  compiler alignment) so the next cycle can proceed after c1859 without
  recycling an exhausted objective.

The next run must still be treated as fixture evidence until the evaluation
ladder reaches `n≥20` with held-out, adversarial, OOD, and RICO suites. Lean,
cache identity, and supervisor liveness are not current learning blockers.

## c1860 measurement closeout

c1860 exercised the new size-matched semantic-contrast plus compiler-alignment
objective. The candidate completed its 20-step scratch train and smoke decode,
but the outer hard cap interrupted the matched control during evaluation before
it wrote a scoreboard. The candidate therefore cannot be compared or called a
learning win. Its standalone smoke result is weak (`struct=.2742`, MPR `.333`,
recall `.333`, binder F1 `.633`, exact AST/canonical `0`, `n=3`) and its p50
latency rose to `3606 ms` with `126` tokens and `52` forwards. The candidate is
size-matched (`1,608,962` parameters) and its higher loss (`17.98` vs control
training `15.09`) does not establish quality harm without a complete control.

The immediate prevention is measurement reliability: replay both frozen arms
under the same bounded stage before changing the objective. After that replay,
the quality blocker to investigate is target/data supervision that can move
exact AST agreement, not another capacity increase or a Lean relaxation.

## c1861 frozen replay closeout

c1861 replayed the exact c1860 control and candidate after the compiler-tree
deadline repair. Both scoreboards completed with zero decode timeouts, proving
the prior control interruption was a harness reliability issue rather than a
training failure. The candidate then showed a matched fixture improvement:
structure `.0575 → .2742`, MPR `0 → .333`, component recall `0 → .333`, and
p50 `16759 → 3626 ms`, with identical `1,608,962` parameters. This is a
narrow learning signal only: smoke `n=3`, exact AST/canonical rates `0`, MPR
and recall below gates, and all held-out/adversarial/OOD/RICO suites missing.
The candidate is therefore rejected for ship and not promoted. The remaining
blocker is target/data coverage and semantic transfer, not Lean, model size,
or a runtime timeout. Next priority is a distinct size-matched quality
objective evaluated on the full ladder; the loop keeps the matched control and
does not recycle the rejected arm.
