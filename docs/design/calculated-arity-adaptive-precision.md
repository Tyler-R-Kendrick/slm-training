# CAP0-01 (SLM-77): calculated arity and adaptive precision

**Status:** specification only. This contract adds no model, solver, quantizer,
experiment, checkpoint, or ship claim. It preregisters terms, proofs, comparator
rules, and falsifiers for later CAP issues; existing decode and deployment behavior
is unchanged.

**Owners and boundaries:** the bounded tree carrier remains in
[grammar-topology-diffusion.md](grammar-topology-diffusion.md), hard next-action
membership remains in [lattice-recursive-search.md](lattice-recursive-search.md),
and verified-completion support and destructive pruning remain in
[verified-scope-solver.md](verified-scope-solver.md). The frozen language bundle and
its declared property tiers remain in [dsl-pack-contract.md](dsl-pack-contract.md).
This document owns only capacity, task-rate, precision, physical-cost, and matched
experiment claims. Sources and fidelity labels live in
[research-lineage.md](research-lineage.md).

## Central non-equivalence

The program must never collapse these four quantities into one claim:

> **exact symbolic-state capacity != task-relevant information rate != neural
> weight/activation precision != deployed system optimum**

An exact count of legal states does not determine how many states a task must
distinguish. A task quotient does not determine the arity of weights or
activations. A nominal arity does not determine packed bytes, latency, energy, or
quality on a real kernel. Any optimum claim must therefore declare, in the same
row or sentence, the state set or task quotient, every `K` and `d` role, structural
bounds, noise model, exact/estimated status, and packing assumption when cost is
involved.

## Declared frame and notation

A report first freezes a frame

`F = (pack, grammar, parser, codec, state_signature, AST/depth/scope/cutoff bounds,
traffic distribution, distortion, noise model)`.

Changing any member creates a different claim.

| Symbol | Meaning | Required declaration |
| --- | --- | --- |
| `Q` | Exact continuation quotient: equivalence classes of bounded compiler states that admit the same verifier-relevant continuation behavior under `F` | State representation, equivalence test, enumeration/completeness method, versions, bounds, and work count |
| `M = \|Q\|` | Number of exact quotient classes | Exact only when `Q` is replayably enumerated; otherwise use an estimate with provenance |
| `M_epsilon` | Task-confusability quotient size after merging states whose distinction changes distortion by at most declared `epsilon` under the declared traffic distribution | Distortion, `epsilon`, estimator, sample design, coverage, confidence interval, and tail metric |
| `R_task(D)` | Task rate-distortion target, `min I(X; Z \| Q)`, where `Q` is exact compiler state, `Z` carries only input-conditioned uncertainty, and the minimum ranges over a declared encoder/representation class subject to expected distortion at most `D` | Distribution of `X`, definition of `Z`, fixed `Q`, encoder class, distortion, estimator, and uncertainty |
| `K_z` | Arity of one latent or decision-code coordinate | Coordinate semantics and length `d_z` |
| `K_w` | Number of representable weight levels after any grouping/scaling | Group size, scale/zero representation, exceptions, and packing |
| `K_a` | Number of representable activation levels | Tensor sites, calibration, clipping, accumulation precision, and kernel |
| `K_E` | Number of representable energy/output-score levels | Scored sites, calibration, numeric format, and decision rule |
| `K_code` | Alphabet size of an explicit error-control code | Code length `d_code`, minimum distance, channel/noise model, encoder, and decoder |
| physical cost | Packed weights/codes plus scales, zero points, masks, norms, indexes, alignment, activations, unpack/routing work, latency, memory traffic, and energy | Hardware, kernel, batch/shape, packing format, accounting scope, and measurement protocol |

The current [`semantic_bits.py`](../../src/slm_training/evals/semantic_bits.py)
computes an empirical corpus-unigram description length for production, choice,
and surface streams. It is useful telemetry but is neither `M_epsilon` nor
`R_task(D)`: it does not condition on `Q`, impose a distortion constraint, declare
an encoder class, or supply coverage, confidence, and tail guarantees.

## Guarantee boundary

The compiler and verified-scope solver own guarantees. Learned modules remain
subordinate:

| Layer | Allowed authority | Forbidden inference |
| --- | --- | --- |
| Compiler / pack | Define hard legality and current next-action membership under declared pack capabilities | A `well_formed_not_behavioral` oracle cannot prove behavior |
| Verified Scope | Prove bounded completion support, carry replayable proof evidence, and authorize destructive removal | `UNKNOWN`, timeout, partial coverage, or a stale version cannot remove a candidate |
| Learned ranker / uncertainty model | Rank live members, estimate uncertainty, propose reversible decisions, or request more compute | A score, margin, low entropy, or latent code never creates legality or proves support |
| Realizer | Fill fields classified surface-only against fixed AST/choice IR, then canonicalize and run the applicable pack oracle | It may not autoregress semantic structure or bypass final verification |

[`CompletionForest`](../../src/slm_training/dsl/grammar/fastpath/compiler_draft.py)
is currently a bounded next-decision projection, not an enumerated and minimized
global `Q`. Likewise,
[`RankedForest.signature`](../../src/slm_training/dsl/grammar/fastpath/lattice_search.py)
hashes coverage and token paths for search telemetry; it omits the full frame and
is not a CAP state signature or proof. CAP does not duplicate the support solver:
feasibility, the support oracle, proof-carrying refinement, and the irreversible
guarantee boundary are owned by the Verified Scope contract.

## Exact finite-state capacity

For a declared exact quotient `Q` with `M = |Q|`, a fixed-length noiseless code of
length `d` over an alphabet of size `K` can injectively name all states iff
`K^d >= M`. This is a cardinality fact only. It says nothing about task distortion,
learnability, geometry, error robustness, packed cost, or a neural optimum.

The preregistered noiseless capacity comparator is `(K_z=2, d_z=6)` versus
`(K_z=3, d_z=4)` for the same exact `M=41` quotient, fixed bounded frame, zero
symbol errors, exact enumeration, and ideal dense symbol packing reported
separately from physical bytes. Both have enough names (`64` versus `81`). Neither
is called optimal until matched semantic-quality and physical-cost studies run.

## Corrected one-substitution baseline

Here `M=41` messages, code length `d`, alphabet `K_code`, and one arbitrary symbol
substitution require minimum Hamming distance at least three. These rows are exact
coding statements under that channel, not model or deployment results.

| State set / task quotient and bounds | `K_code`, `d_code`, noise, exact status | Result and construction / proof |
| --- | --- | --- |
| Fixed exact 41-message set; no AST semantics beyond its declared enumeration | `K_code=6`, `d_code=4`; one substitution; exact; packing not claimed | **Impossible.** Singleton gives `M <= K_code^(d_code-3+1) = 6^2 = 36 < 41`. The former robust arm `(6,4)` is deleted. |
| Same fixed exact 41-message set | `K_code=7`, `d_code=4`; one substitution; exact; packing not claimed | **Sufficient.** The `[4,2,3]_7` MDS code generated by rows `(1,1,1,1)` and `(0,1,2,3)` has 49 words; any 41-word subset retains distance at least three. |
| Same fixed exact 41-message set | `K_code=3`, `d_code=6`; one substitution; exact computer-assisted coding bound; packing not claimed | **Impossible.** The published exhaustive classification establishes `A_3(6,3) <= 39`; its provenance is required because Singleton alone is insufficient. The former robust arm `(3,6)` is deleted. |
| Same fixed exact 41-message set | `K_code=3`, `d_code=7`; one substitution; exact; packing not claimed | **Sufficient.** A shortened ternary Hamming `[7,4,3]_3` code has 81 words. One parity-check matrix has columns `(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)` over `F_3`; pairwise non-proportional columns exclude weights 1 and 2, and `(1,0,0)+(0,1,0)-(1,1,0)=0` supplies weight 3. Select any 41 words. This code is not MDS. |

The `d=6` impossibility relies on a computer-assisted small-alphabet-code result,
not an elementary analytic bound. Replay must retain the cited theorem and its
provenance; the repository does not currently carry an independent exhaustive
enumerator.

## Boundary predicates and ternary-specific claims

### Strict margin plane

For a declared fixed geometric ternary residual with `R` digits, bounded aggregate
error `E_max`, and decision margin `gamma` expressed in the same exact integer
units, the admissible sufficient condition is strictly

`E_max / (3^R - 1) < gamma / 2`.

Implementations must compare the division-free integer predicate
`2 * E_max < gamma * (pow(3, R) - 1)`. Equality fails. Do not derive the boundary
through floating logarithms or rounded powers; the earlier exact-power boundary bug
came from doing so. Learned scales or non-integer units require an empirical margin
distribution with numeric-error provenance, not this exact predicate.

### Ternary ECOC

For `b` labels, an injective base-3 label needs `ceil(log_3 b)` trits but has no
error-detection guarantee. Append one mod-3 parity trit, giving
`m = ceil(log_3 b) + 1`; then every one-trit substitution violates parity and the
code has minimum distance at least two. This detects one substitution; it does not
correct one. Correction requires minimum distance at least three and the robust
coding table above.

### Balanced ternary

`R` balanced ternary digits with the exact geometric place values
`1, 3, ..., 3^(R-1)` represent exactly `3^R` consecutive integer states. That
cardinality guarantee does not transfer to learned, repeated, zero, unordered, or
otherwise non-geometric scales. Learned scales must report empirical collision,
coverage, reconstruction, margin, and tail-error results; they do not inherit a
`3^R` state guarantee.

## Evidence and certificate ledger

The word **certificate** is reserved for replayable exact evidence or an explicitly
statistical certificate whose estimator, uncertainty, and provenance are recorded.
A teacher margin, low-rank diagnostic, or point estimate is not an exact
certificate.

| Claim class | Exact replay record | Estimated / statistical record |
| --- | --- | --- |
| State count / quotient | Pack, grammar, parser, codec, and state-signature versions; AST/depth/scope/cutoff bounds; canonical states; equivalence procedure; completeness proof; work count; digest | Trace/data IDs; sample design and `n`; coverage; estimator; confidence interval; collision/tail metric; code/checkpoint digest |
| Code capacity / robustness | Message set digest; `K_code`, `d_code`, encoder/decoder; minimum-distance proof or replay; channel model; construction/theorem provenance | Sampled corruption design; seeds and `n`; decoder; error-rate estimate and interval; worst observed tail, explicitly not exhaustive |
| Task quotient / rate | Exact distribution and distortion table, fixed `Q`, declared encoder class, exhaustive optimizer and replay | Traffic distribution/version; distortion and `epsilon`/`D`; estimator; held-out design; interval; coverage and tail error |
| Margin boundary | Exact integer units; geometric scales; `R`; exhaustive `E_max`; strict integer predicate replay | Checkpoint/data IDs; margin/error estimator; calibration split; interval and quantiles; numeric precision |
| Physical cost | Exact format layout and byte accounting for a fixed shape, including metadata/alignment | Hardware/software/kernel versions; warmup/repeats; batch/shape; latency/memory/energy distribution and interval |

Every exact record must name the property tier it establishes. Every statistical
record must remain statistical. Pack-oracle labels are limited to the capability
declared by the pack; syntax success is not meaningful-parse or ship evidence.

## Fair comparator contract

Later experiments must keep each question separate:

1. **Noiseless requirement-matched capacity:** compare `(K_z=2,d_z=6)` with
   `(K_z=3,d_z=4)` on the same exact 41-state quotient and bounded frame.
2. **Robust requirement-matched capacity:** compare valid arms such as
   `(K_code=7,d_code=4)` and `(K_code=3,d_code=7)` under the same one-substitution channel;
   never resurrect `(6,4)` or `(3,6)`.
3. **Equal nominal rate:** match `d * log2(K)` before metadata, and label this an
   analytic ideal-rate study.
4. **Equal physical bytes:** match the full stored and live-memory ledger. This is
   a distinct study; padding, scales, masks, and kernel layout count.
5. **Representation controls:** include optimized four-level-with-zero, symmetric
   four-level, binary-plus-mask, and continuous/soft latent controls beside ternary.

Across trained arms, match activation precision, group size, calibration/adaptation
data, data order, optimizer steps/tokens, seeds, model shape, compiler frame, and
kernel implementation. If a factor cannot be matched, isolate it as a named
ablation. Report syntax only as diagnostic; meaningful parse and the unchanged
multi-suite ship gates remain primary, with latency, memory, and energy reported by
the performance matrix. Nominal rate and equal physical bytes must never be pooled
into one winner.

## Hypothesis ledger

IDs are permanent and must not be reused. A later result appends a verdict rather
than rewriting the prediction or falsifier.

| ID | Operational prediction | Falsifier | Comparator contract and required metrics |
| --- | --- | --- | --- |
| `CAP-H1` | A compiler-owned exact state plus a local legal-action scorer reaches matched semantic quality with fewer learned parameters and lower precision than implicit grammar-state inference. | Implicit state tracking is equal or better after matched state information and deployed cost. | Explicit-`Q` vs implicit-history models; match data/order/steps/shape/kernel; meaningful parse, regret among legal ASTs, parameters, `K_z/K_w/K_a/K_E`, bytes, latency, seeds. |
| `CAP-H2` | `R_task(D)` or `M_epsilon` predicts the latent-capacity phase boundary better than raw `log2(M)`. | Raw exact-state count remains the strongest held-out predictor after action entropy, semantic regret, and margin are controlled. | Same exact `Q`, traffic and model ladder; raw count vs task-rate estimators; meaningful parse, distortion, regret, coverage, confidence and tail errors. |
| `CAP-H3` | An unrestricted optimized four-level-with-zero representation matches or beats ternary at equal physical storage; any ternary win is due to optimization, regularization, entropy, zero support, or execution rather than reconstruction capacity. | Repeated matched runs show ternary strictly superior outside the preregistered equivalence interval. | Ternary, optimized four-level-with-zero, symmetric four-level, binary+mask, continuous/soft; match bytes/calibration/steps/activations/groups/kernel; meaningful parse, gates, latency, energy, seeds. |
| `CAP-H4` | Grammar-stratified calibration by state/action/scope/margin/timestep outperforms random calibration at the same sample count. | Random calibration is indistinguishable after repeated seeds and coverage controls. | Same checkpoint, `n`, quantizer, groups, activations and kernel; meaningful parse/gates, regret, calibration error, state/action coverage, worst-case/tail error. |
| `CAP-H5` | At equal packed bytes, a slightly wider ternary/two-bit model recovers more semantic quality than a narrower high-precision model. | Quality stays precision-dominated and does not recover with width under equal packed bytes and latency. | Width-by-precision frontier with matched data/order/steps and physical ledger; meaningful parse/gates, parameters, bytes, peak memory, latency, energy, seeds. |
| `CAP-H6` | Hybrid precision—low-bit bulk blocks with INT4/INT8 task-sensitive plan projections, state updates, or local heads—improves the quality-cost frontier over uniform ternary. | Sensitivity is broadly distributed and no hybrid schedule improves the replicated frontier. | Uniform vs sensitivity-guided schedules; same checkpoint/data/groups/kernel/budget; per-layer `K_w/K_a/K_E`, meaningful parse/gates, loss, bytes, latency, energy. |
| `CAP-H7` | Runtime posterior entropy, top-two margin, or conditioned sensitivity routes refinement better than raw branching, reducing average compute over structural-state-only routing. | Structural state alone predicts required planes and learned/runtime routing adds only overhead or changes verified outcomes beyond tolerance. | Same compiler/support oracle/ranker; reversible routing only; meaningful parse/gates, verified terminals, `UNKNOWN` rate, planes/nodes/verifier calls, mean/tail latency, seeds. |
| `CAP-H8` | Compiler-aligned deterministic block activation or state-routed micro-experts gives larger measured speedups than incidental unstructured ternary zero-skipping. | Dispatch, batching, or memory overhead removes the speed/energy benefit at matched quality. | Structured routing vs unstructured zero-skip and dense controls; same bytes/quality target/kernel family; meaningful parse/gates, utilization, traffic, latency distribution, energy. |
| `CAP-H9` | Typed-hole or exact-quotient-state corruption reduces invalid-state work only when the quotient graph retains adequate connectivity, conductance, and posterior support. | Graph bottlenecks or information loss require more steps or a larger denoiser than matched AST-subtree corruption. | Quotient-state vs AST-subtree diffusion; same data/model/steps/compute; graph connectivity/conductance/support, invalid work, denoise steps, meaningful parse/gates, bytes, latency. |

`CAP-H3` tests whether an optimized four-level control explains an apparent ternary
effect. Repeated ternary superiority is its falsifier, not its confirmation. No
ledger row asserts general ternary optimality.

## Physical-cost ledger

Every cost table has two non-interchangeable columns:

- **analytic ideal packing:** `N * log2(K)` or the specified dense-code formula,
  with state/task quotient, all `K,d` roles, bounds, and noise model in the row;
- **measured physical system:** packed payload plus scale/bias/zero-point/norm/mask,
  indexes and alignment, activation and accumulator storage, unpack/dequant work,
  memory traffic, routing/verifier work, latency, and energy on a named kernel and
  device.

An ideal packing win is not a deployed-system win. A deployed optimum is local to
the stated hardware, shapes, traffic, quality constraints, compiler frame, and
packing implementation.

## Source review and corrections (2026-07-17)

The project review covered the Linear documents derived from the user-supplied
artifacts named `Kimi_Agent_Kimi CLI Context Issue.zip` and `Pasted text(6).txt`
(Grok follow-up). This repository paraphrases their conclusions; it does not copy
private source text.

The condensed Kimi document source-reports an independent reproduction over 27
quantitative claim classes and 19 demo outputs: 130 bounded ASTs, 351 trie states,
41 minimized states, decision histograms of 162/190/345, exact Hankel rank 40, and
99%-energy rank 32. The typing-rule variant, executable, signature frame, and
checksum needed to replay those numbers are absent from Linear. They are therefore
**source-reported estimates/evidence, not repository certificates**. The raw
86-state claim was not reproduced; this does not by itself invalidate the reported
downstream 41-state calculation, but blocks reuse as an exact local baseline.

Corrections adopted here:

- separate exact state capacity, task rate, neural precision, and deployment cost;
- replace infeasible robust `(6,4)` and `(3,6)` arms with proven sufficient
  `(7,4)` and `(3,7)` arms;
- attribute ternary length-six impossibility to the computer-assisted
  small-alphabet classification, not Singleton;
- use a strict division-free integer margin predicate and distinguish geometric
  balanced-ternary guarantees from learned-scale estimates;
- require distance two plus parity only for one-error detection, and distance
  three for one-error correction;
- correct `CAP-H3` polarity and keep nominal-rate and equal-physical-byte studies
  separate.

Primary sources and transfer boundaries are catalogued in
[research-lineage.md](research-lineage.md). In particular, quantization results on
large pretrained models, latent-image codecs, and automata over restricted algebras
do not establish a local OpenUI optimum.

## Implementation note (2026-07-17)

SLM-77 commits this architecture, claim, and hypothesis contract only. No train,
eval, benchmark, matrix, profile, telemetry, or reproduction run occurred; no
result JSON, AgentV bundle, model-card update, or checkpoint is warranted. Future
runs must use the existing quality/performance matrices, preserve meaningful parse
as the primary metric, retain unchanged ship gates, and document even negative or
partial results under `docs/design/`.
