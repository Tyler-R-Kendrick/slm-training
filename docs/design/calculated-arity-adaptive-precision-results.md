# CAP5-03: Calculated arity and adaptive precision — reproducible evidence package

**Status:** evidence package (SLM-102). This document closes the CAP0–CAP4 campaign
by indexing the exact calculations, controlled fixtures, and honest claims produced
in the lineage. It adds no new model, checkpoint, or ship claim; every substantive
claim links to a committed artifact and states its scope.

**Scope frame:** all claims below are bounded to the declared `bounded-expr` grammar,
the OpenUI production codec, the committed state-signature versions, and the exact or
estimated evidence kind recorded in the claim ledger. Transfer outside these frames is
unverified.

**Quick reproduction:**

```bash
python -m scripts.reproduce_calculated_arity_fixtures \
  --out outputs/runs/cap_repro --verify-expected
```

The command runs on CPU, requires no model downloads, and regenerates the stamped
summary at `outputs/runs/cap_repro/cap_repro_summary.json`.

---

## 1. Problem definition and bounded-domain assumptions

The central non-equivalence is stated in
[calculated-arity-adaptive-precision.md](calculated-arity-adaptive-precision.md):

> exact symbolic-state capacity ≠ task-relevant information rate ≠ neural
> weight/activation precision ≠ deployed system optimum

The campaign studies bounded compiler-constrained domains where:

* the grammar and state signature can be frozen and versioned;
* exact enumeration is possible for small profiles;
* task distortion, precision, and physical cost are kept distinct;
* every learned module is subordinate to the compiler/verifier guarantee boundary.

The target contribution is a *method* for deriving task-relevant representation
requirements and selecting a measured neural/system realization, not a claim that
a single arity or precision is universally optimal.

## 2. Compiler-owned versus learned-state decomposition

The compiler/verifier owns hard legality and completion support. Learned modules
rank live candidates, estimate uncertainty, and propose reversible decisions. This
boundary is enforced in the decode stack
([grammar-fastpath.md](grammar-fastpath.md),
[verified-scope-solver.md](verified-scope-solver.md)) and reflected in every CAP
experiment design.

Evidence:

* CAP2-04 state-ownership ablation
  ([iter-cap2-04-state-ablation-20260718.md](iter-cap2-04-state-ablation-20260718.md))
  compares implicit, explicit, discrete, and compiler-owned state tracks.
* The `CompletionForest` and `RankedForest.signature` are bounded projections, not
  enumerated global `Q`; CAP does not duplicate the verified-scope solver.

## 3. Exact Lark/AST state construction and signature

For the `bounded-expr` fixture the exact analyzer enumerates:

| Quantity | Value | Evidence |
| --- | --- | --- |
| Canonical ASTs | 400 | `tests/test_dsl/test_arity_analysis.py` |
| Raw prefix states | 11 | `docs/design/cap0-02-arity-analyzer-20260718.json` |
| Trie states | 844 | `docs/design/cap0-02-arity-analyzer-20260718.json` |
| Minimized DFA states | 28 | `docs/design/cap0-02-arity-analyzer-20260718.json` |
| Action alphabet | 8 | `docs/design/cap0-02-arity-analyzer-20260718.json` |
| Scope signatures | 3 | `docs/design/cap0-02-arity-analyzer-20260718.json` |
| Capacity `K^d`, d=4 | `min K = 3` | `scripts/reproduce_calculated_arity_fixtures.py` |

The analyzer is byte-stable and torch-free. See
[cap0-02-arity-analyzer-20260718.md](cap0-02-arity-analyzer-20260718.md) for the
full certificate.

## 4. Exact robust coding corrections/constructions

Two small verified constructions anchor the precision story:

| Code | Parameters | Verified property | Evidence |
| --- | --- | --- | --- |
| MDS | `[4,2,3]_7` | 49 codewords, distance 3 | `tests/test_dsl/test_arity_coding.py` |
| Shortened Hamming | `[7,4,3]_3` | 81 codewords, distance 3 | `tests/test_dsl/test_arity_coding.py` |

Bounds checked: Singleton upper bound, Hamming sphere-packing bound, Gilbert-Varshamov
existence, ternary ECOC width, and residual trit-plane counts. See
[cap0-03-coding-precision-20260718.md](cap0-03-coding-precision-20260718.md).

## 5. Task-confusability and conditional-rate methodology

* Task-confusability graph: states that can share a neural representation under a
declared distortion without changing compiler decisions.
* Conditional rate: lower bound on latent bits needed given exact compiler state `Q`,
  a distortion constraint, and an encoder class.
* Fano bound and posterior effective support are computed from empirical state
  profiles.

Evidence:

* [cap1-03-task-quotient-20260718.md](cap1-03-task-quotient-20260718.md)
* [cap1-04-conditional-rate-20260718.md](cap1-04-conditional-rate-20260718.md)
* `tests/test_dsl/test_task_quotient.py`
* `tests/test_dsl/test_conditional_rate.py`

## 6. Latent-code experiments and compiler-ownership ablation

CAP2 ran a fixture matrix across:

* strict K-ary bottleneck;
* mixed-radix FSQ, binary LFQ, learned VQ, and continuous latent controls;
* state-local action heads with ternary ECOC;
* implicit / explicit / discrete / compiler-owned state ownership.

Key result: compiler-owned state and hard legality remain the load-bearing
boundary; latent-codec improvements are measured only inside that boundary. See

* [iter-cap2-01-kary-bottleneck-20260717.md](iter-cap2-01-kary-bottleneck-20260717.md)
* [iter-cap2-02-latent-codec-matrix-20260718.md](iter-cap2-02-latent-codec-matrix-20260718.md)
* [iter-cap2-03-state-local-action-heads-20260718.md](iter-cap2-03-state-local-action-heads-20260718.md)
* [iter-cap2-04-state-ablation-20260718.md](iter-cap2-04-state-ablation-20260718.md)

## 7. Quantization, calibration, and equal-storage results

CAP3 tested:

* reference low-bit quantizers and physical-cost ledger;
* grammar-stratified calibration;
* equal-storage ternary falsification matrix;
* grammar-conditioned quantization sensitivity and mixed-precision allocation;
* width × precision ladder at equal deployed bytes.

Evidence:

* [iter-cap3-01-quantization-20260717.md](iter-cap3-01-quantization-20260717.md)
* [iter-cap3-02-calibration-20260718.md](iter-cap3-02-calibration-20260718.md)
* [iter-cap3-03-ternary-falsification-20260718.md](iter-cap3-03-ternary-falsification-20260718.md)
* [iter-cap3-04-sensitivity-20260718.md](iter-cap3-04-sensitivity-20260718.md)
* [iter-cap3-05-equal-byte-ladder-20260718.md](iter-cap3-05-equal-byte-ladder-20260718.md)

## 8. Mixed precision and width×precision frontier

Mixed-precision allocation is solved as a sensitivity-weighted knapsack over
quantization groups. The equal-byte ladder compares widths and precisions at matched
static bytes. No universal winner emerged; the frontier depends on the grammar
profile and target latency. See
[iter-cap3-04-sensitivity-20260718.md](iter-cap3-04-sensitivity-20260718.md) and
[iter-cap3-05-equal-byte-ladder-20260718.md](iter-cap3-05-equal-byte-ladder-20260718.md).

## 9. Residual planes, routing, structured inference, sparsity, and diffusion

CAP4 arms:

* residual ternary planes;
* compiler-floor + runtime-signal adaptive-plane routing;
* quantized local legal-action energies vs exact lattice inference;
* compiler-routed block sparsity and state-family micro-experts;
* quotient-state diffusion graph connectivity and mixing.

Evidence:

* [iter-cap4-01-residual-quantization-20260718.md](iter-cap4-01-residual-quantization-20260718.md)
* [iter-cap4-02-adaptive-plane-routing-20260718.md](iter-cap4-02-adaptive-plane-routing-20260718.md)
* [iter-cap4-03-quantized-energy-inference-20260719.md](iter-cap4-03-quantized-energy-inference-20260719.md)
* [iter-cap4-04-block-sparsity-20260718.md](iter-cap4-04-block-sparsity-20260718.md)
* [iter-cap4-05-quotient-diffusion-graph-20260718.md](iter-cap4-05-quotient-diffusion-graph-20260718.md)

## 10. Cross-grammar/cutoff phase-law analysis

CAP5-01 built cross-grammar and cutoff scaling families to test whether normalized
structural/task quantities predict model behavior better than raw structural counts.
The phase law is framed as a conditional failure function of capacity slack,
task-rate slack, margin slack, predictive-rank slack, semantic action entropy, and
deployed system cost. See the CAP5-01 artifacts.

## 11. Hardware/system findings

Physical cost is recorded as actual packed bytes, measured latency, memory traffic,
and energy where telemetry exists. Reference/fake-quantization paths and optimized
kernels are reported separately. No cross-hardware universal claim is made. See
[perf-experiment-matrix.md](perf-experiment-matrix.md) and the CAP4 physical-cost
ledgers.

## 12. Negative/null results and abandoned hypotheses

See [cap5-negative-results.md](cap5-negative-results.md) and the existing
[iter-efs0-05-rejected-lever-readjudication-20260719.md](iter-efs0-05-rejected-lever-readjudication-20260719.md)
registry. Every rejected lever retains its reason code and linked evidence.

## 13. Limitations and transfer conditions

* Exact enumeration is limited to small bounded profiles; larger profiles are
  estimated with declared uncertainty.
* Quality/latency/energy claims are scoped to the measured hardware, kernel, and
  batch size.
* OpenUI is the only production pack; other DSL packs are not claimed.
* No learned module overrides the compiler/verifier guarantee boundary.

## 14. Exact reproduction commands and artifact index

| Artifact | Command / file |
| --- | --- |
| Exact arity certificate | `python -m scripts.analyze_grammar_arity --fixture bounded-expr --max-ast-nodes 6 --max-live-bindings 2 --dimensions 4 --out outputs/runs/arity/report.json` |
| Coding/precision checks | `python -m scripts.reproduce_calculated_arity_fixtures --verify-expected` |
| CAP2 bottleneck matrix | `python -m scripts.run_cap2_bottleneck` |
| CAP2 state ablation | `python -m scripts.run_cap2_04_state_ablation` |
| CAP3 ternary falsification | `python -m scripts.run_cap3_03_ternary_falsification` |
| CAP4 fixtures | `scripts/run_residual_trit_fixture.py`, `run_adaptive_plane_fixture.py`, `run_quantized_energy_inference_fixture.py`, `run_block_sparsity_fixture.py`, `run_quotient_diffusion_fixture.py` |
| Full artifact index | [cap5-artifact-index.json](cap5-artifact-index.json) + [cap5-artifact-index.md](cap5-artifact-index.md) |
| Claim ledger | [cap5-claim-ledger.json](cap5-claim-ledger.json) + [cap5-claim-ledger.md](cap5-claim-ledger.md) |

---

## Related documents

* [calculated-arity-adaptive-precision.md](calculated-arity-adaptive-precision.md) — CAP0 contract
* [research-lineage.md](research-lineage.md) — citations and fidelity labels
* [MODEL_CARD.md](../MODEL_CARD.md) — checkpoint roster
* [README.md](../README.md) — model-card summary
* [checkpoint-bucket.md](checkpoint-bucket.md) — durable checkpoint rules
