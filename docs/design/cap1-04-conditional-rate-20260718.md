# CAP1-04 (SLM-84): conditional task rate, Fano bounds, and posterior effective support

**Date:** 2026-07-18. **Status:** wiring / tool-only. This note adds a
Torch-free estimator for state-conditioned task information, Fano lower bounds,
posterior effective support, and a finite rate-distortion curve. It makes **no
model, train, eval, checkpoint, or ship claim**. It preregisters the schema,
CLI, and regression tests; production trace integration is deferred.

Owner package: [`src/slm_training/dsl/analysis/arity/`](../../src/slm_training/dsl/analysis/arity/).
CLI: [`scripts/analyze_conditional_rate.py`](../../scripts/analyze_conditional_rate.py).
Tests: [`tests/test_dsl/test_conditional_rate.py`](../../tests/test_dsl/test_conditional_rate.py).
Depends on: CAP1-03 task quotient ([`cap1-03-task-quotient-20260718.md`](cap1-03-task-quotient-20260718.md)).

## Honesty boundary (read first)

This module estimates `R_task(D) = min I(X; Z | Q)` from **empirical** action
distributions over exact compiler states. It is not a universal scalar arity:

- Estimates are **trace-conditional** and **distortion-conditional**.
- Rate-distortion points come from a finite Blahut-Arimoto solver (fixed
  reproduction alphabet) or from a CAP1-03 quotient tolerance sweep.
- No weight-bit allocation, quantizer design, or model training is performed.
- The `estimated` flag is `True` because trace coverage is not audited.

Per `docs/design/calculated-arity-adaptive-precision.md`, exact symbolic
capacity `|Q|`, task-quotient color count `M_epsilon`, and task rate-distortion
`R_task(D)` remain distinct quantities. This module owns the last of the three.

## Schema

### `TaskDistortionSpec`

Reused verbatim from CAP1-03 (`task_quotient.py`). It declares the action
alignment, policy metric, tolerance, average tolerance, and hard forbidden
confusions that frame every estimate.

### `ConditionalRateReport`

Aggregate output:

| Field | Meaning |
| --- | --- |
| `spec` | The `TaskDistortionSpec` frame. |
| `state_count` | Number of exact states observed. |
| `action_alphabet_size` | Number of distinct aligned action families. |
| `conditional_entropy_bits` | `H(A \| Q)` under empirical state weighting. |
| `mutual_information_bits` | `I(color; A)` when a CAP1-03 quotient is supplied. |
| `fano_bounds` | Aggregate and per-state lower bounds on Bayes error. |
| `posterior_support` | `exp(H)` per state plus aggregate statistics. |
| `rate_distortion_curve` | Pareto frontier of `(distortion, rate_bits)` points. |
| `estimated` | Always `True` for this wiring. |

### `RateDistortionPoint`

One `(distortion, rate_bits, beta, exact)` point. `beta` is the Lagrange
multiplier used by Blahut-Arimoto or the inverse tolerance used by the quotient
sweep.

### `FanoBound`

Finite-class lower bound on error probability:

```text
P_e >= (H(A|Q) - log2(2)) / log2(|A| - 1)
```

with safe handling of `|A| <= 1` and zero-entropy cases.

### `PosteriorEffectiveSupport`

Dynamic support diagnostics:

- `N_eff_Shannon(q) = exp(H(A | Q=q) * ln(2))`
- mean, median, min, max over states

## Algorithms

### Entropy and conditional entropy

Standard Shannon entropy in bits on empirical distributions. Conditional
entropy weights each state by its empirical visit mass.

### Mutual information

Computed from the joint distribution over quotient colors and actions:

```text
I(color; A) = H(color) + H(A) - H(color, A)
```

### Blahut-Arimoto

Finite-alphabet solver for a fixed reproduction alphabet `Z`:

1. Initialize `p(z|x)` uniformly.
2. Iterate:
   - `q(z) = sum_x p(x) p(z|x)`
   - `p(z|x) ∝ q(z) exp(-beta * d(x,z))`
3. Compute `D = E[d]` and `R = I(X;Z)`.
4. Return the Pareto frontier over a beta sweep.

Convergence is tracked; points are labeled `exact=False` if the iteration limit
is hit.

### Quotient sweep

An alternative R(D) estimate that recomputes CAP1-03 colorings at a log-spaced
tolerance grid. Each tolerance yields one `(avg_intra_color_distortion,
H(color_assignment))` point. This curve is always approximate because the
reproduction alphabet is restricted to quotient colors.

## CLI usage

```bash
python -m scripts.analyze_conditional_rate \
  --records records.jsonl \
  --out report.json \
  --markdown-out report.md \
  --task-quotient task_quotient_report.json \
  --policy-metric cross_entropy_regret \
  --policy-tolerance 0.1
```

`records.jsonl` follows the same `AlignedActionRecord` format as the CAP1-03
CLI. `--task-quotient` is optional; when supplied, the color assignment is used
to estimate `I(color; A)`.

## Verification

Regression tests:

```bash
.venv/bin/python -m pytest tests/test_dsl/test_conditional_rate.py -q
```

Policy / lint / diff checks:

```bash
.venv/bin/python -m scripts.repo_policy
.githooks/check-changed
git diff --check
```

Current result: **13 passed** in `tests/test_dsl/test_conditional_rate.py`;
`check-changed` and `repo_policy` clean on the changed files.

## Caveats and next steps

- `value_weight`, `execution_weight`, `semantic_fingerprint_weight`, and
  `horizon` in the distortion spec are placeholders; the wiring only uses action
  distributions.
- CVaR tail bounds are reserved for a future issue.
- Production trace integration is deferred; the CLI currently accepts
  hand-written JSONL records. A future issue will wire `TraceStore` loading and
  derive state fingerprints from decode commits.
- The rate-distortion curve is finite and reproduction-alphabet-dependent; it
  is a lower/upper estimate, not a continuous optimum.
