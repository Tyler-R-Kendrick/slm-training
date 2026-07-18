# CAP1-03 (SLM-83): task-confusability graph / neural state quotient

**Date:** 2026-07-18. **Status:** wiring / tool-only. This note adds a
Torch-free confusability-graph builder and exact graph-coloring quotient for
aligned action records. It makes **no model, train, eval, checkpoint, or ship
claim**. It preregisters the schema, CLI, and regression tests; production trace
integration is deferred.

Owner package: [`src/slm_training/dsl/analysis/arity/`](../../src/slm_training/dsl/analysis/arity/).
CLI: [`scripts/analyze_task_quotient.py`](../../scripts/analyze_task_quotient.py).
Tests: [`tests/test_dsl/test_task_quotient.py`](../../tests/test_dsl/test_task_quotient.py).

## Honesty boundary (read first)

This module computes an abstract graph over **exact compiler state
fingerprints**. It never merges compiler states, edits the grammar, or changes
legality. The output is a coloring witness: which exact states may share one
neural representation under a declared task distortion. The coloring is exact
for graphs up to 128 vertices (branch-and-bound) and heuristic above that.

- The quotient is **frame-relative**: changing the distortion spec, action
  alignment, or trace sample changes the graph and therefore the number of
  colors.
- It is **not** a trained neural compression. It is a pre-compression audit
  tool.
- It is **not** a deployed-system guarantee. Runtime state collisions are a
  separate concern.
- The CLI accepts arbitrary JSONL records; no claim is made about the
  provenance or coverage of those records.

## Schema

### `TaskDistortionSpec`

Versioned distortion specification.

| Field | Meaning |
| --- | --- |
| `spec_id` | Version label for the distortion frame. |
| `action_alignment` | How actions are grouped before comparison (default: `production_family`). |
| `policy_metric` | `js`, `tv`, `cross_entropy_regret`, or `topk_regret`. |
| `policy_tolerance` | Per-pair threshold that refinement enforces. |
| `average_tolerance` | If set, the looser threshold used to build the initial merge graph. |
| `value_weight`, `execution_weight`, `semantic_fingerprint_weight` | Reserved for future composite distances; currently 0. |
| `hard_forbidden_confusions` | Semantic fingerprints that must never share a color. |
| `horizon` | Reserved: `next_action`, `bounded_completion`, or `terminal`. |

### `AlignedActionRecord`

One observed decision at one exact state.

| Field | Meaning |
| --- | --- |
| `state_fingerprint` | Exact compiler-state id. |
| `action_id` | Exact action taken. |
| `aligned_family` | Family after applying `action_alignment`. |
| `probability` | Optional weight; used by profile aggregation when present. |
| `value` | Optional value estimate (reserved). |
| `semantic_fingerprint` | Optional semantic tag for forbidden-confusion checks. |

### `StateProfile`

Aggregated per-state action distribution, family set, visit count, and the most
common non-null semantic fingerprint.

### `ConfusabilityGraph`

Undirected graph over state fingerprints. An edge means "must not share a neural
representation." Edge reasons are stored for diagnostics.

### `QuotientReport`

Aggregate output:

- `state_count`, `edge_count`, `density`
- `coloring`: number of colors, exactness flag, lower/upper bound, algorithm
- `class_size_histogram`: states per color
- `counterexamples`: refinement edges added after the initial graph
- `capacity_feasibility`: for requested `(K, d)` pairs, whether `K^d` covers the
  color count
- `estimated`: always `True` for this wiring (exact graph, but trace coverage is
  not audited)

## Policy distances

All distances operate on aligned action distributions.

- `js`: Jensen–Shannon divergence.
- `tv`: Total variation.
- `cross_entropy_regret`: `CE(p, q) - H(p)`; `inf` if `q` gives zero mass to an
  action `p` visits.
- `topk_regret`: `1 - |top_k(p) ∩ top_k(q)| / k` with `k = 3`.

## Coloring

- Graphs with `<= 128` vertices get exact branch-and-bound coloring.
- Larger graphs fall back to DSATUR heuristic.
- A greedy clique lower bound is computed for both paths.

## Counterexample-guided refinement

If `average_tolerance` is set, the initial graph is built with that looser
threshold. `refine_quotient` then re-checks every within-color pair against the
stricter `policy_tolerance`, adds edges for violations, and recolors. This lets
the quotient first merge states that are similar on average, then split pairs
with excess pairwise regret.

If `average_tolerance` is unset, the initial graph and refinement use the same
threshold, so refinement is a no-op unless the input coloring was produced by a
different criterion.

## CLI usage

```bash
python -m scripts.analyze_task_quotient \
  --records records.jsonl \
  --out report.json \
  --markdown-out report.md \
  --policy-metric tv \
  --policy-tolerance 0.1 \
  --capacities "2:4,3:4,4:4,8:3"
```

`records.jsonl` lines look like:

```json
{"state_fingerprint": "s1", "action_id": "+box", "aligned_family": "component", "probability": 0.9}
{"state_fingerprint": "s1", "action_id": "+stack", "aligned_family": "component", "probability": 0.1}
```

## Verification

Regression tests:

```bash
.venv/bin/python -m pytest tests/test_dsl/test_task_quotient.py -q
```

Policy / lint / diff checks:

```bash
.venv/bin/python -m scripts.repo_policy
.githooks/check-changed
git diff --check
```

Current result: **9 passed** in `tests/test_dsl/test_task_quotient.py`;
`check-changed` and `repo_policy` clean on the changed files.

## Caveats and next steps

- `value_weight`, `execution_weight`, `semantic_fingerprint_weight`, and
  `horizon` are schema placeholders. Only action-distribution distances and
  hard forbidden confusions are implemented.
- `cvar_alpha` / `cvar_tolerance` are reserved for tail-risk refinement.
- Production trace integration is deferred: this wiring accepts hand-written
  JSONL records. A future issue will wire the trace exporter from training
  telemetry.
- The `estimated` flag is `True` because trace coverage is not audited, even
  though the graph coloring itself is exact for small graphs.
