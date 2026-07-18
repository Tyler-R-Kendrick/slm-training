# Grammar experiment matrix (X-series)

Grammar-native diffusion ablations for the TwoTower OpenUI stack. This is the
index for the grammar/topology experiment set; the canonical row tables and
measured results are maintained alongside the quality matrix.

## Where the rows live

* **Runner:** `scripts/run_grammar_matrix.py` (X-series: X0, X1, X6, X9-X22;
  X2-X5/X7/X8 are frozen legacy IDs). `--describe` prints the full resolved row
  definitions without loading a model or data.
* **Row tables:** the `## X matrix (grammar-native diffusion ablations)` section
  of [quality-experiment-matrix.md](quality-experiment-matrix.md).
* **Design:** [grammar-topology-diffusion.md](grammar-topology-diffusion.md).
* **Measured results (JSON, updated in place):**
  [grammar-matrix-results.json](grammar-matrix-results.json) and
  [grammar-scope-matrix-results.json](grammar-scope-matrix-results.json).

## Metrics

The grammar matrix surfaces the parse/meaningful groups (`parse_rate`,
`syntax_parse_rate`, `meaningful_program_v1_rate`, the binding-aware v2 rates)
and the AST/topology groups (`ast_node_f1`, `ast_edge_f1`,
`tree_edit_similarity`, `topology_quality_score`, `topology_structure_score`,
`topology_trace_score`, `topology_efficiency_score`, `topology_composite`,
`topology_telemetry`). The primary metric is meaningful parse, not syntax parse.

Interpretation rules follow `.agents/skills/running-experiment-matrices/SKILL.md`:
compare only across runs that share honesty mode and suite sizes; register a
stable ID + run_id in the runner and the markdown table before recording a
result; keep invalidated historical rows labeled rather than deleted.

## Verified-scope-solver topology rows (VSS4-02)

The topology-diffusion layer is also exercised by the matched verified-solver
matrix, which adds the capsule-aware topology row (R3) and its energy-ranked
sibling (R4). Those rows carry the topology-diffusion metric group (hard-domain
coverage, active holes, model proposals accepted/rejected by exact live-set
projection, reversible remasks/backtracks, certified removals retained across
remask, atomic-batch rollbacks, denoiser NFE/canvas tokens) under the same hard
gates. See [verified-scope-solver-benchmark.md](verified-scope-solver-benchmark.md)
and run:

```bash
python scripts/run_quality_matrix.py --matrix-set verified-solver --describe
```

Grammar/topology solver quality claims stay gated by the existing OpenUI ship
gates (`DEFAULT_SHIP_GATES`); the verified-solver rows never weaken them.
