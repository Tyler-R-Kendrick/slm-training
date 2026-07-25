# Standard evaluation contract: AgentEvals + AgentV

All repository evaluation runs use the [AgentEvals](https://agentevals.io/)
portable JSONL/YAML contract and the canonical
[AgentV](https://agentv.dev/) implementation. The checked-in versions are
`agentv@4.42.4` for the CLI and `@agentv/core@4.42.4` for the TypeScript SDK.
Both are exact pins.

AgentV standardizes execution artifacts; it does not redefine model quality or
produce an aggregate “AgentV” score. The existing honest multi-suite policy in
`ship_gates.py` remains the source of truth for OpenUI readiness. Every model
suite publishes its named domain graders through `@agentv/core`, including
parse, meaningful-program, binding-aware, contract, fidelity, structural,
reward, AST, language, reference-graph, and target-quality metrics. A missing
or mismatched named SDK result fails the evaluation publication.

## Tracked grader metrics

Every listed metric is executed and recorded through the pinned AgentV SDK.
The runner itself is metadata, never a score. `null` means the grader found the
metric inapplicable and its `metric_defined_n` denominator is zero.

| Metric | What it measures |
| --- | --- |
| `parse_rate` | Official OpenUI parse success |
| `meaningful_program_rate` | Meaningful-program verdict |
| `binding_aware_meaningful_v2_rate_strict` | Strict binding-aware meaningfulness |
| `binding_aware_meaningful_v2_rate_coverage_conditioned` | Strict meaningfulness where coverage is known |
| `binding_aware_meaningful_v2_coverage` | Coverage of the strict meaningfulness contract |
| `syntax_parse_rate` | Syntax parser success |
| `raw_syntax_validity` | Raw generated syntax validity |
| `contract_precision` | Correct emitted contract items |
| `contract_recall` | Required contract-item coverage |
| `placeholder_fidelity` | Exact visible-placeholder fidelity |
| `placeholder_fidelity_normalized` | Canonicalized placeholder fidelity |
| `placeholder_validity` | Placeholder legality |
| `exact_match` | Exact canonical program match |
| `structural_similarity` | Program structure similarity |
| `tree_edit_similarity` | AST edit similarity |
| `component_type_recall` | Required component-type coverage |
| `reward_score` | Canonical composite quality reward |
| `ast_node_f1` | AST node F1 |
| `ast_edge_f1` | AST edge F1 |
| `language_validity` | Language-contract validity |
| `canonical_exact` | Canonical serialization exactness |
| `ref_graph_exact` | Reference-graph exactness |
| `target_correctness` | Target correctness |
| `target_efficiency` | Target efficiency |
| `target_composite` | Target composite quality |

## Flow

1. Python evaluators produce the domain metric inputs using the existing harnesses.
2. `publish_agentv_evaluation` writes standard `*.eval.jsonl` cases.
3. `run_agentv_eval.mjs` invokes one named SDK grader per metric through
   `evaluate()` from `@agentv/core`.
4. The source spec and AgentV result bundle land beside the original evidence
   under `<run-dir>/agentv/`.

The `agentv` npm package is retained for the canonical CLI and dashboard. In
the pinned release its published package is CLI-only, so programmatic execution
correctly imports the SDK from `@agentv/core`.

## Coverage

| Evaluation surface | AgentV publication |
| --- | --- |
| `evaluate` / `evaluate_suites` / `evaluate_model` | Five canonical ship-suite cases; absent suites fail |
| Quality, grammar, phase, reproduction, and mid-train model evals | Inherit the shared model-eval publication path |
| `evaluate_loss_suites` report writer | Complete finite diagnostic report; explicitly not a ship claim |
| `evaluate_tasks` | Fixture prediction evidence; fails the AgentV quality case while ship gates are not run |
| `diagnose_eval` | Diagnostic completion and length-budget result; explicitly not a ship claim |
| Pure gate calculators and web read endpoints | No run is created because they evaluate supplied data without executing a model eval |

New evaluation entrypoints must call the shared publisher instead of inventing
another result format. The recorded domain JSON carries the named SDK grader
outputs and denominators; the AgentV bundle is the standard cross-evaluator
run envelope.

## Commands and artifacts

```bash
npm ci
python -m scripts.evaluate_model --run-id <id> --ship-gates
npm run agentv -- dashboard
```

The Python command automatically creates:

```text
outputs/runs/<id>/agentv/
  openui-model-ship-gates-<timestamp>.eval.jsonl
  openui-model-ship-gates-<timestamp>/
    benchmark.json
    index.jsonl
    timing.json
    ... per-case grading and trace files
```

## Measured wiring result

The implementation check is recorded in
[`agentv-sdk-wiring-results.json`](agentv-sdk-wiring-results.json).

| Date | Recipe | Result | Claim |
| --- | --- | --- | --- |
| 2026-07-14 | CPU, steps 0, no model backend; AgentV SDK fixture plus model/loss/task/train-loop harness tests | 34/34 focused tests passed; SDK fixture wrote valid AgentEvals JSONL and AgentV artifacts; dependency audit has 0 high/critical findings | Tooling wiring only; no checkpoint, model score, or ship gate was produced |
