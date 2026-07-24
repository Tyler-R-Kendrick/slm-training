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
