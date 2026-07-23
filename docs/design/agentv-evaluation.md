# Standard evaluation contract: AgentEvals criteria, AgentV runner

All repository evaluation runs use the [AgentEvals](https://agentevals.io/)
portable JSONL/YAML contract and the canonical
[AgentV](https://agentv.dev/) implementation. The checked-in versions are
`agentv@4.42.4` for the CLI and `@agentv/core@4.42.4` for the TypeScript SDK.
Both are exact pins.

AgentEvals assertions are the gate authority. The existing honest multi-suite
thresholds in `ship_gates.py` remain the policy source, and `agentv.py` lowers
the raw metric evidence into required `actual/operator/expected` assertions.
The assertions—not a Python-generated pass boolean—produce the durable
verdict. AgentV runs the spec and publishes artifacts; it is not itself a gate
or model-quality metric. Missing suites remain required failing criteria, so a
successful smoke case can never turn a partial run into a production claim.

`evaluate_ship_gates()` remains available as an explicitly labeled Python
preview for APIs and diagnostics. Only `gates.json` derived from successful
AgentEvals assertion results has `authority: "AgentEvals assertions"`.

## Flow

1. Python evaluators compute domain metrics using the existing harnesses.
2. `publish_agentevals_evaluation` writes standard `*.eval.jsonl` cases whose
   required code-graders compare raw evidence with policy thresholds.
3. `run_agentv_eval.mjs` invokes `evaluate()` from `@agentv/core`; it injects no
   assertion and makes no quality decision of its own.
4. The source spec and AgentV runner bundle land beside the original evidence
   under `<run-dir>/evals/`.
5. `write_ship_gates` projects those assertion results into the compatibility
   `gates.json` shape and records the AgentEvals authority.

The `agentv` npm package is retained for the canonical CLI and dashboard. In
the pinned release its published package is CLI-only, so programmatic execution
correctly imports the SDK from `@agentv/core`.

## Coverage

| Evaluation surface | AgentEvals criteria / AgentV execution |
| --- | --- |
| `evaluate` / `evaluate_suites` / `evaluate_model` | Five canonical ship-suite cases; absent suites fail |
| Quality, grammar, phase, reproduction, and mid-train model evals | Inherit the shared model-eval publication path |
| `evaluate_loss_suites` report writer | Complete finite diagnostic report; explicitly not a ship claim |
| `evaluate_tasks` | Fixture prediction evidence; fails its AgentEvals quality criterion while ship gates are not run |
| `diagnose_eval` | Diagnostic completion and length-budget result; explicitly not a ship claim |
| Pure gate calculators and web read endpoints | No run is created because they evaluate supplied data without executing a model eval |

New evaluation entrypoints must call the shared publisher instead of inventing
another result format. The original domain JSON remains for existing research
tables; the AgentEvals spec is the gate contract and AgentV provides the
standard cross-evaluator run envelope.

## Commands and artifacts

```bash
npm ci
python -m scripts.evaluate_model --run-id <id> --ship-gates
npm run agentv -- dashboard
```

The Python command automatically creates:

```text
outputs/runs/<id>/evals/
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
| 2026-07-23 | CPU, steps 0, no model backend; focused assertion-authority tests with the pinned SDK | AgentEvals JSONL carried required code-graders over raw criteria; the runner produced 1/1 passing fixture criteria with 0 execution errors; 135 focused gate/consumer checks and all 6 interpreted page validations passed. Dashboard production build remained environment-blocked because the locked `@openfeature/web-sdk` dependency was unavailable and automatic review rejected installation before execution. | Tooling and gate-authority wiring only; no checkpoint or model-quality claim |
