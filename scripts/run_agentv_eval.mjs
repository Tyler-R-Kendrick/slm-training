#!/usr/bin/env node

import path from "node:path";
import { pathToFileURL } from "node:url";

const sdkEntry = path.join(process.cwd(), "node_modules/@agentv/core/dist/index.js");
const { evaluate } = await import(pathToFileURL(sdkEntry).href);

// AgentV executes and publishes these named domain metrics; it is not itself a metric.
const TRACKED_METRICS = [
  "parse_rate", "meaningful_program_rate", "binding_aware_meaningful_v2_rate_strict",
  "binding_aware_meaningful_v2_rate_coverage_conditioned", "binding_aware_meaningful_v2_coverage",
  "syntax_parse_rate", "raw_syntax_validity", "contract_precision", "contract_recall",
  "placeholder_fidelity", "placeholder_fidelity_normalized", "placeholder_validity",
  "exact_match", "structural_similarity", "tree_edit_similarity", "component_type_recall",
  "reward_score", "ast_node_f1", "ast_edge_f1", "language_validity", "canonical_exact",
  "ref_graph_exact", "target_correctness", "target_efficiency", "target_composite",
];

function option(name) {
  const index = process.argv.indexOf(name);
  if (index === -1 || !process.argv[index + 1]) {
    throw new Error(`missing required ${name}`);
  }
  return process.argv[index + 1];
}

const specFile = option("--spec");
const outputDir = option("--output-dir");
const experiment = option("--experiment");

function metricAssertion(metric) {
  return ({ output }) => {
    try {
      const payload = JSON.parse(output);
      const value = payload.result?.metrics?.[metric];
      const defined = typeof value === "number" && Number.isFinite(value);
      return {
        name: `openui-metric:${metric}`,
        // The SDK requires a numeric score. This is transport only; metadata is canonical.
        score: defined ? value : 0,
        metadata: {
          suite: payload.suite ?? "unspecified",
          metric,
          value: defined ? value : null,
          defined,
          defined_n: payload.result?.metric_defined_n?.[metric] ?? 0,
        },
      };
    } catch (error) {
      return {
        name: `openui-metric:${metric}`,
        score: 0,
        metadata: { metric, value: null, defined: false, defined_n: 0, error: String(error) },
      };
    }
  };
}

function metricResults(results) {
  const bySuite = {};
  for (const result of results ?? []) {
    for (const score of result.scores ?? []) {
      const metadata = score.details ?? score.metadata ?? {};
      const metric = metadata.metric;
      if (!TRACKED_METRICS.includes(metric)) continue;
      const suite = metadata.suite ?? "unspecified";
      bySuite[suite] ??= {};
      bySuite[suite][metric] = {
        value: metadata.defined ? metadata.value : null,
        defined_n: metadata.defined_n ?? 0,
      };
    }
  }
  return bySuite;
}

const result = await evaluate({
  specFile,
  task: async (input) => input,
  assert: TRACKED_METRICS.map(metricAssertion),
  threshold: 0,
  workers: 1,
  cache: false,
  outputDir,
  experiment,
});

console.log(JSON.stringify({
  summary: result.summary,
  artifacts: result.artifacts,
  metric_results: metricResults(result.results),
}));

if (result.summary.executionErrors > 0) {
  process.exitCode = 2;
}
