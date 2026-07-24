#!/usr/bin/env node

import { pathToFileURL } from "node:url";

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
const sdkRoot = option("--sdk-root");
const sdkUrl = pathToFileURL(
  `${sdkRoot}/node_modules/@agentv/core/dist/index.js`,
);
const { evaluate } = await import(sdkUrl.href);

const result = await evaluate({
  specFile,
  task: async (input) => input,
  assert: [({ output }) => {
    try {
      const payload = JSON.parse(output);
      const checks = payload.checks ?? {};
      const entries = Object.entries(checks);
      const failedChecks = entries
        .filter(([, passed]) => passed !== true)
        .map(([name]) => name);
      const contractPassed = entries.length === 0 || failedChecks.length === 0;
      return {
        name: "openui-domain-gate",
        score: payload.agentv_pass === true && contractPassed ? 1 : 0,
        metadata: {
          claim: payload.claim ?? "unspecified",
          checks,
          failedChecks,
          failures: payload.failures ?? [],
        },
      };
    } catch (error) {
      return {
        name: "openui-domain-gate",
        score: 0,
        metadata: { error: String(error) },
      };
    }
  }],
  threshold: 1,
  workers: 1,
  cache: false,
  outputDir,
  experiment,
});

console.log(JSON.stringify({
  summary: result.summary,
  artifacts: result.artifacts,
}));

if (result.summary.executionErrors > 0) {
  process.exitCode = 2;
}
