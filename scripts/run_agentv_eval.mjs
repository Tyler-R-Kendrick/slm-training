#!/usr/bin/env node

import { evaluate } from "@agentv/core";

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

const result = await evaluate({
  specFile,
  task: async (input) => input,
  assert: [({ output }) => {
    try {
      const payload = JSON.parse(output);
      return {
        name: "openui-domain-gate",
        score: payload.agentv_pass === true ? 1 : 0,
        metadata: {
          claim: payload.claim ?? "unspecified",
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
