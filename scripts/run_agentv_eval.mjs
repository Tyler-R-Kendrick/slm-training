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
