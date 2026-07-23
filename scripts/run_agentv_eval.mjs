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
  threshold: 1,
  workers: 1,
  cache: false,
  outputDir,
  experiment,
});

console.log(JSON.stringify({
  summary: result.summary,
  artifacts: result.artifacts,
  results: result.results.map((item) => ({
    testId: item.testId,
    score: item.score,
    executionStatus: item.executionStatus,
    scores: (item.scores ?? []).map((score) => ({
      name: score.name,
      type: score.type,
      score: score.score,
      assertions: score.assertions,
      details: score.details,
    })),
  })),
}));

if (result.summary.executionErrors > 0) {
  process.exitCode = 2;
}
