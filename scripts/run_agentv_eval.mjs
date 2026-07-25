#!/usr/bin/env node

import { pathToFileURL } from "node:url";
import { randomUUID } from "node:crypto";

function option(name) {
  const index = process.argv.indexOf(name);
  if (index === -1 || !process.argv[index + 1]) {
    throw new Error(`missing required ${name}`);
  }
  return process.argv[index + 1];
}

function optional(name) {
  const index = process.argv.indexOf(name);
  return index === -1 ? undefined : process.argv[index + 1];
}

function langsmithEnabled() {
  return /^(1|true|yes|on)$/i.test(process.env.LANGSMITH_TRACING ?? "")
    && Boolean(process.env.LANGSMITH_API_KEY);
}

function traceUuid(traceId) {
  return traceId && /^[0-9a-f]{32}$/i.test(traceId)
    ? `${traceId.slice(0, 8)}-${traceId.slice(8, 12)}-${traceId.slice(12, 16)}-${traceId.slice(16, 20)}-${traceId.slice(20)}`
    : undefined;
}

async function publishLangSmithSummary({ traceId, runId, experiment, result }) {
  const parentRunId = traceUuid(traceId);
  if (!langsmithEnabled() || !parentRunId) return;
  try {
    const now = new Date().toISOString();
    const endpoint = `${(process.env.LANGSMITH_ENDPOINT || "https://api.smith.langchain.com").replace(/\/$/, "")}/runs`;
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": process.env.LANGSMITH_API_KEY,
        ...(process.env.LANGSMITH_WORKSPACE_ID
          ? { "x-tenant-id": process.env.LANGSMITH_WORKSPACE_ID }
          : {}),
      },
      body: JSON.stringify({
        id: randomUUID(),
        trace_id: parentRunId,
        parent_run_id: parentRunId,
        project_name: process.env.LANGSMITH_PROJECT || "slm-training",
        name: "agentv.publication",
        run_type: "tool",
        inputs: { run_id: runId, experiment },
        outputs: { summary: result.summary },
        start_time: now,
        end_time: now,
        extra: { metadata: { w3c_trace_id: traceId, sdk: "@agentv/core" } },
      }),
      signal: AbortSignal.timeout(500),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    console.warn(`LangSmith AgentV export failed: ${String(error)}`);
  }
}

const specFile = option("--spec");
const outputDir = option("--output-dir");
const experiment = option("--experiment");
const sdkRoot = option("--sdk-root");
const traceId = optional("--trace-id");
const runId = optional("--run-id");
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

await publishLangSmithSummary({ traceId, runId, experiment, result });

console.log(JSON.stringify({
  summary: result.summary,
  artifacts: result.artifacts,
}));

if (result.summary.executionErrors > 0) {
  process.exitCode = 2;
}
