#!/usr/bin/env node
/**
 * JSON-over-stdio CLI for @google/design.md linter.
 *
 * Input:  { "op": "lint", "source": "<DESIGN.md markdown>" }
 * Output: { ok, score, summary, findings, design_system? }
 */
import { lint } from "@google/design.md/linter";

function scoreFromSummary(summary) {
  const errors = Number(summary?.errors || 0);
  const warnings = Number(summary?.warnings || 0);
  const infos = Number(summary?.infos || summary?.info || 0);
  // Errors dominate; warnings lightly penalize; clamp to [0, 1].
  const raw = 1 - Math.min(1, errors * 0.35 + warnings * 0.05 + infos * 0.01);
  return Math.round(raw * 1000) / 1000;
}

function handle(req) {
  const op = req.op || "lint";
  if (op !== "lint") throw new Error(`unknown op: ${op}`);
  const source = req.source;
  if (typeof source !== "string") throw new Error("source must be a string");
  const report = lint(source);
  const summary = report.summary || { errors: 0, warnings: 0, infos: 0 };
  const findings = report.findings || [];
  const score = scoreFromSummary(summary);
  const ok = Number(summary.errors || 0) === 0;
  return {
    ok,
    score,
    summary: {
      errors: Number(summary.errors || 0),
      warnings: Number(summary.warnings || 0),
      infos: Number(summary.infos || summary.info || 0),
    },
    findings,
    design_system: report.designSystem || null,
  };
}

async function main() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  const raw = Buffer.concat(chunks).toString("utf8").trim();
  let req;
  try {
    req = JSON.parse(raw || "{}");
  } catch (e) {
    process.stdout.write(
      JSON.stringify({ ok: false, error: `invalid JSON: ${e.message}` }) + "\n",
    );
    process.exit(1);
  }
  try {
    const result = handle(req);
    process.stdout.write(JSON.stringify(result) + "\n");
    process.exit(result.ok === false && Number(result.summary?.errors || 0) > 0 ? 2 : 0);
  } catch (e) {
    process.stdout.write(
      JSON.stringify({ ok: false, error: e.message || String(e) }) + "\n",
    );
    process.exit(1);
  }
}

main();
