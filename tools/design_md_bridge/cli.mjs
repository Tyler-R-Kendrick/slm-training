#!/usr/bin/env node
/**
 * JSON-over-stdio CLI for @google/design.md linter.
 *
 * One-shot: stdin JSON → stdout JSON
 * REPL:    node cli.mjs --repl  (NDJSON lines)
 *
 * Input:  { "op": "lint", "source": "<DESIGN.md markdown>" }
 * Output: { ok, score, summary, findings, design_system? }
 */
import readline from "node:readline";
import { lint } from "@google/design.md/linter";

function scoreFromSummary(summary) {
  const errors = Number(summary?.errors || 0);
  const warnings = Number(summary?.warnings || 0);
  const infos = Number(summary?.infos || summary?.info || 0);
  const raw = 1 - Math.min(1, errors * 0.35 + warnings * 0.05 + infos * 0.01);
  return Math.round(raw * 1000) / 1000;
}

function handle(req) {
  const op = req.op || "lint";
  if (op === "ping") return { ok: true, pong: true };
  if (op === "quit") return { ok: true, bye: true };
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

async function runOnce() {
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

async function runRepl() {
  const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of rl) {
    const raw = String(line || "").trim();
    if (!raw) continue;
    let req;
    try {
      req = JSON.parse(raw);
    } catch (e) {
      process.stdout.write(
        JSON.stringify({ ok: false, error: `invalid JSON: ${e.message}` }) + "\n",
      );
      continue;
    }
    try {
      const result = handle(req);
      process.stdout.write(JSON.stringify(result) + "\n");
      if (req.op === "quit") break;
    } catch (e) {
      process.stdout.write(
        JSON.stringify({ ok: false, error: e.message || String(e) }) + "\n",
      );
    }
  }
  process.exit(0);
}

async function main() {
  const repl =
    process.argv.includes("--repl") ||
    process.env.DESIGN_MD_BRIDGE_REPL === "1";
  if (repl) await runRepl();
  else await runOnce();
}

main();
