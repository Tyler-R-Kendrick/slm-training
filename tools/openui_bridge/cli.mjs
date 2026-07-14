#!/usr/bin/env node
/**
 * JSON-over-stdio CLI for @openuidev/lang-core.
 *
 * Input (stdin, one JSON object):
 *   { "op": "parse"|"validate"|"serialize"|"prompt"|"schema"|"stream_check", ... }
 *
 * Output (stdout): JSON result object.
 */
import {
  createParser,
  createStreamingParser,
  jsonToOpenUI,
} from "@openuidev/lang-core";
import { library, CONTENT_PROPS, PLACEHOLDER_RE } from "./library.mjs";
import { contractInfo } from "./contract.mjs";

const schema = library.toJSONSchema();
const parser = createParser(schema, library.root);
const FEATURES = [
  "state",
  "query",
  "mutation",
  "action",
  "tool-call",
  "bindings",
  "expressions",
];

function programMetadata(result) {
  return {
    state_declarations: result.stateDeclarations || {},
    query_statements: result.queryStatements || [],
    mutation_statements: result.mutationStatements || [],
  };
}

function collectPlaceholders(node, out = []) {
  if (!node || typeof node !== "object") return out;
  if (Array.isArray(node)) {
    for (const item of node) collectPlaceholders(item, out);
    return out;
  }
  if (node.type === "element") {
    const props = node.props || {};
    for (const [key, value] of Object.entries(props)) {
      if (typeof value === "string" && PLACEHOLDER_RE.test(value)) {
        if (!out.includes(value)) out.push(value);
      } else {
        collectPlaceholders(value, out);
      }
    }
  } else {
    for (const value of Object.values(node)) collectPlaceholders(value, out);
  }
  return out;
}

function contentPolicyErrors(node, path = "root", errors = []) {
  if (!node || typeof node !== "object") return errors;
  if (Array.isArray(node)) {
    node.forEach((item, i) => contentPolicyErrors(item, `${path}[${i}]`, errors));
    return errors;
  }
  if (node.type === "element") {
    const props = node.props || {};
    for (const [key, value] of Object.entries(props)) {
      if (CONTENT_PROPS.has(key)) {
        // Only string values are user-facing copy slots. Arrays/objects (e.g.
        // TabItem.content children) and null/undefined optionals are exempt.
        if (value == null || typeof value !== "string") {
          if (value != null && typeof value === "object") {
            contentPolicyErrors(value, `${path}.${key}`, errors);
          }
          continue;
        }
        if (!PLACEHOLDER_RE.test(value)) {
          errors.push({
            path: `${path}.${key}`,
            code: "placeholder_required",
            message: `content prop '${key}' must be a placeholder like :scope.name, got ${JSON.stringify(value)}`,
          });
        }
      } else {
        contentPolicyErrors(value, `${path}.${key}`, errors);
      }
    }
  }
  return errors;
}

function handle(req) {
  const op = req.op;
  if (!op) throw new Error("missing op");

  if (op === "schema") {
    const tools = req.tools || [];
    const contract = contractInfo(tools);
    return {
      ok: true,
      schema: {
        ...schema,
        "x-openui-lang": { version: "0.5", features: FEATURES },
      },
      library_id: library.id,
      root: library.root,
      ...contract,
    };
  }

  if (op === "prompt") {
    const options = req.options || {};
    const tools = options.tools || [];
    const toolCalls = options.toolCalls ?? tools.length > 0;
    const bindings = options.bindings ?? toolCalls;
    const additionalRules = options.additionalRules || [
      "Use official openuiLibrary components (Stack, Card, TextContent, Button, Form, Input, …).",
      'Stack direction is "row"|"column"; gap is "none"|"xs"|"s"|"m"|"l"|"xl"|"2xl".',
      "Card takes children arrays; TextContent(text, size?) accepts copy slots.",
      "Put placeholders (for example :hero.title) in every static user-facing string prop.",
      "Do not invent literal user-facing copy; a separate model fills placeholders.",
      ...(toolCalls
        ? []
        : ["Do not use Query(), Mutation(), Action(), tool calls, or $bindings in layout-only skeletons."]),
    ];
    const text = library.prompt({
      preamble:
        options.preamble ||
        "You generate placeholder-augmented OpenUI layout skeletons. Content props must be placeholder strings like :hero.title — never marketing copy.",
      additionalRules,
      examples: options.examples,
      toolExamples: options.toolExamples,
      tools,
      toolCalls,
      bindings,
      editMode: options.editMode,
      inlineMode: options.inlineMode,
    });
    return { ok: true, prompt: text, ...contractInfo(tools) };
  }

  if (op === "stream_check") {
    const source = req.source;
    if (typeof source !== "string") throw new Error("source must be a string");
    const streaming = createStreamingParser(schema, library.root);
    const result = streaming.set(source);
    const policyErrors = result.root ? contentPolicyErrors(result.root) : [];
    const errors = [...(result.meta.errors || []), ...policyErrors];
    const ok =
      Boolean(result.root) &&
      !result.meta.incomplete &&
      errors.length === 0 &&
      (result.meta.unresolved || []).length === 0;
    return {
      ok,
      incomplete: Boolean(result.meta.incomplete),
      has_root: Boolean(result.root),
      errors,
      unresolved: result.meta.unresolved || [],
      placeholders: result.root ? collectPlaceholders(result.root) : [],
      serialized: result.root != null && !result.meta.incomplete ? source.trim() : null,
      root: result.root,
      meta: result.meta,
      ...programMetadata(result),
      ...contractInfo(req.tools || []),
    };
  }

  if (op === "parse" || op === "validate") {
    const source = req.source;
    if (typeof source !== "string") throw new Error("source must be a string");
    const result = parser.parse(source);
    const policyErrors = result.root
      ? contentPolicyErrors(result.root)
      : [{ path: "root", code: "no_root", message: "parser produced no root element" }];
    const placeholders = result.root ? collectPlaceholders(result.root) : [];
    const ok =
      Boolean(result.root) &&
      !result.meta.incomplete &&
      (result.meta.errors || []).length === 0 &&
      (op === "parse" || policyErrors.length === 0);

    return {
      ok,
      root: result.root,
      meta: result.meta,
      placeholders,
      policy_errors: policyErrors,
      serialized:
        result.root != null && !result.meta.incomplete ? source.trim() : null,
      ...programMetadata(result),
      ...contractInfo(req.tools || []),
    };
  }

  if (op === "serialize") {
    const root = req.root;
    if (!root) throw new Error("root ElementNode required");
    const source =
      typeof req.source === "string" && req.source.trim()
        ? req.source.trim()
        : jsonToOpenUI(root, library, {
            stateDeclarations: req.state_declarations || {},
          }).trim();
    return { ok: true, source, ...contractInfo(req.tools || []) };
  }

  throw new Error(`unknown op: ${op}`);
}

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8").trim();
}

async function runOnce() {
  const raw = await readStdin();
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
    process.exit(result.ok === false ? 2 : 0);
  } catch (e) {
    process.stdout.write(
      JSON.stringify({ ok: false, error: e.message || String(e) }) + "\n",
    );
    process.exit(1);
  }
}

async function runRepl() {
  const readline = await import("node:readline");
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false,
  });
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
    if (req.op === "ping") {
      process.stdout.write(JSON.stringify({ ok: true, pong: true }) + "\n");
      continue;
    }
    if (req.op === "quit") {
      process.stdout.write(JSON.stringify({ ok: true, bye: true }) + "\n");
      break;
    }
    try {
      const result = handle(req);
      process.stdout.write(JSON.stringify(result) + "\n");
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
    process.env.OPENUI_BRIDGE_REPL === "1";
  if (repl) {
    await runRepl();
  } else {
    await runOnce();
  }
}

main();
