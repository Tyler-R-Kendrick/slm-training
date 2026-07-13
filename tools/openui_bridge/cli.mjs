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

const schema = library.toJSONSchema();
const parser = createParser(schema);

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
    return { ok: true, schema, library_id: library.id, root: library.root };
  }

  if (op === "prompt") {
    const options = req.options || {};
    const text = library.prompt({
      preamble:
        options.preamble ||
        "You generate placeholder-augmented OpenUI layout skeletons. Content props must be placeholder strings like :hero.title — never marketing copy.",
      additionalRules: options.additionalRules || [
        "Use official openuiLibrary components (Stack, Card, TextContent, Button, Form, Input, …).",
        "Stack direction is \"row\"|\"column\"; gap is \"none\"|\"xs\"|\"s\"|\"m\"|\"l\"|\"xl\"|\"2xl\".",
        "Card takes children arrays (e.g. Card([title, body])); TextContent(text, size?) for copy slots.",
        "Put placeholders (e.g. :hero.title, :cta.label) in all user-facing string props (text, label, title, placeholder, alt, …).",
        "Do not invent literal user-facing copy; a separate model fills placeholders later.",
        "Do not use Query(), Mutation(), Action(), or $bindings in layout skeletons.",
      ],
      examples: options.examples,
      toolCalls: false,
      bindings: false,
    });
    return { ok: true, prompt: text };
  }

  if (op === "stream_check") {
    const source = req.source;
    if (typeof source !== "string") throw new Error("source must be a string");
    const streaming = createStreamingParser(schema);
    const result = streaming.push(source);
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
      serialized:
        result.root != null && !result.meta.incomplete
          ? jsonToOpenUI(result.root, library).trim()
          : null,
      root: result.root,
      meta: result.meta,
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
        result.root != null ? jsonToOpenUI(result.root, library).trim() : null,
    };
  }

  if (op === "serialize") {
    const root = req.root;
    if (!root) throw new Error("root ElementNode required");
    return { ok: true, source: jsonToOpenUI(root, library).trim() };
  }

  throw new Error(`unknown op: ${op}`);
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
    process.exit(result.ok === false ? 2 : 0);
  } catch (e) {
    process.stdout.write(
      JSON.stringify({ ok: false, error: e.message || String(e) }) + "\n",
    );
    process.exit(1);
  }
}

main();
