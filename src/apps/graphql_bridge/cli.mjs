#!/usr/bin/env node
/**
 * JSON-over-stdio CLI for graphql-js — the F2 GraphQL pack's validity oracle.
 *
 * Input (stdin, one JSON object):
 *   { "op": "parse"|"validate"|"canonicalize"|"schema_symbols"|"version",
 *     "source"?: str, "schema_sdl"?: str }
 *
 * Output (stdout): one JSON result object. Mirrors src/apps/openui_bridge.
 *
 * Ops:
 *   parse           — syntax only: { ok, canonical?, errors? }
 *   validate        — parse + (when schema_sdl given) full graphql-js
 *                     validation against the schema: fields must exist,
 *                     arguments type-check, fragments apply. The schema IS
 *                     the symbol table (F2 scope rules).
 *   canonicalize    — print(parse(source)): graphql-js's printer emits a
 *                     stable normal form (indentation, argument layout).
 *   schema_symbols  — { types: { TypeName: [fieldNames...] } } from the SDL —
 *                     the scope/symbol context for corpus generation.
 *   version         — { graphql: "16.x.y" }
 */
import { readFileSync } from "node:fs";
import {
  buildSchema,
  parse,
  print,
  validate,
  version as graphqlVersion,
  isObjectType,
  isInterfaceType,
} from "graphql";

function fail(errors) {
  return { ok: false, errors: errors.map((e) => String(e.message || e)) };
}

function opParse(source) {
  try {
    const doc = parse(source);
    return { ok: true, canonical: print(doc).trim() };
  } catch (err) {
    return fail([err]);
  }
}

function opValidate(source, schemaSdl) {
  const parsed = opParse(source);
  if (!parsed.ok || !schemaSdl) return parsed;
  let schema;
  try {
    schema = buildSchema(schemaSdl);
  } catch (err) {
    return fail([`schema: ${err.message || err}`]);
  }
  const errors = validate(schema, parse(source));
  if (errors.length > 0) return fail(errors);
  return parsed;
}

function opSchemaSymbols(schemaSdl) {
  let schema;
  try {
    schema = buildSchema(schemaSdl);
  } catch (err) {
    return fail([`schema: ${err.message || err}`]);
  }
  const types = {};
  for (const [name, type] of Object.entries(schema.getTypeMap())) {
    if (name.startsWith("__")) continue;
    if (isObjectType(type) || isInterfaceType(type)) {
      types[name] = Object.keys(type.getFields());
    }
  }
  return { ok: true, types };
}

function handle(request) {
  const { op, source, schema_sdl: schemaSdl } = request;
  switch (op) {
    case "parse":
      return opParse(source ?? "");
    case "validate":
      return opValidate(source ?? "", schemaSdl);
    case "canonicalize":
      return opParse(source ?? "");
    case "schema_symbols":
      return opSchemaSymbols(schemaSdl ?? "");
    case "version":
      return { ok: true, graphql: graphqlVersion };
    default:
      return fail([`unknown op ${JSON.stringify(op)}`]);
  }
}

const input = readFileSync(0, "utf8");
let request;
try {
  request = JSON.parse(input);
} catch (err) {
  process.stdout.write(JSON.stringify(fail([`bad request: ${err.message}`])));
  process.exit(0);
}
process.stdout.write(JSON.stringify(handle(request)));
