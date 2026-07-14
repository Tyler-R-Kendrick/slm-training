import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

const manifest = JSON.parse(
  readFileSync(new URL("../../grammars/openui_contract.json", import.meta.url), "utf8"),
);

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonical(value[key])]),
    );
  }
  return value;
}

function sha256(value) {
  return createHash("sha256")
    .update(JSON.stringify(canonical(value)))
    .digest("hex");
}

export function contractInfo(toolSchema = []) {
  const inputs = { ...manifest, tool_schema_sha256: sha256(toolSchema) };
  return {
    contract_id: `openui-v${manifest.lang_spec_version}-${sha256(inputs)}`,
    contract_inputs: inputs,
  };
}
