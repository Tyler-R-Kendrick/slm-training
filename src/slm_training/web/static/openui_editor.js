const COMPONENTS = [
  {
    label: "Stack",
    detail: "Layout children in a row or column",
    insertText: 'Stack([], "column")',
    cursorBack: 12,
  },
  {
    label: "Card",
    detail: "Group related child components",
    insertText: "Card([])",
    cursorBack: 2,
  },
  {
    label: "TextContent",
    detail: "Render placeholder-backed text",
    insertText: 'TextContent(":content.text")',
    cursorBack: 2,
  },
  {
    label: "Button",
    detail: "Render a placeholder-backed action",
    insertText: 'Button(":cta.label")',
    cursorBack: 2,
  },
  ...[
    "CardHeader", "Buttons", "Input", "Form", "FormControl", "Label", "TextArea",
    "Select", "SelectItem", "CheckBoxGroup", "CheckBoxItem", "RadioGroup", "RadioItem",
    "SwitchGroup", "SwitchItem", "Slider", "DatePicker", "Image", "ImageBlock",
    "ImageGallery", "Modal", "Tabs", "TabItem", "Callout", "TextCallout", "Separator",
    "Table", "Col",
  ].map((label) => ({
    label,
    detail: "OpenUI component",
    insertText: `${label}()`,
    cursorBack: 1,
  })),
];

const COMPONENT_NAMES = new Set(COMPONENTS.map((item) => item.label));
const ROOT_CONTAINERS = {
  Col: "Table",
  SelectItem: "Select",
  CheckBoxItem: "CheckBoxGroup",
  RadioItem: "RadioGroup",
  SwitchItem: "SwitchGroup",
  TabItem: "Tabs",
};
const WORD_PATTERN = /[A-Za-z_][A-Za-z0-9_]*/g;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function highlightOpenUI(source) {
  const tokenPattern =
    /"(?:\\.|[^"\\])*"|:[A-Za-z_][A-Za-z0-9_.]*|\b[A-Z][A-Za-z0-9]*(?=\s*\()|\b(?:root|true|false|null)\b|\b\d+(?:\.\d+)?\b|[A-Za-z_][A-Za-z0-9_]*(?=\s*=)/g;
  let html = "";
  let cursor = 0;
  for (const match of String(source || "").matchAll(tokenPattern)) {
    const token = match[0];
    const offset = match.index || 0;
    html += escapeHtml(source.slice(cursor, offset));
    let kind = "identifier";
    if (token.startsWith('"')) kind = token.includes(":") ? "placeholder" : "string";
    else if (token.startsWith(":")) kind = "placeholder";
    else if (COMPONENT_NAMES.has(token) || /^[A-Z]/.test(token)) kind = "component";
    else if (["root", "true", "false", "null"].includes(token)) kind = "keyword";
    else if (/^\d/.test(token)) kind = "number";
    html += `<span class="tok-${kind}">${escapeHtml(token)}</span>`;
    cursor = offset + token.length;
  }
  return html + escapeHtml(source.slice(cursor)) + (source.endsWith("\n") ? " " : "");
}

function delimiterErrors(source) {
  const errors = [];
  const stack = [];
  const pairs = { ")": "(", "]": "[", "}": "{" };
  let quote = false;
  let escaped = false;
  let line = 1;
  for (const char of source) {
    if (char === "\n") line += 1;
    if (quote) {
      if (escaped) escaped = false;
      else if (char === "\\") escaped = true;
      else if (char === '"') quote = false;
      continue;
    }
    if (char === '"') {
      quote = true;
      continue;
    }
    if (["(", "[", "{"].includes(char)) stack.push({ char, line });
    if (pairs[char]) {
      const opening = stack.pop();
      if (!opening || opening.char !== pairs[char]) {
        errors.push(`Line ${line}: unmatched ${char}`);
      }
    }
  }
  if (quote) errors.push(`Line ${line}: unterminated string`);
  for (const opening of stack.reverse()) {
    errors.push(`Line ${opening.line}: unclosed ${opening.char}`);
  }
  return errors;
}

function lintOpenUI(source) {
  const text = String(source || "");
  const errors = delimiterErrors(text);
  const warnings = [];
  const definitions = new Map();
  const references = new Set();
  const lines = text.split("\n");

  lines.forEach((rawLine, index) => {
    const line = rawLine.trim();
    const lineNumber = index + 1;
    if (!line) return;
    const assignment = line.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(/);
    if (!assignment) {
      errors.push(`Line ${lineNumber}: expected name = Component(...)`);
      return;
    }
    const [, identifier, component] = assignment;
    if (definitions.has(identifier)) {
      errors.push(`Line ${lineNumber}: ${identifier} is defined more than once`);
    } else {
      definitions.set(identifier, { component, line: lineNumber });
    }
    if (!COMPONENT_NAMES.has(component)) {
      errors.push(`Line ${lineNumber}: unsupported component ${component}`);
    }
    for (const array of line.matchAll(/\[([^\]]*)\]/g)) {
      const children = array[1].replace(/"(?:\\.|[^"\\])*"/g, "");
      for (const wordMatch of children.matchAll(WORD_PATTERN)) {
        const word = wordMatch[0];
        const following = children.slice((wordMatch.index || 0) + word.length).trimStart();
        if (COMPONENT_NAMES.has(word) && following.startsWith("(")) continue;
        if (!["true", "false", "null", "row", "column"].includes(word)) {
          references.add(word);
        }
      }
    }
    if (["TextContent", "Button"].includes(component) && !/"\s*:[A-Za-z_]/.test(line)) {
      warnings.push(`Line ${lineNumber}: ${component} should use a :placeholder string`);
    }
    if (component === "Stack") {
      const direction = line.match(/,\s*"([^"]+)"\s*\)/)?.[1];
      if (direction && !["row", "column"].includes(direction)) {
        errors.push(`Line ${lineNumber}: Stack direction must be "row" or "column"`);
      }
    }
  });

  if (!definitions.has("root")) errors.push("Define exactly one root assignment");
  const root = definitions.get("root");
  const container = ROOT_CONTAINERS[root?.component];
  if (container) {
    errors.push(`Line ${root.line}: root ${root.component} is structural-only; wrap it in ${container}(...)`);
  }
  for (const reference of references) {
    if (!definitions.has(reference)) errors.push(`Undefined component reference: ${reference}`);
  }
  for (const identifier of definitions.keys()) {
    if (identifier !== "root" && !references.has(identifier)) {
      warnings.push(`Unused component: ${identifier}`);
    }
  }

  return {
    valid: errors.length === 0,
    errors: [...new Set(errors)].slice(0, 8),
    warnings: [...new Set(warnings)].slice(0, 8),
  };
}

function tokenRangeAt(value, cursor) {
  const before = String(value || "").slice(0, cursor);
  const match = before.match(/[A-Za-z_][A-Za-z0-9_]*$/);
  return { start: match ? cursor - match[0].length : cursor, end: cursor, prefix: match?.[0] || "" };
}

function completionItems(source, cursor, force = false) {
  const value = String(source || "");
  const range = tokenRangeAt(value, cursor);
  const before = value.slice(0, range.start);
  const definitions = [...value.matchAll(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=/gm)]
    .map((match) => match[1])
    .filter((name) => name !== "root");
  const insideChildren = /\[[^\]]*$/.test(before);
  const afterAssignment = /=\s*$/.test(before);
  let items = insideChildren
    ? definitions.map((label) => ({ label, detail: "Defined component", insertText: label }))
    : COMPONENTS;
  if (!afterAssignment && !insideChildren && !range.prefix && !force) return [];
  if (!afterAssignment && !insideChildren) {
    items = [...COMPONENTS, ...definitions.map((label) => ({
      label,
      detail: "Defined component",
      insertText: label,
    }))];
  }
  const prefix = range.prefix.toLowerCase();
  return items
    .filter((item) => !prefix || item.label.toLowerCase().startsWith(prefix))
    .slice(0, 8)
    .map((item) => ({ ...item, start: range.start, end: range.end }));
}

function applyCompletion(source, suggestion) {
  const insertText = suggestion.insertText || suggestion.label;
  const value = `${source.slice(0, suggestion.start)}${insertText}${source.slice(suggestion.end)}`;
  return {
    value,
    cursor: suggestion.start + insertText.length - (suggestion.cursorBack || 0),
  };
}

export { COMPONENTS, applyCompletion, completionItems, highlightOpenUI, lintOpenUI };
