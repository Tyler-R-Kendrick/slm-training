#!/usr/bin/env node
/** Capture a ProgramSpec render matrix with stable statement/layout metadata. */

import { mkdir, readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
const bundle = resolve(root, "src/slm_training/web/static/preview/preview.js");
const stylesheet = resolve(root, "src/slm_training/web/static/preview/preview.css");
const appStylesheet = resolve(root, "src/slm_training/web/static/styles.css");
const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const request = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
const spec = request.program_spec;
if (!spec?.id || !spec?.canonical_openui || !spec?.ast) {
  throw new Error("program_spec must include id, canonical_openui, and ast");
}
if (!Array.isArray(request.variants) || !request.variants.length) {
  throw new Error("variants must be a non-empty array");
}
const outputDir = resolve(String(request.output_dir || ""));
await mkdir(outputDir, { recursive: true });

function flattenAst(node, parentStatement = null, result = []) {
  if (!node || typeof node !== "object") return result;
  if (node.type === "element" && node.statementId) {
    const entry = {
      statement_name: String(node.statementId),
      parent_statement_name: parentStatement,
      type_name: String(node.typeName || "generic"),
      tokens: [],
      child_statements: [],
    };
    const visitProp = (value) => {
      if (typeof value === "string") entry.tokens.push(value);
      else if (Array.isArray(value)) value.forEach(visitProp);
      else if (value && typeof value === "object" && value.type !== "element") {
        Object.values(value).forEach(visitProp);
      }
    };
    Object.values(node.props || {}).forEach(visitProp);
    result.push(entry);
    const before = result.length;
    Object.values(node.props || {}).forEach((value) => {
      const visitChild = (child) => {
        if (Array.isArray(child)) child.forEach(visitChild);
        else if (child?.type === "element") flattenAst(child, entry.statement_name, result);
        else if (child && typeof child === "object") Object.values(child).forEach(visitChild);
      };
      visitChild(value);
    });
    entry.child_statements = result.slice(before).filter((item) => item.parent_statement_name === entry.statement_name).map((item) => item.statement_name);
  }
  return result;
}

function axisPositions(total, viewport, overlap) {
  const maximum = Math.max(0, total - viewport);
  const step = Math.max(1, viewport - Math.max(0, overlap));
  const positions = [];
  for (let value = 0; value < maximum; value += step) positions.push(value);
  positions.push(maximum);
  return [...new Set(positions)];
}

const astNodes = flattenAst(spec.ast);
const browser = await chromium.launch({ headless: true });
const captures = [];
try {
  for (const variant of request.variants) {
    const page = await browser.newPage({ viewport: { width: variant.width, height: variant.height }, colorScheme: variant.theme });
    const consoleErrors = [];
    const behaviorErrors = [];
    const interactionTrace = [];
    page.on("console", (message) => message.type() === "error" && consoleErrors.push(message.text()));
    page.on("pageerror", (error) => behaviorErrors.push(error.message));
    await page.setContent('<main id="target" class="preview-mount"></main>');
    await page.addStyleTag({ content: await readFile(stylesheet, "utf8") });
    await page.addStyleTag({ content: await readFile(appStylesheet, "utf8") });
    await page.addStyleTag({ content: `html, body { margin: 0; background: ${variant.theme === "dark" ? "#171717" : "#fff"}; }` });
    await page.addScriptTag({ content: await readFile(bundle, "utf8"), type: "module" });
    await page.waitForFunction(() => Boolean(window.OpenUIPreview));
    const source = request.state_sources?.[variant.render_state] || spec.canonical_openui;
    await page.evaluate(({ source, variant: active }) => {
      window.OpenUIPreview.mount(document.querySelector("#target"), {
        source,
        keepPlaceholders: true,
      });
      document.documentElement.style.colorScheme = active.theme;
    }, { source, variant });
    await page.waitForSelector('#target .openui-preview-root[data-parse-ok="1"]');
    await page.locator("#target .openui-preview-root").evaluate((element, active) => {
      element.setAttribute("data-theme", active.theme);
      element.setAttribute("data-render-state", active.render_state);
      element.setAttribute("data-interaction-state", active.interaction_state);
    }, variant);
    await page.evaluate(() => document.fonts?.ready);
    await page.waitForTimeout(100);

    if (variant.interaction_state !== "idle") {
      const target = page.locator("#target button, #target a, #target input, #target select, #target textarea").first();
      if (await target.count()) {
        await target.click();
        interactionTrace.push(`${variant.interaction_state}:${await target.evaluate((element) => element.tagName.toLowerCase())}`);
      }
    }

    const elements = await page.evaluate(({ nodes, programId, renderState }) => {
      const preview = document.querySelector("#target .openui-preview-root");
      const visible = [...preview.querySelectorAll("*")].filter((element) => {
        const rect = element.getBoundingClientRect();
        const style = getComputedStyle(element);
        return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
      });
      const used = new Set();
      const resolved = new Map();
      const roleTags = {
        Button: ["button"], TextInput: ["input", "textarea"], Select: ["select"],
        Image: ["img"], ImageBlock: ["img"], Link: ["a"], TextContent: ["p", "span", "div"],
      };
      const smallest = (candidates) => candidates.sort((left, right) => {
        const a = left.getBoundingClientRect();
        const b = right.getBoundingClientRect();
        return a.width * a.height - b.width * b.height;
      })[0];
      for (const node of [...nodes].reverse()) {
        let element = node.parent_statement_name === null ? preview : null;
        let candidates = visible.filter((element) => !used.has(element));
        const tags = roleTags[node.type_name];
        if (tags) candidates = candidates.filter((element) => tags.includes(element.tagName.toLowerCase()));
        const tokens = node.tokens.filter((token) => token.startsWith(":"));
        if (tokens.length) candidates = candidates.filter((element) => tokens.some((token) => (element.textContent || element.getAttribute("aria-label") || "").includes(token)));
        element ||= smallest(candidates);
        if (!element && node.child_statements.length) {
          const children = node.child_statements.map((name) => resolved.get(name)).filter(Boolean);
          element = children[0]?.parentElement || children[0];
        }
        if (!element) continue;
        used.add(element);
        resolved.set(node.statement_name, element);
        element.setAttribute("data-openui-node-id", `${programId}::${node.statement_name}`);
        element.setAttribute("data-openui-statement", node.statement_name);
      }
      return nodes.map((node) => {
        const element = resolved.get(node.statement_name);
        if (!element) return null;
        const rect = element.getBoundingClientRect();
        const x = rect.x + window.scrollX;
        const y = rect.y + window.scrollY;
        const clipX = Math.max(0, rect.x);
        const clipY = Math.max(0, rect.y);
        const clipRight = Math.min(innerWidth, rect.right);
        const clipBottom = Math.min(innerHeight, rect.bottom);
        const style = getComputedStyle(element);
        const zIndex = Number.parseInt(style.zIndex, 10) || 0;
        const domOrder = visible.indexOf(element);
        const tag = element.tagName.toLowerCase();
        return {
          openui_node_id: `${programId}::${node.statement_name}`,
          statement_name: node.statement_name,
          parent_node_id: node.parent_statement_name ? `${programId}::${node.parent_statement_name}` : null,
          bounding_box: { x, y, width: rect.width, height: rect.height },
          visible_clip: { x: clipX + window.scrollX, y: clipY + window.scrollY, width: Math.max(0, clipRight - clipX), height: Math.max(0, clipBottom - clipY) },
          z_order: zIndex * 1000000 + domOrder,
          semantic_role: element.getAttribute("role") || tag || node.type_name,
          accessible_name: (element.getAttribute("aria-label") || element.textContent || "").trim().slice(0, 200),
          interaction_target: ["button", "a", "input", "select", "textarea"].includes(tag),
          render_state: renderState,
        };
      }).filter(Boolean);
    }, { nodes: astNodes, programId: spec.id, renderState: variant.render_state });

    const key = `${variant.width}x${variant.height}.${variant.theme}.${variant.render_state}.${variant.interaction_state}`.replace(/[^A-Za-z0-9_.-]/g, "_");
    const fixed = `${key}.fixed.png`;
    const full = `${key}.full.png`;
    await page.evaluate(() => scrollTo(0, 0));
    await page.screenshot({ path: join(outputDir, fixed) });
    await page.screenshot({ path: join(outputDir, full), fullPage: true });
    const dimensions = await page.evaluate(() => ({ width: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight }));
    const scrollTiles = [];
    for (const scrollY of axisPositions(dimensions.height, variant.height, request.tile_overlap || 0)) {
      for (const scrollX of axisPositions(dimensions.width, variant.width, request.tile_overlap || 0)) {
        await page.evaluate(({ x, y }) => scrollTo(x, y), { x: scrollX, y: scrollY });
        const screenshot = `${key}.tile-${scrollX}-${scrollY}.png`;
        await page.screenshot({ path: join(outputDir, screenshot) });
        scrollTiles.push({ screenshot, scroll_x: scrollX, scroll_y: scrollY, width: variant.width, height: variant.height });
      }
    }
    captures.push({
      program_id: spec.id,
      variant,
      fixed_screenshot: fixed,
      full_page_screenshot: full,
      scroll_tiles: scrollTiles,
      elements,
      interaction_trace: interactionTrace,
      console_errors: consoleErrors,
      behavior_errors: behaviorErrors,
    });
    await page.close();
  }
} finally {
  await browser.close();
}
process.stdout.write(JSON.stringify({ captures }));
