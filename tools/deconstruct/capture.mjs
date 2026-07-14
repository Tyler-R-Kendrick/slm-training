import fs from "node:fs";
import path from "node:path";
import { chromium } from "@playwright/test";

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const input = JSON.parse(Buffer.concat(chunks).toString("utf8"));
const url = new URL(input.url);
if (!["http:", "https:"].includes(url.protocol)) {
  throw new Error(`unsupported URL protocol: ${url.protocol}`);
}

const viewport = input.viewport || { width: 1440, height: 1000 };
const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext({
    viewport,
    userAgent: input.user_agent,
  });
  const page = await context.newPage();
  await page.goto(url.href, { waitUntil: "networkidle", timeout: 45_000 });

  const elements = await page.evaluate(() => {
    const roleFor = (element) => {
      const explicit = element.getAttribute("role");
      if (explicit) return explicit;
      const roles = {
        A: "link",
        BUTTON: "button",
        FORM: "form",
        H1: "heading",
        H2: "heading",
        H3: "heading",
        IMG: "img",
        INPUT: "textbox",
        LI: "listitem",
        MAIN: "main",
        NAV: "navigation",
        SELECT: "combobox",
        TABLE: "table",
        TEXTAREA: "textbox",
      };
      return roles[element.tagName] || "generic";
    };
    const domPath = (element) => {
      const parts = [];
      let current = element;
      while (current && current.nodeType === Node.ELEMENT_NODE && parts.length < 8) {
        const siblings = [...current.parentElement?.children || []].filter(
          (item) => item.tagName === current.tagName,
        );
        const suffix = siblings.length > 1 ? `:nth-of-type(${siblings.indexOf(current) + 1})` : "";
        parts.unshift(`${current.tagName.toLowerCase()}${suffix}`);
        current = current.parentElement;
      }
      return parts.join(" > ");
    };

    const candidates = [...document.body.querySelectorAll("*")].filter((element) => {
      const rect = element.getBoundingClientRect();
      const style = getComputedStyle(element);
      return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
    });
    const ids = new Map(candidates.map((element, index) => [element, `node_${index}`]));
    return candidates.map((element, index) => {
      const rect = element.getBoundingClientRect();
      const tag = element.tagName;
      const form = ["INPUT", "SELECT", "TEXTAREA"].includes(tag)
        ? {
            type: element.getAttribute("type") || tag.toLowerCase(),
            value: element.value || "",
            checked: Boolean(element.checked),
            disabled: Boolean(element.disabled),
            required: Boolean(element.required),
          }
        : {};
      const affordances = [];
      if (tag === "A") affordances.push("navigate");
      if (["BUTTON", "SUMMARY"].includes(tag) || element.onclick) affordances.push("click");
      if (["INPUT", "SELECT", "TEXTAREA"].includes(tag)) affordances.push("input");
      if (element.closest("form") && ["BUTTON", "INPUT"].includes(tag)) affordances.push("submit");
      const text = [...element.childNodes]
        .filter((node) => node.nodeType === Node.TEXT_NODE)
        .map((node) => node.textContent.trim())
        .filter(Boolean)
        .join(" ")
        .slice(0, 500);
      return {
        id: ids.get(element),
        role: roleFor(element),
        accessible_name:
          element.getAttribute("aria-label") ||
          element.getAttribute("alt") ||
          element.getAttribute("title") ||
          "",
        text,
        bbox: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
        visible: true,
        parent_id: ids.get(element.parentElement) || null,
        form,
        repeated_region: Boolean(element.closest("ul,ol,table,tbody,[role=list]")),
        affordances,
        responsive_state: innerWidth < 640 ? "mobile" : innerWidth < 1024 ? "tablet" : "desktop",
        dom_path: domPath(element),
        screenshot_ref: `node_${index}`,
        dsl_node: element.getAttribute("data-openui-node") || null,
      };
    });
  });

  if (input.screenshot_path) {
    fs.mkdirSync(path.dirname(input.screenshot_path), { recursive: true });
    await page.screenshot({ path: input.screenshot_path, fullPage: true });
  }
  const accessibilityTree = await page.locator("body").ariaSnapshot();
  const interactive = elements
    .filter((element) => element.affordances.length)
    .map((element) => `observed:${element.id}:${element.affordances.join("+")}`);
  process.stdout.write(
    JSON.stringify({
      source_url: input.url,
      dom_snapshot: await page.content(),
      accessibility_tree: accessibilityTree,
      elements,
      viewport,
      responsive_state: viewport.width < 640 ? "mobile" : viewport.width < 1024 ? "tablet" : "desktop",
      screenshot_path: input.screenshot_path || null,
      interaction_trace: interactive,
    }),
  );
} finally {
  await browser.close();
}
