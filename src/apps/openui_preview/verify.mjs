#!/usr/bin/env node
/** Render one or more isolated OpenUI programs and emit runtime evidence. */

import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../../..");
const bundle = await readFile(resolve(root, "src/slm_training/web/static/preview/preview.js"), "utf8");
const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const request = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
const sources = Array.isArray(request.sources) ? request.sources : [request.source];
if (!sources.length || sources.some((source) => typeof source !== "string" || !source.trim())) {
  throw new Error("source must be a non-empty string");
}

const browser = await chromium.launch({ headless: true });
async function render(source) {
  const page = await browser.newPage();
  page.setDefaultTimeout(5_000);
  const consoleErrors = [];
  const behaviorErrors = [];
  const interactionTrace = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => behaviorErrors.push(error.message));
  try {
    await page.setContent('<main id="target"></main>');
    await page.evaluate(() => {
      if (typeof globalThis.crypto?.randomUUID === "function") return;
      Object.defineProperty(globalThis.crypto, "randomUUID", {
        configurable: true,
        value: () => crypto.getRandomValues(new Uint8Array(16)).join(""),
      });
    });
    await page.addScriptTag({ content: bundle, type: "module" });
    await page.waitForFunction(() => Boolean(window.OpenUIPreview));
    await page.evaluate((value) => {
      window.OpenUIPreview.mount(document.querySelector("#target"), { source: value });
    }, source);
    await page.waitForSelector("#target .openui-preview-root");
    await page.waitForTimeout(100);
    if (request.seed_console_error) await page.evaluate(() => console.error("seeded console error"));
    const button = page.locator("#target button").first();
    if ((await button.count()) > 0) {
      if (request.seed_behavior_error) {
        await button.evaluate((element) => element.addEventListener("click", () => {
          throw new Error("seeded behavior error");
        }, { once: true }));
      }
      await button.click();
      interactionTrace.push("click:button");
      await page.waitForTimeout(50);
    } else if (request.seed_behavior_error) {
      behaviorErrors.push("seeded behavior error: no button rendered");
    }
    const rendered = await page.evaluate(() => {
      const preview = document.querySelector("#target .openui-preview-root");
      if (preview?.getAttribute("data-parse-ok") !== "1") return false;
      return [...preview.querySelectorAll("*")].some((element) => {
        if (element.classList.contains("openui-preview-sr") || element.classList.contains("openui-preview-empty")) return false;
        const bounds = element.getBoundingClientRect();
        const style = getComputedStyle(element);
        return bounds.width > 0 && bounds.height > 0 && style.display !== "none" && style.visibility !== "hidden";
      });
    });
    return { rendered, console_errors: consoleErrors, behavior_errors: behaviorErrors, interaction_trace: interactionTrace };
  } catch (error) {
    return { rendered: false, console_errors: consoleErrors, behavior_errors: [...behaviorErrors, String(error?.message || error)], interaction_trace: interactionTrace };
  } finally {
    await page.close();
  }
}
try {
  const results = [];
  for (let index = 0; index < sources.length; index += 32) {
    results.push(...await Promise.all(sources.slice(index, index + 32).map(render)));
  }
  process.stdout.write(JSON.stringify(Array.isArray(request.sources) ? results : results[0]));
} finally {
  await browser.close();
}
