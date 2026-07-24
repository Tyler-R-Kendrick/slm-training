#!/usr/bin/env node
/** Render one OpenUI program and emit runtime/behavior evidence as JSON. */

import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../../..");
const bundle = resolve(root, "src/slm_training/web/static/preview/preview.js");

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const request = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
if (typeof request.source !== "string" || !request.source.trim()) {
  throw new Error("source must be a non-empty string");
}

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  const consoleErrors = [];
  const behaviorErrors = [];
  const interactionTrace = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => behaviorErrors.push(error.message));

  await page.setContent('<main id="target"></main>');
  await page.evaluate(() => {
    // Chromium's data: documents expose Web Crypto but omit randomUUID. The
    // official chart components use it for element ids, so mirror browsers
    // that implement the standard API before loading the preview bundle.
    if (typeof globalThis.crypto?.randomUUID === "function") return;
    Object.defineProperty(globalThis.crypto, "randomUUID", {
      configurable: true,
      value: () => {
        const bytes = globalThis.crypto.getRandomValues(new Uint8Array(16));
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        const hex = [...bytes].map((byte) => byte.toString(16).padStart(2, "0"));
        return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
      },
    });
  });
  await page.addScriptTag({
    content: await readFile(bundle, "utf8"),
    type: "module",
  });
  await page.waitForFunction(() => Boolean(window.OpenUIPreview));
  await page.evaluate((source) => {
    window.OpenUIPreview.mount(document.querySelector("#target"), { source });
  }, request.source);
  await page.waitForSelector("#target .openui-preview-root");
  await page.waitForTimeout(100);

  if (request.seed_console_error) {
    await page.evaluate(() => console.error("seeded console error"));
  }

  const button = page.locator("#target button").first();
  if ((await button.count()) > 0) {
    if (request.seed_behavior_error) {
      await button.evaluate((element) => {
        element.addEventListener(
          "click",
          () => {
            throw new Error("seeded behavior error");
          },
          { once: true },
        );
      });
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
  process.stdout.write(
    JSON.stringify({
      rendered,
      console_errors: consoleErrors,
      behavior_errors: behaviorErrors,
      interaction_trace: interactionTrace,
    }),
  );
} finally {
  await browser.close();
}
