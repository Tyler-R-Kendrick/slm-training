import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const FEEDBACK = path.resolve("outputs/annotations/feedback.jsonl");

function feedbackCount(): number {
  if (!fs.existsSync(FEEDBACK)) return 0;
  return fs
    .readFileSync(FEEDBACK, "utf8")
    .split("\n")
    .filter((l) => l.trim()).length;
}

async function waitForSampleReady(page: import("@playwright/test").Page) {
  await expect(page.locator("#badge")).not.toHaveText(/loading|generating/i, {
    timeout: 90_000,
  });
  await expect(page.locator("#promptText")).not.toHaveText(/Loading|…/);
  await expect(page.locator("#badge")).toHaveText("valid");
  await expect(page.locator("#badge")).not.toHaveText(/invalid|error/i);
}

test.describe("annotate playground", () => {
  test("desktop: thumbs icons + grade stays on sample", async ({ page }, testInfo) => {
    const before = feedbackCount();
    await page.goto("/playground/classic");
    await page.evaluate(() => localStorage.setItem("twotower_annotate_view", "render"));
    await page.reload();

    await expect(page.getByText("TwoTower")).toBeVisible();
    await expect(page.locator("#card")).toBeVisible();

    // Thumbs up/down symbols (not arrow glyphs).
    await expect(page.locator("#btnUp .thumb-icon")).toHaveText("👍");
    await expect(page.locator("#btnDown .thumb-icon")).toHaveText("👎");
    await expect(page.locator(".brand-sub")).toContainText("👍");
    await expect(page.locator(".brand-sub")).toContainText("👎");
    await expect(page.locator("#btnUp")).not.toContainText("↑");
    await expect(page.locator("#btnDown")).not.toContainText("↓");

    await waitForSampleReady(page);

    // A valid sample is present; empty-stack "Ready" false-positive is fixed.
    await expect(page.locator("#promptText")).not.toHaveText("…");
    await expect(page.locator("#badge")).toHaveText("valid");

    const indexBefore = (await page.locator("#indexPill").innerText()).trim();
    const promptBefore = (await page.locator("#promptText").innerText()).trim();

    await page.keyboard.type("looks structured");
    await page.locator("#note").press("Escape");
    await page.locator("#card").focus();
    await page.keyboard.press("ArrowUp");

    await expect(page.locator("#status")).toContainText(/Saved thumbs up|Saving 👍/i, {
      timeout: 30_000,
    });

    // Grade must not advance to the next sample.
    await expect(page.locator("#indexPill")).toHaveText(indexBefore);
    await expect(page.locator("#promptText")).toHaveText(promptBefore);

    await expect.poll(() => feedbackCount(), { timeout: 15_000 }).toBeGreaterThan(before);

    await page.screenshot({
      path: path.join(
        testInfo.project.outputDir,
        `annotate-thumbs-grade-stay-${testInfo.project.name}.png`
      ),
      fullPage: true,
    });
  });

  test("desktop: request + rendered default + dsl toggle", async ({ page }, testInfo) => {
    await page.goto("/playground/classic");
    await page.evaluate(() => localStorage.setItem("twotower_annotate_view", "render"));
    await page.reload();

    await expect(page.locator("#btnViewRender")).toBeVisible();
    await expect(page.locator("#btnViewDsl")).toBeVisible();

    await waitForSampleReady(page);

    const prompt = (await page.locator("#promptText").innerText()).trim();
    expect(prompt.length).toBeGreaterThan(40);
    expect(prompt.toLowerCase()).not.toBe("hero card with title and body");
    expect(prompt.toLowerCase()).not.toBe("primary call to action button");
    expect(prompt.toLowerCase()).not.toBe("build primary call to action button");

    await expect(page.locator("#panelRender")).toBeVisible();
    await expect(page.locator("#panelDsl")).toBeHidden();
    await expect(page.locator("#preview .openui-preview-root")).toBeVisible({
      timeout: 30_000,
    });

    await page.screenshot({
      path: path.join(
        testInfo.project.outputDir,
        `annotate-desktop-ready-${testInfo.project.name}.png`
      ),
      fullPage: true,
    });

    await page.locator("#btnViewDsl").click();
    await expect(page.locator("#panelDsl")).toBeVisible();
    await expect(page.locator("#panelRender")).toBeHidden();
    await expect(page.locator("#output")).toContainText(/root\s*=/);

    await page.screenshot({
      path: path.join(
        testInfo.project.outputDir,
        `annotate-desktop-dsl-${testInfo.project.name}.png`
      ),
      fullPage: true,
    });

    await page.locator("#btnViewRender").click();
    await expect(page.locator("#panelRender")).toBeVisible();

    await page.evaluate(() => {
      const src = [
        'root = Stack([hero], "column")',
        'hero_title = TextContent(":hero.title")',
        'hero_body = TextContent(":hero.body")',
        "hero = Card([hero_title, hero_body])",
      ].join("\n");
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const api = (window as any).OpenUIPreview;
      api.mount(document.getElementById("preview"), { source: src });
    });
    await expect(page.locator("#preview")).toContainText(/Title|Body/i, { timeout: 10_000 });
  });

  test("desktop: arrow down grades without advancing", async ({ page }) => {
    await page.goto("/playground/classic");
    await page.evaluate(() => localStorage.setItem("twotower_annotate_view", "render"));
    await page.reload();

    await waitForSampleReady(page);

    const indexBefore = (await page.locator("#indexPill").innerText()).trim();

    await page.locator("#card").focus();
    await page.keyboard.press("ArrowDown");

    await expect(page.locator("#status")).toContainText(/Saved thumbs down|Saving 👎/i, {
      timeout: 30_000,
    });
    await expect(page.locator("#indexPill")).toHaveText(indexBefore);
  });

  test("mobile: preview default + grade targets", async ({ page }, testInfo) => {
    await page.goto("/playground/classic");
    await page.evaluate(() => localStorage.setItem("twotower_annotate_view", "render"));
    await page.reload();
    await expect(page.locator("#btnUp")).toBeVisible();
    await expect(page.locator("#btnDown")).toBeVisible();
    await expect(page.locator("#btnUp .thumb-icon")).toHaveText("👍");
    await expect(page.locator("#btnDown .thumb-icon")).toHaveText("👎");
    await expect(page.locator("#panelRender")).toBeVisible();
    await waitForSampleReady(page);

    const upBox = await page.locator("#btnUp").boundingBox();
    expect(upBox).toBeTruthy();
    expect((upBox?.height || 0) >= 40).toBeTruthy();

    await page.screenshot({
      path: path.join(
        testInfo.project.outputDir,
        `annotate-mobile-${testInfo.project.name}.png`
      ),
      fullPage: true,
    });
  });
});
