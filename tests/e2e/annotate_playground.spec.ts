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

test.describe("annotate playground", () => {
  test("desktop: render preview, note, grade, navigate", async ({ page }, testInfo) => {
    const before = feedbackCount();
    await page.goto("/");
    await expect(page.getByText("TwoTower")).toBeVisible();
    await expect(page.locator("#card")).toBeVisible();
    await expect(page.locator("#preview")).toBeVisible();

    await expect(page.locator("#badge")).not.toHaveText(/loading|generating/i, {
      timeout: 90_000,
    });
    await expect(page.locator("#promptText")).not.toHaveText(/Loading|…/);

    // Renderer island mounted (valid samples show UI; invalid show status text).
    await expect(page.locator("#preview .openui-preview-root")).toBeVisible({
      timeout: 30_000,
    });
    await expect
      .poll(async () => {
        const root = page.locator("#preview .openui-preview-root");
        if ((await root.count()) === 0) return "";
        const text = (await page.locator("#preview").innerText()).trim();
        const kids = await page.locator("#preview .openui-preview-root *").count();
        return text || (kids > 0 ? "rendered" : "");
      }, { timeout: 30_000 })
      .not.toEqual("");

    // Force a known-good OpenUI render to assert demo-like visual output.
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

    await page.screenshot({
      path: path.join(
        testInfo.project.outputDir,
        `annotate-desktop-ready-${testInfo.project.name}.png`
      ),
      fullPage: true,
    });

    await page.keyboard.type("looks structured");
    await expect(page.locator("#note")).toHaveValue(/looks structured/);

    await page.locator("#note").press("Escape");
    await page.locator("#card").focus();
    await page.keyboard.press("ArrowUp");

    await expect(page.locator("#status")).toContainText(/Saved thumbs up|Saving/i, {
      timeout: 30_000,
    });

    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("ArrowLeft");

    await page.screenshot({
      path: path.join(
        testInfo.project.outputDir,
        `annotate-desktop-after-grade-${testInfo.project.name}.png`
      ),
      fullPage: true,
    });

    await expect.poll(() => feedbackCount(), { timeout: 15_000 }).toBeGreaterThan(before);
  });

  test("mobile: preview + large grade targets visible", async ({ page }, testInfo) => {
    await page.goto("/");
    await expect(page.locator("#btnUp")).toBeVisible();
    await expect(page.locator("#btnDown")).toBeVisible();
    await expect(page.locator("#preview")).toBeVisible();
    await expect(page.locator("#badge")).not.toHaveText(/loading/i, { timeout: 90_000 });

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
