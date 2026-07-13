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
  test("desktop: load, note, grade, navigate", async ({ page }, testInfo) => {
    const before = feedbackCount();
    await page.goto("/");
    await expect(page.getByText("TwoTower")).toBeVisible();
    await expect(page.locator("#card")).toBeVisible();

    // Wait until a real sample is ready (not generating forever).
    await expect(page.locator("#badge")).not.toHaveText(/loading|generating/i, {
      timeout: 90_000,
    });
    await expect(page.locator("#promptText")).not.toHaveText(/Loading|…/);

    await page.screenshot({
      path: path.join(
        testInfo.project.outputDir,
        `annotate-desktop-ready-${testInfo.project.name}.png`
      ),
      fullPage: true,
    });

    // Type to focus note
    await page.keyboard.type("looks structured");
    await expect(page.locator("#note")).toHaveValue(/looks structured/);

    await page.locator("#note").press("Escape");
    await page.locator("#card").focus();
    await page.keyboard.press("ArrowUp");

    await expect(page.locator("#status")).toContainText(/Saved thumbs up|Saving/i, {
      timeout: 30_000,
    });

    // Navigate
    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("ArrowLeft");

    await page.screenshot({
      path: path.join(
        testInfo.project.outputDir,
        `annotate-desktop-after-grade-${testInfo.project.name}.png`
      ),
      fullPage: true,
    });

    // Prefer asserting feedback grew; generation may be slow but grade should persist.
    await expect.poll(() => feedbackCount(), { timeout: 15_000 }).toBeGreaterThan(before);
  });

  test("mobile: large grade targets visible", async ({ page }, testInfo) => {
    await page.goto("/");
    await expect(page.locator("#btnUp")).toBeVisible();
    await expect(page.locator("#btnDown")).toBeVisible();
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
