import { test, expect } from "@playwright/test";

test.describe("mission control dashboard", () => {
  test("renders overview and navigates to checkpoints", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".page-title")).toHaveText(/Mission Control/);
    await expect(page.locator(".tile").first()).toBeVisible();

    await page.locator(".nav-link", { hasText: "Checkpoints" }).click();
    await expect(page.locator(".page-title")).toHaveText(/Checkpoints/);
    await expect(page.locator(".gate-matrix")).toBeVisible({ timeout: 10_000 });
  });

  test("editing a gate threshold re-evaluates live", async ({ page }) => {
    await page.goto("/checkpoints");
    await expect(page.getByText("GATES PASS")).toBeVisible({ timeout: 10_000 });

    // Raise smoke structural_similarity threshold above the actual value.
    const smoke = page.locator(".thr-suite", { hasText: "smoke" });
    await smoke.locator("input").nth(1).fill("0.99");

    // The pure-compute /api/gates/evaluate endpoint recolors the matrix.
    await expect(page.getByText("GATES FAIL")).toBeVisible({ timeout: 5_000 });
  });

  test("react playground renders inside the SPA shell", async ({ page }) => {
    await page.goto("/playground");
    await expect(page.locator(".page-title")).toHaveText(/Playground/);
    await expect(page.locator(".pg-card")).toBeVisible();
  });

  test("training data is visible and dataset creation is simple", async ({ page }) => {
    await page.goto("/data");
    await expect(page.locator(".page-title")).toHaveText("Training Data");
    await expect(page.getByText("Training examples")).toBeVisible();
    await expect(page.locator(".dtable tbody tr").first()).toBeVisible({ timeout: 10_000 });
    await expect(page.locator(".data-prompt").first()).not.toBeEmpty();
    await expect(page.locator(".data-target").first()).not.toBeEmpty();

    await expect(page.getByText("Create a training dataset")).toBeVisible();
    await expect(page.getByLabel("Dataset name")).toBeVisible();
    await expect(page.getByText("Variation recipe")).toHaveCount(0);
    await expect(page.getByText("Fuzzy deduplication")).toHaveCount(0);
  });

  test("classic annotate playground stays reachable as a fallback", async ({ page }) => {
    await page.goto("/playground/classic");
    await expect(page.getByText("TwoTower")).toBeVisible();
  });
});
