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

  test("classic annotate playground stays reachable", async ({ page }) => {
    await page.goto("/playground");
    await expect(page.getByText("TwoTower")).toBeVisible();
  });
});
