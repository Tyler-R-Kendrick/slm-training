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

  test("run detail charts loss, marks collapse, and explains phase improvements", async ({ page }) => {
    await page.route("**/api/runs/demo/rl-traces**", (route) => route.fulfill({ json: { traces: [], total: 0, count: 0, invalid_rows: 0 } }));
    await page.route("**/api/runs/demo", (route) => route.fulfill({ json: {
      run_id: "demo",
      provenance: "live",
      train_summary: { steps: 6, last_loss: 3, finished_at: "2026-07-15T00:00:00Z" },
      scoreboard: {}, manifest: null, gates: null, telemetry: null, matrix_result: null,
      insights: {
        run_id: "demo", source_fingerprint: "a".repeat(64), enrichment: { generated: { summary: "The final batch coincides with a loss spike.", causes: [] } },
        loss: { status: "collapsed", points: [{ step: 1, loss: 1 }, { step: 5, loss: 1 }, { step: 6, loss: 3 }], events: [{ kind: "spike", step: 6, loss: 3, severity: "critical", finding: "Loss spiked above its rolling baseline.", suggestion: "Test a lower learning rate." }] },
        phases: [{ label: "denoiser", value: 72, help: "Profile model forwards first, then rerun quality guardrails." }],
      },
    } }));
    await page.goto("/runs/demo");
    await expect(page.getByText("Loss over time")).toBeVisible();
    await expect(page.locator(".loss-line")).toBeVisible();
    await expect(page.locator(".loss-marker-critical circle")).toHaveCount(1);
    await expect(page.locator(".bar-help")).toHaveAttribute("aria-label", /rerun quality guardrails/);
    await expect(page.getByText("The final batch coincides with a loss spike.")).toBeVisible();
  });
});
