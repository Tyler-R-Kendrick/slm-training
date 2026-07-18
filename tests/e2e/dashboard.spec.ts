import { test, expect } from "@playwright/test";

test.describe("mission control dashboard", () => {
  test("renders overview and navigates to checkpoints", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".page-title")).toHaveText(/Mission control/);
    await expect(page.locator(".verdict")).toBeVisible();
    await expect(page.locator(".tile").first()).toBeVisible();

    await page.locator(".nav-link", { hasText: "Checkpoints" }).click();
    await expect(page.locator(".page-title")).toHaveText(/Checkpoints/);
    await expect(page.locator(".gate-matrix")).toBeVisible({ timeout: 10_000 });
  });

  test("renderer toggle swaps interpreted for compiled Overview", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".mode-btn.active")).toHaveText(/Interpreted/);
    await page.locator(".mode-btn", { hasText: "Compiled" }).click();
    await expect(page.locator(".mode-btn.active")).toHaveText(/Compiled/);
    await expect(page.locator(".page-title")).toHaveText(/Mission control/);
    await expect(page.locator(".verdict")).toBeVisible();
  });

  test("Experiments derives metric columns in both renderer modes", async ({ page }) => {
    await page.route("**/api/scoreboards/research", (route) => route.fulfill({ json: {
      metric_columns: [{
        key: "smoke__component_type_recall",
        suite: "smoke",
        metric: "component_type_recall",
        label: "smoke type recall",
      }],
      results: [{ id: "dynamic-column", suites: { smoke: { component_type_recall: 0.42 } } }],
    } }));

    await page.goto("/experiments");
    await expect(page.getByRole("columnheader", { name: "smoke type recall" })).toBeVisible();
    await expect(page.getByRole("cell", { name: "0.42" })).toBeVisible();

    await page.locator(".mode-btn", { hasText: "Compiled" }).click();
    await expect(page.getByRole("columnheader", { name: "smoke type recall" })).toBeVisible();
    await expect(page.getByRole("cell", { name: "0.42" })).toBeVisible();
  });

  test("editing a gate threshold re-evaluates live", async ({ page }) => {
    const passingSuite = {
      n: 1,
      meaningful_program_rate: 1,
      structural_similarity: 0.5,
      component_type_recall: 0.5,
      placeholder_fidelity: 0.5,
      reward_score: 0.5,
    };
    await page.route("**/api/scoreboards/quality", (route) => route.fulfill({ json: {
      results: [{ id: "passing-fixture", run_id: "passing-fixture", suites: {
        smoke: passingSuite,
        held_out: passingSuite,
        adversarial: passingSuite,
        ood: passingSuite,
        rico_held: passingSuite,
      } }],
    } }));
    await page.goto("/checkpoints");
    await expect(page.getByText("GATES PASS")).toBeVisible({ timeout: 10_000 });

    // Raise smoke structural_similarity threshold above the actual value.
    const smoke = page.locator(".thr-suite", { hasText: "smoke" });
    await smoke.locator("label", { hasText: "structural_similarity" }).locator("input").fill("0.99");

    // The pure-compute /api/gates/evaluate endpoint recolors the matrix.
    await expect(page.getByText("GATES FAIL")).toBeVisible({ timeout: 5_000 });
  });

  test("react playground renders inside the SPA shell", async ({ page }) => {
    await page.goto("/playground");
    await expect(page.locator(".page-title")).toHaveText(/Playground/);
    await expect(page.locator(".pg-card")).toBeVisible();
  });

  test("training data exhausts record pages and exposes complete details", async ({ page }) => {
    const records = [
      { id: "row-1", prompt: "First prompt", openui: 'TextContent("first")', source: "fixture", split: "train", meta: { quality: 1 } },
      { id: "row-2", prompt: "Second prompt", openui: 'TextContent("second")', source: "fixture", split: "train", meta: { quality: 2 } },
      { id: "row-3", prompt: "Third prompt", openui: 'TextContent("third")', source: "generated", split: "train", meta: { quality: 3 } },
    ];
    await page.route("**/api/data/train/v1/records?**", (route) => {
      const url = new URL(route.request().url());
      const query = (url.searchParams.get("q") || "").toLowerCase();
      const filtered = query ? records.filter((record) => JSON.stringify(record).toLowerCase().includes(query)) : records;
      const offset = Number(url.searchParams.get("offset") || 0);
      route.fulfill({ json: {
        version: "v1", count: filtered.length, offset, limit: 500,
        sources: ["fixture", "generated"], records: filtered.slice(offset, offset + 2),
      } });
    });
    await page.route("**/api/data/train", (route) => route.fulfill({ json: {
      version: "v1", versions: ["v1"], record_count: 3, storage: "committed", stats: { record_count: 3 },
    } }));
    await page.route("**/api/data/test", (route) => route.fulfill({ json: { suites: {} } }));

    await page.goto("/data");
    await expect(page.locator(".data-browser-count")).toHaveText("Showing all 3 of 3");
    await expect(page.getByRole("button", { name: "View" })).toHaveCount(3);
    await page.getByPlaceholder("id, prompt, source, or OpenUI").fill("Third");
    await expect(page.locator(".data-browser-count")).toHaveText("Showing all 1 of 1");
    await page.getByRole("button", { name: "View" }).click();
    await expect(page.locator(".record-detail")).toContainText('"quality": 3');
  });

  test("retired classic URL redirects to the React playground", async ({ page }) => {
    await page.goto("/playground/classic");
    await expect(page).toHaveURL(/\/playground$/);
    await expect(page.locator(".pg-card")).toBeVisible();
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
