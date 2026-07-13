import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 120_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  retries: 0,
  reporter: [["list"], ["html", { open: "never", outputFolder: "outputs/annotations/playwright-report" }]],
  use: {
    baseURL: process.env.PLAYGROUND_URL || "http://127.0.0.1:8765",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "desktop-chrome", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile-chrome", use: { ...devices["Pixel 7"] } },
  ],
  webServer: {
    command: "python3 -m scripts.serve_playground --host 127.0.0.1 --port 8765",
    url: "http://127.0.0.1:8765/api/health",
    reuseExistingServer: true,
    timeout: 180_000,
  },
  outputDir: "outputs/annotations/test-results",
});
