import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const FEEDBACK = path.resolve("outputs/data/annotation/feedback.jsonl");

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
  test("busy and boundary states disable the correct controls", async ({ page }) => {
    const source = [
      'root = Stack([hero], "column")',
      'hero_title = TextContent(":hero.title")',
      'hero = Card([hero_title])',
    ].join("\n");
    await page.addInitScript(() => {
      // @ts-expect-error Test-only counter.
      window.__browserCreateCalls = 0;
      // @ts-expect-error Test double for Chrome's built-in Prompt API.
      window.LanguageModel = {
        availability: async () => "available",
        create: async () => {
          // @ts-expect-error Test-only counter.
          window.__browserCreateCalls += 1;
          return {
            prompt: async () =>
              JSON.stringify({ passed: true, score: 0.95, reasons: ["Useful layout"] }),
            destroy: () => {},
          };
        },
      };
    });
    await page.route("**/api/server-attempt", async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 2_000));
      const body = route.request().postDataJSON();
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          prompt: "A deliberately delayed sample for state verification",
          openui: source,
          serialized: source,
          valid: true,
          error: null,
          source: "server",
          attempt: {
            id: `server_${body.attempt}`,
            source: "server",
            attempt: body.attempt,
            valid: true,
            openui: source,
            prior_failures: body.prior_failures,
          },
        }),
      });
    });
    await page.route("**/api/generation-review", async (route) => {
      const body = route.request().postDataJSON();
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          ok: true,
          id: `review_${body.attempt}`,
          passed: true,
          score: 0.95,
          reasons: ["Useful layout"],
          error: null,
        }),
      });
    });
    let keyboardGradeAuthorization: string | undefined;
    let annotationCalls = 0;
    await page.route("**/api/annotate", async (route) => {
      keyboardGradeAuthorization = route.request().headers().authorization;
      annotationCalls += 1;
      if (annotationCalls === 1) {
        await route.fulfill({
          status: 503,
          contentType: "application/json",
          body: JSON.stringify({ detail: "temporary annotation store failure" }),
        });
        return;
      }
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ ok: true, id: "feedback_keyboard", rating: "up" }),
      });
    });

    await page.goto("/playground");
    await expect(page.locator("#badge")).toHaveText(/loading|generating|rendering/i);
    for (const id of [
      "#btnUp",
      "#btnDown",
      "#btnPrev",
      "#btnNext",
      "#btnViewRender",
      "#btnViewDsl",
      "#note",
    ]) {
      await expect(page.locator(id)).toBeDisabled();
    }
    await expect(page.locator("#activityLog")).toContainText(/training-model pipeline started/i);

    await expect(page.locator("#badge")).toHaveText("valid", { timeout: 10_000 });
    await expect(page.locator("#btnPrev")).toBeDisabled();
    await expect(page.locator("#btnUp")).toBeEnabled();
    await expect(page.locator("#modelSource")).toContainText(/training model.*browser-approved/i);
    await page.locator(".pg-advanced summary").click();
    await page.locator("#annotationToken").fill("keyboard-secret");
    await page.locator("#card").focus();
    await page.keyboard.press("ArrowUp");
    await expect.poll(() => keyboardGradeAuthorization).toBe("Bearer keyboard-secret");
    await expect(page.locator("#error")).toContainText("temporary annotation store failure");
    await expect(page.locator("#status")).toHaveText("");
    await expect(page.locator("#btnUp")).toBeEnabled();

    await page.locator("#card").focus();
    await page.keyboard.press("ArrowUp");
    await expect(page.locator("#btnUp")).toBeDisabled();
    await expect(page.locator("#status")).toContainText("Saved thumbs up");
    await expect(page.locator("#btnUp")).toBeEnabled({ timeout: 2_000 });

    await expect(page.locator("#btnNext")).toBeEnabled({ timeout: 10_000 });
    await page.locator("#btnNext").click();
    await expect(page.locator("#btnNext")).toBeDisabled();
    await expect
      .poll(() => page.evaluate(() => (window as any).__browserCreateCalls))
      .toBe(1);
    await expect(page.locator("#activityLog")).toContainText(/reused for every sample/i);
    await expect.poll(() => page.locator("#activityLog").evaluate((element) => element.scrollTop > 0)).toBe(true);
  });

  test("browser fallback carries failures and stores all three attempts", async ({ page }) => {
    const browserOutputs = [
      "not openui",
      "root = Missing([thing])",
      'root = Card([title])\ntitle = TextContent(":hero.title")',
    ];
    // Concurrent prefetch interleaves samples, so the double keys on the TASK
    // marker instead of call order: reviews (for warm-queue samples) pass,
    // and only fallback GENERATE calls consume the scripted outputs.
    await page.addInitScript((outputs) => {
      let index = 0;
      // @ts-expect-error Test double for Chrome's built-in Prompt API.
      window.LanguageModel = {
        availability: async () => "available",
        create: async () => ({
          prompt: async (input: string) =>
            String(input).startsWith("TASK: REVIEW")
              ? JSON.stringify({ passed: true, score: 0.9, reasons: ["Useful layout"] })
              : outputs[index++],
          destroy: () => {},
        }),
      };
    }, browserOutputs);

    const FALLBACK_PROMPT = "A browser fallback card";
    let promptCalls = 0;
    await page.route("**/api/prompt/next**", async (route) => {
      promptCalls += 1;
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ prompt: promptCalls === 1 ? FALLBACK_PROMPT : `A server-success prefetch ${promptCalls}` }),
      });
    });
    await page.route("**/api/server-attempt", async (route) => {
      const body = route.request().postDataJSON();
      // Only the fallback sample fails; warm-queue samples succeed regardless
      // of arrival order.
      if (body.prompt !== FALLBACK_PROMPT) {
        const openui = 'root = Button(":cta.label")';
        await route.fulfill({
          contentType: "application/json",
          body: JSON.stringify({
            prompt: body.prompt,
            openui,
            serialized: openui,
            valid: true,
            source: "server",
            attempt: {
              id: `server_${body.attempt}_prefetch`,
              source: "server",
              attempt: body.attempt,
              valid: true,
              openui,
              prior_failures: body.prior_failures,
            },
          }),
        });
        return;
      }
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          prompt: FALLBACK_PROMPT,
          openui: `bad server output ${body.attempt}`,
          valid: false,
          error: `server failure ${body.attempt}`,
          source: "server",
          attempt: {
            id: `server_${body.attempt}`,
            source: "server",
            attempt: body.attempt,
            valid: false,
            error: `server failure ${body.attempt}`,
            openui: `bad server output ${body.attempt}`,
            prior_failures: body.prior_failures,
          },
        }),
      });
    });

    const browserBodies: Array<{ attempt: number; prior_failures: string[]; openui: string }> = [];
    await page.route("**/api/generation-attempt", async (route) => {
      const body = route.request().postDataJSON();
      browserBodies.push(body);
      const valid = body.attempt === 3;
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          ok: true,
          id: `browser_${body.attempt}`,
          valid,
          error: valid ? null : `browser failure ${body.attempt}`,
          serialized: valid ? body.openui : null,
          training_path: `attempt-${body.attempt}.json`,
          storage: "test",
        }),
      });
    });
    await page.route("**/api/generation-review", async (route) => {
      const body = route.request().postDataJSON();
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          ok: true,
          id: `review_${body.attempt}`,
          passed: true,
          score: 0.9,
          reasons: ["Useful baseline-quality layout"],
          error: null,
        }),
      });
    });

    await page.goto("/playground");
    await expect(page.locator("#badge")).toHaveText("valid", { timeout: 15_000 });
    await expect(page.locator("#activityLog")).toContainText(/browser attempt 3\/3 succeeded/i);
    await expect(page.locator("#modelSource")).toContainText(/browser baseline.*fallback/i);
    expect(browserBodies).toHaveLength(3);
    expect(browserBodies.map((body) => body.prior_failures.length)).toEqual([3, 4, 5]);
    expect(browserBodies.map((body) => body.openui)).toEqual(browserOutputs);
  });

  test("browser baseline rejects a lint-clean candidate before display", async ({ page }) => {
    // Concurrent prefetch interleaves reviews across samples, so the double
    // judges the candidate content itself: every attempt-1 card (title only)
    // is rejected and every attempt-2 card (with :hero.body) passes.
    await page.addInitScript(() => {
      // @ts-expect-error Test double for Chrome's built-in Prompt API.
      window.LanguageModel = {
        availability: async () => "available",
        create: async () => ({
          prompt: async (input: string) =>
            String(input).includes(":hero.body")
              ? JSON.stringify({
                  passed: true,
                  score: 0.91,
                  reasons: ["Complete request-aligned hierarchy"],
                })
              : JSON.stringify({
                  passed: false,
                  score: 0.35,
                  reasons: ["Missing requested body content"],
                }),
          destroy: () => {},
        }),
      };
    });
    const serverBodies: Array<{ attempt: number; prior_failures: string[] }> = [];
    await page.route("**/api/server-attempt", async (route) => {
      const body = route.request().postDataJSON();
      serverBodies.push(body);
      const openui =
        body.attempt === 1
          ? 'root = Card([title])\ntitle = TextContent(":hero.title")'
          : 'root = Card([title, body])\ntitle = TextContent(":hero.title")\nbody = TextContent(":hero.body")';
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          prompt: "Hero card with title and body",
          openui,
          serialized: openui,
          valid: true,
          source: "server",
          attempt: {
            id: `server_${body.attempt}`,
            source: "server",
            attempt: body.attempt,
            valid: true,
            openui,
            prior_failures: body.prior_failures,
          },
        }),
      });
    });
    const reviews: Array<{ attempt: number; passed: boolean; score: number }> = [];
    let reviewCount = 0;
    await page.route("**/api/generation-review", async (route) => {
      const body = route.request().postDataJSON();
      reviews.push(body);
      reviewCount += 1;
      // Store echoes the client verdict, like the real endpoint.
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          ok: true,
          id: `review_${reviewCount}`,
          passed: body.passed === true,
          score: body.score,
          reasons: body.reasons,
          error: body.passed === true ? null : (body.reasons || []).join("; "),
        }),
      });
    });

    await page.goto("/playground");
    await expect(page.locator("#badge")).toHaveText("valid", { timeout: 15_000 });
    await expect(page.locator("#modelSource")).toContainText(/training model.*browser-approved/i);
    await expect(page.locator("#output")).toContainText(":hero.body");
    const attemptOne = reviews.filter((review) => review.attempt === 1);
    const attemptTwo = reviews.filter((review) => review.attempt === 2);
    expect(attemptOne.length).toBeGreaterThan(0);
    expect(attemptTwo.length).toBeGreaterThan(0);
    expect(attemptOne.every((review) => review.passed === false)).toBe(true);
    expect(attemptTwo.every((review) => review.passed === true)).toBe(true);
    const retries = serverBodies.filter((body) => body.attempt === 2);
    expect(retries.length).toBeGreaterThan(0);
    expect(retries.every((body) => body.prior_failures.join(" ").includes("Missing requested body content"))).toBe(true);
  });

  test("human edits preview at a stable height and persist correction identity", async ({ page }) => {
    const generated =
      'root = Card([title])\ntitle = TextContent(":hero.title")';
    const corrected =
      'root = Card([title, body])\n' +
      'title = TextContent(":hero.title")\n' +
      'body = TextContent(":hero.body")';
    await page.addInitScript(() => {
      // @ts-expect-error Test double for Chrome's built-in Prompt API.
      window.LanguageModel = {
        availability: async () => "available",
        create: async () => ({
          prompt: async () =>
            JSON.stringify({ passed: true, score: 0.94, reasons: ["Useful layout"] }),
          destroy: () => {},
        }),
      };
    });
    await page.route("**/api/server-attempt", async (route) => {
      const body = route.request().postDataJSON();
      const identities = {
        request_generator: {
          kind: "system",
          provider: "slm-training",
          id: "prompt-bank-composer:v1",
          model: "prompt-bank-composer",
        },
        output_generator: {
          kind: "model",
          provider: "slm-training",
          id: "twotower:last.pt",
          model: "twotower",
        },
      };
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          prompt: "Hero card with title and body",
          openui: generated,
          serialized: generated,
          valid: true,
          source: "server",
          identities,
          attempt: {
            id: `server_edit_${body.attempt}`,
            source: "server",
            attempt: body.attempt,
            valid: true,
            openui: generated,
            prior_failures: body.prior_failures,
            identities,
          },
        }),
      });
    });
    await page.route("**/api/generation-review", async (route) => {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          ok: true,
          id: "review_edit",
          passed: true,
          score: 0.94,
          reasons: ["Useful layout"],
          error: null,
        }),
      });
    });
    let annotationBody: any = null;
    let annotationAuthorization: string | undefined;
    await page.route("**/api/annotate", async (route) => {
      annotationBody = route.request().postDataJSON();
      annotationAuthorization = route.request().headers().authorization;
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          ok: true,
          id: "fb_human_correction",
          rating: "up",
          openui: annotationBody.openui,
          human_corrected: true,
          identities: annotationBody.identities,
        }),
      });
    });

    await page.goto("/playground");
    await expect(page.locator("#badge")).toHaveText("valid", { timeout: 15_000 });
    await page.locator("#annotatorIdentity").fill("alice@example.com");
    await page.locator("#annotatorIdentity").blur();
    await page.locator(".pg-advanced summary").click();
    await page.locator("#annotationToken").fill("test-secret");
    const renderHeight = await page.locator(".view-panels").evaluate((el) => el.clientHeight);

    await page.locator("#btnViewDsl").click();
    await page.locator("#output").fill("root = Card([missing])");
    await expect(page.locator("#dslDiagnostics")).toHaveClass(/error/);
    await expect(page.locator("#dslDiagnosticList")).toContainText(
      "Undefined component reference: missing"
    );

    await page.locator("#output").fill('root = Col(":hero.title", "success")');
    await expect(page.locator("#dslDiagnosticList")).toContainText(
      "root Col is structural-only; wrap it in Table(...)"
    );
    await page.locator("#output").fill(
      'root = Table([Col(":hero.title", "success")])'
    );
    await expect(page.locator("#dslDiagnosticState")).toContainText("Valid", {
      timeout: 5_000,
    });

    await page.locator("#output").fill("root = Sta");
    await page.locator("#output").press("Control+Space");
    await expect(page.locator("#dslAutocomplete")).toBeVisible();
    await expect(page.locator("#dslAutocomplete")).toContainText("Stack");
    await expect(page.locator("#output")).toHaveAttribute("aria-activedescendant", "dslCompletion0");
    await expect(page.locator("#dslCompletion0")).toHaveAttribute("aria-selected", "true");
    await page.locator("#output").press("Enter");
    await expect(page.locator("#output")).toHaveValue(/root = Stack\(\[\], "column"\)/);

    await page.locator("#output").fill(corrected);
    await expect(page.locator("#dslDiagnosticState")).toContainText("Valid", {
      timeout: 5_000,
    });
    await expect(page.locator("#correctionActions")).toBeHidden();
    const dslHeight = await page.locator(".view-panels").evaluate((el) => el.clientHeight);
    expect(dslHeight).toBe(renderHeight);

    await page.locator(".nav-link", { hasText: "Training Data" }).click();
    await expect(page).toHaveURL(/\/playground$/);
    await expect(page.locator("#status")).toContainText("Save or discard the correction before leaving");

    await page.locator("#btnViewRender").click();
    await expect(page.locator("#correctionActions")).toBeVisible();
    await expect(page.locator("#btnSaveCorrection")).toBeEnabled();
    await expect(page.locator("#preview")).toContainText(/Body/i);
    await page.locator("#btnSaveCorrection").click();
    await expect.poll(() => annotationBody).not.toBeNull();

    expect(annotationBody.human_corrected).toBe(true);
    expect(annotationBody.original_openui).toBe(generated);
    expect(annotationBody.openui).toBe(corrected);
    expect(annotationBody.generation_id).toContain("server_edit_");
    expect(annotationBody.identities.output_generator.model).toBe("twotower");
    expect(annotationBody.identities.correction_author.id).toBe("alice@example.com");
    expect(annotationAuthorization).toBe("Bearer test-secret");
  });

  test("desktop: icon grading persists without advancing", async ({ page }, testInfo) => {
    const before = feedbackCount();
    await page.goto("/playground");
    await page.evaluate(() => localStorage.setItem("twotower_annotate_view", "render"));
    await page.reload();

    await expect(page.locator(".page-title")).toHaveText(/Playground/);
    await expect(page.locator("#card")).toBeVisible();

    await expect(page.locator("#btnUp svg")).toBeVisible();
    await expect(page.locator("#btnDown svg")).toBeVisible();
    await expect(page.locator("#btnUp")).toContainText("Up");
    await expect(page.locator("#btnDown")).toContainText("Down");

    await waitForSampleReady(page);

    // A valid sample is present; empty-stack "Ready" false-positive is fixed.
    await expect(page.locator("#promptText")).not.toHaveText("…");
    await expect(page.locator("#badge")).toHaveText("valid");

    const indexBefore = (await page.locator("#indexPill").innerText()).trim();
    await page.keyboard.type("looks structured");
    await page.locator("#note").press("Enter");
    await expect(page.locator("#note")).not.toBeFocused();
    await expect(page.locator("#card")).toBeFocused();
    await expect(page.locator("#status")).toContainText("Note ready");
    await page.keyboard.press("ArrowUp");

    await expect(page.locator("#status")).toContainText(/Saved thumbs up|Saving approval/i, {
      timeout: 30_000,
    });

    await expect(page.locator("#indexPill")).toHaveText(indexBefore);

    await expect.poll(() => feedbackCount(), { timeout: 15_000 }).toBeGreaterThan(before);

    await page.screenshot({
      path: path.join(
        testInfo.project.outputDir,
        `annotate-grade-stays-${testInfo.project.name}.png`
      ),
      fullPage: true,
    });
  });

  test("desktop: request + rendered default + dsl toggle", async ({ page }, testInfo) => {
    await page.goto("/playground");
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
    await expect(page.locator("#output")).toHaveValue(/root\s*=/);

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
    await page.goto("/playground");
    await page.evaluate(() => localStorage.setItem("twotower_annotate_view", "render"));
    await page.reload();

    await waitForSampleReady(page);

    const indexBefore = (await page.locator("#indexPill").innerText()).trim();

    await page.locator("#card").focus();
    await page.keyboard.press("ArrowDown");

    await expect(page.locator("#status")).toContainText(/Saved thumbs down|Saving rejection/i, {
      timeout: 30_000,
    });
    await expect(page.locator("#indexPill")).toHaveText(indexBefore);
  });

  test("mobile: preview default + grade targets", async ({ page }, testInfo) => {
    await page.goto("/playground");
    await page.evaluate(() => localStorage.setItem("twotower_annotate_view", "render"));
    await page.reload();
    await expect(page.locator("#btnUp")).toBeVisible();
    await expect(page.locator("#btnDown")).toBeVisible();
    await expect(page.locator("#btnUp svg")).toBeVisible();
    await expect(page.locator("#btnDown svg")).toBeVisible();
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
