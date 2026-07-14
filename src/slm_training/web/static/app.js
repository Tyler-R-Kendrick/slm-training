import {
  OPENUI_REVIEW_SCHEMA,
  buildBrowserGradePrompt,
  buildBrowserRepairPrompt,
  browserAccelerationCapabilities,
  cleanOpenUIResponse,
  createBrowserModelSession,
  parseBrowserGradeResponse,
} from "./browser_inference.js?v=20260713-4";
import {
  applyCompletion,
  completionItems,
  highlightOpenUI,
  lintOpenUI,
} from "./openui_editor.js?v=20260713-1";

const cardEl = document.getElementById("card");
const promptTextEl = document.getElementById("promptText");
const outputEl = document.getElementById("output");
const errorEl = document.getElementById("error");
const badgeEl = document.getElementById("badge");
const modelSourceEl = document.getElementById("modelSource");
const statusEl = document.getElementById("status");
const noteEl = document.getElementById("note");
const designMdEl = document.getElementById("design_md");
const annotationTokenEl = document.getElementById("annotationToken");
const annotatorIdentityEl = document.getElementById("annotatorIdentity");
const grammarEl = document.getElementById("grammar");
const keepPlaceholdersEl = document.getElementById("keepPlaceholders");
const previewEl = document.getElementById("preview");
const panelRender = document.getElementById("panelRender");
const panelDsl = document.getElementById("panelDsl");
const btnViewRender = document.getElementById("btnViewRender");
const btnViewDsl = document.getElementById("btnViewDsl");
const indexPillEl = document.getElementById("indexPill");
const btnUp = document.getElementById("btnUp");
const btnDown = document.getElementById("btnDown");
const btnPrev = document.getElementById("btnPrev");
const btnNext = document.getElementById("btnNext");
const btnSaveCorrection = document.getElementById("btnSaveCorrection");
const btnDiscardCorrection = document.getElementById("btnDiscardCorrection");
const correctionActionsEl = document.getElementById("correctionActions");
const correctionStatusEl = document.getElementById("correctionStatus");
const activityLogEl = document.getElementById("activityLog");
const dslHighlightEl = document.getElementById("dslHighlight");
const dslAutocompleteEl = document.getElementById("dslAutocomplete");
const dslDiagnosticsEl = document.getElementById("dslDiagnostics");
const dslDiagnosticStateEl = document.getElementById("dslDiagnosticState");
const dslDiagnosticListEl = document.getElementById("dslDiagnosticList");
const dslLintMountEl = document.getElementById("dslLintMount");

const PREFETCH = 2;
const MAX_LOG_ENTRIES = 80;
const SESSION_KEY = "twotower_annotate_session";
const VIEW_KEY = "twotower_annotate_view";
const ANNOTATION_TOKEN_KEY = "twotower_annotation_token";
const ANNOTATOR_IDENTITY_KEY = "twotower_annotator_identity";

if (annotationTokenEl) {
  annotationTokenEl.value = sessionStorage.getItem(ANNOTATION_TOKEN_KEY) || "";
}

if (annotatorIdentityEl) {
  let annotatorId = localStorage.getItem(ANNOTATOR_IDENTITY_KEY);
  if (!annotatorId) {
    annotatorId = `annotator-${Math.random().toString(36).slice(2, 8)}`;
    localStorage.setItem(ANNOTATOR_IDENTITY_KEY, annotatorId);
  }
  annotatorIdentityEl.value = annotatorId;
}

/** @type {"render"|"dsl"} */
let activeView = localStorage.getItem(VIEW_KEY) === "dsl" ? "dsl" : "render";

function sessionId() {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = `s_${Math.random().toString(36).slice(2, 10)}_${Date.now().toString(36)}`;
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

/** @type {{prompt:string, openui:string, serialized?:string|null, originalOpenui?:string, draftOpenui?:string|null, dirty?:boolean, valid:boolean, error?:string|null, status:"ready"|"loading"|"error", source?:"server"|"browser"|null, generationId?:string|null, identities?:Record<string,object>, note?:string}[]} */
const stack = [];
let index = 0;
let busyGrade = false;
let prefetching = false;
let previewRendering = false;
let previewRenderToken = 0;
let renderedItem = null;
let lintTimer = null;
let lintToken = 0;
let completions = [];
let completionIndex = 0;
const browserModelState = {
  session: null,
  promise: null,
  runtime: null,
  availability: null,
  error: null,
};

function current() {
  return stack[index] || null;
}

function annotatorIdentity() {
  const id = (annotatorIdentityEl?.value || "").trim() || sessionId();
  return {
    kind: "user",
    provider: "playground-annotator",
    id,
    display_name: id,
  };
}

function browserModelIdentity(role = "output") {
  const runtime = browserModelState.runtime || "browser";
  const transformers = runtime.startsWith("transformers-js:");
  return {
    kind: "model",
    provider: transformers ? "huggingface-transformers-js" : "browser-built-in-ai",
    id: transformers ? "onnx-community/gemma-3-270m-it-ONNX" : "prompt-api-default",
    model: transformers
      ? "gemma-3-270m-it-ONNX"
      : role === "review"
        ? "prompt-api-baseline-reviewer"
        : "prompt-api-default",
    runtime,
  };
}

function displayedOpenUI(item) {
  if (!item) return "";
  return item.dirty ? item.draftOpenui || "" : item.serialized || item.openui || "";
}

function isCurrentBusy() {
  const item = current();
  return !item || item.status === "loading" || previewRendering;
}

function appendLog(message, level = "info") {
  if (!activityLogEl) return;
  const entry = document.createElement("li");
  entry.className = `activity-entry ${level}`;
  const now = new Date();

  const time = document.createElement("time");
  time.dateTime = now.toISOString();
  time.textContent = new Intl.DateTimeFormat([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(now);

  const text = document.createElement("span");
  text.textContent = message;
  entry.append(time, text);
  activityLogEl.append(entry);
  while (activityLogEl.children.length > MAX_LOG_ENTRIES) {
    activityLogEl.firstElementChild?.remove();
  }
  activityLogEl.scrollTop = activityLogEl.scrollHeight;
}

function preloadBrowserModel() {
  if (browserModelState.promise) return browserModelState.promise;
  const acceleration = browserAccelerationCapabilities();
  appendLog(
    `Browser acceleration: ${acceleration.promptApi ? "built-in Prompt API" : acceleration.webnn ? "WebNN/NPU preferred" : acceleration.webgpu ? "WebGPU preferred" : "WASM fallback"}.`
  );
  appendLog(
    `Local compute: ${acceleration.hardwareConcurrency} logical cores; ` +
      `${acceleration.wasmThreads} WASM thread${acceleration.wasmThreads === 1 ? "" : "s"}; ` +
      `${acceleration.crossOriginIsolated ? "cross-origin isolated" : "single-thread compatibility mode"}.`
  );
  appendLog("Initializing browser baseline model once for this page…");
  browserModelState.promise = createBrowserModelSession({
    mode: "shared",
    onProgress(progress) {
      const previousStatus = browserModelState.availability;
      browserModelState.availability = progress.status;
      if (progress.status === "downloading") {
        if (previousStatus !== "downloading") {
          appendLog("Downloading and caching browser baseline model assets…");
        }
      } else if (progress.status !== previousStatus) {
        appendLog(`Browser baseline availability: ${progress.status}.`);
      }
    },
  })
    .then((created) => {
      browserModelState.session = created.session;
      browserModelState.runtime = created.runtime;
      browserModelState.availability = created.availability;
      appendLog(
        `Browser baseline ready (${created.runtime}); it will be reused for every sample.`,
        "success"
      );
      return created.session;
    })
    .catch((error) => {
      browserModelState.error = error;
      appendLog(
        `Browser baseline initialization failed — ${error?.message || String(error)}`,
        "error"
      );
      return null;
    });
  return browserModelState.promise;
}

async function getBrowserModelSession() {
  const session = browserModelState.session || (await preloadBrowserModel());
  if (!session) {
    throw browserModelState.error || new Error("Browser baseline model is unavailable");
  }
  return session;
}

async function withBrowserModelSession(callback) {
  const baseSession = await getBrowserModelSession();
  const session = baseSession.clone ? await baseSession.clone() : baseSession;
  try {
    return await callback(session);
  } finally {
    if (session !== baseSession) session.destroy?.();
  }
}

function syncControls(item = current()) {
  const busy = !item || item.status === "loading" || previewRendering;
  const editing = !!item?.dirty;
  const gradable =
    !!item &&
    item.status === "ready" &&
    item.valid &&
    !item.renderError &&
    !editing &&
    !busyGrade &&
    !busy;

  btnUp.disabled = !gradable;
  btnDown.disabled = !gradable;
  btnPrev.disabled = busy || editing || index === 0;
  btnNext.disabled = busy || editing || index >= stack.length - 1;
  btnViewRender.disabled = busy;
  btnViewDsl.disabled = busy;
  noteEl.disabled = busy;
  outputEl.disabled = busy || item?.status !== "ready" || !item?.valid;
  correctionActionsEl.hidden = !editing || activeView !== "render";
  btnDiscardCorrection.disabled = !editing || busyGrade;
  const diagnostics = item?.dslDiagnostics;
  btnSaveCorrection.disabled =
    !editing ||
    busyGrade ||
    busy ||
    !!item?.renderError ||
    diagnostics?.pending ||
    diagnostics?.valid !== true;
  if (editing) {
    correctionStatusEl.textContent = diagnostics?.pending
      ? "Checking corrected OpenUI…"
      : diagnostics?.valid === false || item?.renderError
      ? "Fix the DSL error or discard this correction"
      : "Unsaved human correction";
  }
  cardEl.classList.toggle("is-busy", busy);
  cardEl.classList.toggle("is-editing", editing);
  cardEl.setAttribute("aria-busy", busy ? "true" : "false");
}

function renderModelSource(item) {
  modelSourceEl.className = "model-source";
  cardEl.classList.toggle("source-training", item?.source === "server");
  cardEl.classList.toggle("source-baseline", item?.source === "browser");
  if (item?.phase === "browser-review") {
    modelSourceEl.textContent = "Browser baseline reviewing";
    modelSourceEl.classList.add("review");
  } else if (item?.source === "server") {
    modelSourceEl.textContent =
      item.status === "ready"
        ? "Training model · browser-approved"
        : `Training model · attempt ${item.attempt || 1}/3`;
    modelSourceEl.classList.add("training");
  } else if (item?.source === "browser") {
    modelSourceEl.textContent =
      item.status === "ready"
        ? "Browser baseline · fallback"
        : `Browser baseline · attempt ${item.attempt || 1}/3`;
    modelSourceEl.classList.add("baseline");
  } else {
    modelSourceEl.textContent = "Selecting model";
    modelSourceEl.classList.add("pending");
  }
}

function setOutput(text) {
  if (document.activeElement !== outputEl) outputEl.value = text || "";
  syncEditorHighlight();
  outputEl.style.animation = "none";
  void outputEl.offsetWidth;
  outputEl.style.animation = "";
}

function syncEditorHighlight() {
  const code = dslHighlightEl?.querySelector("code");
  if (!code) return;
  code.innerHTML = highlightOpenUI(outputEl.value);
  dslHighlightEl.scrollTop = outputEl.scrollTop;
  dslHighlightEl.scrollLeft = outputEl.scrollLeft;
}

function renderDslDiagnostics(item = current()) {
  if (!dslDiagnosticsEl || !dslDiagnosticStateEl || !dslDiagnosticListEl) return;
  const diagnostics = item?.dslDiagnostics || {
    valid: !!item?.valid,
    pending: false,
    errors: item?.valid ? [] : [item?.error || "OpenUI is not valid"],
    warnings: [],
  };
  const errors = diagnostics.errors || [];
  const warnings = diagnostics.warnings || [];
  dslDiagnosticsEl.classList.remove("valid", "warning", "error");
  dslDiagnosticsEl.classList.add(errors.length ? "error" : warnings.length ? "warning" : "valid");
  dslDiagnosticStateEl.textContent = diagnostics.pending
    ? "Checking OpenUI syntax…"
    : errors.length
      ? `${errors.length} ${errors.length === 1 ? "error" : "errors"}`
      : warnings.length
        ? `Valid with ${warnings.length} ${warnings.length === 1 ? "warning" : "warnings"}`
        : "Valid OpenUI";
  dslDiagnosticListEl.replaceChildren();
  for (const [kind, messages] of [["error", errors], ["warning", warnings]]) {
    for (const message of messages) {
      const entry = document.createElement("li");
      entry.className = `diagnostic-${kind}`;
      entry.textContent = message;
      dslDiagnosticListEl.append(entry);
    }
  }
  outputEl.setAttribute("aria-invalid", errors.length ? "true" : "false");
}

async function validateDslWithRenderer(item, source, staticResult, token) {
  try {
    const api = await waitForPreviewApi();
    if (token !== lintToken || current() !== item || displayedOpenUI(item) !== source) return;
    api.mount(dslLintMountEl, { source, keepPlaceholders: true });
    await afterPreviewCommit();
    if (token !== lintToken || current() !== item || displayedOpenUI(item) !== source) return;
    const root = dslLintMountEl.querySelector(".openui-preview-root");
    const parseOk = root?.getAttribute("data-parse-ok") === "1";
    const errors = [...staticResult.errors];
    if (!parseOk) {
      const parserMessage = root?.textContent?.trim() || "OpenUI renderer rejected this syntax";
      if (!errors.includes(parserMessage)) errors.push(parserMessage);
    }
    item.dslDiagnostics = {
      valid: parseOk && errors.length === 0,
      pending: false,
      errors,
      warnings: staticResult.warnings,
    };
  } catch (error) {
    if (token !== lintToken || current() !== item) return;
    item.dslDiagnostics = {
      valid: false,
      pending: false,
      errors: [...staticResult.errors, error?.message || String(error)],
      warnings: staticResult.warnings,
    };
  }
  renderDslDiagnostics(item);
  if (activeView === "render" && item.dslDiagnostics.valid) {
    item.renderError = null;
    renderedItem = null;
    render();
  } else {
    syncControls(item);
  }
}

function scheduleDslValidation(item, source) {
  if (!item) return;
  clearTimeout(lintTimer);
  const token = ++lintToken;
  const result = lintOpenUI(source);
  item.dslDiagnostics = { ...result, pending: true };
  renderDslDiagnostics(item);
  syncControls(item);
  lintTimer = setTimeout(() => {
    void validateDslWithRenderer(item, source, result, token);
  }, 180);
}

function closeCompletions() {
  completions = [];
  completionIndex = 0;
  dslAutocompleteEl.hidden = true;
  dslAutocompleteEl.replaceChildren();
  outputEl.removeAttribute("aria-activedescendant");
}

function renderCompletions(items) {
  completions = items;
  completionIndex = Math.min(completionIndex, Math.max(items.length - 1, 0));
  dslAutocompleteEl.replaceChildren();
  if (!items.length) {
    closeCompletions();
    return;
  }
  items.forEach((suggestion, suggestionIndex) => {
    const option = document.createElement("button");
    option.type = "button";
    option.id = `dslCompletion${suggestionIndex}`;
    option.className = `completion-option${suggestionIndex === completionIndex ? " is-selected" : ""}`;
    option.setAttribute("role", "option");
    option.setAttribute("aria-selected", suggestionIndex === completionIndex ? "true" : "false");
    option.tabIndex = -1;
    const label = document.createElement("span");
    label.className = "completion-label";
    label.textContent = suggestion.label;
    const detail = document.createElement("span");
    detail.className = "completion-detail";
    detail.textContent = suggestion.detail || "OpenUI suggestion";
    option.append(label, detail);
    option.addEventListener("pointerdown", (event) => event.preventDefault());
    option.addEventListener("click", () => acceptCompletion(suggestionIndex));
    dslAutocompleteEl.append(option);
  });
  dslAutocompleteEl.hidden = false;
  outputEl.setAttribute("aria-activedescendant", `dslCompletion${completionIndex}`);
}

function updateCompletions(force = false) {
  if (outputEl.disabled || document.activeElement !== outputEl) {
    closeCompletions();
    return;
  }
  completionIndex = 0;
  renderCompletions(completionItems(outputEl.value, outputEl.selectionStart, force));
}

function selectCompletion(nextIndex) {
  if (!completions.length) return;
  completionIndex = (nextIndex + completions.length) % completions.length;
  renderCompletions(completions);
  dslAutocompleteEl.querySelector(".is-selected")?.scrollIntoView({ block: "nearest" });
}

function acceptCompletion(selectedIndex = completionIndex) {
  const suggestion = completions[selectedIndex];
  if (!suggestion) return;
  const completed = applyCompletion(outputEl.value, suggestion);
  outputEl.value = completed.value;
  outputEl.setSelectionRange(completed.cursor, completed.cursor);
  closeCompletions();
  outputEl.dispatchEvent(new Event("input", { bubbles: true }));
  outputEl.focus();
}

function setView(view) {
  activeView = view === "dsl" ? "dsl" : "render";
  localStorage.setItem(VIEW_KEY, activeView);
  const showRender = activeView === "render";
  panelRender.classList.toggle("is-active", showRender);
  panelDsl.classList.toggle("is-active", !showRender);
  panelRender.hidden = !showRender;
  panelDsl.hidden = showRender;
  btnViewRender.classList.toggle("is-active", showRender);
  btnViewDsl.classList.toggle("is-active", !showRender);
  btnViewRender.setAttribute("aria-selected", showRender ? "true" : "false");
  btnViewDsl.setAttribute("aria-selected", showRender ? "false" : "true");
  if (showRender) {
    renderedItem = null;
  } else {
    previewRenderToken += 1;
    previewRendering = false;
  }
  render();
}

function waitForPreviewApi(timeoutMs = 15000) {
  if (window.OpenUIPreview?.mount) return Promise.resolve(window.OpenUIPreview);
  return new Promise((resolve, reject) => {
    const t0 = performance.now();
    const tick = () => {
      if (window.OpenUIPreview?.mount) {
        resolve(window.OpenUIPreview);
        return;
      }
      if (performance.now() - t0 > timeoutMs) {
        reject(new Error("OpenUI preview bundle failed to load"));
        return;
      }
      requestAnimationFrame(tick);
    };
    tick();
  });
}

function updatePreview(item) {
  if (!previewEl || activeView !== "render") return;
  const api = window.OpenUIPreview;
  if (!api?.mount) {
    previewEl.innerHTML =
      '<p class="openui-preview-empty">Loading renderer…</p>';
    return;
  }
  if (!item || item.status === "loading") {
    api.mount(previewEl, { source: null });
    return;
  }
  const source = displayedOpenUI(item);
  try {
    api.mount(previewEl, {
      source,
      keepPlaceholders: !!keepPlaceholdersEl?.checked,
    });
  } catch (err) {
    previewEl.innerHTML = `<p class="openui-preview-empty">Render error: ${
      err?.message || err
    }</p>`;
    throw err;
  }
}

function afterPreviewCommit() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(resolve));
  });
}

async function startPreviewRender(item) {
  const token = ++previewRenderToken;
  previewRendering = true;
  item.renderError = null;
  appendLog(`Sample ${index + 1}: rendering preview…`);
  syncControls(item);
  try {
    if (item.dirty && item.dslDiagnostics?.valid !== true) {
      throw new Error(
        item.dslDiagnostics?.errors?.[0] || "OpenUI must be valid before it can be previewed"
      );
    }
    updatePreview(item);
    await afterPreviewCommit();
    if (token !== previewRenderToken || current() !== item) return;
    const root = previewEl.querySelector(".openui-preview-root");
    if (!root) throw new Error("preview did not mount");
    if (root.getAttribute("data-parse-ok") === "0") {
      throw new Error(root.textContent?.trim() || "preview parse failed");
    }
    renderedItem = item;
    previewRendering = false;
    appendLog(`Sample ${index + 1}: preview rendered.`, "success");
  } catch (err) {
    if (token !== previewRenderToken || current() !== item) return;
    item.renderError = err?.message || String(err);
    renderedItem = item;
    previewRendering = false;
    appendLog(`Sample ${index + 1}: preview render failed — ${item.renderError}`, "error");
  }
  render();
}

function render() {
  const item = current();
  if (item?.status === "ready" && item.valid && !item.dslDiagnostics) {
    item.dslDiagnostics = { ...lintOpenUI(displayedOpenUI(item)), pending: false };
  }
  renderModelSource(item);
  const needsPreview =
    activeView === "render" &&
    !!item &&
    item.status === "ready" &&
    item.valid &&
    (!item.dirty || item.dslDiagnostics?.valid === true) &&
    renderedItem !== item &&
    !previewRendering;
  const previewBlocked =
    activeView === "render" &&
    !!item?.dirty &&
    item.dslDiagnostics?.valid !== true;
  if (previewBlocked && renderedItem !== item) {
    window.OpenUIPreview?.unmount?.(previewEl);
    const message = document.createElement("p");
    message.className = "openui-preview-empty";
    message.textContent = item.dslDiagnostics?.pending
      ? "Checking the corrected OpenUI before preview…"
      : "Preview blocked until the OpenUI errors are fixed.";
    previewEl.replaceChildren(message);
    renderedItem = item;
  }
  if (needsPreview) void startPreviewRender(item);
  syncControls(item);
  indexPillEl.textContent = `${Math.min(index + 1, Math.max(stack.length, 1))} / ${Math.max(stack.length, 1)}`;
  if (!item) {
    promptTextEl.textContent = "Loading…";
    setOutput("// waiting for sample");
    badgeEl.textContent = "loading";
    badgeEl.className = "badge";
    errorEl.hidden = true;
    if (activeView === "render") updatePreview(null);
    renderDslDiagnostics(null);
    return;
  }
  promptTextEl.textContent = item.prompt;
  setOutput(displayedOpenUI(item) || "// empty");
  renderDslDiagnostics(item);
  noteEl.value = item.note || "";
  if (item.status === "loading" && activeView === "render") updatePreview(item);
  if (item.status === "loading") {
    badgeEl.textContent = "generating";
    badgeEl.className = "badge";
    errorEl.hidden = true;
  } else if (item.dirty && item.dslDiagnostics?.pending) {
    badgeEl.textContent = "checking DSL";
    badgeEl.className = "badge";
    errorEl.hidden = true;
  } else if (item.dirty && item.dslDiagnostics?.valid === false) {
    badgeEl.textContent = "invalid DSL";
    badgeEl.className = "badge bad";
    errorEl.hidden = false;
    errorEl.textContent = item.dslDiagnostics.errors?.[0] || "Fix the OpenUI syntax";
  } else if (item.status === "error" || item.renderError) {
    badgeEl.textContent = "error";
    badgeEl.className = "badge bad";
    errorEl.hidden = false;
    errorEl.textContent = item.renderError || item.error || "Generation failed";
  } else if (previewRendering) {
    badgeEl.textContent = "rendering";
    badgeEl.className = "badge";
    errorEl.hidden = true;
  } else if (item.valid) {
    badgeEl.textContent = "valid";
    badgeEl.className = "badge ok";
    errorEl.hidden = true;
  } else {
    badgeEl.textContent = "invalid";
    badgeEl.className = "badge bad";
    errorEl.hidden = false;
    errorEl.textContent = item.error || "Validation failed";
  }
}

async function fetchServerAttempt(prompt, attempt, priorFailures, requestIdentity = null) {
  const design_md = (designMdEl?.value || "").trim();
  const body = {
    session_id: sessionId(),
    grammar_constrained: !!grammarEl?.checked,
    design_md: design_md || null,
    auto_prompt: !prompt,
    attempt,
    prior_failures: priorFailures,
    request_identity: requestIdentity,
  };
  if (prompt) body.prompt = prompt;
  const res = await fetch("/api/server-attempt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Sample failed");
  }
  return {
    prompt: data.prompt,
    openui: data.openui,
    serialized: data.serialized,
    valid: !!data.valid,
    error: data.error || null,
    status: "loading",
    source: "server",
    attempt,
    attemptRecord: data.attempt,
    generationId: data.attempt?.id || null,
    identities: data.identities || data.attempt?.identities || {},
    note: "",
  };
}

async function persistBrowserReview(candidate, judgement, priorFailures) {
  const res = await fetch("/api/generation-review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      generation_id: candidate.attemptRecord.id,
      prompt: candidate.prompt,
      openui: candidate.serialized || candidate.openui,
      attempt: candidate.attempt,
      passed: judgement.passed,
      score: judgement.score,
      reasons: judgement.reasons,
      prior_failures: priorFailures,
      session_id: sessionId(),
      meta: {
        runtime: browserModelState.runtime || "unavailable",
        availability: browserModelState.availability || "unknown",
        role: "baseline_gate",
        identities: {
          ...(candidate.identities || {}),
          reviewer: browserModelIdentity("review"),
        },
      },
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Browser review could not be stored");
  return data;
}

async function gradeServerCandidate(candidate, priorFailures) {
  let judgement;
  try {
    const gradePrompt = buildBrowserGradePrompt(
      candidate.prompt,
      candidate.serialized || candidate.openui,
      candidate.attempt
    );
    const response = await withBrowserModelSession(async (session) => {
      try {
        return await session.prompt(gradePrompt, {
          responseConstraint: OPENUI_REVIEW_SCHEMA,
        });
      } catch (error) {
        if (error?.name !== "NotSupportedError") throw error;
        return session.prompt(gradePrompt);
      }
    });
    judgement = parseBrowserGradeResponse(response);
    if (judgement.passed && judgement.score < 0.7) {
      judgement.passed = false;
      judgement.reasons.push(
        `Browser baseline score ${judgement.score.toFixed(2)} is below the 0.70 gate`
      );
    }
  } catch (error) {
    judgement = {
      passed: false,
      score: 0,
      reasons: [`Browser baseline review failed: ${error?.message || String(error)}`],
    };
  }
  return persistBrowserReview(candidate, judgement, priorFailures);
}

function formatAttemptFailure(attempt) {
  const output = String(attempt.openui || "").trim();
  const outputContext = output ? ` Output was: ${output.slice(0, 1200)}` : "";
  return `${attempt.source} attempt ${attempt.attempt} failed: ${
    attempt.error || "unknown failure"
  }.${outputContext}`;
}

async function persistBrowserAttempt({
  prompt,
  openui,
  attempt,
  error,
  priorFailures,
  runtime,
  availability,
  identities,
}) {
  const design_md = (designMdEl?.value || "").trim();
  const res = await fetch("/api/generation-attempt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      openui,
      attempt,
      error: error || null,
      prior_failures: priorFailures,
      design_md: design_md || null,
      session_id: sessionId(),
      meta: {
        runtime: runtime || "unavailable",
        availability: availability || "unknown",
        identities,
      },
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Browser attempt could not be stored for training");
  }
  return data;
}

async function browserFallback(sample, slot) {
  const failureReasons = [...(sample.failureReasons || [])];
  if (!failureReasons.length) {
    for (const attempt of sample.attempts || []) {
      if (!attempt.valid) failureReasons.push(formatAttemptFailure(attempt));
    }
  }
  let lastError = sample.error || "The real model exhausted three attempts";

  appendLog(
    `Sample ${slot + 1}: real model exhausted; switching to browser inference.`,
    "warning"
  );
  try {
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      const priorFailures = [...failureReasons];
      stack[slot] = {
        ...sample,
        openui: "",
        serialized: null,
        valid: false,
        status: "loading",
        source: "browser",
        phase: "browser-generation",
        attempt,
      };
      render();
      appendLog(`Sample ${slot + 1}: browser attempt ${attempt}/3 started.`);
      let openui = "";
      let inferenceError = null;
      const startedAt = performance.now();
      const heartbeat = window.setInterval(() => {
        const elapsed = Math.round((performance.now() - startedAt) / 1000);
        appendLog(
          `Sample ${slot + 1}: browser attempt ${attempt}/3 is still running (${elapsed}s elapsed).`
        );
      }, 5_000);
      try {
        const response = await withBrowserModelSession((session) =>
          session.prompt(buildBrowserRepairPrompt(sample.prompt, priorFailures, attempt))
        );
        openui = cleanOpenUIResponse(response);
      } catch (error) {
        inferenceError = error?.message || String(error);
      } finally {
        window.clearInterval(heartbeat);
      }

      const stored = await persistBrowserAttempt({
        prompt: sample.prompt,
        openui,
        attempt,
        error: inferenceError,
        priorFailures,
        runtime: browserModelState.runtime,
        availability: browserModelState.availability,
        identities: {
          ...(sample.identities || {}),
          output_generator: browserModelIdentity("output"),
        },
      });
      const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
      if (stored.valid) {
        appendLog(
          `Sample ${slot + 1}: browser attempt ${attempt}/3 succeeded after ${elapsed}s; training record ${stored.id}.`,
          "success"
        );
        return {
          prompt: sample.prompt,
          openui: stored.serialized || openui,
          serialized: stored.serialized || openui,
          valid: true,
          error: null,
          status: "ready",
          source: "browser",
          attempt,
          generationId: stored.id,
          identities: {
            ...(sample.identities || {}),
            output_generator: browserModelIdentity("output"),
          },
          originalOpenui: stored.serialized || openui,
          browserApproved: true,
          note: "",
        };
      }
      lastError = stored.error || inferenceError || "Browser output failed validation";
      failureReasons.push(
        `browser attempt ${attempt} failed: ${lastError}.${
          openui ? ` Output was: ${openui.slice(0, 1200)}` : ""
        }`
      );
      appendLog(
        `Sample ${slot + 1}: browser attempt ${attempt}/3 failed — ${lastError}; training record ${stored.id}.`,
        "error"
      );
    }
  } finally {
    // Retain the page-scoped browser model for all later samples.
  }
  return {
    prompt: sample.prompt,
    openui: "",
    serialized: null,
    valid: false,
    error: lastError,
    status: "error",
    source: null,
    note: "",
  };
}

async function trainingModelPipeline(placeholder, slot) {
  const attempts = [];
  const failureReasons = [];
  let prompt = null;
  let requestIdentity = null;
  let identities = {};
  let lastError = "The training model exhausted three attempts";

  try {
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      Object.assign(placeholder, {
        prompt: prompt || "Selecting a request…",
        openui: "",
        serialized: null,
        valid: false,
        error: null,
        status: "loading",
        source: "server",
        phase: "server-generation",
        attempt,
      });
      render();
      appendLog(`Sample ${slot + 1}: training model attempt ${attempt}/3 started.`);
      let candidate;
      try {
        candidate = await fetchServerAttempt(
          prompt,
          attempt,
          failureReasons,
          requestIdentity
        );
      } catch (error) {
        lastError = error?.message || String(error);
        failureReasons.push(`training model attempt ${attempt} request failed: ${lastError}`);
        appendLog(
          `Sample ${slot + 1}: training model attempt ${attempt}/3 request failed — ${lastError}`,
          "error"
        );
        continue;
      }
      prompt = candidate.prompt;
      identities = candidate.identities || identities;
      requestIdentity = identities.request_generator || requestIdentity;
      placeholder.prompt = prompt;
      const attemptRecord = candidate.attemptRecord;
      attempts.push(attemptRecord);
      appendLog(
        `Sample ${slot + 1}: training model attempt ${attempt}/3 ${
          candidate.valid ? "passed lint" : `failed lint — ${candidate.error}`
        }; training record ${attemptRecord.id}.`,
        candidate.valid ? "success" : "error"
      );
      if (!candidate.valid) {
        lastError = candidate.error || "Training-model output failed linting";
        failureReasons.push(formatAttemptFailure(attemptRecord));
        continue;
      }

      Object.assign(placeholder, {
        openui: "",
        serialized: null,
        phase: "browser-review",
      });
      render();
      appendLog(
        `Sample ${slot + 1}: browser baseline grading training attempt ${attempt}/3.`
      );
      const review = await gradeServerCandidate(candidate, failureReasons);
      appendLog(
        `Sample ${slot + 1}: browser baseline ${
          review.passed ? "approved" : "rejected"
        } training attempt ${attempt}/3 (score ${review.score.toFixed(2)}); review record ${
          review.id
        }.`,
        review.passed ? "success" : "warning"
      );
      if (review.passed) {
        return {
          ...candidate,
          openui: candidate.serialized || candidate.openui,
          serialized: candidate.serialized || candidate.openui,
          valid: true,
          error: null,
          status: "ready",
          source: "server",
          phase: "ready",
          originalOpenui: candidate.serialized || candidate.openui,
          identities: {
            ...(candidate.identities || {}),
            reviewer: browserModelIdentity("review"),
          },
          browserApproved: true,
          browserReview: review,
          note: "",
        };
      }
      lastError = review.error || review.reasons.join("; ");
      failureReasons.push(
        `browser baseline rejected training attempt ${attempt} (score ${review.score.toFixed(
          2
        )}): ${review.reasons.join("; ")}. Output was: ${(
          candidate.serialized || candidate.openui
        ).slice(0, 1200)}`
      );
    }
  } finally {
    // Browser inference is preloaded once and shared by every pipeline on this page.
  }

  return browserFallback(
    {
      prompt: prompt || placeholder.prompt,
      openui: "",
      valid: false,
      error: lastError,
      status: "loading",
      source: null,
      attempts,
      failureReasons,
      identities,
      note: "",
    },
    slot
  );
}

async function ensurePrefetch() {
  if (prefetching) return;
  prefetching = true;
  try {
    while (stack.length - index - 1 < PREFETCH) {
      const placeholder = {
        prompt: "…",
        openui: "",
        valid: false,
        status: "loading",
        note: "",
      };
      stack.push(placeholder);
      const slot = stack.length - 1;
      appendLog(`Sample ${slot + 1}: queued for generation.`);
      render();
      let sample;
      const startedAt = performance.now();
      appendLog(
        `Sample ${slot + 1}: training-model pipeline started; browser baseline gate required.`
      );
      try {
        sample = await trainingModelPipeline(placeholder, slot);
      } catch (error) {
        sample = {
          prompt: "Generation request failed",
          openui: "",
          valid: false,
          error: error?.message || String(error),
          status: "error",
          source: null,
          note: "",
        };
      }
      stack[slot] = sample;
      render();
      if (sample.valid) {
        const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
        appendLog(
          `Sample ${slot + 1}: ${sample.source} output ready after ${elapsed}s.`,
          "success"
        );
      } else {
        statusEl.textContent = `All generation attempts failed (${sample.error || "unknown error"})`;
        appendLog(
          `Sample ${slot + 1}: all server and browser attempts exhausted — ${
            sample.error || "unknown error"
          }`,
          "error"
        );
      }
      render();
      if (!sample.valid) break;
    }
  } finally {
    prefetching = false;
    appendLog("Generation queue is idle.");
    render();
  }
}

async function go(delta) {
  if (isCurrentBusy()) return;
  if (current()?.dirty) {
    statusEl.textContent = "Save or discard the correction before browsing";
    return;
  }
  const next = index + delta;
  if (next < 0) {
    statusEl.textContent = "At first sample";
    return;
  }
  if (next >= stack.length) {
    await ensurePrefetch();
  }
  if (next >= stack.length) {
    statusEl.textContent = "No sample ready yet";
    return;
  }
  if (current()) current().note = noteEl.value;
  index = next;
  renderedItem = null;
  previewRenderToken += 1;
  previewRendering = false;
  render();
  void ensurePrefetch();
}

async function persistHumanAnnotation(
  item,
  rating,
  { openui = item.serialized || item.openui, originalOpenui = null, humanCorrected = false } = {}
) {
  const design_md = (designMdEl?.value || "").trim();
  const annotationToken = (annotationTokenEl?.value || "").trim();
  const headers = { "Content-Type": "application/json" };
  if (annotationToken) headers.Authorization = `Bearer ${annotationToken}`;
  const humanIdentity = annotatorIdentity();
  const identities = {
    ...(item.identities || {}),
    annotator: humanIdentity,
  };
  if (humanCorrected) identities.correction_author = humanIdentity;
  const res = await fetch("/api/annotate", {
    method: "POST",
    headers,
    body: JSON.stringify({
      prompt: item.prompt,
      openui,
      rating,
      description: (noteEl.value || "").trim() || null,
      design_md: design_md || null,
      valid: item.valid,
      session_id: sessionId(),
      generation_id: item.generationId || item.attemptRecord?.id || null,
      original_openui: originalOpenui,
      human_corrected: humanCorrected,
      identities,
      meta: {
        source: "annotate_playground",
        generation_source: item.source,
        browser_baseline: item.source === "browser",
        browser_gate_passed: !!item.browserApproved,
        browser_review_id: item.browserReview?.id || null,
        browser_review_score: item.browserReview?.score ?? null,
        usable_for_test_data: rating === "up" && !!item.valid,
        human_corrected: humanCorrected,
        view: activeView,
      },
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Annotate failed");
  return data;
}

function waitForSwipeAnimation() {
  return new Promise((resolve) => window.setTimeout(resolve, 285));
}

async function swipeAway(rating) {
  cardEl.classList.remove("swipe-right", "swipe-left");
  void cardEl.offsetWidth;
  cardEl.classList.add(rating === "up" ? "swipe-right" : "swipe-left");
  await waitForSwipeAnimation();
  cardEl.classList.remove("swipe-right", "swipe-left");
  cardEl.style.transform = "";
  cardEl.style.opacity = "";
}

async function grade(rating) {
  const item = current();
  if (!item || item.status !== "ready" || busyGrade || item.dirty) return;
  busyGrade = true;
  statusEl.textContent = rating === "up" ? "Saving approval…" : "Saving rejection…";
  syncControls(item);
  appendLog(`Saving thumbs ${rating} for sample ${index + 1}…`);
  try {
    const data = await persistHumanAnnotation(item, rating);
    statusEl.textContent =
      rating === "up"
        ? `Saved thumbs up · ${data.id}`
        : `Saved thumbs down · ${data.id}`;
    appendLog(`Saved thumbs ${rating} · ${data.id}`, "success");
    noteEl.value = "";
    item.note = "";
    await swipeAway(rating);
  } catch (err) {
    statusEl.textContent = "";
    errorEl.hidden = false;
    errorEl.textContent = err.message || String(err);
    appendLog(`Annotation failed — ${err.message || String(err)}`, "error");
  } finally {
    busyGrade = false;
    syncControls();
  }
}

async function saveCorrection() {
  const item = current();
  if (
    !item?.dirty ||
    item.status !== "ready" ||
    item.renderError ||
    item.dslDiagnostics?.pending ||
    item.dslDiagnostics?.valid !== true ||
    busyGrade
  ) return;
  const corrected = (item.draftOpenui || "").trim();
  const original = item.originalOpenui || item.serialized || item.openui;
  busyGrade = true;
  statusEl.textContent = "Saving human correction…";
  syncControls(item);
  try {
    const data = await persistHumanAnnotation(item, "up", {
      openui: corrected,
      originalOpenui: original,
      humanCorrected: true,
    });
    item.openui = data.openui || corrected;
    item.serialized = data.openui || corrected;
    item.draftOpenui = null;
    item.dirty = false;
    item.renderError = null;
    item.dslDiagnostics = { ...lintOpenUI(item.openui), pending: false };
    item.humanCorrected = true;
    item.identities = data.identities || item.identities;
    appendLog(`Saved human correction · ${data.id}`, "success");
    statusEl.textContent = `Correction saved · ${data.id}`;
    noteEl.value = "";
    item.note = "";
    await swipeAway("up");
  } catch (err) {
    errorEl.hidden = false;
    errorEl.textContent = err.message || String(err);
    statusEl.textContent = "Correction was not saved";
    appendLog(`Correction failed — ${err.message || String(err)}`, "error");
  } finally {
    busyGrade = false;
    syncControls();
  }
}

function discardCorrection() {
  const item = current();
  if (!item?.dirty || busyGrade) return;
  item.draftOpenui = null;
  item.dirty = false;
  item.renderError = null;
  item.dslDiagnostics = { ...lintOpenUI(displayedOpenUI(item)), pending: false };
  clearTimeout(lintTimer);
  lintToken += 1;
  closeCompletions();
  renderedItem = null;
  previewRenderToken += 1;
  previewRendering = false;
  appendLog(`Discarded correction for sample ${index + 1}.`, "warning");
  statusEl.textContent = "Correction discarded";
  render();
}

function noteFocused() {
  return document.activeElement === noteEl;
}

function textInputFocused() {
  const el = document.activeElement;
  if (!el || el === document.body || el === cardEl) return false;
  const tag = (el.tagName || "").toLowerCase();
  if (tag === "textarea" || tag === "input" || tag === "select") return true;
  if (el.isContentEditable) return true;
  return false;
}

function onKeyDown(event) {
  if (noteFocused()) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (current()) current().note = noteEl.value;
      noteEl.blur();
      cardEl.focus();
      statusEl.textContent = "Note ready · use a grading hotkey";
    } else if (event.key === "Escape") {
      event.preventDefault();
      noteEl.blur();
      cardEl.focus();
    }
    return;
  }
  if (textInputFocused()) {
    return;
  }
  if (isCurrentBusy()) {
    if (
      ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "d", "D", "r", "R"].includes(
        event.key
      ) || event.key.length === 1
    ) {
      event.preventDefault();
    }
    return;
  }
  if (
    current()?.dirty &&
    ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)
  ) {
    event.preventDefault();
    statusEl.textContent = "Save or discard the correction before swiping";
    return;
  }
  if (event.key === "ArrowUp") {
    event.preventDefault();
    void grade("up");
  } else if (event.key === "ArrowDown") {
    event.preventDefault();
    void grade("down");
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    void go(-1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    void go(1);
  } else if (event.key === "Tab") {
    // Only swap views from the card/body chrome — never steal Tab from
    // buttons, view tabs, or other focusable controls.
    const focus = document.activeElement;
    if (focus === cardEl || focus === document.body || focus == null) {
      event.preventDefault();
      setView(activeView === "render" ? "dsl" : "render");
    }
  } else if (event.key === "d" || event.key === "D") {
    event.preventDefault();
    setView("dsl");
  } else if (event.key === "r" || event.key === "R") {
    event.preventDefault();
    setView("render");
  } else if (event.key.length === 1 && !event.metaKey && !event.ctrlKey && !event.altKey) {
    noteEl.focus();
  }
}

let pointerStart = null;
cardEl.addEventListener("pointerdown", (event) => {
  if (
    isCurrentBusy() ||
    current()?.dirty ||
    event.target.closest?.("button, textarea, input, select, a, summary")
  ) {
    return;
  }
  pointerStart = { id: event.pointerId, x: event.clientX, y: event.clientY };
  cardEl.classList.add("is-dragging");
  cardEl.setPointerCapture?.(event.pointerId);
});

cardEl.addEventListener("pointermove", (event) => {
  if (!pointerStart || pointerStart.id !== event.pointerId) return;
  const dx = event.clientX - pointerStart.x;
  const dy = event.clientY - pointerStart.y;
  if (Math.abs(dx) < 8 || Math.abs(dx) < Math.abs(dy)) return;
  event.preventDefault();
  const rotation = Math.max(-7, Math.min(7, dx / 24));
  cardEl.style.transform = `translateX(${dx}px) rotate(${rotation}deg)`;
  cardEl.style.opacity = String(Math.max(0.68, 1 - Math.abs(dx) / 650));
});

function finishPointerSwipe(event) {
  if (!pointerStart || pointerStart.id !== event.pointerId) return;
  const dx = event.clientX - pointerStart.x;
  const dy = event.clientY - pointerStart.y;
  pointerStart = null;
  cardEl.classList.remove("is-dragging");
  cardEl.releasePointerCapture?.(event.pointerId);
  cardEl.style.transform = "";
  cardEl.style.opacity = "";
  if (Math.abs(dx) >= 72 && Math.abs(dx) > Math.abs(dy)) {
    void grade(dx > 0 ? "up" : "down");
  }
}

cardEl.addEventListener("pointerup", finishPointerSwipe);
cardEl.addEventListener("pointercancel", finishPointerSwipe);

btnUp.addEventListener("click", () => void grade("up"));
btnDown.addEventListener("click", () => void grade("down"));
btnPrev.addEventListener("click", () => void go(-1));
btnNext.addEventListener("click", () => void go(1));
btnViewRender.addEventListener("click", () => setView("render"));
btnViewDsl.addEventListener("click", () => setView("dsl"));
btnSaveCorrection.addEventListener("click", () => void saveCorrection());
btnDiscardCorrection.addEventListener("click", discardCorrection);
window.addEventListener("keydown", onKeyDown);

noteEl.addEventListener("input", () => {
  if (current()) current().note = noteEl.value;
});

outputEl.addEventListener("input", () => {
  const item = current();
  if (!item || item.status !== "ready") return;
  const original = item.originalOpenui || item.serialized || item.openui || "";
  item.originalOpenui = original;
  const draft = outputEl.value;
  item.dirty = draft.trim() !== original.trim();
  item.draftOpenui = item.dirty ? draft : null;
  item.renderError = null;
  renderedItem = null;
  syncEditorHighlight();
  scheduleDslValidation(item, draft);
  updateCompletions(false);
  statusEl.textContent = item.dirty
    ? "Correction drafted · valid OpenUI can be previewed"
    : "Correction cleared";
  syncControls(item);
});

outputEl.addEventListener("scroll", syncEditorHighlight);
outputEl.addEventListener("click", () => updateCompletions(false));
outputEl.addEventListener("blur", () => setTimeout(closeCompletions, 100));
outputEl.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.code === "Space") {
    event.preventDefault();
    updateCompletions(true);
    return;
  }
  if (!completions.length) return;
  if (event.key === "ArrowDown") {
    event.preventDefault();
    selectCompletion(completionIndex + 1);
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    selectCompletion(completionIndex - 1);
  } else if (event.key === "Enter" || event.key === "Tab") {
    event.preventDefault();
    acceptCompletion();
  } else if (event.key === "Escape") {
    event.preventDefault();
    closeCompletions();
  }
});

async function boot() {
  appendLog("Playground starting.");
  void preloadBrowserModel();
  setView(activeView);
  statusEl.textContent = "Loading renderer…";
  appendLog("Loading OpenUI renderer…");
  try {
    await waitForPreviewApi();
    appendLog("OpenUI renderer ready.", "success");
  } catch (err) {
    statusEl.textContent = err.message || String(err);
    appendLog(`Renderer error — ${err.message || String(err)}`, "error");
  }
  statusEl.textContent = "Prefetching samples…";
  appendLog("Starting sample prefetch.");
  render();
  await ensurePrefetch();
  const ready = current()?.status === "ready" && current()?.valid;
  if (ready) {
    statusEl.textContent = "Ready · swipe or use thumbs · arrows browse · Tab changes view";
  } else if (!statusEl.textContent || statusEl.textContent.startsWith("Prefetching")) {
    statusEl.textContent = "Waiting for valid sample…";
  }
  cardEl.focus();
}

keepPlaceholdersEl?.addEventListener("change", () => {
  renderedItem = null;
  render();
});

annotationTokenEl?.addEventListener("change", () => {
  const token = annotationTokenEl.value.trim();
  if (token) sessionStorage.setItem(ANNOTATION_TOKEN_KEY, token);
  else sessionStorage.removeItem(ANNOTATION_TOKEN_KEY);
});

annotatorIdentityEl?.addEventListener("change", () => {
  const id = annotatorIdentityEl.value.trim() || sessionId();
  annotatorIdentityEl.value = id;
  localStorage.setItem(ANNOTATOR_IDENTITY_KEY, id);
  appendLog(`Annotator identity set to ${id}.`, "success");
});

window.addEventListener("pagehide", () => {
  clearTimeout(lintTimer);
  lintToken += 1;
  window.OpenUIPreview?.unmount?.(dslLintMountEl);
  browserModelState.session?.destroy?.();
  browserModelState.session = null;
  browserModelState.promise = null;
  browserModelState.error = null;
});

window.addEventListener("pageshow", (event) => {
  if (event.persisted) void preloadBrowserModel();
});

boot();
