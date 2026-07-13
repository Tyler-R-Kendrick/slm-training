const cardEl = document.getElementById("card");
const promptTextEl = document.getElementById("promptText");
const outputEl = document.getElementById("output");
const errorEl = document.getElementById("error");
const badgeEl = document.getElementById("badge");
const statusEl = document.getElementById("status");
const noteEl = document.getElementById("note");
const designMdEl = document.getElementById("design_md");
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

const PREFETCH = 2;
const SESSION_KEY = "twotower_annotate_session";
const VIEW_KEY = "twotower_annotate_view";

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

/** @type {{prompt:string, openui:string, serialized?:string|null, valid:boolean, error?:string|null, status:"ready"|"loading"|"error", note?:string}[]} */
const stack = [];
let index = 0;
let busyGrade = false;
let prefetching = false;

function current() {
  return stack[index] || null;
}

function setOutput(text) {
  const code = outputEl.querySelector("code") || outputEl;
  code.textContent = text || "";
  outputEl.style.animation = "none";
  void outputEl.offsetWidth;
  outputEl.style.animation = "";
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
  if (showRender) updatePreview(current());
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
  const source = item.serialized || item.openui || "";
  try {
    api.mount(previewEl, {
      source,
      keepPlaceholders: !!keepPlaceholdersEl?.checked,
    });
  } catch (err) {
    previewEl.innerHTML = `<p class="openui-preview-empty">Render error: ${
      err?.message || err
    }</p>`;
  }
}

function render() {
  const item = current();
  indexPillEl.textContent = `${Math.min(index + 1, Math.max(stack.length, 1))} / ${Math.max(stack.length, 1)}`;
  if (!item) {
    promptTextEl.textContent = "Loading…";
    setOutput("// waiting for sample");
    badgeEl.textContent = "loading";
    badgeEl.className = "badge";
    errorEl.hidden = true;
    updatePreview(null);
    return;
  }
  promptTextEl.textContent = item.prompt;
  setOutput(item.serialized || item.openui || "// empty");
  noteEl.value = item.note || "";
  updatePreview(item);
  if (item.status === "loading") {
    badgeEl.textContent = "generating";
    badgeEl.className = "badge";
    errorEl.hidden = true;
  } else if (item.status === "error") {
    badgeEl.textContent = "error";
    badgeEl.className = "badge bad";
    errorEl.hidden = false;
    errorEl.textContent = item.error || "Generation failed";
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

async function fetchSample(prompt = null) {
  const design_md = (designMdEl?.value || "").trim();
  const body = {
    session_id: sessionId(),
    grammar_constrained: !!grammarEl?.checked,
    design_md: design_md || null,
    auto_prompt: !prompt,
  };
  if (prompt) body.prompt = prompt;
  const res = await fetch("/api/sample", {
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
    status: "ready",
    note: "",
  };
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
      render();
      try {
        const sample = await fetchSample(null);
        stack[slot] = sample;
      } catch (err) {
        stack[slot] = {
          prompt: "(generation error)",
          openui: "",
          valid: false,
          error: err.message || String(err),
          status: "error",
          note: "",
        };
      }
      render();
    }
  } finally {
    prefetching = false;
  }
}

async function go(delta) {
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
  render();
  void ensurePrefetch();
}

async function grade(rating) {
  const item = current();
  if (!item || item.status !== "ready" || busyGrade) return;
  busyGrade = true;
  statusEl.textContent = rating === "up" ? "Saving ↑…" : "Saving ↓…";
  btnUp.disabled = true;
  btnDown.disabled = true;
  try {
    const design_md = (designMdEl?.value || "").trim();
    const res = await fetch("/api/annotate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: item.prompt,
        openui: item.openui,
        rating,
        description: (noteEl.value || "").trim() || null,
        design_md: design_md || null,
        valid: item.valid,
        session_id: sessionId(),
        meta: {
          source: "annotate_playground",
          usable_for_test_data: true,
          view: activeView,
        },
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Annotate failed");
    statusEl.textContent =
      rating === "up"
        ? `Saved thumbs up · ${data.id}`
        : `Saved thumbs down · ${data.id}`;
    cardEl.classList.remove("flash-up", "flash-down");
    void cardEl.offsetWidth;
    cardEl.classList.add(rating === "up" ? "flash-up" : "flash-down");
    noteEl.value = "";
    if (current()) current().note = "";
    await go(1);
  } catch (err) {
    statusEl.textContent = "";
    errorEl.hidden = false;
    errorEl.textContent = err.message || String(err);
  } finally {
    busyGrade = false;
    btnUp.disabled = false;
    btnDown.disabled = false;
  }
}

function noteFocused() {
  return document.activeElement === noteEl;
}

function onKeyDown(event) {
  if (noteFocused()) {
    if (event.key === "Escape") {
      event.preventDefault();
      noteEl.blur();
      cardEl.focus();
    }
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

let touchX = null;
let touchY = null;
cardEl.addEventListener(
  "touchstart",
  (e) => {
    const t = e.changedTouches[0];
    touchX = t.clientX;
    touchY = t.clientY;
  },
  { passive: true }
);
cardEl.addEventListener(
  "touchend",
  (e) => {
    if (touchX == null || touchY == null) return;
    const t = e.changedTouches[0];
    const dx = t.clientX - touchX;
    const dy = t.clientY - touchY;
    touchX = touchY = null;
    if (Math.abs(dx) < 48 && Math.abs(dy) < 48) return;
    if (Math.abs(dx) > Math.abs(dy)) {
      if (dx < 0) void go(1);
      else void go(-1);
    } else {
      if (dy < 0) void grade("up");
      else void grade("down");
    }
  },
  { passive: true }
);

btnUp.addEventListener("click", () => void grade("up"));
btnDown.addEventListener("click", () => void grade("down"));
btnPrev.addEventListener("click", () => void go(-1));
btnNext.addEventListener("click", () => void go(1));
btnViewRender.addEventListener("click", () => setView("render"));
btnViewDsl.addEventListener("click", () => setView("dsl"));
window.addEventListener("keydown", onKeyDown);

noteEl.addEventListener("input", () => {
  if (current()) current().note = noteEl.value;
});

async function boot() {
  setView(activeView);
  statusEl.textContent = "Loading renderer…";
  try {
    await waitForPreviewApi();
  } catch (err) {
    statusEl.textContent = err.message || String(err);
  }
  statusEl.textContent = "Prefetching samples…";
  render();
  await ensurePrefetch();
  statusEl.textContent = "Ready · ↑/↓ grade · ←/→ navigate · R/D view · type to note";
  cardEl.focus();
}

keepPlaceholdersEl?.addEventListener("change", () => updatePreview(current()));

boot();
