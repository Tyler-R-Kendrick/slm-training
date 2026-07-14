import React, { useEffect, useReducer, useRef, useState } from "react";
import { useCaps } from "../caps";

// Faithful React port of the classic annotate playground (web/static/app.js):
// prefetch two valid samples ahead, grade up/down (stays on sample), keyboard +
// swipe, Rendered/DSL toggle, live OpenUI preview. The mutable sample stack is
// held in refs (as the imperative original does) so the async prefetch loop
// never reads stale state; a reducer tick drives re-renders.

const PREFETCH = 2;
const SESSION_KEY = "twotower_annotate_session";
const VIEW_KEY = "twotower_annotate_view";

interface Sample {
  prompt: string;
  openui: string;
  serialized?: string | null;
  valid: boolean;
  error?: string | null;
  status: "loading" | "ready" | "error";
  note: string;
}

function sessionId(): string {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = `s_${Math.random().toString(36).slice(2, 10)}_${Date.now().toString(36)}`;
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

function ensurePreviewLib(): Promise<any> {
  const w = window as any;
  if (w.OpenUIPreview?.mount) return Promise.resolve(w.OpenUIPreview);
  if (!document.getElementById("openui-preview-css")) {
    const link = document.createElement("link");
    link.id = "openui-preview-css";
    link.rel = "stylesheet";
    link.href = "/static/preview/preview.css";
    document.head.appendChild(link);
  }
  if (!document.getElementById("openui-preview-lib")) {
    const s = document.createElement("script");
    s.id = "openui-preview-lib";
    s.type = "module";
    s.src = "/static/preview/preview.js";
    document.head.appendChild(s);
  }
  return new Promise((resolve, reject) => {
    const t0 = performance.now();
    const tick = () => {
      if (w.OpenUIPreview?.mount) return resolve(w.OpenUIPreview);
      if (performance.now() - t0 > 15000) return reject(new Error("OpenUI preview bundle failed to load"));
      requestAnimationFrame(tick);
    };
    tick();
  });
}

export function Playground() {
  const caps = useCaps();
  const stackRef = useRef<Sample[]>([]);
  const idxRef = useRef(0);
  const busyRef = useRef(false);
  const prefetchingRef = useRef(false);
  const [, forceRender] = useReducer((x) => x + 1, 0);
  const [view, setView] = useState<"render" | "dsl">(
    localStorage.getItem(VIEW_KEY) === "dsl" ? "dsl" : "render"
  );
  const [status, setStatus] = useState("Loading renderer…");
  const [flash, setFlash] = useState<"" | "up" | "down">("");

  const noteRef = useRef<HTMLTextAreaElement>(null);
  const designRef = useRef<HTMLTextAreaElement>(null);
  const grammarRef = useRef<HTMLInputElement>(null);
  const keepRef = useRef<HTMLInputElement>(null);
  const previewRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);

  const current = () => stackRef.current[idxRef.current] || null;

  function updatePreview() {
    const el = previewRef.current;
    if (!el || view !== "render") return;
    const api = (window as any).OpenUIPreview;
    if (!api?.mount) {
      el.innerHTML = '<p class="openui-preview-empty">Loading renderer…</p>';
      return;
    }
    const item = current();
    if (!item || item.status === "loading") {
      api.mount(el, { source: null });
      return;
    }
    try {
      api.mount(el, { source: item.serialized || item.openui || "", keepPlaceholders: !!keepRef.current?.checked });
    } catch (e: any) {
      el.innerHTML = `<p class="openui-preview-empty">Render error: ${e?.message || e}</p>`;
    }
  }

  async function fetchSample(prompt: string | null = null): Promise<Sample> {
    const design_md = (designRef.current?.value || "").trim();
    const body: any = {
      session_id: sessionId(),
      grammar_constrained: !!grammarRef.current?.checked,
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
    if (!res.ok) throw new Error(data.detail || "Sample failed");
    return {
      prompt: data.prompt,
      openui: data.openui,
      serialized: data.serialized,
      valid: !!data.valid,
      error: data.error || null,
      status: data.valid ? "ready" : "error",
      note: "",
    };
  }

  async function ensurePrefetch() {
    if (prefetchingRef.current) return;
    prefetchingRef.current = true;
    try {
      while (stackRef.current.length - idxRef.current - 1 < PREFETCH) {
        stackRef.current = [...stackRef.current, { prompt: "…", openui: "", valid: false, status: "loading", note: "" }];
        const slot = stackRef.current.length - 1;
        forceRender();
        let sample: Sample | null = null;
        let lastErr: string | null = null;
        for (let t = 0; t < 6; t += 1) {
          try {
            const fetched = await fetchSample(null);
            if (fetched.valid && fetched.status === "ready") {
              sample = fetched;
              break;
            }
            lastErr = fetched.error || "Invalid OpenUI (discarded)";
          } catch (err: any) {
            lastErr = err?.message || String(err);
          }
        }
        if (sample) {
          const next = [...stackRef.current];
          next[slot] = sample;
          stackRef.current = next;
        } else {
          stackRef.current = stackRef.current.slice(0, -1);
          if (idxRef.current >= stackRef.current.length && stackRef.current.length > 0) {
            idxRef.current = stackRef.current.length - 1;
          }
          setStatus(lastErr ? `Waiting for valid sample (${lastErr})` : "Waiting for valid sample…");
          break;
        }
        forceRender();
      }
    } finally {
      prefetchingRef.current = false;
    }
  }

  async function go(delta: number) {
    const next = idxRef.current + delta;
    if (next < 0) return setStatus("At first sample");
    if (next >= stackRef.current.length) await ensurePrefetch();
    if (next >= stackRef.current.length) return setStatus("No sample ready yet");
    const cur = current();
    if (cur && noteRef.current) cur.note = noteRef.current.value;
    idxRef.current = next;
    forceRender();
    void ensurePrefetch();
  }

  async function grade(rating: "up" | "down") {
    const item = current();
    if (!item || item.status !== "ready" || busyRef.current) return;
    busyRef.current = true;
    setStatus(rating === "up" ? "Saving 👍…" : "Saving 👎…");
    try {
      const design_md = (designRef.current?.value || "").trim();
      const res = await fetch("/api/annotate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: item.prompt,
          openui: item.openui,
          rating,
          description: (noteRef.current?.value || "").trim() || null,
          design_md: design_md || null,
          valid: item.valid,
          session_id: sessionId(),
          meta: { source: "annotate_playground", usable_for_test_data: rating === "up" && !!item.valid, view },
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Annotate failed");
      setStatus(rating === "up" ? `Saved thumbs up · ${data.id}` : `Saved thumbs down · ${data.id}`);
      setFlash("");
      requestAnimationFrame(() => setFlash(rating));
      if (noteRef.current) noteRef.current.value = "";
      item.note = "";
    } catch (err: any) {
      setStatus(err?.message || String(err));
    } finally {
      busyRef.current = false;
    }
  }

  // boot: load preview lib, prefetch, focus.
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        await ensurePreviewLib();
      } catch (e: any) {
        if (alive) setStatus(e?.message || String(e));
      }
      if (!alive) return;
      setStatus("Prefetching samples…");
      await ensurePrefetch();
      if (!alive) return;
      const ready = current()?.status === "ready" && current()?.valid;
      if (ready) setStatus("Ready · 👍/👎 grade · ←/→ navigate · Tab view · type to note");
      cardRef.current?.focus();
    })();
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // re-render the preview when the sample or view changes.
  useEffect(() => {
    updatePreview();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  });

  // keyboard
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const active = document.activeElement;
      if (active === noteRef.current) {
        if (e.key === "Escape") {
          e.preventDefault();
          noteRef.current?.blur();
          cardRef.current?.focus();
        }
        return;
      }
      const tag = (active?.tagName || "").toLowerCase();
      if (active && active !== document.body && active !== cardRef.current && ["textarea", "input", "select"].includes(tag)) return;
      if (e.key === "ArrowUp") { e.preventDefault(); void grade("up"); }
      else if (e.key === "ArrowDown") { e.preventDefault(); void grade("down"); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); void go(-1); }
      else if (e.key === "ArrowRight") { e.preventDefault(); void go(1); }
      else if (e.key === "Tab") {
        if (active === cardRef.current || active === document.body || active == null) {
          e.preventDefault();
          switchView(view === "render" ? "dsl" : "render");
        }
      } else if (e.key === "d" || e.key === "D") { e.preventDefault(); switchView("dsl"); }
      else if (e.key === "r" || e.key === "R") { e.preventDefault(); switchView("render"); }
      else if (e.key.length === 1 && !e.metaKey && !e.ctrlKey && !e.altKey) { noteRef.current?.focus(); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view]);

  function switchView(v: "render" | "dsl") {
    setView(v);
    localStorage.setItem(VIEW_KEY, v);
  }

  // swipe
  const touch = useRef<{ x: number; y: number } | null>(null);
  function onTouchStart(e: React.TouchEvent) {
    const t = e.changedTouches[0];
    touch.current = { x: t.clientX, y: t.clientY };
  }
  function onTouchEnd(e: React.TouchEvent) {
    if (!touch.current) return;
    const t = e.changedTouches[0];
    const dx = t.clientX - touch.current.x;
    const dy = t.clientY - touch.current.y;
    touch.current = null;
    if (Math.abs(dx) < 48 && Math.abs(dy) < 48) return;
    if (Math.abs(dx) > Math.abs(dy)) void go(dx < 0 ? 1 : -1);
    else void grade(dy < 0 ? "up" : "down");
  }

  const item = current();
  const badge = !item ? "loading" : item.status === "loading" ? "generating" : item.status === "error" ? "error" : item.valid ? "valid" : "invalid";
  const total = Math.max(stackRef.current.length, 1);
  const posn = Math.min(idxRef.current + 1, total);

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Playground</h1>
        <p className="page-sub">
          Grade OpenUI samples — 👍 good · 👎 bad · ←/→ navigate · Tab toggles view · type to note.
          Feedback flows to <span className="mono">outputs/annotations/</span>.
          Classic page: <a className="runlink" href="/playground/classic">/playground/classic</a>.
        </p>
      </div>

      <section
        className={`card pg-card ${flash ? `flash-${flash}` : ""}`}
        ref={cardRef}
        tabIndex={0}
        onTouchStart={onTouchStart}
        onTouchEnd={onTouchEnd}
        aria-label="Annotation sample"
      >
        <div className="pg-meta">
          <span className="mono" style={{ color: "var(--text-mute)" }}>{posn} / {total}</span>
          <span className={`pill pill-${badge === "valid" ? "passed" : badge === "invalid" || badge === "error" ? "failed" : "idle"}`}>{badge}</span>
        </div>

        <p className="tile-label">Request</p>
        <p className="pg-prompt">{item ? item.prompt : "Loading…"}</p>

        <div className="view-toggle">
          <button className={`view-btn ${view === "render" ? "is-active" : ""}`} onClick={() => switchView("render")}>Rendered</button>
          <button className={`view-btn ${view === "dsl" ? "is-active" : ""}`} onClick={() => switchView("dsl")}>DSL</button>
        </div>

        {view === "render" ? (
          <div className="openui-preview" ref={previewRef} />
        ) : (
          <pre className="logstream-body" style={{ maxHeight: "42vh" }}>{item?.serialized || item?.openui || "// empty"}</pre>
        )}

        {item?.status === "error" && <p className="error-note">{item.error || "Generation failed"}</p>}

        <label className="tile-label" htmlFor="pg-note">Note (optional)</label>
        <textarea id="pg-note" ref={noteRef} rows={2} className="pg-note" placeholder="Type to annotate… Esc to blur" />

        <div className="pg-grade">
          <button className="btn btn-ember pg-down" onClick={() => void grade("down")}>👎 Down</button>
          <button className="btn pg-nav" onClick={() => void go(-1)}>←</button>
          <button className="btn pg-nav" onClick={() => void go(1)}>→</button>
          <button className="btn btn-primary pg-up" onClick={() => void grade("up")}>Up 👍</button>
        </div>
        <p className="hint" style={{ minHeight: "1.2em" }}>{status}</p>
      </section>

      <details className="card" style={{ marginTop: "1rem" }}>
        <summary style={{ cursor: "pointer", color: "var(--text-dim)" }}>Advanced</summary>
        <label className="tile-label" htmlFor="pg-design" style={{ marginTop: "0.6rem", display: "block" }}>DESIGN.md (optional)</label>
        <textarea id="pg-design" ref={designRef} rows={3} className="pg-note" placeholder="Paste DESIGN.md to condition generation" />
        <label className="thr-metric" style={{ flexDirection: "row", gap: "0.4rem", marginTop: "0.5rem" }}>
          <input type="checkbox" ref={grammarRef} defaultChecked /> Grammar guard
        </label>
        <label className="thr-metric" style={{ flexDirection: "row", gap: "0.4rem" }}>
          <input type="checkbox" ref={keepRef} onChange={updatePreview} /> Keep :placeholders in preview
        </label>
        {!caps.execution && <p className="hint">Note: generation needs a running model server (read-only deploys can't generate).</p>}
      </details>
    </div>
  );
}
