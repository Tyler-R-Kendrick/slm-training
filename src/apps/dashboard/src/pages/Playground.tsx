import React, { useEffect, useReducer, useRef, useState } from "react";
import { useCaps } from "../caps";

const PREFETCH = 2;
const MAX_LOG_ENTRIES = 80;
const SESSION_KEY = "twotower_annotate_session";
const VIEW_KEY = "twotower_annotate_view";
const ANNOTATION_TOKEN_KEY = "twotower_annotation_token";
const ANNOTATOR_IDENTITY_KEY = "twotower_annotator_identity";
const DIFFUSION_GLYPHS = ["·", "░", "▒", "▓", "#", "∷", "□", "◇"];

type View = "render" | "dsl";
type Source = "server" | "browser";

interface Diagnostics {
  valid: boolean;
  pending: boolean;
  errors: string[];
  warnings: string[];
}

interface Sample {
  prompt: string;
  openui: string;
  serialized?: string | null;
  originalOpenui?: string;
  draftOpenui?: string | null;
  dirty?: boolean;
  valid: boolean;
  error?: string | null;
  renderError?: string | null;
  status: "loading" | "ready" | "error";
  source?: Source | null;
  phase?: "server-generation" | "browser-review" | "browser-generation" | "ready";
  attempt?: number;
  attemptRecord?: any;
  generationId?: string | null;
  identities?: Record<string, any>;
  browserApproved?: boolean;
  browserReview?: any;
  attempts?: any[];
  failureReasons?: string[];
  dslDiagnostics?: Diagnostics;
  note: string;
}

interface LogEntry {
  id: number;
  time: string;
  message: string;
  level: "info" | "success" | "warning" | "error";
}

function DiffusionCanvas() {
  const [phase, setPhase] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => setPhase((value) => value + 1), 180);
    return () => window.clearInterval(timer);
  }, []);
  return (
    <div className="diffusion-canvas" role="status" aria-label="Diffusion pass in progress">
      <div className="diffusion-head">
        <span className="mono">DIFFUSION PASS</span>
        <span className="diffusion-pulse" aria-hidden="true" />
        <span className="hint">resolving changing blocks</span>
      </div>
      <div className="diffusion-field" aria-hidden="true">
        {Array.from({ length: 48 }, (_, index) => {
          const age = (phase + index * 3) % 17;
          const glyph = DIFFUSION_GLYPHS[(phase + index * 5) % DIFFUSION_GLYPHS.length];
          return <span className={`diffusion-token token-age-${Math.min(age, 5)}`} key={index}>{glyph}</span>;
        })}
      </div>
      <p className="hint diffusion-caption">The layout is taking shape; unstable tokens stay visibly provisional until the pass settles.</p>
    </div>
  );
}

function sessionId(): string {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = `s_${Math.random().toString(36).slice(2, 10)}_${Date.now().toString(36)}`;
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

function defaultAnnotator(): string {
  let id = localStorage.getItem(ANNOTATOR_IDENTITY_KEY);
  if (!id) {
    id = `annotator-${Math.random().toString(36).slice(2, 8)}`;
    localStorage.setItem(ANNOTATOR_IDENTITY_KEY, id);
  }
  return id;
}

function displayedOpenUI(item: Sample | null): string {
  if (!item) return "";
  return item.dirty ? item.draftOpenui || "" : item.serialized || item.openui || "";
}

function waitForPreviewApi(timeoutMs = 15_000): Promise<any> {
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
    const script = document.createElement("script");
    script.id = "openui-preview-lib";
    script.type = "module";
    script.src = "/static/preview/preview.js";
    document.head.appendChild(script);
  }
  return new Promise((resolve, reject) => {
    const started = performance.now();
    const tick = () => {
      if (w.OpenUIPreview?.mount) return resolve(w.OpenUIPreview);
      if (performance.now() - started > timeoutMs) return reject(new Error("OpenUI preview bundle failed to load"));
      requestAnimationFrame(tick);
    };
    tick();
  });
}

const browserModuleUrl = "/static/browser_inference.js?v=20260713-4";
const editorModuleUrl = "/static/openui_editor.js?v=20260713-1";
let browserModulePromise: Promise<any> | null = null;
let editorModulePromise: Promise<any> | null = null;
const loadBrowserModule = () => browserModulePromise ||= import(/* @vite-ignore */ browserModuleUrl);
const loadEditorModule = () => editorModulePromise ||= import(/* @vite-ignore */ editorModuleUrl);

async function responseJSON(res: Response, fallback: string): Promise<any> {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${fallback} (${res.status})`);
  return data;
}

function abortError(): DOMException {
  return new DOMException("Playground generation cancelled", "AbortError");
}

function isAbortError(error: any): boolean {
  return error?.name === "AbortError";
}

const afterPreviewCommit = () => new Promise<void>((resolve) => {
  requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
});

export function Playground() {
  const caps = useCaps();
  const stackRef = useRef<Sample[]>([]);
  const indexRef = useRef(0);
  const busyGradeRef = useRef(false);
  const prefetchingRef = useRef(false);
  const prefetchOwnerRef = useRef<AbortSignal | null>(null);
  const activeControllerRef = useRef<AbortController | null>(null);
  const previewRenderingRef = useRef(false);
  const previewRenderTokenRef = useRef(0);
  const renderedItemRef = useRef<Sample | null>(null);
  const lintTimerRef = useRef<number | null>(null);
  const lintTokenRef = useRef(0);
  const logIdRef = useRef(0);
  const browserRef = useRef<any>({ session: null, promise: null, runtime: null, availability: null, error: null, waiters: new Set<AbortSignal>() });
  const editorRef = useRef<any>(null);
  const [, forceRender] = useReducer((value) => value + 1, 0);

  const [view, setView] = useState<View>(() => localStorage.getItem(VIEW_KEY) === "dsl" ? "dsl" : "render");
  const viewRef = useRef<View>(view);
  const [status, setStatus] = useState("Loading renderer…");
  const [uiError, setUiError] = useState("");
  const [flash, setFlash] = useState<"" | "up" | "down">("");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [note, setNote] = useState("");
  const [designMd, setDesignMd] = useState("");
  const [grammar, setGrammar] = useState(true);
  const [keepPlaceholders, setKeepPlaceholders] = useState(false);
  const [annotationToken, setAnnotationToken] = useState(() => sessionStorage.getItem(ANNOTATION_TOKEN_KEY) || "");
  const [annotator, setAnnotator] = useState(defaultAnnotator);
  const [completions, setCompletions] = useState<any[]>([]);
  const [completionIndex, setCompletionIndex] = useState(0);

  const previewRef = useRef<HTMLDivElement>(null);
  const lintMountRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLElement>(null);
  const outputRef = useRef<HTMLTextAreaElement>(null);
  const highlightRef = useRef<HTMLPreElement>(null);
  const completionRef = useRef<HTMLDivElement>(null);
  const pointerRef = useRef<{ id: number; x: number; y: number } | null>(null);
  const grammarValueRef = useRef(grammar);
  const designMdValueRef = useRef(designMd);
  const annotationTokenValueRef = useRef(annotationToken);
  const annotatorValueRef = useRef(annotator);
  const noteValueRef = useRef(note);
  grammarValueRef.current = grammar;
  designMdValueRef.current = designMd;
  annotationTokenValueRef.current = annotationToken;
  annotatorValueRef.current = annotator;
  noteValueRef.current = note;

  const current = () => stackRef.current[indexRef.current] || null;

  function appendLog(message: string, level: LogEntry["level"] = "info") {
    const entry: LogEntry = {
      id: ++logIdRef.current,
      time: new Intl.DateTimeFormat([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date()),
      message,
      level,
    };
    setLogs((existing) => [...existing, entry].slice(-MAX_LOG_ENTRIES));
  }

  function browserModelIdentity(role = "output") {
    const runtime = browserRef.current.runtime || "browser";
    const transformers = runtime.startsWith("transformers-js:");
    return {
      kind: "model",
      provider: transformers ? "huggingface-transformers-js" : "browser-built-in-ai",
      id: transformers ? "onnx-community/gemma-3-270m-it-ONNX" : "prompt-api-default",
      model: transformers ? "gemma-3-270m-it-ONNX" : role === "review" ? "prompt-api-baseline-reviewer" : "prompt-api-default",
      runtime,
    };
  }

  function annotatorIdentity() {
    const id = annotatorValueRef.current.trim() || sessionId();
    return { kind: "user", provider: "playground-annotator", id, display_name: id };
  }

  async function preloadBrowserModel(signal: AbortSignal) {
    signal.throwIfAborted();
    const state = browserRef.current;
    state.waiters.add(signal);
    try {
      if (state.session) return state.session;
      if (!state.promise) {
        let creation: Promise<any>;
        creation = loadBrowserModule().then(async (browser) => {
          if (![...state.waiters].some((waiter: AbortSignal) => !waiter.aborted)) throw abortError();
          const acceleration = browser.browserAccelerationCapabilities();
          appendLog(`Browser acceleration: ${acceleration.promptApi ? "built-in Prompt API" : acceleration.webnn ? "WebNN/NPU preferred" : acceleration.webgpu ? "WebGPU preferred" : "WASM fallback"}.`);
          appendLog(`Local compute: ${acceleration.hardwareConcurrency} logical cores; ${acceleration.wasmThreads} WASM thread${acceleration.wasmThreads === 1 ? "" : "s"}; ${acceleration.crossOriginIsolated ? "cross-origin isolated" : "single-thread compatibility mode"}.`);
          appendLog("Initializing browser baseline model once for this page…");
          return browser.createBrowserModelSession({
            mode: "shared",
            onProgress(progress: any) {
              if (![...state.waiters].some((waiter: AbortSignal) => !waiter.aborted)) return;
              const previous = state.availability;
              state.availability = progress.status;
              if (progress.status === "downloading" && previous !== "downloading") appendLog("Downloading and caching browser baseline model assets…");
              else if (progress.status !== previous) appendLog(`Browser baseline availability: ${progress.status}.`);
            },
          });
        }).then((created: any) => {
          const active = [...state.waiters].some((waiter: AbortSignal) => !waiter.aborted);
          if (!active || state.promise !== creation) {
            created.session?.destroy?.();
            throw abortError();
          }
          state.session = created.session;
          state.runtime = created.runtime;
          state.availability = created.availability;
          state.promise = null;
          appendLog(`Browser baseline ready (${created.runtime}); it will be reused for every sample.`, "success");
          return created.session;
        }).catch((error: any) => {
          if (state.promise === creation) state.promise = null;
          if (!isAbortError(error)) {
            state.error = error;
            if ([...state.waiters].some((waiter: AbortSignal) => !waiter.aborted)) appendLog(`Browser baseline initialization failed — ${error?.message || String(error)}`, "error");
          }
          throw error;
        });
        state.promise = creation;
      }
      const session = await state.promise;
      signal.throwIfAborted();
      return session;
    } finally {
      state.waiters.delete(signal);
    }
  }

  async function withBrowserModelSession(callback: (session: any) => Promise<any>, signal: AbortSignal) {
    signal.throwIfAborted();
    const state = browserRef.current;
    const base = state.session || await preloadBrowserModel(signal);
    signal.throwIfAborted();
    if (!base) throw state.error || new Error("Browser baseline model is unavailable");
    const session = base.clone ? await base.clone() : base;
    if (signal.aborted) {
      if (session !== base) session.destroy?.();
      throw abortError();
    }
    try {
      const result = await callback(session);
      signal.throwIfAborted();
      return result;
    } finally {
      if (session !== base) session.destroy?.();
    }
  }

  async function fetchServerAttempt(prompt: string | null, attempt: number, priorFailures: string[], requestIdentity: any, signal: AbortSignal) {
    signal.throwIfAborted();
    const body: any = {
      session_id: sessionId(), grammar_constrained: grammarValueRef.current, design_md: designMdValueRef.current.trim() || null,
      auto_prompt: !prompt, attempt, prior_failures: priorFailures, request_identity: requestIdentity,
    };
    if (prompt) body.prompt = prompt;
    const res = await fetch("/api/server-attempt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal });
    const data = await responseJSON(res, "Server attempt failed");
    signal.throwIfAborted();
    return {
      prompt: data.prompt, openui: data.openui, serialized: data.serialized, valid: !!data.valid,
      error: data.error || null, status: "loading" as const, source: "server" as const, attempt,
      attemptRecord: data.attempt, generationId: data.attempt?.id || null,
      identities: data.identities || data.attempt?.identities || {}, note: "",
    };
  }

  async function persistBrowserReview(candidate: Sample, judgement: any, priorFailures: string[], signal: AbortSignal) {
    signal.throwIfAborted();
    const res = await fetch("/api/generation-review", {
      method: "POST", headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({
        generation_id: candidate.attemptRecord?.id || candidate.generationId,
        prompt: candidate.prompt, openui: candidate.serialized || candidate.openui,
        attempt: candidate.attempt, passed: judgement.passed, score: judgement.score,
        reasons: judgement.reasons, prior_failures: priorFailures, session_id: sessionId(),
        meta: { runtime: browserRef.current.runtime || "unavailable", availability: browserRef.current.availability || "unknown", role: "baseline_gate", identities: { ...(candidate.identities || {}), reviewer: browserModelIdentity("review") } },
      }),
    });
    const data = await responseJSON(res, "Browser review could not be stored");
    signal.throwIfAborted();
    return data;
  }

  async function gradeServerCandidate(candidate: Sample, priorFailures: string[], signal: AbortSignal) {
    signal.throwIfAborted();
    const browser = await loadBrowserModule();
    signal.throwIfAborted();
    let judgement: any;
    try {
      const gradePrompt = browser.buildBrowserGradePrompt(candidate.prompt, candidate.serialized || candidate.openui, candidate.attempt);
      const response = await withBrowserModelSession(async (session) => {
        try {
          return await session.prompt(gradePrompt, { responseConstraint: browser.OPENUI_REVIEW_SCHEMA });
        } catch (error: any) {
          if (error?.name !== "NotSupportedError") throw error;
          return session.prompt(gradePrompt);
        }
      }, signal);
      signal.throwIfAborted();
      judgement = browser.parseBrowserGradeResponse(response);
      if (judgement.passed && judgement.score < 0.7) {
        judgement.passed = false;
        judgement.reasons.push(`Browser baseline score ${judgement.score.toFixed(2)} is below the 0.70 gate`);
      }
    } catch (error: any) {
      if (isAbortError(error)) throw error;
      judgement = { passed: false, score: 0, reasons: [`Browser baseline review failed: ${error?.message || String(error)}`] };
    }
    return persistBrowserReview(candidate, judgement, priorFailures, signal);
  }

  function formatAttemptFailure(attempt: any) {
    const output = String(attempt?.openui || "").trim();
    return `${attempt?.source || "server"} attempt ${attempt?.attempt || "?"} failed: ${attempt?.error || "unknown failure"}.${output ? ` Output was: ${output.slice(0, 1200)}` : ""}`;
  }

  async function persistBrowserAttempt(args: any, signal: AbortSignal) {
    signal.throwIfAborted();
    const res = await fetch("/api/generation-attempt", {
      method: "POST", headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({
        prompt: args.prompt, openui: args.openui, attempt: args.attempt, error: args.error || null,
        prior_failures: args.priorFailures, design_md: designMdValueRef.current.trim() || null, session_id: sessionId(),
        meta: { runtime: args.runtime || "unavailable", availability: args.availability || "unknown", identities: args.identities },
      }),
    });
    const data = await responseJSON(res, "Browser attempt could not be stored for training");
    signal.throwIfAborted();
    return data;
  }

  async function browserFallback(sample: Sample, slot: number, signal: AbortSignal): Promise<Sample> {
    signal.throwIfAborted();
    const browser = await loadBrowserModule();
    signal.throwIfAborted();
    const failureReasons = [...(sample.failureReasons || [])];
    if (!failureReasons.length) for (const attempt of sample.attempts || []) if (!attempt.valid) failureReasons.push(formatAttemptFailure(attempt));
    let lastError = sample.error || "The real model exhausted three attempts";
    appendLog(`Sample ${slot + 1}: real model exhausted; switching to browser inference.`, "warning");
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      signal.throwIfAborted();
      const priorFailures = [...failureReasons];
      stackRef.current[slot] = { ...sample, openui: "", serialized: null, valid: false, status: "loading", source: "browser", phase: "browser-generation", attempt };
      forceRender();
      appendLog(`Sample ${slot + 1}: browser attempt ${attempt}/3 started.`);
      let openui = "";
      let inferenceError: string | null = null;
      const started = performance.now();
      const heartbeat = window.setInterval(() => appendLog(`Sample ${slot + 1}: browser attempt ${attempt}/3 is still running (${Math.round((performance.now() - started) / 1000)}s elapsed).`), 5_000);
      try {
        const response = await withBrowserModelSession((session) => session.prompt(browser.buildBrowserRepairPrompt(sample.prompt, priorFailures, attempt)), signal);
        signal.throwIfAborted();
        openui = browser.cleanOpenUIResponse(response);
      } catch (error: any) {
        if (isAbortError(error)) throw error;
        inferenceError = error?.message || String(error);
      } finally {
        window.clearInterval(heartbeat);
      }
      signal.throwIfAborted();
      const stored = await persistBrowserAttempt({
        prompt: sample.prompt, openui, attempt, error: inferenceError, priorFailures,
        runtime: browserRef.current.runtime, availability: browserRef.current.availability,
        identities: { ...(sample.identities || {}), output_generator: browserModelIdentity("output") },
      }, signal);
      signal.throwIfAborted();
      const elapsed = ((performance.now() - started) / 1000).toFixed(1);
      if (stored.valid) {
        appendLog(`Sample ${slot + 1}: browser attempt ${attempt}/3 succeeded after ${elapsed}s; training record ${stored.id}.`, "success");
        return {
          prompt: sample.prompt, openui: stored.serialized || openui, serialized: stored.serialized || openui,
          valid: true, error: null, status: "ready", source: "browser", phase: "ready", attempt,
          generationId: stored.id, identities: { ...(sample.identities || {}), output_generator: browserModelIdentity("output") },
          originalOpenui: stored.serialized || openui, browserApproved: true, note: "",
        };
      }
      lastError = stored.error || inferenceError || "Browser output failed validation";
      failureReasons.push(`browser attempt ${attempt} failed: ${lastError}.${openui ? ` Output was: ${openui.slice(0, 1200)}` : ""}`);
      appendLog(`Sample ${slot + 1}: browser attempt ${attempt}/3 failed — ${lastError}; training record ${stored.id}.`, "error");
    }
    return { prompt: sample.prompt, openui: "", serialized: null, valid: false, error: lastError, status: "error", source: null, note: "" };
  }

  async function trainingModelPipeline(placeholder: Sample, slot: number, signal: AbortSignal): Promise<Sample> {
    signal.throwIfAborted();
    const attempts: any[] = [];
    const failureReasons: string[] = [];
    let prompt: string | null = null;
    let requestIdentity: any = null;
    let identities: Record<string, any> = {};
    let lastError = "The training model exhausted three attempts";
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      signal.throwIfAborted();
      Object.assign(placeholder, { prompt: prompt || "Selecting a request…", openui: "", serialized: null, valid: false, error: null, status: "loading", source: "server", phase: "server-generation", attempt });
      forceRender();
      appendLog(`Sample ${slot + 1}: training model attempt ${attempt}/3 started.`);
      let candidate: Sample;
      try {
        candidate = await fetchServerAttempt(prompt, attempt, failureReasons, requestIdentity, signal);
        signal.throwIfAborted();
      } catch (error: any) {
        if (isAbortError(error)) throw error;
        lastError = error?.message || String(error);
        failureReasons.push(`training model attempt ${attempt} request failed: ${lastError}`);
        appendLog(`Sample ${slot + 1}: training model attempt ${attempt}/3 request failed — ${lastError}`, "error");
        continue;
      }
      prompt = candidate.prompt;
      identities = candidate.identities || identities;
      requestIdentity = identities.request_generator || requestIdentity;
      placeholder.prompt = prompt;
      attempts.push(candidate.attemptRecord);
      appendLog(`Sample ${slot + 1}: training model attempt ${attempt}/3 ${candidate.valid ? "passed lint" : `failed lint — ${candidate.error}`}; training record ${candidate.attemptRecord?.id || "unknown"}.`, candidate.valid ? "success" : "error");
      if (!candidate.valid) {
        lastError = candidate.error || "Training-model output failed linting";
        failureReasons.push(formatAttemptFailure(candidate.attemptRecord));
        continue;
      }
      Object.assign(placeholder, { openui: "", serialized: null, phase: "browser-review" });
      forceRender();
      appendLog(`Sample ${slot + 1}: browser baseline grading training attempt ${attempt}/3.`);
      const review = await gradeServerCandidate(candidate, failureReasons, signal);
      signal.throwIfAborted();
      appendLog(`Sample ${slot + 1}: browser baseline ${review.passed ? "approved" : "rejected"} training attempt ${attempt}/3 (score ${Number(review.score).toFixed(2)}); review record ${review.id}.`, review.passed ? "success" : "warning");
      if (review.passed) {
        const openui = candidate.serialized || candidate.openui;
        return {
          ...candidate, openui, serialized: openui, valid: true, error: null, status: "ready", source: "server", phase: "ready",
          originalOpenui: openui, identities: { ...(candidate.identities || {}), reviewer: browserModelIdentity("review") },
          browserApproved: true, browserReview: review, note: "",
        };
      }
      lastError = review.error || (review.reasons || []).join("; ");
      failureReasons.push(`browser baseline rejected training attempt ${attempt} (score ${Number(review.score).toFixed(2)}): ${(review.reasons || []).join("; ")}. Output was: ${(candidate.serialized || candidate.openui).slice(0, 1200)}`);
    }
    signal.throwIfAborted();
    return browserFallback({ prompt: prompt || placeholder.prompt, openui: "", valid: false, error: lastError, status: "loading", source: null, attempts, failureReasons, identities, note: "" }, slot, signal);
  }

  async function ensurePrefetch(signal: AbortSignal) {
    signal.throwIfAborted();
    if (prefetchingRef.current && prefetchOwnerRef.current && !prefetchOwnerRef.current.aborted) return;
    prefetchingRef.current = true;
    prefetchOwnerRef.current = signal;
    try {
      while (stackRef.current.length - indexRef.current - 1 < PREFETCH) {
        signal.throwIfAborted();
        const placeholder: Sample = { prompt: "…", openui: "", valid: false, status: "loading", note: "" };
        stackRef.current.push(placeholder);
        const slot = stackRef.current.length - 1;
        appendLog(`Sample ${slot + 1}: queued for generation.`);
        forceRender();
        appendLog(`Sample ${slot + 1}: training-model pipeline started; browser baseline gate required.`);
        const started = performance.now();
        let sample: Sample;
        try {
          sample = await trainingModelPipeline(placeholder, slot, signal);
          signal.throwIfAborted();
        } catch (error: any) {
          if (isAbortError(error)) throw error;
          sample = { prompt: placeholder.prompt || "Generation request failed", openui: "", valid: false, error: error?.message || String(error), status: "error", source: null, note: "" };
        }
        stackRef.current[slot] = sample;
        renderedItemRef.current = null;
        forceRender();
        if (slot === indexRef.current) setNote(sample.note || "");
        if (sample.valid) appendLog(`Sample ${slot + 1}: ${sample.source} output ready after ${((performance.now() - started) / 1000).toFixed(1)}s.`, "success");
        else {
          setStatus(`All generation attempts failed (${sample.error || "unknown error"})`);
          setUiError(sample.error || "All generation attempts failed");
          appendLog(`Sample ${slot + 1}: all server and browser attempts exhausted — ${sample.error || "unknown error"}`, "error");
          break;
        }
      }
    } finally {
      if (prefetchOwnerRef.current === signal) {
        prefetchingRef.current = false;
        prefetchOwnerRef.current = null;
        if (!signal.aborted) {
          appendLog("Generation queue is idle.");
          forceRender();
        }
      }
    }
  }

  async function persistHumanAnnotation(item: Sample, rating: "up" | "down", options: any = {}) {
    const openui = options.openui || item.serialized || item.openui;
    const humanIdentity = annotatorIdentity();
    const identities: Record<string, any> = { ...(item.identities || {}), annotator: humanIdentity };
    if (options.humanCorrected) identities.correction_author = humanIdentity;
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (annotationTokenValueRef.current.trim()) headers.Authorization = `Bearer ${annotationTokenValueRef.current.trim()}`;
    const res = await fetch("/api/annotate", {
      method: "POST", headers,
      body: JSON.stringify({
        prompt: item.prompt, openui, rating, description: noteValueRef.current.trim() || null, design_md: designMdValueRef.current.trim() || null,
        valid: item.valid, session_id: sessionId(), generation_id: item.generationId || item.attemptRecord?.id || null,
        original_openui: options.originalOpenui || null, human_corrected: !!options.humanCorrected, identities,
        meta: {
          source: "annotate_playground", generation_source: item.source, browser_baseline: item.source === "browser",
          browser_gate_passed: !!item.browserApproved, browser_review_id: item.browserReview?.id || null,
          browser_review_score: item.browserReview?.score ?? null, usable_for_test_data: rating === "up" && !!item.valid,
          human_corrected: !!options.humanCorrected, view: viewRef.current,
        },
      }),
    });
    return responseJSON(res, "Annotate failed");
  }

  async function grade(rating: "up" | "down") {
    const item = current();
    if (!item || item.status !== "ready" || busyGradeRef.current || item.dirty || item.renderError) return;
    busyGradeRef.current = true;
    setStatus(rating === "up" ? "Saving approval…" : "Saving rejection…");
    setUiError("");
    forceRender();
    appendLog(`Saving thumbs ${rating} for sample ${indexRef.current + 1}…`);
    try {
      const data = await persistHumanAnnotation(item, rating);
      setStatus(rating === "up" ? `Saved thumbs up · ${data.id}` : `Saved thumbs down · ${data.id}`);
      appendLog(`Saved thumbs ${rating} · ${data.id}`, "success");
      item.note = "";
      setNote("");
      setFlash(rating);
      window.setTimeout(() => setFlash(""), 285);
    } catch (error: any) {
      setUiError(error?.message || String(error));
      appendLog(`Annotation failed — ${error?.message || String(error)}`, "error");
    } finally {
      busyGradeRef.current = false;
      forceRender();
    }
  }

  async function go(delta: number) {
    const signal = activeControllerRef.current?.signal;
    if (!signal || signal.aborted) return;
    const item = current();
    if (!item || item.status === "loading" || previewRenderingRef.current) return;
    if (item.dirty) return setStatus("Save or discard the correction before browsing");
    const next = indexRef.current + delta;
    if (next < 0) return setStatus("At first sample");
    if (next >= stackRef.current.length) await ensurePrefetch(signal);
    signal.throwIfAborted();
    if (next >= stackRef.current.length) return setStatus("No sample ready yet");
    item.note = noteValueRef.current;
    indexRef.current = next;
    renderedItemRef.current = null;
    previewRenderTokenRef.current += 1;
    previewRenderingRef.current = false;
    setNote(current()?.note || "");
    setUiError("");
    forceRender();
    void ensurePrefetch(signal).catch((error) => {
      if (!isAbortError(error)) appendLog(`Generation queue failed — ${error?.message || String(error)}`, "error");
    });
  }

  function switchView(next: View) {
    viewRef.current = next;
    setView(next);
    localStorage.setItem(VIEW_KEY, next);
    if (next === "render") renderedItemRef.current = null;
    else {
      previewRenderTokenRef.current += 1;
      previewRenderingRef.current = false;
    }
    forceRender();
  }

  function syncHighlight(value = outputRef.current?.value || "") {
    const code = highlightRef.current?.querySelector("code");
    if (!code || !editorRef.current) return;
    code.innerHTML = editorRef.current.highlightOpenUI(value);
    if (outputRef.current && highlightRef.current) {
      highlightRef.current.scrollTop = outputRef.current.scrollTop;
      highlightRef.current.scrollLeft = outputRef.current.scrollLeft;
    }
  }

  async function validateDslWithRenderer(item: Sample, source: string, staticResult: any, token: number) {
    try {
      const api = await waitForPreviewApi();
      if (token !== lintTokenRef.current || current() !== item || displayedOpenUI(item) !== source || !lintMountRef.current) return;
      api.mount(lintMountRef.current, { source, keepPlaceholders: true });
      await afterPreviewCommit();
      if (token !== lintTokenRef.current || current() !== item || displayedOpenUI(item) !== source) return;
      const root = lintMountRef.current.querySelector(".openui-preview-root");
      const errors = [...staticResult.errors];
      if (root?.getAttribute("data-parse-ok") !== "1") {
        const message = root?.textContent?.trim() || "OpenUI renderer rejected this syntax";
        if (!errors.includes(message)) errors.push(message);
      }
      item.dslDiagnostics = { valid: errors.length === 0, pending: false, errors, warnings: staticResult.warnings };
    } catch (error: any) {
      if (token !== lintTokenRef.current || current() !== item) return;
      item.dslDiagnostics = { valid: false, pending: false, errors: [...staticResult.errors, error?.message || String(error)], warnings: staticResult.warnings };
    }
    if (viewRef.current === "render" && item.dslDiagnostics.valid) renderedItemRef.current = null;
    forceRender();
  }

  function scheduleDslValidation(item: Sample, source: string) {
    if (lintTimerRef.current != null) window.clearTimeout(lintTimerRef.current);
    const token = ++lintTokenRef.current;
    const result = editorRef.current.lintOpenUI(source);
    item.dslDiagnostics = { ...result, pending: true };
    forceRender();
    lintTimerRef.current = window.setTimeout(() => void validateDslWithRenderer(item, source, result, token), 180);
  }

  function onDslChange(value: string) {
    const item = current();
    if (!item || item.status !== "ready" || !editorRef.current) return;
    const original = item.originalOpenui || item.serialized || item.openui || "";
    item.originalOpenui = original;
    item.dirty = value.trim() !== original.trim();
    item.draftOpenui = item.dirty ? value : null;
    item.renderError = null;
    renderedItemRef.current = null;
    syncHighlight(value);
    scheduleDslValidation(item, value);
    updateCompletions(value, false);
    setStatus(item.dirty ? "Correction drafted · valid OpenUI can be previewed" : "Correction cleared");
  }

  function updateCompletions(value = outputRef.current?.value || "", force = false) {
    if (!editorRef.current || !outputRef.current || outputRef.current.disabled || document.activeElement !== outputRef.current) {
      setCompletions([]);
      return;
    }
    setCompletionIndex(0);
    setCompletions(editorRef.current.completionItems(value, outputRef.current.selectionStart, force));
  }

  function acceptCompletion(selected = completionIndex) {
    const suggestion = completions[selected];
    const item = current();
    if (!suggestion || !item || !editorRef.current || !outputRef.current) return;
    const completed = editorRef.current.applyCompletion(outputRef.current.value, suggestion);
    item.draftOpenui = completed.value;
    setCompletions([]);
    onDslChange(completed.value);
    forceRender();
    requestAnimationFrame(() => {
      outputRef.current?.focus();
      outputRef.current?.setSelectionRange(completed.cursor, completed.cursor);
    });
  }

  async function saveCorrection() {
    const item = current();
    if (!item?.dirty || item.status !== "ready" || item.renderError || item.dslDiagnostics?.pending || item.dslDiagnostics?.valid !== true || busyGradeRef.current) return;
    const corrected = (item.draftOpenui || "").trim();
    const original = item.originalOpenui || item.serialized || item.openui;
    busyGradeRef.current = true;
    setStatus("Saving human correction…");
    forceRender();
    try {
      const data = await persistHumanAnnotation(item, "up", { openui: corrected, originalOpenui: original, humanCorrected: true });
      item.openui = data.openui || corrected;
      item.serialized = data.openui || corrected;
      item.draftOpenui = null;
      item.dirty = false;
      item.renderError = null;
      item.dslDiagnostics = { ...editorRef.current.lintOpenUI(item.openui), pending: false };
      item.identities = data.identities || item.identities;
      renderedItemRef.current = null;
      setNote("");
      item.note = "";
      setStatus(`Correction saved · ${data.id}`);
      appendLog(`Saved human correction · ${data.id}`, "success");
      setFlash("up");
      window.setTimeout(() => setFlash(""), 285);
    } catch (error: any) {
      setUiError(error?.message || String(error));
      setStatus("Correction was not saved");
      appendLog(`Correction failed — ${error?.message || String(error)}`, "error");
    } finally {
      busyGradeRef.current = false;
      forceRender();
    }
  }

  function discardCorrection() {
    const item = current();
    if (!item?.dirty || busyGradeRef.current || !editorRef.current) return;
    item.draftOpenui = null;
    item.dirty = false;
    item.renderError = null;
    item.dslDiagnostics = { ...editorRef.current.lintOpenUI(displayedOpenUI(item)), pending: false };
    if (lintTimerRef.current != null) window.clearTimeout(lintTimerRef.current);
    lintTokenRef.current += 1;
    setCompletions([]);
    renderedItemRef.current = null;
    previewRenderTokenRef.current += 1;
    previewRenderingRef.current = false;
    setStatus("Correction discarded");
    appendLog(`Discarded correction for sample ${indexRef.current + 1}.`, "warning");
    forceRender();
  }

  async function startPreviewRender(item: Sample) {
    const signal = activeControllerRef.current?.signal;
    if (!signal || signal.aborted) return;
    const token = ++previewRenderTokenRef.current;
    previewRenderingRef.current = true;
    item.renderError = null;
    appendLog(`Sample ${indexRef.current + 1}: rendering preview…`);
    forceRender();
    try {
      const api = await waitForPreviewApi();
      signal.throwIfAborted();
      if (!previewRef.current) throw new Error("preview did not mount");
      api.mount(previewRef.current, { source: displayedOpenUI(item), keepPlaceholders });
      await afterPreviewCommit();
      signal.throwIfAborted();
      if (token !== previewRenderTokenRef.current || current() !== item) return;
      const root = previewRef.current.querySelector(".openui-preview-root");
      if (!root) throw new Error("preview did not mount");
      if (root.getAttribute("data-parse-ok") === "0") throw new Error(root.textContent?.trim() || "preview parse failed");
      renderedItemRef.current = item;
      appendLog(`Sample ${indexRef.current + 1}: preview rendered.`, "success");
    } catch (error: any) {
      if (isAbortError(error)) return;
      if (token !== previewRenderTokenRef.current || current() !== item) return;
      item.renderError = error?.message || String(error);
      renderedItemRef.current = item;
      appendLog(`Sample ${indexRef.current + 1}: preview render failed — ${item.renderError}`, "error");
    } finally {
      if (!signal.aborted && token === previewRenderTokenRef.current) {
        previewRenderingRef.current = false;
        forceRender();
      }
    }
  }

  useEffect(() => {
    const item = current();
    if (view !== "render" || !item || item.status !== "ready" || !item.valid || item.dslDiagnostics?.valid === false || renderedItemRef.current === item || previewRenderingRef.current) return;
    void startPreviewRender(item);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  });

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    activeControllerRef.current = controller;
    let alive = true;
    appendLog("Playground starting.");
    void preloadBrowserModel(signal).catch((error) => {
      if (!isAbortError(error) && !signal.aborted) appendLog(`Browser baseline initialization failed — ${error?.message || String(error)}`, "error");
    });
    (async () => {
      try {
        editorRef.current = await loadEditorModule();
        signal.throwIfAborted();
        appendLog("OpenUI editor ready.", "success");
        await waitForPreviewApi();
        signal.throwIfAborted();
        appendLog("OpenUI renderer ready.", "success");
      } catch (error: any) {
        if (isAbortError(error)) return;
        if (alive) {
          setUiError(error?.message || String(error));
          appendLog(`Renderer error — ${error?.message || String(error)}`, "error");
        }
      }
      if (!alive) return;
      setStatus("Prefetching samples…");
      try {
        await ensurePrefetch(signal);
      } catch (error: any) {
        if (isAbortError(error)) return;
        if (alive) {
          setUiError(error?.message || String(error));
          appendLog(`Generation queue failed — ${error?.message || String(error)}`, "error");
        }
        return;
      }
      if (signal.aborted) return;
      if (!alive) return;
      if (current()?.status === "ready" && current()?.valid) setStatus("Ready · swipe or use thumbs · arrows browse · Tab changes view");
      else if (!current()) setStatus("Waiting for valid sample…");
      cardRef.current?.focus();
    })();
    return () => {
      alive = false;
      controller.abort();
      if (activeControllerRef.current === controller) activeControllerRef.current = null;
      if (prefetchOwnerRef.current === signal) {
        prefetchingRef.current = false;
        prefetchOwnerRef.current = null;
      }
      if (lintTimerRef.current != null) window.clearTimeout(lintTimerRef.current);
      lintTokenRef.current += 1;
      previewRenderTokenRef.current += 1;
      (window as any).OpenUIPreview?.unmount?.(lintMountRef.current);
      if (![...browserRef.current.waiters].some((waiter: AbortSignal) => !waiter.aborted)) {
        browserRef.current.session?.destroy?.();
        browserRef.current.session = null;
        if (!browserRef.current.promise) browserRef.current.error = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const active = document.activeElement;
      if (active?.id === "note") {
        if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
          event.preventDefault();
          if (current()) current()!.note = noteValueRef.current;
          (active as HTMLElement).blur();
          cardRef.current?.focus();
          setStatus("Note ready · use a grading hotkey");
        } else if (event.key === "Escape") {
          event.preventDefault();
          (active as HTMLElement).blur();
          cardRef.current?.focus();
        }
        return;
      }
      if (active && active !== document.body && active !== cardRef.current && ["textarea", "input", "select"].includes(active.tagName.toLowerCase())) return;
      const item = current();
      const busy = !item || item.status === "loading" || previewRenderingRef.current;
      if (busy) {
        if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "d", "D", "r", "R"].includes(event.key) || event.key.length === 1) event.preventDefault();
        return;
      }
      if (item.dirty && ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) {
        event.preventDefault();
        setStatus("Save or discard the correction before swiping");
      } else if (event.key === "ArrowUp") { event.preventDefault(); void grade("up"); }
      else if (event.key === "ArrowDown") { event.preventDefault(); void grade("down"); }
      else if (event.key === "ArrowLeft") { event.preventDefault(); void go(-1); }
      else if (event.key === "ArrowRight") { event.preventDefault(); void go(1); }
      else if (event.key === "Tab" && (active === cardRef.current || active === document.body || active == null)) { event.preventDefault(); switchView(viewRef.current === "render" ? "dsl" : "render"); }
      else if (event.key === "d" || event.key === "D") { event.preventDefault(); switchView("dsl"); }
      else if (event.key === "r" || event.key === "R") { event.preventDefault(); switchView("render"); }
      else if (event.key.length === 1 && !event.metaKey && !event.ctrlKey && !event.altKey) document.getElementById("note")?.focus();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function onPointerDown(event: React.PointerEvent<HTMLElement>) {
    const item = current();
    if (!item || item.status === "loading" || previewRenderingRef.current || item.dirty || (event.target as Element).closest("button, textarea, input, select, a, summary")) return;
    pointerRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY };
    cardRef.current?.classList.add("is-dragging");
    cardRef.current?.setPointerCapture?.(event.pointerId);
  }

  function onPointerMove(event: React.PointerEvent<HTMLElement>) {
    const start = pointerRef.current;
    if (!start || start.id !== event.pointerId || !cardRef.current) return;
    const dx = event.clientX - start.x;
    const dy = event.clientY - start.y;
    if (Math.abs(dx) < 8 || Math.abs(dx) < Math.abs(dy)) return;
    event.preventDefault();
    cardRef.current.style.transform = `translateX(${dx}px) rotate(${Math.max(-7, Math.min(7, dx / 24))}deg)`;
    cardRef.current.style.opacity = String(Math.max(0.68, 1 - Math.abs(dx) / 650));
  }

  function finishPointerSwipe(event: React.PointerEvent<HTMLElement>) {
    const start = pointerRef.current;
    if (!start || start.id !== event.pointerId) return;
    const dx = event.clientX - start.x;
    const dy = event.clientY - start.y;
    pointerRef.current = null;
    cardRef.current?.classList.remove("is-dragging");
    cardRef.current?.releasePointerCapture?.(event.pointerId);
    if (cardRef.current) { cardRef.current.style.transform = ""; cardRef.current.style.opacity = ""; }
    if (Math.abs(dx) >= 72 && Math.abs(dx) > Math.abs(dy)) void grade(dx > 0 ? "up" : "down");
  }

  const item = current();
  if (item?.status === "ready" && item.valid && editorRef.current && !item.dslDiagnostics) item.dslDiagnostics = { ...editorRef.current.lintOpenUI(displayedOpenUI(item)), pending: false };
  const previewNeeded = view === "render" && !!item && item.status === "ready" && item.valid && (!item.dirty || item.dslDiagnostics?.valid === true) && renderedItemRef.current !== item;
  const busy = !item || item.status === "loading" || previewRenderingRef.current || previewNeeded;
  const editing = !!item?.dirty;
  const gradable = !!item && item.status === "ready" && item.valid && !item.renderError && !editing && !busyGradeRef.current && !busy;
  const diagnostics = item?.dslDiagnostics || { valid: !!item?.valid, pending: false, errors: item?.valid ? [] : [item?.error || "OpenUI is not valid"], warnings: [] };
  const badge = !item ? "loading" : item.status === "loading" ? "generating" : editing && diagnostics.pending ? "checking DSL" : editing && !diagnostics.valid ? "invalid DSL" : item.status === "error" || item.renderError ? "error" : previewRenderingRef.current || previewNeeded ? "rendering" : item.valid ? "valid" : "invalid";
  const modelSource = item?.phase === "browser-review" ? "Browser baseline reviewing" : item?.source === "server" ? item.status === "ready" ? "Training model · browser-approved" : `Training model · attempt ${item.attempt || 1}/3` : item?.source === "browser" ? item.status === "ready" ? "Browser baseline · fallback" : `Browser baseline · attempt ${item.attempt || 1}/3` : "Selecting model";
  const modelClass = item?.phase === "browser-review" ? "review" : item?.source === "server" ? "training" : item?.source === "browser" ? "baseline" : "pending";
  const total = Math.max(stackRef.current.length, 1);
  const position = Math.min(indexRef.current + 1, total);
  const previewBlocked = view === "render" && !!item?.dirty && diagnostics.valid !== true;
  const saveDisabled = !editing || busyGradeRef.current || busy || !!item?.renderError || diagnostics.pending || diagnostics.valid !== true;

  return (
    <div className="pg-page">
      <div className="page-head">
        <h1 className="page-title">Playground</h1>
        <p className="page-sub">Grade, repair, and retain browser-reviewed OpenUI samples. Feedback flows to <span className="mono">outputs/annotations/</span>.</p>
        <div className="pg-legend" aria-label="Model source legend">
          <span className="pg-source training">Training model · candidate under evaluation</span>
          <span className="pg-source baseline">Browser baseline · on-device reference</span>
        </div>
        <label className="pg-annotator" htmlFor="annotatorIdentity">
          <span>Annotator</span>
          <input id="annotatorIdentity" value={annotator} maxLength={160} autoComplete="username" spellCheck={false} onChange={(event) => setAnnotator(event.target.value)} onBlur={() => {
            const id = annotator.trim() || sessionId();
            setAnnotator(id); localStorage.setItem(ANNOTATOR_IDENTITY_KEY, id); appendLog(`Annotator identity set to ${id}.`, "success");
          }} />
        </label>
      </div>

      <section
        id="card" ref={cardRef} tabIndex={0} aria-label="Annotation sample" aria-busy={busy}
        className={`card pg-card source-${item?.source || "pending"} ${busy ? "is-busy" : ""} ${editing ? "is-editing" : ""} ${flash ? `flash-${flash}` : ""}`}
        onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={finishPointerSwipe} onPointerCancel={finishPointerSwipe}
      >
        <div className="pg-meta">
          <span className="mono" id="indexPill">{position} / {total}</span>
          <div className="pg-state">
            <span id="modelSource" className={`pg-source ${modelClass}`}>{modelSource}</span>
            <span id="badge" className={`pill pill-${badge === "valid" ? "passed" : badge === "invalid" || badge === "invalid DSL" || badge === "error" ? "failed" : "idle"}`}>{badge}</span>
          </div>
        </div>

        <div>
          <p className="tile-label">Request</p>
          <p className="pg-prompt" id="promptText">{item ? item.prompt : "Loading…"}</p>
        </div>

        <div className="view-toggle" role="tablist" aria-label="Output view">
          <button id="btnViewRender" type="button" role="tab" aria-selected={view === "render"} disabled={busy} className={`view-btn ${view === "render" ? "is-active" : ""}`} onClick={() => switchView("render")}>Rendered</button>
          <button id="btnViewDsl" type="button" role="tab" aria-selected={view === "dsl"} disabled={busy} className={`view-btn ${view === "dsl" ? "is-active" : ""}`} onClick={() => switchView("dsl")}>DSL</button>
        </div>

        <div className="view-panels">
          <div id="panelRender" className={`view-panel preview-panel ${view === "render" ? "is-active" : ""}`} role="tabpanel" aria-label="Rendered OpenUI" hidden={view !== "render"}>
            {item?.status === "loading" ? <DiffusionCanvas /> : previewBlocked ? <p className="openui-preview-empty">{diagnostics.pending ? "Checking the corrected OpenUI before preview…" : "Preview blocked until the OpenUI errors are fixed."}</p> : <div id="preview" className="openui-preview" ref={previewRef} />}
            <div id="correctionActions" className="pg-correction" hidden={!editing || view !== "render"}>
              <span id="correctionStatus" className="hint">{diagnostics.pending ? "Checking corrected OpenUI…" : !diagnostics.valid || item?.renderError ? "Fix the DSL error or discard this correction" : "Unsaved human correction"}</span>
              <div className="pg-correction-buttons" role="group" aria-label="Correction actions">
                <button id="btnDiscardCorrection" className="btn btn-small" type="button" disabled={!editing || busyGradeRef.current} onClick={discardCorrection} aria-label="Discard correction">Discard</button>
                <button id="btnSaveCorrection" className="btn btn-small btn-primary" type="button" disabled={saveDisabled} onClick={() => void saveCorrection()} aria-label="Save correction as training data">Save correction</button>
              </div>
            </div>
          </div>

          <div id="panelDsl" className={`view-panel source-panel ${view === "dsl" ? "is-active" : ""}`} role="tabpanel" aria-label="OpenUI DSL" hidden={view !== "dsl"}>
            <div className="editor-shell">
              <pre id="dslHighlight" className="dsl-highlight" ref={highlightRef} aria-hidden="true"><code dangerouslySetInnerHTML={{ __html: editorRef.current?.highlightOpenUI(displayedOpenUI(item)) || "" }} /></pre>
              <textarea
                id="output" ref={outputRef} className="dsl-editor" value={displayedOpenUI(item) || "// empty"} rows={12}
                disabled={busy || item?.status !== "ready" || !item?.valid} spellCheck={false} aria-label="Editable OpenUI DSL"
                aria-describedby="dslDiagnostics" aria-autocomplete="list" aria-controls="dslAutocomplete" aria-invalid={diagnostics.errors.length > 0}
                onChange={(event) => onDslChange(event.target.value)} onScroll={() => syncHighlight()} onClick={() => updateCompletions()}
                onBlur={() => window.setTimeout(() => setCompletions([]), 100)}
                onKeyDown={(event) => {
                  if ((event.ctrlKey || event.metaKey) && event.code === "Space") { event.preventDefault(); updateCompletions(event.currentTarget.value, true); return; }
                  if (!completions.length) return;
                  if (event.key === "ArrowDown") { event.preventDefault(); setCompletionIndex((completionIndex + 1) % completions.length); }
                  else if (event.key === "ArrowUp") { event.preventDefault(); setCompletionIndex((completionIndex - 1 + completions.length) % completions.length); }
                  else if (event.key === "Enter" || event.key === "Tab") { event.preventDefault(); acceptCompletion(); }
                  else if (event.key === "Escape") { event.preventDefault(); setCompletions([]); }
                }}
              />
              <div id="dslAutocomplete" ref={completionRef} className="dsl-autocomplete" role="listbox" aria-label="OpenUI suggestions" hidden={!completions.length}>
                {completions.map((completion, suggestionIndex) => <button key={`${completion.label}-${suggestionIndex}`} type="button" role="option" aria-selected={suggestionIndex === completionIndex} className={`completion-option ${suggestionIndex === completionIndex ? "is-selected" : ""}`} onPointerDown={(event) => event.preventDefault()} onClick={() => acceptCompletion(suggestionIndex)}><span>{completion.label}</span><small>{completion.detail || "OpenUI suggestion"}</small></button>)}
              </div>
            </div>
            <div id="dslDiagnostics" className={`dsl-diagnostics ${diagnostics.errors.length ? "error" : diagnostics.warnings.length ? "warning" : "valid"}`} role="status" aria-live="polite">
              <div className="diagnostic-summary"><span id="dslDiagnosticState">{diagnostics.pending ? "Checking OpenUI syntax…" : diagnostics.errors.length ? `${diagnostics.errors.length} ${diagnostics.errors.length === 1 ? "error" : "errors"}` : diagnostics.warnings.length ? `Valid with ${diagnostics.warnings.length} ${diagnostics.warnings.length === 1 ? "warning" : "warnings"}` : "Valid OpenUI"}</span><span>Ctrl/⌘ Space for suggestions</span></div>
              <ul id="dslDiagnosticList">{diagnostics.errors.map((error) => <li className="diagnostic-error" key={error}>{error}</li>)}{diagnostics.warnings.map((warning) => <li className="diagnostic-warning" key={warning}>{warning}</li>)}</ul>
            </div>
          </div>
        </div>
        <div id="dslLintMount" className="dsl-lint-mount" ref={lintMountRef} aria-hidden="true" />

        {(uiError || item?.renderError || item?.status === "error") && <p id="error" className="error-note">{uiError || item?.renderError || item?.error || "Generation failed"}</p>}
        <label className="tile-label" htmlFor="note">Note (optional)</label>
        <textarea id="note" rows={2} className="pg-note" value={note} disabled={busy} placeholder="Type to annotate… Enter to finish; Shift+Enter for a new line" onChange={(event) => { setNote(event.target.value); if (current()) current()!.note = event.target.value; }} />

        <div className="pg-grade" role="group" aria-label="Swipe or grade sample">
          <button id="btnPrev" className="btn pg-nav" type="button" disabled={busy || editing || indexRef.current === 0} onClick={() => void go(-1)} aria-label="Previous sample">←</button>
          <button id="btnDown" className="btn btn-ember pg-down" type="button" disabled={!gradable} onClick={() => void grade("down")} aria-label="Thumbs down"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 10v8M4 18h6l4 3V14h3.5a2 2 0 0 0 2-1.7l.8-5A2 2 0 0 0 18.3 5H12l-2 5H4a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1Z" /></svg>Down</button>
          <button id="btnUp" className="btn btn-primary pg-up" type="button" disabled={!gradable} onClick={() => void grade("up")} aria-label="Thumbs up">Up<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 14V6M4 6h6l4-3v7h3.5a2 2 0 0 1 2 1.7l.8 5a2 2 0 0 1-2 2.3H12l-2-5H4a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1Z" /></svg></button>
          <button id="btnNext" className="btn pg-nav" type="button" disabled={busy || editing || indexRef.current >= stackRef.current.length - 1} onClick={() => void go(1)} aria-label="Next sample">→</button>
        </div>
        <p id="status" className="hint pg-status" role="status">{status}</p>

        <section className="pg-activity" aria-labelledby="activityTitle">
          <div className="pg-activity-head"><p id="activityTitle">Activity log</p><span>● live</span></div>
          <ol id="activityLog" className="pg-activity-log" role="log" aria-live="polite" aria-relevant="additions">
            {logs.map((entry) => <li className={`pg-activity-entry ${entry.level}`} key={entry.id}><time>{entry.time}</time><span>{entry.message}</span></li>)}
          </ol>
        </section>
      </section>

      <details className="card pg-advanced">
        <summary>Advanced</summary>
        <label className="tile-label" htmlFor="annotationToken">Annotation token</label>
        <input id="annotationToken" className="pg-input" type="password" value={annotationToken} placeholder="Required for authorized deployed annotations" autoComplete="current-password" onChange={(event) => {
          const token = event.target.value; setAnnotationToken(token);
          if (token.trim()) sessionStorage.setItem(ANNOTATION_TOKEN_KEY, token.trim()); else sessionStorage.removeItem(ANNOTATION_TOKEN_KEY);
        }} />
        <label className="tile-label" htmlFor="design_md">DESIGN.md (optional)</label>
        <textarea id="design_md" rows={3} className="pg-note" value={designMd} onChange={(event) => setDesignMd(event.target.value)} placeholder="Paste DESIGN.md to condition generation" />
        <label className="pg-toggle"><input type="checkbox" id="grammar" checked={grammar} onChange={(event) => setGrammar(event.target.checked)} /> Grammar guard</label>
        <label className="pg-toggle"><input type="checkbox" id="keepPlaceholders" checked={keepPlaceholders} onChange={(event) => { setKeepPlaceholders(event.target.checked); renderedItemRef.current = null; forceRender(); }} /> Keep :placeholders in preview</label>
        {!caps.execution && <p className="hint">Generation needs a running model server; read-only deploys cannot generate.</p>}
      </details>
    </div>
  );
}
