const OPENUI_SYSTEM_PROMPT = `You generate valid OpenUI DSL and nothing else.

Return plain DSL without Markdown fences or explanation. Use only these forms:
root = Stack([child_a, child_b], "column")
child_a = Card([title, body])
title = TextContent(":section.title")
body = TextContent(":section.body")
child_b = Button(":cta.label")

Rules:
- Assign root exactly once.
- Define every identifier referenced by Stack or Card exactly once.
- Use Stack, Card, TextContent, and Button components.
- Stack direction is "column" or "row".
- User-facing content must be a quoted placeholder like ":hero.title".
- Never emit prose, comments, Markdown, JSON, HTML, or CSS.`;

const OPENUI_REVIEW_SYSTEM_PROMPT = `You are the browser baseline reviewer for OpenUI training candidates.
Judge whether a candidate is good enough to show a human annotator. Be strict about request fidelity,
useful component hierarchy, complete placeholder-backed content, and obvious structural mistakes.
Syntactic linting has already run, but you must reject empty, trivial, incoherent, or request-ignoring layouts.
Return only JSON with: passed (boolean), score (number from 0 to 1), and reasons (array of concise strings).`;

const RUN_INSIGHTS_SYSTEM_PROMPT = `You explain supplied OpenUI training-run evidence.
Deterministic events are authoritative observations; possible causes are hypotheses. Cite only evidence
paths present in the input. Recommend bounded experiments, never commands, automatic queue actions,
or weaker quality gates. Return only JSON with summary, causes, and phase_suggestions. Each cause has
category (collapse, data, optimization, or unknown), title, rationale, evidence (string paths),
suggestion, confidence (0 to 1), and optional event_step. Each phase suggestion has phase and suggestion.`;

const OPENUI_BROWSER_SYSTEM_PROMPT = `You are the on-device browser baseline for OpenUI. Every user
request starts with a TASK marker and you must follow the matching output contract.

For TASK: GENERATE, return only a complete OpenUI DSL program without Markdown or explanation. Use
Stack, Card, TextContent, and Button; assign root exactly once; define every referenced identifier;
use only "column" or "row" Stack directions; and express user-facing content as quoted placeholders
such as ":hero.title".

For TASK: REVIEW, judge whether the training candidate is good enough to show a human annotator.
Return only JSON with passed (boolean), score (number from 0 to 1), and reasons (array of concise
strings). Be strict about request fidelity, useful hierarchy, complete placeholder-backed content,
and structural quality. Reject empty, trivial, incoherent, or request-ignoring layouts.`;

const OPENUI_REVIEW_SCHEMA = {
  type: "object",
  required: ["passed", "score", "reasons"],
  additionalProperties: false,
  properties: {
    passed: { type: "boolean" },
    score: { type: "number", minimum: 0, maximum: 1 },
    reasons: {
      type: "array",
      minItems: 1,
      maxItems: 6,
      items: { type: "string" },
    },
  },
};

// One CDN outage (or a proxy that blocks one host) must not take down the
// whole browser baseline: the runtime is imported from the first CDN that
// responds, in this order.
const TRANSFORMERS_JS_URLS = [
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0",
  "https://unpkg.com/@huggingface/transformers@4.2.0/dist/transformers.min.js",
];
// Browser baseline model. SmolLM2-360M-Instruct is the same family this
// repo's HF-context experiments freeze (docs/MODEL_CARD.md), and its official
// repo ships ONNX exports sized for a one-time cached download: q4f16 ≈ 273MB
// for WebGPU, q8 ≈ 365MB for WASM. The q8 export uses plain integer
// matmul/gather ops, so it runs on the WASM EP — unlike block-quantized q4
// exports (gemma-3-270m's embed_tokens needs the GatherBlockQuantized op the
// WASM backend does not implement, and its q8 export is a 545MB download).
// The smaller SmolLM2-135M was tried first and could not hold the DSL or the
// review-JSON format even with few-shot anchoring; 360M holds both.
const TRANSFORMERS_JS_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct";
// Weight format per execution provider. The q4 variant needs the ORT contrib
// ops MatMulNBits + GatherBlockQuantized: the WASM EP rejects it
// (`GatherBlockQuantized(1)` observed 2026-07-15) and the WebNN EP cannot map
// contrib quant ops onto WebNN graph ops, so only WebGPU may use a 4-bit
// variant (q4f16, fp16 activations — its smallest working download). WebNN
// devices get plain fp16 (native DirectML type) and WASM gets q8, which uses
// standard integer ops.
const TRANSFORMERS_DEVICE_DTYPES = {
  "webnn-npu": "fp16",
  "webnn-gpu": "fp16",
  webgpu: "q4f16",
  wasm: "q8",
};
// ORT WebGPU kernels for this class of model need 64KB of workgroup storage;
// software adapters (SwiftShader) top out at 32KB and WASM beats them anyway.
const WEBGPU_MIN_WORKGROUP_STORAGE = 65536;
// One remembered working backend per model, so later visits initialize the
// proven device/dtype directly instead of re-walking (and re-downloading)
// failed candidates.
const INFERENCE_PROFILE_KEY = "twotower_browser_inference_profile_v1";
// An OpenUI program or review JSON fits well inside 192 new tokens; a small
// cap is what keeps WASM-CPU generation inside an interactive budget.
const GENERATION_MAX_NEW_TOKENS = 192;
const PROMPT_TIMEOUTS_MS = {
  "prompt-api": 30_000,
  "legacy-prompt-api": 30_000,
  "transformers-js:webnn-npu": 90_000,
  "transformers-js:webnn-gpu": 90_000,
  "transformers-js:webgpu": 90_000,
  "transformers-js:wasm": 240_000,
};

function browserPromptTimeoutMs(runtime) {
  // Runtime stamps carry the dtype (`transformers-js:<device>:<dtype>`), so
  // budgets match on the device prefix.
  const value = String(runtime || "");
  for (const [prefix, timeout] of Object.entries(PROMPT_TIMEOUTS_MS)) {
    if (value === prefix || value.startsWith(`${prefix}:`)) return timeout;
  }
  return 240_000;
}

function browserAccelerationCapabilities() {
  const promptApi = Boolean(
    globalThis.LanguageModel?.create || globalThis.ai?.languageModel?.create
  );
  const webnn = Boolean(globalThis.navigator?.ml);
  const webgpu = Boolean(globalThis.navigator?.gpu);
  // Ladder order: NPU first (the only path to it), then WebGPU before
  // webnn-gpu — both target the same silicon, but WebGPU runs the q4f16
  // variant while webnn-gpu would first fetch the larger fp16 weights.
  const devices = [];
  if (webnn) devices.push("webnn-npu");
  if (webgpu) devices.push("webgpu");
  if (webnn) devices.push("webnn-gpu");
  devices.push("wasm");
  const hardwareConcurrency = Math.max(
    1,
    Number(globalThis.navigator?.hardwareConcurrency) || 1
  );
  const wasmThreads = globalThis.crossOriginIsolated
    ? Math.max(1, Math.min(4, hardwareConcurrency - 1 || 1))
    : 1;
  return {
    promptApi,
    webnn,
    webgpu,
    crossOriginIsolated: Boolean(globalThis.crossOriginIsolated),
    hardwareConcurrency,
    wasmThreads,
    devices,
    dtypes: Object.fromEntries(
      devices.map((device) => [device, TRANSFORMERS_DEVICE_DTYPES[device] || "q8"])
    ),
    preferred: promptApi ? "prompt-api" : devices[0],
  };
}

async function assertWebnnBackend(device) {
  const ml = globalThis.navigator?.ml;
  if (!ml?.createContext) {
    throw new Error("navigator.ml.createContext is unavailable");
  }
  const deviceType = device === "webnn-npu" ? "npu" : "gpu";
  // Probe the MLContext before pipeline() so an absent WebNN backend fails
  // here instead of after a multi-hundred-megabyte weight download.
  const context = await ml.createContext({ deviceType });
  if (!context) {
    throw new Error(`WebNN did not create a ${deviceType} context`);
  }
}

function readInferenceProfile() {
  try {
    const raw = globalThis.localStorage?.getItem(INFERENCE_PROFILE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed?.model !== TRANSFORMERS_JS_MODEL) return null;
    if (typeof parsed.device !== "string" || typeof parsed.dtype !== "string") return null;
    return parsed;
  } catch {
    return null;
  }
}

function writeInferenceProfile(profile) {
  try {
    if (!profile) globalThis.localStorage?.removeItem(INFERENCE_PROFILE_KEY);
    else
      globalThis.localStorage?.setItem(
        INFERENCE_PROFILE_KEY,
        JSON.stringify({ ...profile, model: TRANSFORMERS_JS_MODEL })
      );
  } catch {
    // Private browsing without storage: worst case is one extra probe.
  }
}

async function webgpuCandidate() {
  const gpu = globalThis.navigator?.gpu;
  if (!gpu?.requestAdapter) return null;
  let adapter = null;
  try {
    adapter = await gpu.requestAdapter();
  } catch {
    return null;
  }
  if (!adapter) return null;
  const storage = Number(adapter.limits?.maxComputeWorkgroupStorageSize || 0);
  if (storage && storage < WEBGPU_MIN_WORKGROUP_STORAGE) {
    return {
      skip: `WebGPU workgroup storage ${storage} is below the model requirement ${WEBGPU_MIN_WORKGROUP_STORAGE}`,
    };
  }
  // fp16 shading needs the shader-f16 feature; fall back to the fp32-activations
  // q4 export when the adapter cannot compile f16 kernels.
  const dtype = adapter.features?.has?.("shader-f16") ? "q4f16" : "q4";
  return { device: "webgpu", dtype };
}

async function transformersCandidates(onProgress) {
  const capabilities = browserAccelerationCapabilities();
  const candidates = [];
  const skipped = [];
  for (const device of capabilities.devices) {
    if (device === "webgpu") {
      const probed = await webgpuCandidate();
      if (probed?.device) candidates.push(probed);
      else if (probed?.skip) {
        skipped.push(`webgpu: ${probed.skip}`);
        onProgress({ status: `skipping webgpu — ${probed.skip}`, progress: null });
      }
      continue;
    }
    // WebNN rungs stay in the walk; assertWebnnBackend probes them at
    // initialization time, before any weight download.
    candidates.push({ device, dtype: capabilities.dtypes[device] });
  }
  const remembered = readInferenceProfile();
  if (remembered) {
    const index = candidates.findIndex(
      (candidate) =>
        candidate.device === remembered.device && candidate.dtype === remembered.dtype
    );
    if (index > 0) candidates.unshift(candidates.splice(index, 1)[0]);
  }
  return { candidates, skipped };
}

function browserLanguageModelApi() {
  if (globalThis.LanguageModel?.create) {
    return { api: globalThis.LanguageModel, runtime: "prompt-api" };
  }
  if (globalThis.ai?.languageModel?.create) {
    return { api: globalThis.ai.languageModel, runtime: "legacy-prompt-api" };
  }
  return null;
}

function generatedText(output) {
  const value = output?.[0]?.generated_text;
  if (Array.isArray(value)) {
    for (let index = value.length - 1; index >= 0; index -= 1) {
      if (value[index]?.role === "assistant") return String(value[index]?.content || "");
    }
    return "";
  }
  return String(value || output?.generated_text || "");
}

// Few-shot exchanges keep the small baseline on-format: sub-1B instruct
// models imitate the preceding assistant turns far more reliably than they
// follow prose instructions. Selected per request via the TASK marker the
// builders stamp on every prompt.
const GENERATE_FEWSHOT = [
  {
    role: "user",
    content:
      "TASK: GENERATE\nCreate an OpenUI layout for this request:\nA welcome hero with a title, a short body, and one call-to-action button.",
  },
  {
    role: "assistant",
    content:
      'root = Stack([hero, action], "column")\n' +
      "hero = Card([title, body])\n" +
      'title = TextContent(":hero.title")\n' +
      'body = TextContent(":hero.body")\n' +
      'action = Button(":cta.label")',
  },
  {
    role: "user",
    content:
      "TASK: GENERATE\nCreate an OpenUI layout for this request:\nA two-column pricing row with the plan summary on the left and a buy button on the right.",
  },
  {
    role: "assistant",
    content:
      'root = Stack([summary, buy], "row")\n' +
      "summary = Card([plan, detail])\n" +
      'plan = TextContent(":plan.name")\n' +
      'detail = TextContent(":plan.detail")\n' +
      'buy = Button(":plan.cta")',
  },
];
const REVIEW_FEWSHOT = [
  {
    role: "user",
    content:
      "TASK: REVIEW\nReview training-model attempt 1 of 3 against the user request.\n\n" +
      "USER REQUEST:\nA checkout summary card with order totals and a pay button.\n\n" +
      'TRAINING-MODEL OPENUI:\nroot = Stack([only], "row")\nonly = TextContent(":x.y")\n\n' +
      "Pass only if this is useful baseline-quality work worth showing a human annotator.\n" +
      "Reply with one JSON object only.",
  },
  {
    role: "assistant",
    content:
      '{"passed": false, "score": 0.2, "reasons": ["a single text node ignores the requested totals and pay button"]}',
  },
  {
    role: "user",
    content:
      "TASK: REVIEW\nReview training-model attempt 2 of 3 against the user request.\n\n" +
      "USER REQUEST:\nA welcome hero with a title, body copy, and one button.\n\n" +
      "TRAINING-MODEL OPENUI:\n" +
      'root = Stack([hero, action], "column")\nhero = Card([title, body])\n' +
      'title = TextContent(":hero.title")\nbody = TextContent(":hero.body")\naction = Button(":cta.label")\n\n' +
      "Pass only if this is useful baseline-quality work worth showing a human annotator.\n" +
      "Reply with one JSON object only.",
  },
  {
    role: "assistant",
    content:
      '{"passed": true, "score": 0.85, "reasons": ["hierarchy covers the title, body, and action with placeholder-backed content"]}',
  },
];

function fewshotFor(content) {
  const text = String(content || "");
  if (text.startsWith("TASK: REVIEW")) return REVIEW_FEWSHOT;
  if (text.startsWith("TASK: GENERATE")) return GENERATE_FEWSHOT;
  return [];
}

async function importTransformers(onProgress) {
  const failures = [];
  for (const url of TRANSFORMERS_JS_URLS) {
    try {
      return await import(url);
    } catch (error) {
      const host = new URL(url).host;
      failures.push(`${host}: ${error?.message || String(error)}`);
      onProgress({ status: `failed runtime import from ${host} — trying next CDN`, progress: null });
    }
  }
  throw new Error(`transformers.js runtime unreachable (${failures.join("; ")})`);
}

async function createTransformersSession(systemPrompt, onProgress) {
  const capabilities = browserAccelerationCapabilities();
  const { env, pipeline } = await importTransformers(onProgress);
  // Cache API persistence: model files download once per browser profile and
  // every later visit (or backend retry) replays them from local storage.
  env.useBrowserCache = true;
  const wasm = env.backends?.onnx?.wasm;
  if (wasm) {
    wasm.simd = true;
    wasm.numThreads = capabilities.wasmThreads;
  }

  const { candidates, skipped } = await transformersCandidates(onProgress);
  let generator = null;
  let selected = null;
  let selectedIndex = -1;
  const failures = [...skipped];
  let disposed = false;
  async function initializeFrom(startIndex) {
    for (let index = startIndex; index < candidates.length; index += 1) {
      if (disposed) throw new Error("Browser inference session has been disposed");
      const candidate = candidates[index];
      const label = `${candidate.device} (${candidate.dtype})`;
      onProgress({ status: `trying ${label}`, progress: null });
      try {
        if (candidate.device.startsWith("webnn")) await assertWebnnBackend(candidate.device);
        generator = await pipeline("text-generation", TRANSFORMERS_JS_MODEL, {
          device: candidate.device,
          dtype: candidate.dtype,
          progress_callback(info) {
            const raw = Number(info?.progress);
            const progress = Number.isFinite(raw) ? (raw > 1 ? raw / 100 : raw) : null;
            if (progress !== null) onProgress({ status: "downloading", progress });
            else if (info?.status === "ready") {
              onProgress({ status: `ready ${label}`, progress: null });
            }
          },
        });
        if (disposed) {
          await generator?.dispose?.();
          generator = null;
          throw new Error("Browser inference session has been disposed");
        }
        selected = candidate;
        selectedIndex = index;
        writeInferenceProfile(candidate);
        return;
      } catch (error) {
        if (disposed) throw error;
        const reason = error?.message || String(error);
        failures.push(`${label}: ${reason}`);
        onProgress({ status: `failed ${label} — ${reason}`, progress: null });
        const remembered = readInferenceProfile();
        if (remembered?.device === candidate.device && remembered?.dtype === candidate.dtype) {
          writeInferenceProfile(null);
        }
      }
    }
    throw new Error(`No browser inference backend initialized (${failures.join("; ")})`);
  }
  await initializeFrom(0);
  // The underlying ORT session cannot run two generations at once, so prompts
  // from concurrent sample pipelines are serialized FIFO.
  let promptChain = Promise.resolve();
  async function runPrompt(content) {
    if (disposed) throw new Error("Browser inference session has been disposed");
    for (;;) {
      try {
        if (disposed) throw new Error("Browser inference session has been disposed");
        const output = await generator(
          [
            { role: "system", content: systemPrompt },
            ...fewshotFor(content),
            { role: "user", content: String(content || "") },
          ],
          {
            max_new_tokens: GENERATION_MAX_NEW_TOKENS,
            do_sample: false,
            repetition_penalty: 1.1,
            return_full_text: false,
          }
        );
        if (disposed) throw new Error("Browser inference session has been disposed");
        return generatedText(output);
      } catch (error) {
        if (disposed) throw new Error("Browser inference session has been disposed");
        const reason = error?.message || String(error);
        failures.push(`${selected?.device} inference: ${reason}`);
        onProgress({ status: `failed ${selected?.device} inference — ${reason}`, progress: null });
        const failedProfile = selected;
        await generator?.dispose?.();
        generator = null;
        selected = null;
        const remembered = readInferenceProfile();
        if (
          remembered?.device === failedProfile?.device &&
          remembered?.dtype === failedProfile?.dtype
        ) {
          writeInferenceProfile(null);
        }
        if (disposed) throw new Error("Browser inference session has been disposed");
        await initializeFrom(selectedIndex + 1);
        if (disposed) {
          await generator?.dispose?.();
          generator = null;
          throw new Error("Browser inference session has been disposed");
        }
      }
    }
  }
  return {
    device: selected.device,
    dtype: selected.dtype,
    model: TRANSFORMERS_JS_MODEL,
    session: {
      prompt(content) {
        const turn = promptChain.then(() => runPrompt(content));
        promptChain = turn.catch(() => {});
        return turn;
      },
      destroy() {
        disposed = true;
        void generator?.dispose?.();
      },
    },
  };
}

async function createBrowserModelSession({ mode = "generate", onProgress = () => {} } = {}) {
  const systemPrompt =
    mode === "insights"
      ? RUN_INSIGHTS_SYSTEM_PROMPT
      : mode === "shared"
      ? OPENUI_BROWSER_SYSTEM_PROMPT
      : mode === "review"
        ? OPENUI_REVIEW_SYSTEM_PROMPT
        : OPENUI_SYSTEM_PROMPT;
  const native = browserLanguageModelApi();
  if (native) {
    const { api, runtime } = native;
    const options = {
      initialPrompts: [{ role: "system", content: systemPrompt }],
      expectedInputs: [{ type: "text", languages: ["en"] }],
      expectedOutputs: [{ type: "text", languages: ["en"] }],
      monitor(monitor) {
        monitor.addEventListener("downloadprogress", (event) => {
          const progress = Number(event.loaded || 0);
          onProgress({ status: "downloading", progress });
        });
      },
    };
    try {
      const availability = api.availability
        ? await api.availability({
            expectedInputs: options.expectedInputs,
            expectedOutputs: options.expectedOutputs,
          })
        : "available";
      onProgress({ status: availability, progress: null });
      if (availability === "unavailable") {
        throw new Error("Browser LanguageModel API reported that its model is unavailable");
      }
      try {
        return { session: await api.create(options), runtime, availability };
      } catch (error) {
        if (runtime !== "legacy-prompt-api") throw error;
        return {
          session: await api.create({ initialPrompts: options.initialPrompts }),
          runtime,
          availability,
        };
      }
    } catch (error) {
      onProgress({
        status: `failed ${runtime} — ${error?.message || String(error)}`,
        progress: null,
      });
    }
  }

  const created = await createTransformersSession(systemPrompt, onProgress);
  return {
    session: created.session,
    runtime: `transformers-js:${created.device}:${created.dtype}`,
    availability: "available",
    model: created.model,
    dtype: created.dtype,
  };
}

function buildRunInsightsPrompt(report) {
  const evidence = {
    run_id: report?.run_id,
    loss: report?.loss,
    phases: report?.phases,
    deterministic_insights: report?.insights,
  };
  return `Explain this completed run. Return only the requested JSON.\n\n${JSON.stringify(evidence)}`;
}

function parseRunInsightsResponse(value) {
  let text = String(value || "").trim();
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced) text = fenced[1].trim();
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start >= 0 && end > start) text = text.slice(start, end + 1);
  const parsed = JSON.parse(text);
  const summary = String(parsed.summary || "").trim().slice(0, 1600);
  if (!summary) throw new Error("browser insights returned no summary");
  const categories = new Set(["collapse", "data", "optimization", "unknown"]);
  const causes = (Array.isArray(parsed.causes) ? parsed.causes : []).slice(0, 8).map((cause) => {
    const title = String(cause?.title || "").trim().slice(0, 160);
    const rationale = String(cause?.rationale || "").trim().slice(0, 1200);
    const suggestion = String(cause?.suggestion || "").trim().slice(0, 1200);
    if (!title || !rationale || !suggestion) throw new Error("browser insights returned an incomplete cause");
    const confidence = Math.max(0, Math.min(1, Number(cause.confidence)));
    if (!Number.isFinite(confidence)) throw new Error("browser insights returned invalid confidence");
    const eventStep = Number(cause.event_step);
    return {
      category: categories.has(cause.category) ? cause.category : "unknown",
      title,
      rationale,
      evidence: (Array.isArray(cause.evidence) ? cause.evidence : []).map(String).slice(0, 8),
      suggestion,
      confidence,
      event_step: Number.isInteger(eventStep) && eventStep >= 0 ? eventStep : null,
    };
  });
  const phase_suggestions = (Array.isArray(parsed.phase_suggestions) ? parsed.phase_suggestions : [])
    .slice(0, 12)
    .map((item) => ({
      phase: String(item?.phase || "").trim().slice(0, 120),
      suggestion: String(item?.suggestion || "").trim().slice(0, 1200),
    }))
    .filter((item) => item.phase && item.suggestion);
  return { summary, causes, phase_suggestions };
}

function buildBrowserGradePrompt(prompt, openui, attempt) {
  return `TASK: REVIEW
Review training-model attempt ${attempt} of 3 against the user request.

USER REQUEST:
${prompt}

TRAINING-MODEL OPENUI:
${openui}

Pass only if this is useful baseline-quality work worth showing a human annotator.
Reply with one JSON object only — no prose, no Markdown. Your whole reply must
start with { and end with }, with exactly the keys passed, score, and reasons.`;
}

function parseBrowserGradeResponse(value) {
  let text = String(value || "").trim();
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced) text = fenced[1].trim();
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start >= 0 && end > start) text = text.slice(start, end + 1);
  const parsed = JSON.parse(text);
  const score = Math.max(0, Math.min(1, Number(parsed.score)));
  if (!Number.isFinite(score)) throw new Error("browser review returned an invalid score");
  const reasons = Array.isArray(parsed.reasons)
    ? parsed.reasons.map((reason) => String(reason).trim()).filter(Boolean).slice(0, 6)
    : [];
  if (!reasons.length) throw new Error("browser review returned no reasons");
  return { passed: parsed.passed === true, score, reasons };
}

function buildBrowserRepairPrompt(prompt, failureReasons, attempt) {
  const failures = failureReasons.length
    ? failureReasons.map((reason, index) => `${index + 1}. ${reason}`).join("\n")
    : "None.";
  return `TASK: GENERATE
Create an OpenUI layout for this request:
${prompt}

This is browser attempt ${attempt} of 3.
Previous server and browser attempts failed for these reasons:
${failures}

Correct every listed failure. Return only a complete valid OpenUI DSL program.`;
}

function cleanOpenUIResponse(value) {
  let text = String(value || "").trim();
  const fenced = text.match(/```(?:openui|text|plaintext)?\s*([\s\S]*?)```/i);
  if (fenced) text = fenced[1].trim();
  const root = text.search(/^\s*root\s*=/m);
  if (root > 0) text = text.slice(root).trim();
  return text;
}

export {
  OPENUI_BROWSER_SYSTEM_PROMPT,
  OPENUI_SYSTEM_PROMPT,
  OPENUI_REVIEW_SCHEMA,
  OPENUI_REVIEW_SYSTEM_PROMPT,
  RUN_INSIGHTS_SYSTEM_PROMPT,
  TRANSFORMERS_DEVICE_DTYPES,
  TRANSFORMERS_JS_MODEL,
  buildBrowserGradePrompt,
  buildBrowserRepairPrompt,
  buildRunInsightsPrompt,
  browserAccelerationCapabilities,
  browserPromptTimeoutMs,
  cleanOpenUIResponse,
  createBrowserModelSession,
  parseBrowserGradeResponse,
  parseRunInsightsResponse,
};
