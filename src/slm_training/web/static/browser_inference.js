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

const TRANSFORMERS_JS_URL =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
const TRANSFORMERS_JS_MODEL = "onnx-community/gemma-3-270m-it-ONNX";

function browserAccelerationCapabilities() {
  const promptApi = Boolean(
    globalThis.LanguageModel?.create || globalThis.ai?.languageModel?.create
  );
  const webnn = Boolean(globalThis.navigator?.ml);
  const webgpu = Boolean(globalThis.navigator?.gpu);
  const devices = [];
  if (webnn) devices.push("webnn-npu", "webnn-gpu");
  if (webgpu) devices.push("webgpu");
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
    preferred: promptApi ? "prompt-api" : devices[0],
  };
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

async function createTransformersSession(systemPrompt, onProgress) {
  const capabilities = browserAccelerationCapabilities();
  const { env, pipeline } = await import(TRANSFORMERS_JS_URL);
  env.useBrowserCache = true;
  const wasm = env.backends?.onnx?.wasm;
  if (wasm) {
    wasm.simd = true;
    wasm.numThreads = capabilities.wasmThreads;
  }

  let generator = null;
  let selectedDevice = null;
  const failures = [];
  for (const device of capabilities.devices) {
    onProgress({ status: `trying ${device}`, progress: null });
    try {
      generator = await pipeline("text-generation", TRANSFORMERS_JS_MODEL, {
        device,
        dtype: "q4",
        progress_callback(info) {
          const raw = Number(info?.progress);
          const progress = Number.isFinite(raw) ? (raw > 1 ? raw / 100 : raw) : null;
          if (progress !== null) onProgress({ status: "downloading", progress });
          else if (info?.status === "ready") {
            onProgress({ status: `ready ${device}`, progress: null });
          }
        },
      });
      selectedDevice = device;
      break;
    } catch (error) {
      const reason = error?.message || String(error);
      failures.push(`${device}: ${reason}`);
      onProgress({ status: `failed ${device} — ${reason}`, progress: null });
    }
  }
  if (!generator || !selectedDevice) {
    throw new Error(`No browser inference backend initialized (${failures.join("; ")})`);
  }
  let disposed = false;
  return {
    device: selectedDevice,
    session: {
      async prompt(content) {
        if (disposed) throw new Error("Browser inference session has been disposed");
        const output = await generator(
          [
            { role: "system", content: systemPrompt },
            { role: "user", content: String(content || "") },
          ],
          { max_new_tokens: 512, do_sample: false, return_full_text: false }
        );
        return generatedText(output);
      },
      destroy() {
        disposed = true;
        void generator.dispose?.();
      },
    },
  };
}

async function createBrowserModelSession({ mode = "generate", onProgress = () => {} } = {}) {
  const systemPrompt =
    mode === "shared"
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
    runtime: `transformers-js:${created.device}`,
    availability: "available",
  };
}

function buildBrowserGradePrompt(prompt, openui, attempt) {
  return `TASK: REVIEW
Review training-model attempt ${attempt} of 3 against the user request.

USER REQUEST:
${prompt}

TRAINING-MODEL OPENUI:
${openui}

Pass only if this is useful baseline-quality work worth showing a human annotator.
Return only the requested JSON judgement.`;
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
  buildBrowserGradePrompt,
  buildBrowserRepairPrompt,
  browserAccelerationCapabilities,
  cleanOpenUIResponse,
  createBrowserModelSession,
  parseBrowserGradeResponse,
};
