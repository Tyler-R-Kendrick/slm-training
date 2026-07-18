# Browser inference options (webapp)

Decision record for how the webapp runs models in the browser, and what would
have to change to adopt an alternative runtime. Updated 2026-07-18.

## Roles the webapp serves

| Role | Model | Where it runs |
| --- | --- | --- |
| Training-model candidate | TwoTower checkpoint | Server: PyTorch CPU, or committed ONNX (`onnx_inference.py`) when torch is absent |
| Browser baseline (generate / review / run insights) | Prompt API model, else `onnx-community/gemma-3-270m-it-ONNX` | Browser: built-in Prompt API, else transformers.js |

The browser baseline is the annotate-playground quality gate and fallback
generator, so it must initialize on ordinary hardware without server help.

## Hardware context: Windows-on-Snapdragon, training under WSL2

Training on this project's dev machine runs through WSL2 on Windows ARM64
(Snapdragon X). WSL2 exposes neither the Hexagon NPU nor Adreno GPU compute to
Linux PyTorch, so local training is CPU-only. The **native Windows browser is
the only local bridge to that silicon**: WebNN reaches the NPU/GPU through
DirectML, and WebGPU reaches the Adreno GPU directly. That makes the webapp's
WebNN/WebGPU ladder the local acceleration story for inference. No browser
runtime accelerates *training*; the accelerated-training options remain
[HF Jobs](hf-jobs-train.md) / pods (`remote_train`), unchanged by this doc.

## What the 2026-07-15 reproduction proved broken

The [real Windows Chrome run](runtime-performance.md#playground-load-reproduction-2026-07-15)
showed the pre-fix ladder was not functional on exactly this hardware:

- every device used `dtype: "q4"`, whose `MatMulNBits` + `GatherBlockQuantized`
  contrib ops the WASM execution provider rejects (`GatherBlockQuantized(1)`),
  so the terminal WASM rung could never work;
- WebGPU was correctly skipped (adapter workgroup storage 32,768 < 65,536), but
  that left no working rung at all — the page fell to the non-training wiring
  fixture;
- WebNN devices would begin a full weight download before discovering the
  backend was absent.

## Current design (2026-07-18)

`src/slm_training/web/static/browser_inference.js` now pins one weight format
per execution provider and probes backends before downloading:

| Device | dtype | Why |
| --- | --- | --- |
| `webnn-npu` | `fp16` | WebNN cannot map ORT contrib quant ops; fp16 is DirectML-native |
| `webgpu` | `q4f16` | 4-bit weights + fp16 activations; smallest working WebGPU download (~273 MB) |
| `webnn-gpu` | `fp16` | Same silicon as WebGPU but larger weights, so it ranks below `webgpu` |
| `wasm` | `q8` | `model_quantized.onnx` uses standard integer ops the WASM EP executes |

Ladder order `webnn-npu → webgpu → webnn-gpu → wasm`; the Prompt API, when
available, still short-circuits the ladder entirely. WebNN rungs call
`navigator.ml.createContext({deviceType})` first so an absent backend fails in
milliseconds instead of after a multi-hundred-MB fetch. The WebGPU 65,536-byte
workgroup-storage guard from the 2026-07-15 run is retained. The session
runtime string now records device **and** dtype
(`transformers-js:webgpu:q4f16`) so attempt/review records say which variant
produced the data. All variants exist in the model repo
(`model_fp16.onnx` 570 MB, `model_q4f16.onnx` 273 MB, `model_quantized.onnx`
545 MB, verified against the HF tree 2026-07-18).

**Evidence (wiring-level, this container):** Node capability smoke over the
module (full/webgpu-only/wasm-only navigators produce the intended
device→dtype plans), `pytest tests/test_web` 75 passed, Playwright
desktop-chrome 16/16 and mobile-chrome 16/16 against the served SPA. Real
NPU/GPU initialization still needs a native Windows Chrome/Edge run (below);
transformers.js pin `@huggingface/transformers@4.2.0` confirmed latest on npm.

## Options considered

| Option | Accelerators | Model format | Verdict |
| --- | --- | --- | --- |
| Chrome/Edge built-in Prompt API | Managed by browser | Built-in (Gemini Nano class) | **Keep as first choice** — zero download, but availability is flag/OS dependent |
| transformers.js 4.2.0 (ONNX Runtime Web) | WebNN-NPU, WebGPU, WebNN-GPU, WASM | ONNX (our exports included) | **Current choice** — only runtime covering NPU *and* GPU *and* universal fallback with our 270 MB-class model |
| LiteRT.js `@litertjs/core` 2.5.3 (updated 2026-07-17) | WebGPU, WebNN (NPU), XNNPack WASM | `.tflite` | Credible alternative; parity on accelerator targets, but requires `ai-edge-torch` conversion of PyTorch models and brings no LLM orchestration (KV cache, chat template, sampling) — transformers.js keeps that for free |
| LiteRT-LM JS `@litert-lm/core` 0.14.0 | WebGPU only (early preview) | `.litertlm`, **only** Gemma 4 `E2B`/`E4B` web builds | Not adoptable yet: smallest supported model is ~2 B effective params vs our 270 M baseline, no WebNN, no WASM fallback. **Watch** — revisit when general `.litertlm` + WebNN land |
| WebLLM (MLC) | WebGPU only | MLC-compiled | No NPU path, bespoke model compilation — no advantage over the current ladder |
| TwoTower ONNX in-browser (onnxruntime-web directly) | WebGPU/WebNN/WASM | Committed `.context.onnx` / `.denoiser.onnx` | Future option: artifacts already export, but the grammar-constrained decode loop (`dfa_admits_token`, force-emit, certify) lives in Python and would need a JS port; server ONNX stays until that is worth owning |

## Decision

1. Keep **Prompt API → transformers.js ladder** for the browser baseline; the
   per-EP dtype table above is the supported configuration.
2. Keep the **TwoTower training model on the server** (PyTorch, ONNX
   fallback); do not port the grammar decode loop to JS now.
3. **LiteRT-LM is the designated re-evaluation target**, with three concrete
   triggers: general `.litertlm` loading, WebNN support in the LM layer, and a
   ≤500 MB web model class. LiteRT.js core alone does not justify a `.tflite`
   conversion pipeline while the ONNX ladder covers the same accelerators.
4. Training stays out of the browser: WSL2 has no NPU/GPU path, and none of
   these runtimes train. Use HF Jobs for accelerated training.

## Verifying on the target hardware (Windows ARM64)

1. Edge stable or Chrome ≥ current stable; for NPU check `edge://flags` /
   `chrome://flags` → "Enables WebNN API" state and an installed Qualcomm NPU
   driver (Copilot+ requirement); `chrome://gpu` should list WebGPU as
   hardware-accelerated.
2. Open the deployed playground; the activity log now prints the plan up
   front: `Backend plan: webnn-npu (fp16) → webgpu (q4f16) → webnn-gpu (fp16)
   → wasm (q8).`
3. Expected on Snapdragon X today: `webnn-npu` fails fast at context creation
   or session init (decoder shapes are dynamic; WebNN wants static graphs),
   `webgpu` is skipped while the Adreno adapter reports 32 KB workgroup
   storage, and **`wasm (q8)` must now initialize and generate** where q4
   previously could not — ending fixture-only sessions.
4. Record the run (device reached, per-rung failure strings from the activity
   log) in [runtime-performance.md](runtime-performance.md) as fixture-demo
   runtime evidence.
