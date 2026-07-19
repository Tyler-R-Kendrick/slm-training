# SLM-108 — External 1–7B constrained-decoding semantic ceiling (wiring)

SLM-108 adds the missing external-model control for the OpenUI compiler/DSL
stack. The goal is to run off-the-shelf 1–7B instruct/code models through the
same compiler-owned legal candidate space and verifier stack, then compare
semantic quality and cost with the best tiny SLM under matched prompts, suites,
decode/search budgets, and judges.

This iteration lands the harness wiring only. No real 1–7B model was loaded and
no ship claim is made.

## What changed

- Added a provider-neutral `ExternalLegalActionScorer` interface in
  `src/slm_training/models/external_scorer.py`.
- Implemented a `TransformersCausalLMScorer` adapter that loads a pinned HF
  causal/instruct model, scores compiler-legal actions and complete candidates,
  and never mutates legality.
- Added `ExternalScorePolicy` so the existing eval-only score-policy path can
  rerank candidates with an external model.
- Added the `external-ceiling` matrix set under
  `src/slm_training/harnesses/experiments/external_ceiling_matrix.py` with arms
  A–E:

| Arm | Model | Decode | Status in fixture |
| --- | --- | --- | --- |
| A | tiny SLM | constrained | not_run (baseline) |
| B | HuggingFaceTB/SmolLM2-135M | constrained | fixture |
| C | Qwen/Qwen2.5-7B-Instruct | constrained | not_run |
| D | HuggingFaceTB/SmolLM2-135M | unconstrained + postvalidation | fixture |
| E | HuggingFaceTB/SmolLM2-135M | complete-candidate rerank | fixture |

- Wired the matrix set into `scripts/run_quality_matrix.py` as
  `--matrix-set external-ceiling`.
- Added `scripts/run_external_ceiling.py` for dedicated fixture/frontier runs.

## Fixture run

```bash
python -m scripts.run_external_ceiling \
  --mode fixture \
  --output-dir outputs/runs/slm108_external_ceiling \
  --checkpoint-reference-uri hf://buckets/TKendrick/OpenUI/checkpoints/slm108_baseline/ref.json
```

```bash
python -m scripts.run_quality_matrix \
  --matrix-set external-ceiling \
  --mode fixture \
  --run-root outputs/runs/slm108_external_ceiling_qm \
  --checkpoint-reference-uri hf://buckets/TKendrick/OpenUI/checkpoints/slm108_baseline/ref.json
```

The fixture scored two synthetic requests per fixture arm using a deterministic
fake scorer. All fixture arms produced a ranked candidate set without errors.
Frontier arms A and C are intentionally `not_run` because they require a durable
baseline checkpoint (SLM-103) and a GPU host, respectively.

## Results

| Metric | Fixture value |
| --- | --- |
| Status | fixture |
| Fixture arms run | B, D, E |
| Records per fixture arm | 2 |
| Candidates scored per fixture arm | 2 |
| Strict binding-aware meaning | 0.0 (fixture placeholder) |
| Non-empty rate | 1.0 |
| Fallback/timeout/OOM | 0 |

No meaningful-program rate is reported because the fixture uses a fake scorer;
the number is a wiring placeholder, not evidence.

## Honest caveats

- This is a wiring and schema iteration only.
- No actual 1–7B weights were loaded; the fake scorer exists so CI can exercise
  the interface without GPUs or network downloads.
- Arm A (tiny SLM baseline) and arm C (6–7B model) are `not_run` pending
  durable checkpoint provenance and GPU host allocation.
- Ship gates are intentionally not claimed.

## Next step

Run the frontier campaign on a GPU host with pinned durable checkpoints and the
EFS0 canonical comparison stack (SLM-103 through SLM-106). The harness is ready
to switch arm B and any available 6–7B arm from `fixture` to `frontier` once the
artifacts and host are available.

Machine-readable evidence is in
[the SLM-108 JSON](iter-slm108-external-ceiling-20260719.json).
