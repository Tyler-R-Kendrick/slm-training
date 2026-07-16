# Model card — OpenUI TwoTower / grammar-diffusion

Canonical card for checkpoints produced by this repo. Agents **must update
this file whenever a new checkpoint is created or promoted** (full train,
remote train, bootstrap demo, or matrix champion intended for reuse), then
mirror a short summary into [`README.md`](../README.md) → “Model card (summary)”.

Storage: durable full-run weights live in
[`hf://buckets/TKendrick/OpenUI`](https://huggingface.co/buckets/TKendrick/OpenUI)
(`checkpoints/<run_id>/`). Local/git fixture demo:
`src/slm_training/resources/checkpoints/playground_demo/`.

Related: [checkpoint-bucket.md](design/checkpoint-bucket.md),
[adversarial-review.md](design/adversarial-review.md),
[quality-experiment-matrix.md](design/quality-experiment-matrix.md).

---

## Current checkpoint roster

| Role | Run id | Kind | Location | Status |
| --- | --- | --- | --- | --- |
| Playground demo | `playground_demo` | Fixture wiring | `src/slm_training/resources/checkpoints/playground_demo/last.pt` (git) | Demo only — **not** a ship claim |
| Restructure CPU verify | `restructure_cpu_scratch_v0` | Fixture scratch train | `outputs/runs/restructure_cpu_scratch_v0/checkpoints/last.pt` (local) | Train OK; smoke parse **0.0** @ 80 steps — **not** a ship claim ([results](design/restructure-cpu-train-results.json)) |
| Local DirectML verify | `local_directml_adreno_20260714` | Local GPU scratch train | `outputs/runs/local_directml_adreno_20260714/checkpoints/last.pt` (local) | Adreno DirectML train/checkpoint OK @ 5 steps; not evaluated — **not** a ship claim ([results](design/local-directml-train-results.json)) |
| Overnight retrain | `overnight_retrain_200` | CPU scratch train | `/tmp/slm-training-overnight/outputs/runs/overnight_retrain_200/checkpoints/last.pt` (local) | 200 steps; all honest suites parse 0.0 — **not promotable or ship** |
| Overnight retrain extended | `overnight_retrain_1000` | CPU scratch train | `/tmp/slm-training-overnight/outputs/runs/overnight_retrain_1000/checkpoints/last.pt` (local) | 1,000 steps; smoke parse 0.0 at steps 200/400/600/800/1000 — **not promotable or ship** |
| E120 singleton diagnostic | `e120_unsandboxed` | CPU scratch decoder diagnostic | `outputs/runs/iter-e120-unsandboxed-20260715/e120_unsandboxed/checkpoints/last.pt` (local) | 8 steps; guarded singleton/root/arity path verified; `rico_held n=1` parse 0.0 — **not promotable or ship** |
| E121 judged-corpus E53 iteration | `qx_e53_honest_v5_champion` | CPU scratch judged-corpus iteration | `outputs/runs/iter-e121d-e53-judged-20260715/qx_e53_honest_v5_champion/checkpoints/last.pt` (local) | 405 judge-approved records; 8 train + 30 trust-gate steps; bounded smoke parse 0.0 with decode timeout — **not promotable or ship** |
| E123 judged-corpus 32-step iteration | `e123_judged_32step_b` | CPU scratch judged-corpus iteration | `outputs/runs/iter-e123b-judged-20260715/e123_judged_32step_b/checkpoints/last.pt` (local) | 405 judge-approved records; loss 10.97; smoke parse 0.0 with unconstrained fallback and canvas cap — **not promotable or ship** ([results](design/iter-e123-judged-corpus-32step-20260715.md)) |
| E127 schema/slot-contract iteration | `e127_judged_schema_slots` | CPU scratch judged-corpus iteration | `outputs/runs/iter-e127-schema-slots-20260715/e127_judged_schema_slots/checkpoints/last.pt` (local) | 405 judged records; loss 10.71; placeholder validity 0.55 / normalized fidelity 0.25, parse 0.0 — **not promotable or ship** ([results](design/iter-e127-schema-slots-20260715.md)) |
| E128 schema/slot 64-step iteration | `e128_judged_schema_slots_64` | CPU scratch judged-corpus iteration | `outputs/runs/iter-e128-schema-slots-20260715/e128_judged_schema_slots_64/checkpoints/last.pt` (local) | 405 judged records; loss 15.03; higher LTR/fidelity weights regressed placeholder signals; parse 0.0 — **not promotable or ship** ([results](design/iter-e128-schema-slots-64step-20260715.md)) |
| E129 schema/slot 64-step low-weight control | `e129_judged_schema_slots_64_lowweights` | CPU scratch judged-corpus iteration | `outputs/runs/iter-e129-schema-slots-20260715/e129_judged_schema_slots_64_lowweights/checkpoints/last.pt` (local) | 405 judged records; loss 9.89; placeholder/parse 0.0; longer training did not reproduce E127 — **not promotable or ship** ([results](design/iter-e129-schema-slots-64low-20260715.md)) |
| E130 schema/slot seed-1 control | `e130_judged_schema_slots_seed1` | CPU scratch judged-corpus iteration | `outputs/runs/iter-e130-schema-slots-20260715/e130_judged_schema_slots_seed1/checkpoints/last.pt` (local) | 405 judged records; seed 1 loss 15.28; parse and placeholder signals 0.0 — **not promotable or ship** ([results](design/iter-e130-schema-slots-seed1-20260715.md)) |
| E132 generation-focused mixture | `e132_generation_focus` | CPU scratch judged-corpus mixture iteration | `outputs/runs/iter-e132-generation-focus-20260715/e132_generation_focus/checkpoints/last.pt` (local) | 405 judged records; three-prompt smoke parse/placeholder 0.0; task reweighting rejected — **not promotable or ship** ([results](design/iter-e132-generation-focus-20260715.md)) |
| E133 no-fused-LTR path | `e133_no_fuse_ltr` | CPU scratch judged-corpus training-path iteration | `outputs/runs/iter-e133-no-fuse-ltr-20260715/e133_no_fuse_ltr/checkpoints/last.pt` (local) | 405 judged records; three-prompt parse/structure 0.0 with one timeout; fused LTR retained — **not promotable or ship** ([results](design/iter-e133-no-fuse-ltr-20260715.md)) |
| E135 HF context control | `e135_hf_context_control` | CPU HF-context control | `outputs/runs/iter-e135-hf-context-20260715/e135_hf_context_control/checkpoints/last.pt` (local) | Frozen SmolLM2-135M, 8 steps; 3-prompt parse 0.0 but structural similarity 0.2422 / placeholder validity 0.3167; **not promotable or ship** ([results](design/iter-e135-hf-context-control-20260715.md)) |
| E136 HF context 32-step control | `e136_hf_context_32` | CPU HF-context control | `outputs/runs/iter-e136-hf-context-20260715/e136_hf_context_32/checkpoints/last.pt` (local) | Frozen SmolLM2-135M, 32 steps; parse/placeholder 0.0 and structural similarity 0.0825; **not promotable or ship** ([results](design/iter-e136-hf-context-32step-20260715.md)) |
| E137 HF context 16-step midpoint | `e137_hf_context_16` | CPU HF-context control | `outputs/runs/iter-e137-hf-context-20260715/e137_hf_context_16/checkpoints/last.pt` (local) | Frozen SmolLM2-135M, 16 steps; placeholder validity 0.40 / structural similarity 0.2142, parse 0.0; **not promotable or ship** ([results](design/iter-e137-hf-context-16step-20260715.md)) |
| E138 HF context seed-1 8-step control | `e138_hf_context_seed1_8` | CPU HF-context seed variance control | `outputs/runs/iter-e138-hf-seed1-20260715/e138_hf_context_seed1_8/checkpoints/last.pt` (local) | Frozen SmolLM2-135M, seed 1, 8 steps; placeholder validity 0.0 / structural similarity 0.1683, parse 0.0; **not promotable or ship** ([results](design/iter-e138-hf-seed1-8step-20260715.md)) |
| E139 HF context seed-2 8-step control | `e139_hf_context_seed2_8` | CPU HF-context seed variance control | `outputs/runs/iter-e139-hf-seed2-20260715/e139_hf_context_seed2_8/checkpoints/last.pt` (local) | Frozen SmolLM2-135M, seed 2, 8 steps; placeholder validity/structure/parse 0.0 with two timeouts; **not promotable or ship** ([results](design/iter-e139-hf-seed2-8step-20260715.md)) |
| E173 schema-context 32-step control | `e173-schema-context-32step` | CPU HF-context semantic control | `outputs/runs/e173-schema-context-32step/checkpoints/last.pt` (local) | Schema/slot context enabled, loss 11.0876; bounded probe syntax 1.0 but meaningful parse 0.0; **not promotable or ship** ([results](design/iter-e173-schema-context-20260716.md)) |
| E174 unfrozen-context 8-step control | `e174-unfrozen-context-8step` | CPU HF-context semantic control | `outputs/runs/e174-unfrozen-context-8step/checkpoints/last.pt` (local) | Unfrozen context, loss 39.4253; bounded probe syntax 0.0 and parse 0.0; rejected control, **not promotable or ship** ([results](design/iter-e174-unfrozen-context-20260716.md)) |
| E175 retrieval 8-step control | `e175-retrieval-8step` | CPU HF-context retrieval control | `outputs/runs/e175-retrieval-8step/checkpoints/last.pt` (local) | Retrieval k=4, loss 27.9708; bounded syntax/parse 0.0; rejected control, **not promotable or ship** ([results](design/iter-e175-retrieval-20260716.md)) |
| E176 broad-corpus 8-step control | `e176-broad-corpus-8step` | CPU HF-context corpus control | `outputs/runs/e176-broad-corpus-8step/checkpoints/last.pt` (local) | 1,417-record corpus, loss 34.0464; bounded syntax/parse 0.0; rejected control, **not promotable or ship** ([results](design/iter-e176-broad-corpus-20260716.md)) |
| E174 unfrozen-context 8-step control | `e174-unfrozen-context-8step` | CPU HF-context semantic control | `outputs/runs/e174-unfrozen-context-8step/checkpoints/last.pt` (local) | Unfrozen context, loss 39.4253; bounded probe syntax 0.0 and parse 0.0; rejected control, **not promotable or ship** ([results](design/iter-e174-unfrozen-context-20260716.md)) |
| Matrix honest champion (scratch) | `qx_e53_*` (V6 E53 family) | CPU scratch matrix clear | Primarily `outputs/runs/` (+ docs matrix JSON) | Honest `--ship-gates` on limited `rico_held` n; **not** production HF ship |
| P13 fixture E50 control | `qx_e50_core_remask` | CPU scratch, fixture corpus | `/tmp/slm17-e50-fixture-honest/` (local) | Matched control; held 0.08 / RICO 0.0667 fidelity; parse 0.0, not ship |
| P13 integrated E50 candidate | `qx_e50_core_remask` | CPU scratch, integrated corpus | `/tmp/slm17-e50-new-honest/` (local) | Strict fidelity gain on both smoke suites; parse 0.0, not promotable or ship |
| Frozen X2 baseline | `gx_x2_codec` seeds 0/1/2 | Retired fixed-canvas grammar diffusion | `/tmp/slm-training-fixed-baseline/outputs/topology_baseline/` (local) | 80 steps; all suites parse/fidelity/structure/reward 0.0; comparison only, not ship |
| Topology implementation smoke | `grammar_diffusion_overfit` | CPU scratch fixture topology v2 | pytest temporary checkpoint (local) | 200 steps; smoke n=2 parse/fidelity 0.5, topology composite 0.482; not reusable or ship |
| Topology X9/X14 confirmation | `gx_x9_topology_base`, `gx_x14_buffer` seeds 0/1/2 | CPU scratch topology v2 matrix | `/tmp/slm-training-grammar-topology/outputs/topology_confirm_4bf964d/` (local) | 200 steps; all 6 fail multi-suite gates; not promoted or synced |
| Production HF ship | — | — | `hf://buckets/TKendrick/OpenUI/checkpoints/<run_id>/` | **None registered yet** — fill this row after the first full HF sync |

Update the table in place when a checkpoint is written or superseded. Keep
invalidated / superseded rows in **Checkpoint history** below.

---

## Intended use

- Generate **placeholder OpenUI** layout programs (`openuiLibrary` syntax) from
  natural-language prompts, optionally conditioned on DESIGN.md.
- Train / eval harness research for TwoTower masked diffusion and
  grammar-diffusion codecs with honest multi-suite ship gates.

**Not intended:** production UI without human review; treating fixture-demo or
scratch-matrix clears as production readiness; silent gold-placeholder channels.

---

## Architecture (serving defaults)

| Piece | Default / notes |
| --- | --- |
| Model | TwoTower (context tower + MaskGIT-style denoiser); optional `grammar_diffusion` |
| Context | HF frozen backbone (`HuggingFaceTB/SmolLM2-135M`) for full ship track; scratch for matrix/CI demos |
| Output tokenizer | Compositional `OpenUITokenizer` (default) or V5 lexer (`DSLNativeTokenizer`) |
| Decode | Grammar-constrained LTR / MaskGIT + repair levers (see design docs) |
| Topology experiment | `grammar_diffusion` v2: typed production-tree expansion/contraction with bounded active nodes; no fixed canvas |
| Eval gates | Multi-suite `--ship-gates` (parse, structural, `placeholder_fidelity`, reward) |

---

## How to load

```bash
# Fixture demo (annotate playground)
python -m scripts.serve_playground
# → src/slm_training/resources/checkpoints/playground_demo/last.pt

# Full-run checkpoint from the OpenUI bucket (after sync)
hf buckets sync \
  hf://buckets/TKendrick/OpenUI/checkpoints/<run_id> \
  ./outputs/runs/<run_id>/checkpoints

python -m scripts.evaluate_model \
  --test-dir outputs/data/eval/v1 \
  --run-id <run_id> \
  --ship-gates
```

Sidecars required next to `*.pt`: `.tokenizer.json`, `.meta.json`
(optional `.context.tokenizer.json`).

---

## Training data

| Split | Source | Notes |
| --- | --- | --- |
| Train | `outputs/data/train/v1` (all sources + quality synth) for ship | Fixture upsample = demo only |
| Eval | `outputs/data/eval/v1` suites: smoke, held_out, adversarial, ood, `rico_held` | Ship claims need full `rico_held` (1500) when asserted |

Leakage: structural fingerprints + train/test isolation
([adversarial-review.md](design/adversarial-review.md)).

---

## Evaluation (fill per checkpoint)

| Suite | n | parse | fidelity | struct | reward | Pass? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| smoke (`restructure_cpu_scratch_v0`) | 3 | 0.0 | 0.0 | 0.31 | 0.0 | No — fixture scratch wiring |
| not run (`local_directml_adreno_20260714`) | 0 | — | — | — | — | No — hardware/checkpoint validation only |
| held_out | | | | | | |
| adversarial | | | | | | |
| ood | | | | | | |
| rico_held | | | | | | |
| `rico_held` (`e120_unsandboxed`, diagnostic subset) | 1 | 0.0 | 0.375 | 0.0375 | 0.0 | No — 8-step scratch; 64-token incomplete program |
| `smoke` (`qx_e53_honest_v5_champion`, E121 diagnostic subset) | 1 | 0.0 | 0.0 | 0.0 | 0.0 | No — one 5-second constrained-decode timeout; not a ship evaluation |
| `smoke` (`e123_judged_32step_b`, E123 diagnostic subset) | 1 | 0.0 | 0.1917 | 0.0 | 0.0 | No — unconstrained retry/canvas cap; not a ship evaluation |
| `smoke` (`e127_judged_schema_slots`, E127 diagnostic subset) | 1 | 0.0 | 0.0 | 0.1917 | 0.0 | No — placeholder signals improved but output did not parse; not a ship evaluation |
| `smoke` (`e128_judged_schema_slots_64`, E128 diagnostic subset) | 1 | 0.0 | 0.0 | 0.1542 | 0.0 | No — higher loss weights regressed placeholder signal; not a ship evaluation |
| `smoke` (`e129_judged_schema_slots_64_lowweights`, E129 diagnostic subset) | 1 | 0.0 | 0.0 | 0.1542 | 0.0 | No — lower-weight control did not reproduce E127; not a ship evaluation |
| `smoke` (`e130_judged_schema_slots_seed1`, E130 diagnostic subset) | 1 | 0.0 | 0.0 | 0.1542 | 0.0 | No — seed-1 control did not reproduce E127; not a ship evaluation |
| `smoke` (`e132_generation_focus`, E132 three-prompt diagnostic) | 3 | 0.0 | 0.0 | 0.1742 | 0.0 | No — task reweighting did not improve quality; not a ship evaluation |
| `smoke` (`e133_no_fuse_ltr`, E133 three-prompt diagnostic) | 3 | 0.0 | 0.0 | 0.0 | 0.0 | No — no-fused-LTR path worsened feedback; not a ship evaluation |
| `smoke` (`e135_hf_context_control`, E135 three-prompt diagnostic) | 3 | 0.0 | 0.0 | 0.2422 | 0.0 | No — HF representation improved signals but did not parse; not a ship evaluation |
| `smoke` (`e136_hf_context_32`, E136 three-prompt diagnostic) | 3 | 0.0 | 0.0 | 0.0825 | 0.0 | No — longer HF run regressed E135; not a ship evaluation |
| `smoke` (`e137_hf_context_16`, E137 three-prompt diagnostic) | 3 | 0.0 | 0.0 | 0.2142 | 0.0 | No — midpoint improved placeholder signal but did not parse; not a ship evaluation |
| `smoke` (`e138_hf_context_seed1_8`, E138 three-prompt diagnostic) | 3 | 0.0 | 0.0 | 0.1683 | 0.0 | No — seed-1 control regressed diagnostic signals; not a ship evaluation |
| `smoke` (`e139_hf_context_seed2_8`, E139 three-prompt diagnostic) | 3 | 0.0 | 0.0 | 0.0 | 0.0 | No — seed-2 control had no quality signal and two timeouts; not a ship evaluation |

Recipe for `restructure_cpu_scratch_v0`: device=cpu, steps=80, context=scratch,
fixture train/test `v0`, `--no-sync-checkpoints`, LTR primary, no DESIGN.md in
context. Host: 4c / 15GB RAM, no CUDA, no `HF_TOKEN` (Jobs/bucket skipped).
Evidence: [restructure-cpu-train-results.json](design/restructure-cpu-train-results.json).

Recipe for `local_directml_adreno_20260714`: Qualcomm Adreno X1-85 via
Torch-DirectML (`privateuseone:0`), 5 steps, batch 4, 585-record remediated
corpus, scratch context, 924,386 trainable parameters, no AMP/compile, and
`--no-sync-checkpoints`. Last loss was 61.2962; no eval suite or ship gates ran.
AdamW `aten::lerp.Scalar_out` fell back to CPU. The checkpoint loaded in the CPU
playground, but a real generation did not return within 120 seconds, so it is not
a viable playground candidate. Evidence:
[local-directml-train-results.json](design/local-directml-train-results.json).

Record device, steps, context backend, honesty mode (`honest_slot_contract`),
and whether gates used `--ship-gates`. Link
`docs/design/*-results.json` / run `gates.json` when available.

**Known honest fixture clears (not production):** V6 E50/E53/E55 on CPU scratch
with limited `rico_held` n — see
[quality-experiment-matrix.md](design/quality-experiment-matrix.md).

### P13 matched smoke (SLM-17)

| Checkpoint | Suite | n | Parse | Fidelity | Struct | Reward | Pass? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixture E50 | `held_out` | 5 | 0.0 | 0.08 | 0.0 | 0.0 | No |
| integrated E50 | `held_out` | 5 | 0.0 | 0.12 | 0.0 | 0.0 | Signal only; +0.04 |
| fixture E50 | `rico_held` | 5 | 0.0 | 0.0667 | 0.0 | 0.0 | No |
| integrated E50 | `rico_held` | 5 | 0.0 | 0.10 | 0.0 | 0.0 | Signal only; +0.0333 |

Recipe: E50 on CPU scratch, 80 train steps, batch 4, lr `3e-4`, seed 0,
honest slot contract, four-step best-of-1 decode, no template fill or
DESIGN.md context, and unchanged gates. Checkpoints are local scratch
artifacts with explicit no-sync rationale; this is a bounded matched data
signal, not a full HF-context train or reusable promotion.
Evidence: [data-synthesis.md](design/data-synthesis.md) and
[data-synthesis-results.json](design/data-synthesis-results.json).

### Grammar topology implementation smoke

| Checkpoint | Suite | n | Parse | Fidelity | Struct | Topology composite | Pass? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| retired X2, seeds 0/1/2 | all five | 19 per seed | 0.0 | 0.0 | 0.0 | unavailable | No |
| topology v2 fixture overfit | smoke | 2 | 0.5 | 0.5 | 0.225 | 0.4820 | Wiring only; no ship |
| X9 confirmation median | smoke | 3 | 0.0 | 0.333 | 0.098 | 0.414 | No |
| X9 confirmation median | held_out | 5 | 0.0 | 0.0 | 0.0 | 0.372 | No |
| X9 confirmation median | adversarial | 4 | 0.0 | 0.0 | 0.0 | 0.472 | No |
| X9 confirmation median | ood | 4 | 0.0 | 0.083 | 0.108 | 0.330 | No |
| X9 confirmation median | rico_held | 3 | 0.667 | 0.125 | 0.317 | 0.464 | No — limited slice; other suites fail |
| X14 confirmation median | smoke | 3 | 0.0 | 0.0 | 0.309 | 0.298 | No |
| X14 confirmation median | held_out | 5 | 0.0 | 0.0 | 0.251 | 0.277 | No |
| X14 confirmation median | adversarial | 4 | 0.0 | 0.0 | 0.291 | 0.285 | No |
| X14 confirmation median | ood | 4 | 0.0 | 0.0 | 0.237 | 0.278 | No |
| X14 confirmation median | rico_held | 3 | 0.0 | 0.042 | 0.078 | 0.233 | No |

X2 used CPU scratch, 80 steps, batch 4, seeds 0/1/2, the 1,165-record
curriculum corpus, limited remediated suites, and no checkpoint sync. All three
AgentV bundles ran without execution errors and all ship gates failed. Topology v2
used CPU scratch, 200 steps, batch 2, learning rate `3e-3`, two fixture records,
and honest request slot contracts; AgentV ran 5 checks with zero passes. Neither
checkpoint family was promoted or uploaded. Evidence:
[grammar-fixed-canvas-baseline-results.json](design/grammar-fixed-canvas-baseline-results.json)
and [grammar-topology-smoke-results.json](design/grammar-topology-smoke-results.json).
The X9/X14 confirmation used the same 1,165-record curriculum corpus, CPU scratch
context, 200 steps, batch 4, learning rate `3e-4`, 16 generation phases, and seeds
0/1/2. Six AgentV bundles completed. All checkpoints are local short-budget matrix
artifacts with an explicit no-sync rationale; no reusable champion was designated.
Evidence: [grammar-matrix-results.json](design/grammar-matrix-results.json).

---

## Limitations & honesty

- Smoke parse alone is a canary, not generalization.
- Soft `placeholder_validity` is diagnostic; ship on `placeholder_fidelity`.
- Inventory must come from the user-visible prompt / DESIGN.md under
  `honest_slot_contract=True` (no silent `gold.placeholders`).
- Scratch + short steps ≠ HF + full `rico_held` production claim.

---

## Checkpoint history

| Date (UTC) | Run id | Bucket / path | Metric headline | Notes |
| --- | --- | --- | --- | --- |
| (seed) | `playground_demo` | `src/slm_training/resources/checkpoints/playground_demo/` | wiring demo | Committed fixture; regenerate via `bootstrap_playground` |
| 2026-07-14 | `restructure_cpu_scratch_v0` | `outputs/runs/restructure_cpu_scratch_v0/` (local) | smoke parse 0.0 @ 80 steps; last_loss≈6.97 | Post-restructure CPU budget verify; not ship |
| 2026-07-14 | `restructure_cpu_scratch_v0_cont` | `outputs/runs/restructure_cpu_scratch_v0_cont/` (local) | resume +200 scratch steps; smoke parse still 0.0 | Continues v0; HF Jobs still blocked on missing HF_TOKEN |
| 2026-07-14 | `qx_e0_baseline` (P13 superseded) | `outputs/slm17/matrix-smoke-baseline/` (local) | `rico_held n=3` parse/fidelity 0.0 | Fixture probe; not comparable to E50; scratch/no-sync |
| 2026-07-14 | `qx_e50_core_remask` (P13 superseded) | `outputs/slm17/matrix-smoke-champion/` (local) | `rico_held n=3` parse/fidelity 1.0 | System-recipe probe, not a matched data signal; scratch/no-sync |
| 2026-07-14 | fixture `qx_e50_core_remask` (P13 final) | `/tmp/slm17-e50-fixture-honest/` (local) | held 0.08 / RICO 0.0667 fidelity; parse 0.0 | Equal-recipe fixture control; scratch/no-sync; not ship |
| 2026-07-14 | integrated `qx_e50_core_remask` (P13 final) | `/tmp/slm17-e50-new-honest/` (local) | held 0.12 / RICO 0.10 fidelity; parse 0.0 | Strict two-suite data signal; scratch/no-sync; not promotable or ship |
| 2026-07-14 | `local_directml_adreno_20260714` | `outputs/runs/local_directml_adreno_20260714/` (local) | DirectML train completed @ 5 steps; last_loss≈61.30 | Adreno GPU/checkpoint wiring; one AdamW op used CPU fallback; CPU generation timed out at 120s; no eval/ship claim |
| 2026-07-15 | `overnight_retrain_200` | `/tmp/slm-training-overnight/outputs/runs/overnight_retrain_200/` (local) | 200 CPU scratch steps; last_loss≈6.64; all suites parse 0.0 | Full honest eval with AgentV bundle; no promotion; decode-path investigation continues |
| 2026-07-15 | `overnight_retrain_1000` | `/tmp/slm-training-overnight/outputs/runs/overnight_retrain_1000/` (local) | 1,000 CPU scratch steps; last_loss≈1.12; smoke parse 0.0 at every checkpoint | Extended training did not improve generation quality; no promotion |
| 2026-07-15 | `gx_x2_codec` seeds 0/1/2 | `/tmp/slm-training-fixed-baseline/outputs/topology_baseline/` (local) | all five suites parse/fidelity/structure/reward 0.0 | Frozen format-v1 comparison; AgentV complete; not promoted or synced |
| 2026-07-15 | topology `grammar_diffusion_overfit` | pytest temporary local checkpoint | smoke n=2 parse/fidelity 0.5; topology composite 0.4820 | Implementation smoke only; temporary checkpoint, not promoted or synced |
| 2026-07-15 | `gx_x9_topology_base` seeds 0/1/2 | `/tmp/slm-training-grammar-topology/outputs/topology_confirm_4bf964d/` (local) | RICO n=3 median parse 0.667, but held/adversarial/OOD parse 0.0 | 200-step CPU scratch confirmation; all seeds fail multi-suite gates; not promoted or synced |
| 2026-07-15 | `gx_x14_buffer` seeds 0/1/2 | `/tmp/slm-training-grammar-topology/outputs/topology_confirm_4bf964d/` (local) | all-suite median parse 0.0 | 200-step CPU scratch confirmation; all seeds fail; not promoted or synced |
| 2026-07-16 | `qx_e53_honest_v5_champion` (E121) | `outputs/runs/iter-e121d-e53-judged-20260715/` (local) | judged corpus 405; smoke n=1 parse/fidelity/structure/reward 0.0; decode timeout | Explicit corpus precedence and evaluator tuple bugs fixed; scratch-only; no promotion |
| 2026-07-16 | `e123_judged_32step_b` (E123) | `outputs/runs/iter-e123b-judged-20260715/` (local) | 32 CPU scratch steps; loss 10.97; smoke parse 0.0, structural similarity 0.1917, 26.75s p50; fallback/canvas cap | Longer training did not improve generation; generation-recipe investigation next; no promotion |
| 2026-07-16 | `e127_judged_schema_slots` (E127) | `outputs/runs/iter-e127-schema-slots-20260715/` (local) | 32 CPU scratch steps; loss 10.71; placeholder validity 0.55 / normalized fidelity 0.25; parse 0.0 | Schema/slot conditioning improves placeholder signal but not syntax; no promotion |
| 2026-07-16 | `e128_judged_schema_slots_64` (E128) | `outputs/runs/iter-e128-schema-slots-20260715/` (local) | 64 CPU scratch steps; loss 15.03; placeholder validity/fidelity 0.0; parse 0.0 | Higher LTR/fidelity weights regressed E127; no promotion |
| 2026-07-16 | `e129_judged_schema_slots_64_lowweights` (E129) | `outputs/runs/iter-e129-schema-slots-20260715/` (local) | 64 CPU scratch steps; loss 9.89; placeholder validity/fidelity 0.0; parse 0.0 | Lower-weight control failed to reproduce E127; data/variance investigation next; no promotion |
| 2026-07-16 | `e130_judged_schema_slots_seed1` (E130) | `outputs/runs/iter-e130-schema-slots-20260715/` (local) | 32 CPU scratch steps, seed 1; loss 15.28; placeholder validity/fidelity 0.0; parse 0.0 | E127 not reproducible; multi-example/task-composition feedback next; no promotion |
| 2026-07-16 | `e132_generation_focus` (E132) | `outputs/runs/iter-e132-generation-focus-20260715/` (local) | 32 CPU scratch steps; three-prompt parse/placeholder 0.0; structural similarity 0.1742 | Task reweighting rejected; architecture/representation or synthesis contract next; no promotion |
| 2026-07-16 | `e133_no_fuse_ltr` (E133) | `outputs/runs/iter-e133-no-fuse-ltr-20260715/` (local) | 32 CPU scratch steps; three-prompt parse/structure 0.0; one 15s timeout | No-fused-LTR path rejected; fused LTR retained; no promotion |
| 2026-07-16 | `e135_hf_context_control` (E135) | `outputs/runs/iter-e135-hf-context-20260715/` (local) | 8 CPU steps with frozen SmolLM2-135M; three-prompt parse 0.0, structural 0.2422, placeholder validity 0.3167; one timeout | HF context is leading representation hypothesis; longer cached control next; no promotion |
| 2026-07-16 | `e136_hf_context_32` (E136) | `outputs/runs/iter-e136-hf-context-20260715/` (local) | 32 CPU steps with frozen SmolLM2-135M; parse/placeholder 0.0, structural 0.0825 | Longer HF training regressed; checkpoint selection/supervision alignment next; no promotion |
| 2026-07-16 | `e137_hf_context_16` (E137) | `outputs/runs/iter-e137-hf-context-20260715/` (local) | 16 CPU steps with frozen SmolLM2-135M; parse 0.0, placeholder validity 0.40, structural 0.2142 | Non-monotonic HF trajectory; explicit early checkpoint selection next; no promotion |
| 2026-07-16 | `e138_hf_context_seed1_8` (E138) | `outputs/runs/iter-e138-hf-seed1-20260715/` (local) | 8 CPU steps with frozen SmolLM2-135M, seed 1; parse 0.0, placeholder validity 0.0, structural 0.1683 | Seed variance is material; multi-seed selection before corpus/loss changes; no promotion |
| 2026-07-16 | `e139_hf_context_seed2_8` (E139) | `outputs/runs/iter-e139-hf-seed2-20260715/` (local) | 8 CPU steps with frozen SmolLM2-135M, seed 2; parse/placeholder/structure 0.0, two timeouts | Seed-0 signal remains unexplained; decoder/checkpoint diagnosis next; no promotion |
| 2026-07-16 | `e173-schema-context-32step` (E173) | `outputs/runs/e173-schema-context-32step/` (local) | 32 CPU steps with frozen HF context plus schema/slot context; loss 11.0876; bounded syntax 1.0, parse 0.0 | Context-only control did not recover semantic hierarchy; no promotion |
| 2026-07-16 | `e174-unfrozen-context-8step` (E174) | `outputs/runs/e174-unfrozen-context-8step/` (local) | 8 CPU steps with unfrozen HF context; loss 39.4253; bounded syntax 0.0, parse 0.0 | Rejected control; retain frozen context; no promotion |
| 2026-07-16 | `e175-retrieval-8step` (E175) | `outputs/runs/e175-retrieval-8step/` (local) | 8 CPU steps with frozen HF context, schema context, retrieval k=4; loss 27.9708; bounded syntax/parse 0.0 | Retrieval rejected; improve semantic coverage; no promotion |
| 2026-07-16 | `e176-broad-corpus-8step` (E176) | `outputs/runs/e176-broad-corpus-8step/` (local) | 8 CPU steps on 1,417-record prompt-contract corpus; loss 34.0464; bounded syntax/parse 0.0 | Broad corpus rejected; targeted judge-gated augmentation next; no promotion |

Append a row for every new or replaced checkpoint. Do not delete history.

---

## Agent checklist (after each checkpoint)

1. Sync durable weights (HF bucket for full runs) —
   [checkpoint-bucket.md](design/checkpoint-bucket.md).
2. Update **Current checkpoint roster** + **Evaluation** + **Checkpoint history**
   in this file.
3. Refresh the **Model card (summary)** section in [`README.md`](../README.md)
   (keep it short; link here for detail).
4. Point measured-results / matrix docs at the new run id when relevant.
5. Commit docs with the checkpoint-producing change.
