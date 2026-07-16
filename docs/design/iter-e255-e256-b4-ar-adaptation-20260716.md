# E255–E256 — B4 AR→diffusion adaptation baseline, DiffuLLaMA-style (2026-07-16)

Fixture-grade wiring campaign for Track B4. Machine-readable evidence:
[quality-matrix-results-iter-v10-b4-20260716.json](quality-matrix-results-iter-v10-b4-20260716.json)
(matched pair) and
[quality-matrix-results-iter-v10-b4-lr3e5-20260716.json](quality-matrix-results-iter-v10-b4-lr3e5-20260716.json)
(LR-confound probe). Code:
[`src/slm_training/models/hf_denoiser.py`](../../src/slm_training/models/hf_denoiser.py).
Linear SLM-24.

## Question

DiffuGPT/DiffuLLaMA ([arXiv:2410.17891](https://arxiv.org/abs/2410.17891))
showed AR→diffusion adaptation can beat from-scratch masked-diffusion training.
Our whole stack assumes a **from-scratch** TwoTower denoiser. B4 asks: does
initializing the denoiser from a pretrained AR model (SmolLM2-135M, already the
frozen context tower) beat the from-scratch denoiser on the same data — before
more representation work is stacked on the from-scratch assumption?

## What was built (Adapted, not reproduced)

`HFDenoiserTower` (`denoiser_backend="hf"` on `TwoTowerConfig` /
`ModelBuildConfig` / matrix `Experiment`):

- the pretrained causal-LM backbone runs with **full bidirectional visibility**
  via an explicit 4D attention mask (only the drop-the-causal-mask move of
  DiffuLLaMA is reused; their attention-mask annealing, shift operation, and
  training recipe are NOT reproduced);
- fresh OpenUI-vocabulary embeddings with a weight-tied `lm_head` replace the
  backbone's vocabulary;
- the context tower's hiddens are linearly projected and **prepended as prefix
  states** (the backbone has no cross-attention, unlike `DenoiserTower`);
- the class implements the exact `DenoiserTower` interface
  (`forward`/`encode`/`project`/`set_runtime_symbol_features` plus
  `.tok`/`.kind`/`.lm_head`/`.max_len`/`.layers`), so masking, training, every
  decode path, and checkpoint round-trips work unchanged.

Regression tests: bidirectionality (a later-position token change moves
earlier-position logits — a causal backbone would not), `encode`/`project`
candidate-gathering equivalence, TwoTower train step with backbone gradients,
save/`from_checkpoint` round-trip preserving `denoiser_backend`, and V10
registration asserting the pair differs **only** in `denoiser_backend`
(`tests/test_models/test_hf_denoiser.py`,
`tests/test_scripts/test_quality_matrix_v10.py`).

## Recipe

`run_quality_matrix --matrix v10 --scratch-control --steps 200 --device cpu
--context-backend scratch --no-design-md-context --rico-limit 3` (lr 3e-4,
batch 4, seed 0), fixture v1 corpus (108 records, `--source fixture`), suites
smoke 3 / held_out 5 / adversarial 4 / ood 4 / **rico_held 0** (fixture corpus
has no RICO records). Parallel MaskGIT decode on both rows (LTR compiler-tree
per-token decode at 135M params is not CPU-tractable). AgentV published per
row.

## Results (wiring evidence only)

Both rows fail the honest gates — syntax/meaningful parse 0.0 from real
placeholder-policy rejections, as across all fixture-scale runs. Secondary
signals, smoke/held_out/adversarial/ood:

| Row | Trainable | Train loss @200 | Structural similarity | component_type_recall | Decode p50/record |
| --- | ---: | ---: | --- | --- | --- |
| E255 scratch control | 1.1M | 3.75 | 0.30 / 0.32 / 0.28 / 0.37 | 0.25 / 0.22 / 0.75 / 0.56 | ~15s |
| E256 AR-adapted (lr 3e-4) | 135M | 8.51 | 0.16 / 0.09 / 0.07 / 0.16 | 0 / 0.10 / 0 / 0.19 | 30–37s |
| E256 AR-adapted (lr 3e-5 probe) | 135M | 9.72 | 0.08 / 0.02 / 0.18 / 0.10 | 0 / 0 / 0 / 0 | 17–21s |

- **At this budget the adaptation loses to the matched from-scratch control on
  every signal.** The lr=3e-5 probe rules out "3e-4 destroyed the pretrained
  weights" as the sole explanation — the lower LR converges even less (loss
  9.72). A 1.1M scratch model simply fits a 108-record corpus far faster than a
  135M adaptation can move in 200 CPU steps.
- The wiring is proven end-to-end: backbone gradients flow, bidirectional
  attention verified, all suites + AgentV + honest gates run on both rows, and
  the checkpoint (incl. 135M denoiser weights) round-trips.

## Verdict and honesty

- **The B4 question is NOT answered.** 200 CPU steps on a 108-record fixture
  corpus is orders of magnitude below any adaptation budget in DiffuLLaMA;
  this campaign can neither confirm nor kill the from-scratch assumption. The
  decisive row is the same invocation on a GPU host: real corpus, matched
  compute per arm (not matched steps), per-arm LR selection, full suites
  (`rico_held` n=1500).
- Fixture/scratch runs are wiring evidence only; no ship gate weakened,
  nothing promoted. The E256 checkpoint stays local (`outputs/`, gitignored) —
  ~540MB, not synced.
- Known simplifications vs the paper (documented, not hidden): no
  attention-mask annealing, no shift operation, prefix-conditioning instead of
  their pure single-sequence recipe, fresh (not inherited) vocabulary
  embeddings because the OpenUI lexer vocabulary shares no id space with the
  SmolLM2 tokenizer.
