# E277 — A2 ASAp-style distribution-aware constrained MaskGIT decode (2026-07-17)

Decode-lever wiring + fixture-grade eval overlay, not a train/ship run. Code:
[`models/parallel_decode.py`](../../src/slm_training/models/parallel_decode.py)
(`AsapLedger`),
[`models/twotower.py`](../../src/slm_training/models/twotower.py)
(`_generate_maskgit_one` wiring), config knob `asap_decode` threaded
`TwoTowerConfig` → `ModelBuildConfig` → factory runtime override. Linear SLM-38.

## What and why

Grammar-Aligned Decoding / ASAp (Park et al., NeurIPS 2024;
[2405.21047](https://arxiv.org/abs/2405.21047)) proves that plain
constraint-mask-and-renormalize decoding distorts the sampled distribution
toward grammatical-but-low-quality outputs: the constrained pick conditions on
*local* legality, never on whether the model's preferred mass leads anywhere.
A1 (E248) diagnosed the valid-but-empty wall as partly this distortion
(`length_bias_constraint_distortion`). A2 is the direct fix candidate: remove
*observed* violating mass from the proposal adaptively instead of pretending
the constraint mask is distribution-neutral.

## Mechanism (trie → canvas position)

ASAp's bookkeeping unit is the AR prefix-trie node; the MaskGIT adaptation
uses the canvas position:

- **Feedback**: `admit_fill` rejections and grammar stream hard-error remasks
  call `AsapLedger.penalize(position, token, p_model)` with the model
  probability mass of the violating token at that position.
- **Proposal**: the next constrained pick at that position sees
  `logit + log(1 - removed_mass)` (log-domain removal, implicit
  renormalization over survivors) via `adjust_logits_row`.
- **Ordering**: `adjusted_confidence` gives unmask selection the
  post-removal max-probability, so a position whose argmax keeps dying no
  longer outranks positions where model and grammar agree.

Approximation recorded honestly (class docstring): positions are coarser than
prefixes — a penalty persists after unrelated canvas changes. With
remask-don't-replace semantics this errs toward exploring alternatives rather
than repeating an observed dead end. No sampling-until-acceptance loop and no
convergence guarantee are inherited from the paper.

## Recipe

Row E277 (`--matrix v12`) is **eval-only**: routed through the frozen E255
checkpoint (`--parent .../qx_e255_b4_scratch_control/checkpoints/best_weighted_nll.pt`)
so the pair differs only in `asap_decode` (registration test
`tests/test_scripts/test_quality_matrix_v14.py` enforces the matched-pair
property). Same fixture v1 corpus (108 records), CPU, scratch context tower,
no DESIGN.md context, `--rico-limit 3`, suites smoke 3 / held_out 5 /
adversarial 4 / ood 4 / rico_held 0.

Container note: the session environment exports
`NODE_OPTIONS="--import tsx" ...`, which this Node build rejects (exit 9,
"--import tsx is not allowed in NODE_OPTIONS") — killing the OpenUI bridge
and zeroing every parse metric (a vacuous-guardrail hard error per the matrix
skill). The run overrides `NODE_OPTIONS` explicitly; treat any all-zero
scoreboard without this override as invalid, not as a result.

## Results (fixture-grade, CPU, 2026-07-17)

JSON: [quality-matrix-results-iter-v14-a2-20260717.json](quality-matrix-results-iter-v14-a2-20260717.json)
(scoreboards, gates, AgentV envelope under `outputs/runs/qx_e277_a2_asap_decode/`).
Two runs (pre/post telemetry counters) produced byte-identical metrics —
the eval-only overlay is deterministic.

**The ledger demonstrably drives decode**: `asap_penalties` / distinct
penalized positions per suite — smoke 204/32, held_out 334/53, adversarial
278/46, ood 285/45. Constrained proposals at this checkpoint hit admit
rejections and stream remasks constantly, so plain renormalization was
re-proposing observed dead-end mass on every revisit; A2 removes it.

| Suite (n) | Metric | E255 control | E277 A2 |
| --- | --- | ---: | ---: |
| smoke (3) | structural_similarity | 0.300 | 0.265 |
| held_out (5) | structural_similarity | 0.323 | 0.248 |
| adversarial (4) | structural_similarity | 0.281 | **0.370** |
| ood (4) | structural_similarity | 0.372 | 0.278 |
| all | syntax / meaningful parse | 0.0 | 0.0 |
| all | placeholder_fidelity / reward | 0.0 | 0.0 |

Honest gates fail on both rows (14 threshold failures each), unchanged from
every fixture-scale run: the placeholder-policy rejection wall dominates and
Track A's target distortion binds at frontier scale, not at a 200-step
108-record checkpoint. Secondary deltas are mixed at n≤5 — adversarial up 9
points, the rest down 3–9 — noise-level movement, not a verdict.

## Honesty

Wiring + fixture-overlay evidence only. No checkpoint promoted, no ship gate
touched, no ship claim. The fixture budget (108-record corpus, 200-step E255
checkpoint, tiny suites) can neither confirm nor kill the ASAp hypothesis for
the E224+ wall — the real A2 verdict needs the frontier checkpoints on a GPU
host with the standard n=1500 RICO bar.
