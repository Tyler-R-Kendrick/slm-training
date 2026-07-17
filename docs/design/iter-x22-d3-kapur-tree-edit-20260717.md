# X22 — D3 Kapur-style tree-edit baseline vs X9 (2026-07-17)

Fixture-grade matched screening for Track D3 (Linear SLM-31). Machine-readable
evidence:
[grammar-matrix-results-iter-x22-kapur-20260717.json](grammar-matrix-results-iter-x22-kapur-20260717.json).
Code: [`src/slm_training/models/tree_edit_diffusion.py`](../../src/slm_training/models/tree_edit_diffusion.py)
(`model_name="tree_edit_diffusion"`, matrix row X22).

## What was built

A baseline faithful to the *mechanism* of "Diffusion On Syntax Trees For
Program Synthesis" (Kapur, Jenner, Russell; NeurIPS 2024,
[arXiv:2405.20519](https://arxiv.org/abs/2405.20519)) — our closest prior art,
previously only **Adapted** (the X-series borrowed the
denoise-typed-syntax-topology idea but reproduced neither the all-valid
forward process nor the search):

- **All-valid state space**: forward noise is a chain of 1–4 small
  validity-preserving edits (replace a component type, add a spurious leaf,
  remove a referenced leaf), each re-verified through the real parser before
  acceptance. Every intermediate training state parses — no mask tokens, no
  invalid canvases.
- **Inverse-edit policy**: the network (transformer over program tokens with
  prompt cross-attention) is supervised on the inverse edit of the last
  mutation — Kapur's repair-step objective — factorized over bounded heads
  (action / statement / component / placeholder slot). Clean states are
  supervised as STOP with full value.
- **Value-guided beam search at decode**: search starts from a minimal valid
  seed program and expands top policy edits per state, re-scoring children
  with the value head and keeping the top beam; STOP freezes a candidate.
  Every emitted program is valid **by construction**.
- **Stated boundary (observation channel)**: the paper conditions on a
  rendered image compared to the target render; text-to-UI generation has no
  target render, so policy/value condition on the prompt context instead.
  Mechanism faithful, observation domain-translated — recorded in
  research-lineage.md and the source manifest.

The component inventory derives from the fixed lexer grammar vocabulary (36
components), so checkpoint round-trips keep head sizes stable; statements
whose surface defies the `name = Comp(...)` shape are never mutated
(fail-closed skip).

## Matched fixture screening (X22 vs X9, wiring evidence)

Recipe: `--only X9,X22 --seeds 0 --steps 80 --gen-steps 8 --device cpu
--context-backend scratch --rico-limit 3`; identical corpus (train v1, 108
records), identical suites and unchanged honest gates.

| Suite (n) | metric | X9 topology base | X22 Kapur baseline |
| --- | --- | --- | --- |
| smoke (3) | meaningful / syntax / struct | 0.0 / 0.0 / 0.0 | 0.0 / **1.0** / 0.32 |
| held_out (5) | meaningful / syntax / struct | 0.0 / 0.0 / 0.0 | **0.2** / **1.0** / 0.32 |
| adversarial (4) | meaningful / syntax / struct | 0.0 / 0.0 / 0.0 | **0.25** / **1.0** / 0.35 |
| ood (4) | meaningful / syntax / struct | 0.0 / 0.0 / 0.0 | **0.25** / **1.0** / 0.37 |
| rico_held (3) | meaningful / syntax / struct | 0.0 / 0.0 / 0.0 | 0.0 / **1.0** / 0.09 |

X22 additionally posts placeholder fidelity 0.12–0.53 and reward 0.68–0.81
across suites (X9: 0.0 everywhere at this recipe). X22 **passed the
adversarial and ood gate rows outright** (all three floors each) — the first
row in this program with nonzero meaningful parse at the 80-step CPU fixture
budget. Both rows still fail the full gate battery (X22 fails smoke
meaningful/struct/recall, held_out meaningful/recall, and all rico_held
rows), so **nothing ships**. Exact meaningful/recall values were reproduced
deterministically by re-evaluating the saved checkpoint
(`gx_x22_kapur_tree_edit_reeval` scoreboard).

## Reading the result honestly

- **Syntax parse 1.0 is a property of the method, not learned skill** — the
  search only ever visits valid programs. The honest signal is meaningful
  parse and recall, where learning must put the *right* content in.
- The nonzero meaningful parse supports the paper's core claim transferring
  to this domain: keeping every state valid and spending model capacity on
  semantic edits (rather than re-learning syntax) helps at tiny scale. It is
  simultaneously an adverse data point for the X-series' mask-node
  representation at matched budget.
- **Comparison caveats**: X9 at 80 steps is far below its documented 200-step
  confirmations (which also failed all gates); single seed; n ≤ 5 per suite;
  the X22 seed state plants one placeholder-bearing leaf, which partially
  seeds placeholder fidelity (though not held_out/ood meaningful parse). A
  decisive D3 verdict needs the standard 3-seed screening + 200-step
  confirmation, then frontier scale.
- The edit space covers component replace/add/remove over the statement
  grammar; it cannot express state/query/action statements (V05 builtins are
  never mutated and never generated). RICO's 0.0 meaningful parse reflects
  exactly that coverage gap plus layout depth.

## Verification

- `tests/test_models/test_tree_edit_diffusion.py`:
  `test_mutations_preserve_validity_and_inverse_restores` (every mutated
  state parses; the recorded inverse edit restores a valid same-shape
  program) and `test_training_loss_decode_all_valid_and_checkpoint`
  (finite/backprop loss, decode output parses, checkpoint round-trip
  reproduces outputs exactly).
- Full-corpus training sweep (108 records) with zero skipped statements;
  `repo_policy`, `check-changed`, `ruff`, `git diff --check` clean.

## Honesty and limits

Fixture/scratch wiring evidence only — one seed, 80 CPU steps, tiny suites;
no ship claim, no gate weakened, nothing promoted. The lineage tag moves to
**Faithful (mechanism)** with the observation-channel boundary stated; the
render-feedback half of the paper remains unreproduced because the domain
has no target render at generation time.
