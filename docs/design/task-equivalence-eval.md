# Task-balanced mixture and equivalence evaluation

## Status and claim boundary

SLM-15 adds task-balanced online sampling, corpus/search diagnostics,
availability-aware per-task metrics, L3–L5 equivalence scoring, and deterministic
generalization slices. The recorded runs below are **fixture wiring only**. They
did not train a model, fit RegMix coefficients, evaluate a checkpoint, or run
ship gates.

## Task-balanced mixture contract

Mixture manifest v2 has two independent maps:

- `task_weights`: target probability for generation,
  repair/completion/inpaint, patch/edit, state/behavior, and noop/adversarial;
- `weights`: source-family priors used only after a task group is selected.

The sampler draws task → available family → row with the training-loop RNG.
Thus a family with 100 rows and one with one row can still receive equal task
mass. Records without explicit `meta.task` remain unclassified; they are never
silently treated as generation. V1 manifests omit `task_weights` and retain the
legacy family-only path.

`run_mixture_search` now covers every configured family before repeating scales,
keeps task weights on probes/proposals, profiles task/family/ProgramSpec and
grammar-component coverage, carries NLL learning-curve points for scored runs,
and reports whether a regression has more observations than features. The
denoising evaluator emits reconcilable per-record, per-family, and per-task NLL
plus the existing near-certain memorization diagnostic. The frozen
`loss_suite_v1.json` is unchanged.

## Per-task evidence contract

`evals/task_scoreboard.py` parses each full in-memory prediction and emits metric
objects with `value`, `n`, `status`, and `reason`. Missing prediction evidence is
`unavailable` (`null`), never a zero or pass.

Always-eligible metrics are language validity, semantic-AST node/edge F1,
restricted ordered-tree edit similarity, and canonicalized reference-graph
exactness. Canonical exact match is available only when the official lang-core
serializer is active; Lark input-identity serialization is not relabeled as
canonicalization.

Repair/edit suites add target/apply correctness and consume explicit
prediction-side minimality, preservation, localization, noop, undo/redo, and
multi-turn evidence. Visual, behavior, and diffusion metrics likewise require
prediction evidence. The evaluator never credits gold runtime/render evidence
to a prediction.

## L3–L5 equivalence

- L3: required/forbidden constraint satisfaction.
- L4: L3 plus prediction-produced behavior evidence.
- L5: L4 plus prediction-produced render evidence.

Exact string/AST match stays diagnostic and is excluded from the equivalence
aggregate. An L4/L5 row with missing behavior/render evidence remains
unavailable rather than being scored from the gold.

## Generalization slices

`evals/generalization.py` first applies the existing ID, split-group, prompt,
OpenUI, structural, and pair fingerprints. Only decontaminated held-out rows are
classified into unseen component pair/triple, deeper tree, longer program, new
edit composition, new domain/site, and new contract-version slices. Existing
`held_out`/`ood` split values remain unchanged.

## Measured fixture wiring — 2026-07-14

Artifacts: [`task-mixture-wiring-results.json`](task-mixture-wiring-results.json)
and [`task-eval-wiring-results.json`](task-eval-wiring-results.json).

| Run | Recipe | Result | Decision |
| --- | --- | --- | --- |
| task-mixture dry wiring | CPU; 0 train steps; 5 fixture rows; one row in each task group; no checkpoint or NLL score | 81 probes emitted across 24 configured families; 5/5 task groups classified; 0 unclassified; 1 structural/ProgramSpec family; 19 configured families intentionally absent from the tiny fixture | CLI/probe/diagnostic wiring passes. RegMix tuning is **not run** until the real producer families and validation slices land. |
| task/equivalence wiring | CPU; 0 train steps; 5 synthetic prediction-evidence cases across generation, repair, edit, behavior | AST/ref/tree metrics emitted; L3/L4 equivalence evidence `n=2`, fixture score 1.0; 8 metric instances unavailable; canonical exact unavailable because official lang-core serialization was not active | Null/coverage behavior passes. Values are fixture self-consistency, not model quality. |

No checkpoint was created, so the model card and README model-card summary do
not change. A future scored mixture run must append weighted-NLL curves here;
any readiness claim still requires the full honest multi-suite ship-gate flow.
