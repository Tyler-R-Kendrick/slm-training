# SLM-298 — local capacity × context × complexity curriculum (2026-07-25)

Evidence: [JSON](iter-slm298-capacity-context-curriculum-20260725.json).

Status: **rejected diagnostic; not promotable or ship.** The completed local
cells all produce syntax-valid constrained output, but strict meaningful-program
rate and binder-reference F1 are both 0.0. Syntax alone is not the primary
metric.

## Recipe

The preregistered local factorial fixes `d_model` 32/64, scratch trainable versus
scratch frozen context, flat versus AST/reference-derived A/B/C curriculum, and
seeds 0/1/2. Every arm uses the same 520-record strict symbol-only snapshot
(`a118e70f…f1eabc`), 5,000 target tokens, CPU, lexer output, constrained native
and compiler evaluation, and a disjoint locked diagnostic `n=1`. AgentV was
published for each completed cell. No human-rating gate, remote replay, HF job,
or checkpoint sync was used.

The strict data build retained 520 records without weakening gates. Its feedback
flags high rejection, nine eval-overlap rejections, and 27 placeholder-contract
violations; rejected rows were not re-admitted. The source-family cleanup is a
separate follow-up, not a post-hoc change to this experiment.

## Results

| Coverage | Syntax | Strict meaningful / binder F1 | Interpretation |
| --- | ---: | ---: | --- |
| 20 completed cells | 1.0 | 0.0 / 0.0 | No semantic signal despite legal syntax |
| d64 complete 2×2×3 factorial | 1.0 | 0.0 / 0.0 | Ordered−flat and trainable−frozen effects are 0.0; degenerate 95% CIs `[0.0, 0.0]` |
| d32 paired seeds 0/1 | 1.0 | 0.0 / 0.0 | No effect estimate claimed: all four seed-2 cells exceeded the fixed cap |

The four d32 seed-2 cells were each retried serially and exceeded the derived
170-second interrupt limit during constrained completion/repair. They have no
result JSON and are **non-evidence**, not zero-filled observations. This leaves
the d32 three-seed replication incomplete. The complete d64 factorial still
falsifies the local hypothesis that capacity, trainability, or curriculum would
lift the primary semantic metric under this bounded recipe.

## Decision

Do not promote any SLM-298 checkpoint, change a default, or claim readiness.
Keep the shared harness repair (ordered views preserve every output-contract
field) and use the locked-cap limitation as input to a separately preregistered
evaluation-efficiency investigation. Ship gates were not run: locked `n=1` is
wiring/diagnostic evidence only.
