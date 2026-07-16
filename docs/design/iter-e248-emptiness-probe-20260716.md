# E248 — A1 emptiness probe: is valid-but-empty a constraint-distortion artifact? (2026-07-16)

Diagnostic, not a train/ship run. Machine-readable evidence:
[iter-e248-emptiness-probe-20260716.json](iter-e248-emptiness-probe-20260716.json).
Code: [`src/slm_training/evals/emptiness_probe.py`](../../src/slm_training/evals/emptiness_probe.py),
CLI [`scripts/probe_emptiness.py`](../../scripts/probe_emptiness.py). Linear SLM-20.

## Question

MODEL_CARD E224–E236 report **syntax parse = 1.0 with meaningful parse ≈ 0**:
the constrained decoder emits grammatically valid but trivial/empty layouts, and
component/edge/binder supervision (E231–E236) *learns* yet does not change the
decoded output. Grammar-Aligned Decoding / ASAp (Park et al., NeurIPS 2024) shows
that ranking grammar-valid completions by model score distorts the model
distribution; with a length prior the shortest valid program (the empty
document) can become the argmax. **Is the emptiness a decode-time length-bias /
constraint-distortion effect, or a genuine content-modeling failure?**

## Method

For each held-out record, score the model's fully-masked (MaskGIT first-step,
mean-field) reconstruction NLL of two grammar-valid programs under the same
context (reusing the `denoising_nll` masking/scoring primitives):

- `y_pop` — the gold populated program (`ExampleRecord.openui`);
- `y_empty` — the deterministic minimal valid program for the active DSL
  (first of `root = Stack([])`, `root = Stack([], "column")`,
  `root = Container([])`, … that validates through the official parser).

Decomposition (the point of the probe):

- **total** sequence NLL — what a score-ranking constrained decoder actually
  compares; shorter is cheaper, so this exposes the length bias;
- **per-token** NLL — length-controlled; if the populated program is cheaper per
  token, the model *does* prefer real content locally.

Verdict rule: empty preferred on total but not per-token ⇒
`length_bias_constraint_distortion` (fix at decode time — A2/A4/E2). Empty
preferred per-token too ⇒ `content_modeling_failure` (needs representation /
training work — Tracks B/C/D).

## Result (wiring evidence only)

Recipe: CPU, `playground_demo` fixture checkpoint (`last.pt`), grammar
`openui`, eval suites from `resources/data/eval/remediated`, `--limit 12`,
AgentV bundle published (`@agentv/core` 4.42.4). **This is the only committed
checkpoint; it is a wiring fixture, not an E224+ diagnostic**, so the numbers
below prove the probe works, not a property of the frontier models.

| Suite | n | empty pref (total) | empty pref (per-token) | mean margin/token (nats) | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| smoke | 3 | 1.00 | 1.00 | 7.85 | content_modeling_failure |
| held_out | 5 | 1.00 | 1.00 | 12.14 | content_modeling_failure |
| adversarial | 4 | 1.00 | 1.00 | 7.61 | content_modeling_failure |
| ood | 4 | 1.00 | 1.00 | 12.71 | content_modeling_failure |

On the fixture demo model the empty program is cheaper on **both** axes — as
expected for a wiring checkpoint that never learned layout content. The large
positive per-token margins confirm the probe separates the two axes with signal
to spare.

## Verdict and next steps

- **The probe is implemented, tested, and validated end-to-end** (four unit
  tests; deterministic; honest AgentV envelope with `claim=diagnostic_not_ship`).
- **The real A1 question is unanswered here**: it requires running
  `scripts/probe_emptiness.py` against the local **E224+ checkpoints**, which are
  gitignored under `outputs/` and absent from a fresh clone. That run decides
  whether the frontier wall is decode-time length-bias (⇒ prioritize A2 ASAp
  reweighting, A4 min-content contracts, E2 semantic-density gates) or a
  content-modeling failure (⇒ prioritize Tracks B/C/D representation work).
- Until that run exists, no claim is made about the cause of the E224+ wall.

## Honesty

Fixture/scratch checkpoints are wiring evidence only; this changes no ship gate
and promotes nothing. `content_modeling_failure` on the demo model is not a
finding about any trained frontier checkpoint.
