# E121 judged-corpus E53 iteration (2026-07-16)

E121 is the first bounded V6 E53 iteration wired explicitly to the committed
`remediated_roots_judged` corpus. The prior invocation silently used the stale
`outputs/train_data/v1_curriculum` snapshot; the matrix harness now maps an
explicit `--train-dir` to curriculum input unless the caller supplies a separate
curriculum path.

The CPU scratch train used 405 judge-approved records for 8 steps, followed by
the E53 trust gate’s 30 steps. Final training loss was **92.6579** and weighted
held-out NLL was **31.1429**. The checkpoint is local-only and not promotable.
Training telemetry is persisted in the run directory; `loss_suites` dominated
44.35% of the 25,993 ms measured cycle, followed by backward (15.99%) and
forward (15.58%). The loss-suite AgentEvals record is complete and explicitly
diagnostic, not a ship evaluation.

The five-suite diagnostic matrix (`smoke`, `held_out`, `adversarial`, `ood`,
`rico_held`, each capped at 3 and `rico_held` n=3) exceeded the 120-second CPU
wall limit under E53 `best_of_n=4` before suite rows were emitted. A bounded
one-record smoke evaluation with two decode steps, `best_of_n=1`, chosen-token
verification, top-k 4, and a 5-second per-record timeout produced parse **0.0**,
fidelity **0.0**, structural similarity **0.0**, reward **0.0**, and one decode
timeout at 5,001 ms. This is a valid negative diagnostic result, not evidence
that the model passed or that the parser is wrong.

A later completed E121c matrix artifact emitted all five suite rows before the
same CPU budget was exhausted. It recorded parse rates from 0.667 to 1.0 and
structural similarity from 0.3512 to 0.6529, but placeholder fidelity was
**0.0 in every suite**, so the ship gate still failed. The artifact is kept as
durable negative evidence at
`quality-matrix-results-iter-e121c-e53-judged-20260715.json`.

During the run, evaluation exposed and fixed three harness defects: a duplicate
`--output-tokenizer` CLI registration, a missing `(predictions, evidence)` tuple
return from the `generate_with_stats` path, and silent replacement of an
explicit training corpus by the default curriculum snapshot. Regression tests
cover the single-record stats path. The next loop should profile or simplify
constrained generation, then rerun a longer judged-corpus train; no ship claim
is made from E121.

Evidence: [iter-e121-judged-corpus-e53-20260715.json](iter-e121-judged-corpus-e53-20260715.json),
the training run under `outputs/runs/iter-e121d-e53-judged-20260715/`, and the
AgentV smoke bundle under
`outputs/runs/iter-e121d-e53-judged-20260715/iter-e121d-smoke-timeout-diagnostic/agentv/`.
