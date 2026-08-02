# Autotrain c1797: semantic-contrast frozen replay incomplete

**Verdict:** shared runtime incompleteness, not a semantic-contrast model
rejection. The exact c1796 checkpoints and manifests were reused. Both arms
completed held-out but timed out on one of the same three smoke positions, so
the preregistered comparison remains non-scoreable.

| Reused arm | Source params | Smoke complete | Held-out complete | Decision |
| --- | ---: | ---: | ---: | --- |
| semantic contrast 0.25 | 1,608,962 | 2/3 (1 timeout) | 5/5 | incomplete |
| matched control 0.0 | 1,608,962 | 2/3 (1 timeout) | 5/5 | incomplete |

No training ran and no checkpoint was created. The replay is content-bound to
the c1796 manifests and source checkpoint hashes (`feb745fc…efb4` treatment,
`8ad84a58…84be0` control). The source recipe remains CPU scratch, 22 steps,
batch 2, seed 101796, 835 matched contrast pairs, margin 1, and fraction .5.

The v76 evaluator allocated a 37 s cumulative wall but silently serialized a
TwoTower that exposes both production batch requests and single-record stats.
That left only about 4.20–4.70 s per sequential smoke record. Both arms timed
out once, while all five held-out documents completed. Their partial metrics
cover the same counts but still exclude a timed-out record and therefore do
not authorize a quality delta. AgentV completed both bundles with zero
execution errors; both gate reports fail. Lean is
`not_applicable:retry_measurement` because no champion exists.

The original terminal projection incorrectly tested only whether the candidate
timed out, then described the result as “candidate-only” without proving the
control completed. Campaign harness v95 requires exclusivity before a runtime
arm can be rejected and carries the source trainable-parameter count into
reused outcomes. Eval harness v77 restores I4 batched request execution even
when single-record stats are also available; `collect_decode_stats` retains
row-tagged evidence, and each scoreboard now discloses configured/max batch
size and chunk count.

Next priority: replay the exact pair once under eval v77 after the canonical
repair. A complete result may be compared. If both arms remain incomplete,
continue runtime diagnosis; if exactly one arm alone reproduces a timeout,
retire only that arm. Promotion, Lean preflight, and ship remain closed.

Machine evidence:
[`autotrain-cycle-1797-semantic-contrast-replay-incomplete.json`](autotrain-cycle-1797-semantic-contrast-replay-incomplete.json).
