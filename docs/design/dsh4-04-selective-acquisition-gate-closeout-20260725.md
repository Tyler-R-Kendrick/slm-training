# SLM-389 / DSH4-04: selective operator-state acquisition entry-gate closeout (dsh4_04_gate_closeout)

Matrix set: `dsh4-selective-acquisition` · Version: `slm389-v1` · Status: **closeout**
Decision: **not_authorized** — no production code.

## Gate assessment

DSH4-04's own Decision statement carries an explicit entry condition, not a
general question: *"Determine which compiler or topology states merit
expensive teacher supervision **after legal-action distillation has
demonstrated value**."* The only completed distillation experiment in this
line is SLM-388 (DSH4-03), and it did not demonstrate that value.

| Gate | Issue | Required | Observed | Passed |
| --- | --- | --- | --- | --- |
| DSH4-03 | SLM-388 | legal-action distillation demonstrates value over the nondistilled certified controller | `no_go_defer` — no teacher-informed arm cleared its acceptance gates ([json](dsh4-03-operator-verifier-kd-defer-fixture-20260725.json)) | **False** |

Detail, from the DSH4-03 evidence:

* `teacher_argmax` / `offline_conditional_kd` (pure teacher imitation)
  changed 100% of eligible decisions but were **wrong 28/30 times**.
* `verifier_ranking_kd` converged to the **same** top-1 pick as the
  nondistilled baseline on every eligible decision and was correctly
  **rejected** as prediction-identical rather than credited.
* `verifier_ranking_kd_defer` deferred on 100% of eligible decisions; its
  fallback (raw compiler order) did not itself beat the baseline either.
* `hard_accepted_set`, the arm that *did* improve held-out CAP2-proxy
  quality on every seed, is DSH4-03's disjoint-trained hard-label reference
  **control** — not a distillation arm. Its strength does not establish
  that teacher- or verifier-informed distillation has demonstrated value;
  if anything it shows the nondistilled prior already captures what the
  synthetic teacher does not add.

## Why this is a closeout, not a run

DSH4-04 asks *which acquisition policy* (uniform, student entropy, student
-teacher divergence, low accepted rank, verifier failure cone, high regret,
or a preregistered hybrid) most efficiently spends a fixed teacher-query
budget. Running that comparison presupposes a teacher signal worth
acquiring in the first place. Since DSH4-03 found the available (synthetic)
teacher carries no real signal on its fixture and no verifier-anchored
combination of it clears gates, there is nothing yet for a selective
-acquisition policy to selectively acquire more efficiently than uniform
acquisition would. Per this repository's evidence-before-implementation
contract (AGENTS.md Iron Law: docs/evidence before claims; never build
ahead of an unmet gate), DSH4-04's architecture — acquisition-reason
telemetry, the seven-arm comparator, the paired held-out evaluation — is
not authorized to be implemented against this entry condition.

## Consequences

- No selective-acquisition policy, uniform-vs-selective comparator, or
  query-budget harness is implemented.
- No config, checkpoint, or production default changes.
- DSH4-04 (SLM-389) stays closed pending a distillation line that clears
  DSH4-03-equivalent gates.

## Reopening conditions

Reopen only when a legal-action distillation experiment — a DSH4-03 rerun
with a non-synthetic/real teacher, or a successor issue — returns a `go`
verdict: at least one teacher-informed arm (`teacher_argmax`,
`offline_conditional_kd`, `verifier_ranking_kd`, or
`verifier_ranking_kd_defer`) clears every DSH4-03 acceptance gate (`>=5%`
of eligible choices changed, correct changes exceed wrong changes, held
-out CAP2-proxy or the frozen `dsh3-13` suite improves across seeds, and
the comparison is not prediction-identical-rejected).

## Verification

This is a design-only closeout: no harness code, tests, or `versions.json`
component changed. The cited evidence (`dsh4-03-operator-verifier-kd-defer
-fixture-20260725.json`) is reused unmodified from the already-merged-to
-branch SLM-388 change; no new run was executed.
