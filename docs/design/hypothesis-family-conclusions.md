# Hypothesis-family conclusions (closed approaches, open goals)

**Status:** WP-4 shipped — `policy.v2.json` `conclusion_policy` block,
`slm_training.autoresearch.conclusions`, append-only
`closed_approaches.v1.json`, and the `concluded_family` preflight plugin.

## Motivation (RC4)

The harness-evolution architecture review
([`harness-evolution-architecture-review-20260809.md`](harness-evolution-architecture-review-20260809.md),
§RC4) found that **"the loop is engineered never to stop, therefore never to
conclude."** Every hard stop had an automatic bypass — capped families were
un-skipped, bank exhaustion converted into a promote attempt, filler
`compose-*` arms were synthesized, confirmed champions re-confirmed just to
have something to run. Adequately powered null evidence never became a
durable conclusion, so the loop re-proposed levers its own closeout docs had
already declared exhausted.

This module makes the conclusion a first-class, durable, machine-checked
artifact — encoding the repository goal law exactly:

> **A rejected experiment closes an approach, never a goal** (AGENTS.md I14).

## Owner artifacts

| Field | Value |
| --- | --- |
| Policy | `src/slm_training/resources/experiments/autotrain_climb/policy.v2.json` (`conclusion_policy` block; superset of `policy.v1.json`, revision v7 → v8, schema family unchanged `autotrain_climb_policy/v1`) |
| Ledger | `src/slm_training/resources/experiments/autotrain_climb/closed_approaches.v1.json` (`closed_approaches/v1`, append-only) |
| Module | `slm_training.autoresearch.conclusions` |
| Plugin | `slm_training.autoresearch.preflight.concluded_family` (`check_id: concluded_family`) |
| Loader | `conclusions.load_policy()` — reads v2 when present, falls back to v1 with a single logged notice; reuses `climb_policy.load_climb_policy` |

## Family key

A hypothesis family is identified by the **canonicalized sorted lever-key
set** of its evidence records:

1. each key is stripped, lowercased, and underscores map to hyphens
   (`Component_Plan` → `component-plan`);
2. keys are deduplicated and sorted;
3. the family key is the join with `+` (e.g. `bounds+canvas`).

Order and duplicates never change the family; `canvas, Bounds, bounds` and
`bounds, canvas` are the same approach.

### Observed lever taxonomy (restriction)

Family membership is **restricted to the lever taxonomy actually observed in
the committed cross-version evidence ledger**
(`resources/experiments/autotrain_climb/evidence_ledger.v1.json`, arm slugs).
As of this writing that taxonomy is:

`balanced-container-close`, `batch1`, `binder-arity`,
`binder-component-plan`, `binder-topology`, `bounds`, `canvas`,
`compiler-decision-margin`, `compiler-decision-token`, `component-edge`,
`component-edge-margin`, `component-edge-token`, `component-inventory`,
`component-plan`, `component-structure`, `constraint-graph`,
`container-close`, `edge-alignment`, `fidelity`, `mixed-mask`,
`semantic-contrast`, `semantic-contrast-compiler-margin`,
`slot-augmentation`, `slot-component-fidelity-coupling`,
`slot-component-inventory-coupling`, `slot-contract-context`.

The taxonomy is derived at runtime from the committed ledger
(`conclusions.known_lever_taxonomy()`), not hardcoded — rebuilding the
evidence ledger extends it. A record naming any lever outside the taxonomy is
**excluded from closure consideration** (fail open): we never conclude about
levers the durable record has not measured.

## Closure semantics

Governed by `policy.v2.json → conclusion_policy`:

```json
{
  "family_close_after_adequately_powered_failures": 3,
  "adequate_power_requires": { "min_seeds": 8, "decidable": true },
  "closed_families_reopen_on": ["new_lever_key", "harness_version_change"]
}
```

- A record is a **failure** when `outcome ∈ {confirm_failed, ship_rejected}`.
- A failure is **adequately powered** when `n_seeds ≥ min_seeds` **and** its
  measurement was decidable (the sign-test power arithmetic of
  `evidence_ledger.power_feasibility_report`; an undecidable null is not
  evidence of absence).
- When a family accumulates
  `family_close_after_adequately_powered_failures` such failures,
  `conclusions.conclude(records)` appends one `ClosedApproachRecord` to
  `closed_approaches.v1.json`.

`ClosedApproachRecord` fields: `id` (content-addressed — SHA-256 of the
family key plus the sorted closing evidence ids), `family_key`,
`closed_at_run_date`, `evidence_record_ids`, `goal_invariant_served`,
`reopen_conditions` (copied from policy at close time), and
`harness_versions` (the eval-comparability component versions in force at
close time, so `harness_version_change` is checkable later).

### Ledger discipline

- **Append-only** — existing entries are never rewritten or deleted.
- **Stable order** — new entries append in sorted family-key order.
- **Idempotent** — re-running `conclude` on the same records adds nothing
  (same content → same `id` → skipped); a family with a still-binding closure
  is not re-closed even from new evidence.

## The closed-approach-never-closed-goal law

`goal_invariant_served` names the AGENTS.md `I*` decode invariant the closed
approach was serving — **when exactly one** invariant is derivable from the
closing records (explicit `goal_invariant` fields, else `I<n>` tokens in
hypothesis text/reasons). Anything ambiguous or absent records `"unknown"`
rather than guessing: **a closed approach never closes a goal.** The goal
invariant stays open; only this particular lever family is retired, and every
record carries the exact conditions under which it comes back. This is the
data-level form of AGENTS.md I14 ("goals are non-negotiable; approaches are
disposable") and of the CLAUDE.md law "a rejected experiment closes an
approach, never a goal."

## Reopen semantics

`is_family_closed(family_key, current_lever_keys, harness_versions)` returns
`True` only while a closure record is **binding**. It stops binding when a
recorded reopen condition holds:

| Condition | Meaning |
| --- | --- |
| `new_lever_key` | Any current lever key (canonicalized) not in the closing family — the candidate is a *different* approach touching this family, not a replay |
| `harness_version_change` | Any harness component version present in both the record's stored `harness_versions` and the current versions differs — the measurement regime changed, so the old nulls no longer bind |

A record without stored harness versions cannot detect version change and
stays binding on that axis (conservative). After a reopen, `conclude` may
close the family again on fresh evidence — as a **new appended record**, never
by editing the old one.

## Preflight plugin

`preflight/concluded_family.py` exposes the module-level `CHECK`
(`check_id: "concluded_family"`). For a candidate
(`{hypothesis_text, lever_keys, config_fingerprint, n_seeds, steps,
minimum_effect, endpoint_metric}`):

- **block** — the candidate's family has a binding closure; reasons cite the
  `ClosedApproachRecord` id, closing evidence, the served goal invariant
  (explicitly noted as *still open*), and the unmet reopen conditions.
- **pass** — no lever keys, no binding closure, or a reopen condition holds.
- The plugin **never raises**; internal errors fail open with a typed reason
  (a check bug must not silently halt research).

## Worked example

1. Cycles c1801, c1815, c1834 each run the `bounds+canvas` family with 8
   seeds, decidable power, and end `confirm_failed`.
2. `conclude([...])` appends:

   ```json
   {
     "id": "<sha256 of family key + [c1801, c1815, c1834]>",
     "family_key": "bounds+canvas",
     "closed_at_run_date": "2026-08-09",
     "evidence_record_ids": ["c1801", "c1815", "c1834"],
     "goal_invariant_served": "I10",
     "reopen_conditions": ["new_lever_key", "harness_version_change"],
     "harness_versions": { "evals.scoring": "v22", "gates.ship": "openui_ship_gates_v6" }
   }
   ```

3. A later candidate proposing `lever_keys = ["bounds", "canvas"]` is
   **blocked** by `concluded_family`, citing that record id. The I10 goal
   (use-case ladder) remains open.
4. A candidate proposing `["bounds", "canvas", "binder-topology"]` **passes**
   (`new_lever_key`), as does the original family after `evals.scoring` bumps
   to v23 (`harness_version_change`) — at which point three fresh adequately
   powered failures may close it again as a new record.

## Related

- [`autotrain-climb-policy.md`](autotrain-climb-policy.md) — §"Policy v2" for
  the externalized block
- [`harness-evolution-architecture-review-20260809.md`](harness-evolution-architecture-review-20260809.md) — RC4
- [`darkfactory-hillclimb-optimization.md`](darkfactory-hillclimb-optimization.md) — evidence ledger + power gate this builds on
- Tests: `tests/test_autoresearch/test_conclusions_family_closure.py`
