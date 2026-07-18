# LDI3-04 — Immutable remine → intervene → regenerate campaign (SLM-132)

**Date:** 2026-07-18. **Status:** fail-closed campaign config + per-stage fingerprint,
content-addressed stage DAG (resume / immutability / duplicate-safe / invalidating),
admission-gated iteration lifecycle, failure-signature migration tables, deterministic
stop rules, and a bounded **one-iteration wiring-only fixture smoke** — implemented and
tested. **Torch-free fixture path; no model generation/training, no GPU run, no RL, no
frontier quality claim, no ship/default-on decision.**

Unblocked by the merged SLM-128 (structured objectives), SLM-129 (structural-slop
forensics), SLM-131 (counterfactual action evidence), and SLM-126 (removable TwoTower
adapter), all on `main`.

## What this issue delivers

The OpenUI derivative of Auto-Antislop, as an **autoresearch campaign integration — not
a new pipeline and not a new scheduler**. Immutability, content-addressed artifacts, and
the hash-chained event log come from the existing `autoresearch.storage.CampaignStore`;
content hashing from `lineage.records.content_sha`. The four collaborators sit behind
narrow backends so the model-generation/training surface stays out of the CPU smoke.

- `src/slm_training/harnesses/preference/remine_campaign.py`
  - **`RemineCampaignConfig`** — versioned, frozen, **fail-closed** (`from_mapping`
    rejects unknown fields), `fingerprint()` copied into every stage. A deterministic
    `created_at` keeps `CampaignStore.initialize` idempotent on resume.
  - **Stage DAG** `STAGES_ITER0` / `STAGES_ITERN` with a `_Ledger` layered on the store:
    each stage keyed by `(iteration, stage, input_fingerprint)`; a matching prior marker
    is **reused** (resume, duplicate-safe), a changed upstream fingerprint **misses** the
    marker and re-runs the stage (invalidation).
  - **Iteration lifecycle**: iteration 0 evaluates the parent and records
    `train_authorized | repair_evidence | no_safe_direction | expired` — **no training
    unless admission authorizes it**; iteration *n* trains one **fresh removable adapter**
    with explicit parent lineage (never auto-merged), regenerates, and migrates failures.
  - **Failure-signature migration** (`migrate_signatures`) → `repaired / persisted /
    regressed / newly_exposed / unresolved`. A disappearance is `repaired` only when the
    signature was **supported** by admissible evidence; an unsupported disappearance is
    `unresolved` (aggregate disappearance from timeout/fallback is not repair).
  - **Deterministic stop rules** (`evaluate_stop_rules`) — the full predeclared policy,
    order-fixed; default max is two trained iterations.
  - `FixtureBackend` — deterministic, torch-free generation + training for the smoke.
- `scripts/run_remine_campaign.py` — `--describe` (resolve config/stages/arms/identities,
  no run) and `--smoke` (bounded one-iteration wiring-only run under a campaign root).

## Immutability, resume, and duplicate-safety

Every stage writes a content-addressed artifact and an append-only, hash-chained
`remine_stage_completed` event via `CampaignStore`. Re-running the same campaign reuses
every completed stage and **persists a byte-identical manifest** (the manifest excludes
run-vs-reuse counters). Changing an upstream config field re-fingerprints downstream
stages so they re-run. A **completed training that restores the parent** (no repaired
signatures) is recorded honestly as a valid negative — the campaign stops with
`no_meaningful_end_to_end_improvement`, not a fabricated win.

## Fixture smoke evidence

`run_remine_campaign --smoke` (two prompt groups, two seeds) produces: iteration 0
`train_authorized`; iteration 1 trains one adapter and repairs one motif + one gate per
group; iteration 2 trains a lineage-child adapter, finds no further repair, and stops.
Committed manifest: `docs/design/ldi3-04-remine-intervene-campaign-report-20260718.json`
(config fingerprint `6dd6a07b942d7f6d`, status `wiring_only`).

## Verification

```bash
python -m pytest tests/test_harnesses/preference/test_remine_campaign.py -q   # 20 passed
python -m pytest tests/test_harnesses/preference tests/test_autoresearch -q
python -m scripts.run_remine_campaign --describe
python -m scripts.run_remine_campaign --smoke --root outputs/autoresearch
python -m ruff check src/slm_training/harnesses/preference/remine_campaign.py scripts/run_remine_campaign.py
python -m scripts.repo_policy
```

Tests cover: unknown-field / out-of-contract config fail-closed and deterministic
fingerprint; the wiring-only smoke publishes all artifacts + manifest; resume reuses
every stage and matches the manifest (duplicate-safe); a changed upstream config
invalidates downstream reuse; training is skipped on `no_safe_direction`; every
migration category (incl. unsupported-disappearance = unresolved, not repair); every
deterministic stop rule + max-iteration; and explicit adapter lineage with no automatic
merge/stacking.

## Scope / deferred (frontier quality-bearing run)

Real on-policy generation, adapter training, the G0–G12 verifier stack, and the
five-suite/AgentV/locality evaluation are injected as real backends in the frontier run,
which then updates the canonical experiment matrix, research lineage, and model card with
exact commands and cost. This issue commits only the immutable lifecycle scaffolding,
the migration/stop semantics, and the CPU wiring-only smoke; no quality claim is derived
from `--describe` or the fixture. Out of scope here: RL (unless the RL-readiness gate
separately passes), automatic adapter composition/routing, and any SAE/ReFT actuator.
