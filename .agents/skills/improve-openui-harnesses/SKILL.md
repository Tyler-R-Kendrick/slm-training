---
name: improve-openui-harnesses
description: >-
  Improve, extend, debug, or review any canonical OpenUI training harness,
  including autoresearch/hypothesizer, annotations, distillation, experiment
  selection/promotion, model build/evaluation, preference learning,
  quality/retrieval, RL, held-out test data, or training data. Use for changes
  under src/slm_training/harnesses/, src/slm_training/autoresearch/, or their
  scripts, tests, outputs, gates, and experiment integrations.
---

# Improve OpenUI harnesses

Preserve one shared research-to-results system. Improve the existing owner instead
of adding a parallel trainer, evaluator, artifact tree, or policy.

## Workflow

1. Read `AGENTS.md`, `docs/repository-organization.md`, and the design document
   named for the harness.
2. Identify the harness family in [references/harnesses.md](references/harnesses.md)
   and read that entire section before editing.
3. Trace its public script to the library owner, downstream artifact consumers,
   tests, and ship/promotion gates.
4. Change the shared owner. Keep schemas strict, paths canonical, and untrusted
   model/research output behind typed compilation and validation. If the file
   is watched by a component in `src/slm_training/resources/versions.json`
   (metrics, gates, eval/train harnesses, matrices, test-data builder), bump
   that component's version — or append a same-version `no-bump: <reason>`
   history entry for behavior-neutral edits — in the same change
   (`docs/design/version-stamp-contract.md`).
5. Add the smallest regression test that proves the new invariant. For a train,
   eval, benchmark, profile, telemetry, matrix, or reproduction run, also use
   `documenting-experiment-results`; for readiness claims use `honest-ship-eval`.
6. Run the family checks from the reference, `python -m scripts.repo_policy`,
   `python -m scripts.verify_version_stamps --check`, `.githooks/check-changed`,
   and `git diff --check`.

## Shared contracts

- Keep raw campaign evidence in `outputs/autoresearch/<campaign>/`, ordinary run
  evidence in `outputs/runs/<run-id>/`, versioned data in `outputs/data/{train,eval}/`,
  and durable measured results in `docs/design/`.
- Reuse `scripts/train_model.py`, `scripts.evaluate_model.py`, AgentV publication,
  lineage records, and the existing promotion gates. Never build shadow paths.
- Preserve train/eval isolation, immutable data snapshots, honest slot contracts,
  checkpoint/model-card requirements, and fail-closed RL readiness.
- Treat fixture/smoke evidence as wiring only. Never weaken gates to promote it.
- Feed persisted outcomes and diagnoses back through evidence or typed feedback;
  never let a harness rewrite its own code, frozen cases, or acceptance thresholds.
- Add a new harness family only when no listed owner fits, then update the reference,
  repository guide, policy, docs, and tests in the same change.

## Improvement evidence

An improvement is incomplete until the relevant invariant has a focused test and,
when execution occurred, the canonical JSON plus markdown record states recipe,
suite size, result, and honest pass/fail. A self-improvement claim additionally
needs frozen evaluation cases, held-out results, and explicit human promotion.
