# Shared contracts (all phases)

Read once per session; every phase reference assumes these.

## Iron law — docs follow every run

No train / eval / bench / profile / telemetry / matrix / reproduction run
without updating docs: JSON under `docs/design/` **and** the matching markdown
measured-results, with recipe metadata (device, steps, backend, matrix set,
suite `n`, honesty mode) and honest pass/fail vs gates. Numbers only in
`outputs/`, chat, or a PR comment are incomplete work. This applies identically
when a run is launched through wrapper commands. REQUIRED SKILL:
`documenting-experiment-results`.

## Hard run cap (every command in every phase)

Every train / eval / bench / profile / telemetry / matrix / reproduction and
supporting shell command must obey `slm_training.levers` (AGENTS.md "Hard run
cap"). Use its derived interrupt and kill-grace values rather than restating
numeric literals.
The command examples below are shown at fixture / smoke scale for this reason —
scale `--steps` / `--rico-limit` / sizes to fit the cap. Where a harness exposes
it (e.g. `scripts.autoresearch`, `scripts.run_scaling_ladder`), pass
`--max-wall-minutes` defaults to and rejects values above
`slm_training.levers.MAX_RUN_MINUTES`.
A timed-out, interrupted, or killed run is never evidence.

## Model-card duty

Every checkpoint that is created, synced, bootstrapped, or promoted updates
`docs/MODEL_CARD.md` (roster, eval table, recipe, bucket URI, history) **and**
the README "Model card (summary)". A checkpoint without both is incomplete.

## Honesty — fixture demo vs ship

Fixture/scratch/smoke evidence is wiring only. Readiness claims require
`--ship-gates` on the full multi-suite scoreboard (smoke, held_out, rico_held,
adversarial, ood) with explicit suite sizes. Never weaken gates to green CI.
REQUIRED SKILL for readiness language: `honest-ship-eval`.

## RL is fail-closed

RL requires an approved `RLReadinessReport` (frozen five-suite evaluation, full
`rico_held`, honest ship gates, AgentV pass, nonzero reward variance) produced
by `autoresearch validate-rl`. There is no override; a rejected report is
evidence, not an obstacle.

## Canonical roots — no shadow paths

Campaign evidence: `outputs/autoresearch/<campaign>/`. Run evidence:
`outputs/runs/<run-id>/`. Versioned data: `outputs/data/{train,eval}/<version>/`
(immutable once published). Durable results: `docs/design/`. Committed fixtures:
`src/slm_training/resources/`. Reuse the canonical scripts and harnesses —
never build a parallel trainer, evaluator, or artifact tree.

## Isolation & provenance

Preserve train/eval disjointness and structural leakage checks; never fit
training data to holdouts or train on eval-feedback records. Keep checkpoint
hashes, lineage labels, and source/license governance intact. Never
reintroduce silent `gold.placeholders` channels under
`honest_slot_contract=True`.

## Approvals

No paid GPU, remote job, or Hugging Face write without explicit user approval.
`HF_TOKEN` is required for bucket/HF-context work and is never committed.

## Quality/retrieval harness

Library-only capability (`src/slm_training/harnesses/quality/`): retrieval,
curriculum, soft-corruption, compact schema context, adversarial synthesis.
It has no CLI — it is exercised through the data, SFT, preference, and
experiments phases. To change it: `improve-openui-harnesses`.
