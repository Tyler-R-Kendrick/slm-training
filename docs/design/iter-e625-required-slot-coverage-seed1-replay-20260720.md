# E625: seed-1 replay of E620's required-slot-coverage control/treatment

E620 tested `schema_role_slot_decode_weight` (0 control vs 8 treatment) on a
seed=0, 800-step scratch checkpoint and found the treatment kept E615's
fidelity/validity/reward/latency gains but never cleared strict meaning-v2 or
required-slot coverage. E625 asks whether that result is seed-0-specific by
replaying the identical recipe on a fresh seed=1 checkpoint.

## Setup

A clean CPU scratch train completed 800 steps in 34.20 seconds (well under the
three-minute cap), on the same published corpus
(`e530_visible_semantic_roles_r2_20260719`, 244 records) and the same
`twotower` / `choice`-tokenizer / scratch-backend recipe as E620, with only
`--seed 1` changed. Loss reached 4.5998 (E620's seed=0 run reached 4.0680;
loss is not directly comparable across seeds/inits). Checkpoint SHA-256
`86573aa5…dc2b7b866`; local-only, not synced or promoted.

Both eval arms reused this identical checkpoint and E620's full evaluation
policy (`honest_slot_contract`, `slot_contract_constrained_decode`,
`slot_contract_in_context`, public semantic-role schema candidates, retained
semantic-plan weights). Only `schema_role_slot_decode_weight` changed: 0 for
control, 8 for treatment. Both runs emitted AgentEvals JSONL and an AgentV SDK
bundle.

## Measured result

| Metric | Control (seed1) | Treatment (seed1) | Delta | E620 control (seed0) | E620 treatment (seed0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 | 0 | 1.0000 | 1.0000 |
| meaningful v1 | 0.7500 | 0.7500 | 0 | 0.5000 | 0.5000 |
| strict meaning v2 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.0000 |
| v2 coverage | 1.0000 | 1.0000 | 0 | 1.0000 | 1.0000 |
| placeholder fidelity | 0.5750 | 0.7833 | +0.2083 | 0.5083 | 0.5500 |
| placeholder validity | 0.7450 | 0.8700 | +0.1250 | 0.7050 | 0.7300 |
| structural similarity | 0.6167 | 0.6081 | -0.0086 | 0.4569 | 0.4886 |
| component recall | 0.7500 | 0.6667 | -0.0833 | 0.5417 | 0.4792 |
| reward | 0.8358 | 0.8983 | +0.0625 | 0.7910 | 0.8140 |
| AST node / edge F1 | 0.6270 / 0.4167 | 0.5913 / 0.3500 | -0.036 / -0.067 | 0.5556 / 0.3750 | 0.5437 / 0.3750 |
| latency p50 / p95 | 906.5 / 4959.0 ms | 997.5 / 5615.3 ms | +91.0 / +656.3 ms | 3366.95 / 14562.73 ms | 3116.67 / 13704.44 ms |
| AgentV | 0/1 | 0/1 | 0 | 0/1 | 0/1 |

No decode timeout or fallback in either arm. `binding_aware_meaningful_v2_rate_strict`
is 0.0 in all four runs across both seeds (E620 control, E620 treatment, this
control, this treatment) — the strict-meaning null result is seed-robust.

## Per-record reason codes

| Record | Control (seed1) | Treatment (seed1) |
| --- | --- | --- |
| dashboard | prompt_component_missing, required_component_missing, required_placeholder_missing, schema_value_role_mismatch:Callout.variant | unchanged (all four codes persist) |
| gallery | required_placeholder_missing, schema_value_role_mismatch:ImageGallery.images | unchanged (both codes persist) |
| modal | placeholder_semantic_role_mismatch, placeholder_spam, required_placeholder_missing, schema_value_role_mismatch:Modal.open | required_placeholder_missing clears; role_mismatch/spam/schema_value codes persist |
| auth | placeholder_semantic_role_mismatch, placeholder_spam, required_placeholder_missing | required_placeholder_missing and placeholder_spam clear; role_mismatch persists |

(`verifier_g8_failed` fires on every record in both arms and is omitted above
for brevity.) At seed1, treatment clears `required_placeholder_missing` on 2
of 4 records (modal, auth); at seed0 (E620) treatment cleared it on 1 of 4.
Dashboard and gallery keep missing required placeholders in every arm at both
seeds — those two records need component/property closure, not more role
biasing.

## What replicates vs. what doesn't

Replicates across seeds:
- Strict meaning-v2 stays 0/4 in every control and treatment arm (4/4 runs,
  both seeds). The lever does not, by itself, close required-slot coverage.
- Direction of the treatment effect: fidelity, validity, and reward all rise
  under treatment; component-type recall falls. Magnitude is larger at
  seed1 than seed0 (e.g. fidelity delta +0.2083 vs +0.0417), so a single-seed
  point estimate should not be read as the lever's expected effect size.
- Dashboard and gallery's specific failure signatures (prompt-component and
  required-placeholder gaps) persist unchanged across both seeds and both
  decode arms — these look like structural coverage gaps, not decode-weight
  noise.

Does not replicate:
- `structural_similarity`'s sign: treatment is higher than control at seed0
  (0.4569 -> 0.4886) but lower at seed1 (0.6167 -> 0.6081). Treat this metric
  as noisy at `n=4` until a multi-seed mean is computed.
- The count of records still missing required placeholders under treatment
  (3/4 at seed0, 2/4 at seed1) — seed-sensitive, not a stable per-seed
  estimate.

## Decision

Retain E615's `schema_role_slot_decode_weight=8` bias as a decode-time
default candidate: its fidelity/validity/reward benefit direction is
seed-robust across two independent seeds, not an artifact of seed=0. Reject
it as a fix for required-slot coverage or strict meaning-v2: those stay
exactly 0/4 in every one of the four E619/E620/E625 control-and-treatment
runs run so far. The next useful lever is a decode-time bias that floors
still-missing *required slots* the way `semantic_plan_margin_decode_weight`
already floors still-required *plan families* (component types) — targeting
dashboard/gallery's persistent `required_placeholder_missing` failures
specifically — not another seed or duration replay of the existing role-bias
lever.

Honest caveats:

- `n=4` per arm; every "rate" here is a fraction of 4 and single flips move
  it by 0.25. Treat point deltas as directional, not precise, until a larger
  OOD suite or more seeds are run.
- No model was promoted or synced; this is scratch-matrix wiring/comparison
  evidence, not a ship claim.
- Loss is not comparable across seeds (different random init), so the
  seed0-vs-seed1 loss delta is not evidence of anything about the lever.

Evidence:
[iter-e625-required-slot-coverage-seed1-replay-20260720.json](iter-e625-required-slot-coverage-seed1-replay-20260720.json).
