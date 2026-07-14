# Data synthesis definition of done (P13)

Status: **verified for the bounded fixture-backed corpus**. Machine-readable
evidence: [data-synthesis-results.json](data-synthesis-results.json). This is a
CPU scratch/data-integration result, not a production HF-context ship claim.

## Reproduction recipe

The final corpus was built twice with `source=all`, `synthesizer=none`,
`programspec_count=1`, `rico_limit=1`, and DESIGN.md attached without making it
a quality filter. The two 176-record builds produced the same
`content_fingerprint`:
`9359d47176c0ec75fce9389558af69884a6ce4afc47f89fc0f7412987ad4bf17`.
Every kept record carries contract `b8b156526f65d26c`.

The eval build used the committed fixture + local RICO sources, the train
manifest for split-before-derive filtering, and `max_children=4`. It contains
51 decontaminated records: smoke 3, held_out 5, adversarial 4, OOD 4, and
`rico_held` 35. `diagnose_eval` used the modern 256-token LTR budget.

## Verification scoreboard

| Check | Result |
| --- | --- |
| Pinned bridge | 61 positive contract records round-trip the supported component/ref/list/placeholder surface; state/query/mutation/action remain explicitly unsupported by OpenUI 0.2.9 |
| Families | 10/10 expected families present; every family has at least one root parent (82 roots total) |
| Leakage | 0/51 exact, structural, or `split_group_id` collisions with train |
| Diagnostic ceiling | parse, placeholder fidelity/validity, structure, and component recall = 1.0 on smoke/held_out/adversarial/OOD |
| Length | train p95 93; `rico_held` p95/max 190; zero records over the 256-token budget |
| Verifier tiers | Bronze 153, Silver 21, Gold 2; no failing gate on Gold/Silver |
| Governance | Croissant, Data Card, and SPDX emitted; PII=0, secrets=0, instruction-like=0 |
| Edit invariants | 5/5 deltas satisfy `apply(before, delta) = after` and inverse/undo-redo identity |
| Task/equivalence | generation, repair, edit, behavior present; L3-L5 equivalence 1.0 (`n=2`); unavailable metrics stay null |
| Generalization | 51/51 accepted; 32 longer-program, 5 unseen-pair, and 5 unseen-triple slice memberships; no contamination |

## Matched smoke signal

Both final arms used CPU scratch, 20 steps, batch 4, learning rate `3e-4`, seed
17, 16 decode steps, no DESIGN.md context, the same `rico_held` records (`n=3`),
and unchanged ship-gate policy.

| Arm | Data / system | Parse | Placeholder fidelity | Structure | Reward |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixture control | 17 fixture rows; E0 ship-recipe baseline | 0.0 | 0.0 | 0.0 | 0.0 |
| Integrated champion | 176-row integrated corpus; honest E50 CoRe/V5 stack | 1.0 | 1.0 | 0.7512 | 1.0 |

This proves the integrated champion moves off the fidelity-0 failure under a
bounded matched budget. It does **not** isolate data from objective/decoder
effects: a same-E0, 20-step integrated-data control also stayed at fidelity
0.0. The optional attribution control therefore says the measured gain is a
system-level corpus+objective result, not a data-only causal claim.

The narrow matrix summaries themselves remain `pass=false` because four suites
were intentionally omitted and are reported as missing. No gate was lowered.
Promotion remains blocked on full multi-suite, full-size `rico_held`, multiple
seeds, rank stability, and a production HF-context checkpoint.

## Failed iterations retained

1. The first E0-vs-E0 20-step comparison stayed at fidelity 0.0 in both arms.
2. The first test projection had `rico_held` p95 280 against the legacy
   192-token budget. Rebuilding the same decontaminated source with
   `max_children=4` reduced p95/max to 190; the final diagnostic uses the
   champion's 256-token budget.
3. A full-suite E50 decode was stopped after smoke because it was about one
   minute per record. The final quality smoke evaluates only the acceptance
   signal; all-suite correctness is supplied by the separate deterministic
   ceiling/leakage report, not misrepresented as a model ship clear.

## Re-run

Run `scripts.build_train_data` twice, build test data against the first
manifest, run `scripts.evaluate_tasks` and the two bounded quality-matrix arms,
then compose the fail-closed report:

```bash
python -m scripts.verify_data_synthesis \
  --first-train-dir outputs/slm17/corpus/first \
  --second-train-dir outputs/slm17/corpus/second \
  --test-dir outputs/slm17/test/v2 \
  --task-results outputs/slm17/task-results.json \
  --baseline-matrix outputs/slm17/matrix-smoke-baseline-summary.json \
  --champion-matrix outputs/slm17/matrix-smoke-champion-summary.json \
  --ltr-max-tokens 256 \
  --out docs/design/data-synthesis-results.json
```
