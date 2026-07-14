# Data synthesis definition of done (P13)

Status: **NO-GO — corpus integrity passes, matched learned signal fails**.
Machine-readable evidence:
[data-synthesis-results.json](data-synthesis-results.json). This is a bounded
CPU scratch result, not a production HF-context ship claim.

## Reproduction recipe

The verification corpus was built twice with `source=all`,
`synthesizer=none`, `programspec_count=1`, and `rico_limit=1`. Both 176-record
builds produced content fingerprint
`9359d47176c0ec75fce9389558af69884a6ce4afc47f89fc0f7412987ad4bf17`;
every kept row carries contract `b8b156526f65d26c`.

The static eval projection contains 51 decontaminated rows: smoke 3,
held-out 5, adversarial 4, OOD 4, and `rico_held` 35. Diagnostics use the
modern 256-token LTR budget. The matched learned comparison uses separate
fixture (103 rows) and integrated (689 rows) curriculum corpora and the same
E53 recipe on each arm:

- CPU scratch context; seed 0; 80 train steps; batch 4; lr `3e-4`;
- E53 slot-aware trust stage, 30 steps, then the resulting checkpoint;
- schema + slot contract in context, constrained slot decode, and
  `honest_slot_contract=true` (hidden gold inventory forbidden);
- parallel eight-step decode, best-of-1, template fill and LTR repair off;
- no DESIGN.md context; held-out n=5 and `rico_held` n=4;
- unchanged repository ship-gate policy.

The two checkpoints are local scratch artifacts with the matrix no-sync
rationale. They are not reusable production champions and were not uploaded.

## Verification scoreboard

| Check | Result |
| --- | --- |
| Pinned bridge | 122 positive contract rows round-trip the supported component/ref/list/placeholder surface; state/query/mutation/action remain explicitly deferred by OpenUI 0.2.9 |
| Families | 10/10 expected families present; every family has at least one root parent |
| Leakage | 0/51 exact, structural, or `split_group_id` collisions with train |
| Diagnostic ceiling | parse, fidelity/validity, structure, and component recall = 1.0 on smoke/held-out/adversarial/OOD |
| Length | train p95 93; `rico_held` p95/max 190; zero rows over 256 tokens |
| Verifier tiers | Bronze 153, Silver 21, Gold 2; no failing Gold/Silver gate |
| Governance | Croissant, Data Card, and SPDX emitted; PII=0, secrets=0, instruction-like=0 |
| Edit invariants | 5/5 deltas satisfy apply and inverse/undo-redo identity |
| Task/equivalence | generation, repair, edit, behavior, visual wired; L3/L4/L5 fixture equivalence 1.0 (`n=3`) |
| Generalization | 51/51 accepted; longer-program, unseen-pair, and unseen-triple slices populated and clean |
| Matched quality | **Fail:** both arms tie on both required suites; no positive data signal |

## Matched learned result

| Corpus / checkpoint | Suite | n | Parse | Fidelity | Structure | Reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fixture E53 (`7e9cedb3`) | `held_out` | 5 | 0.0 | 0.2 | 0.0 | 0.0 |
| Integrated E53 (`b7446345`) | `held_out` | 5 | 0.0 | 0.2 | 0.0 | 0.0 |
| Fixture E53 (`7e9cedb3`) | `rico_held` | 4 | 0.0 | 0.5278 | 0.0 | 0.0 |
| Integrated E53 (`b7446345`) | `rico_held` | 4 | 0.0 | 0.5278 | 0.0 | 0.0 |

The fidelity deltas are 0.0 on both suites, so SLM-17's required strict
improvement is not met. Both bounded scoreboards also fail unchanged ship
gates: all generated programs fail parsing/structure, and the three omitted
suites are reported missing. The verifier exits nonzero and promotion remains
blocked.

## Superseded probes

1. A 20-step E0-vs-E0 data-only probe stayed at fidelity 0.0.
2. The original PR artifact compared fixture E0 against integrated E50 on
   `rico_held` only. It reached fidelity 1.0, but experiment/decode differences
   meant it was not a matched data comparison. The verifier now rejects that
   shape.
3. A template-fill E53 probe saturated the fixture path and was too slow to
   attribute signal to learned weights.
4. A standalone parallel decode was stopped after held-out because its runtime
   override had `honest_slot_contract=false`. The accepted comparison adds the
   explicit `--honest-slot-contract` flag and forbids hidden gold inventory.

Failed and interrupted attempts are retained here because experiment failures
are evidence, not omitted results.

## Decision and next unblock

P13 does not admit this integrated corpus/checkpoint pair to promotion. Static
corpus correctness is green, but the required learned signal is absent. The
next attempt must improve the model under an equal E53 recipe (preferably a
full HF-context, multi-seed train) and rerun both suites without weakening the
matched-signal rule or ship gates.

Recompose the report after producing equal-recipe matrix summaries:

```bash
rtk python -m scripts.verify_data_synthesis \
  --first-train-dir outputs/slm17/corpus/first \
  --second-train-dir outputs/slm17/corpus/second \
  --test-dir outputs/slm17/test/v2 \
  --task-results docs/design/task-eval-wiring-results.json \
  --baseline-matrix outputs/slm17/matched-e53-fixture-summary.json \
  --champion-matrix outputs/slm17/matched-e53-integrated-summary.json \
  --ltr-max-tokens 256 \
  --out docs/design/data-synthesis-results.json
```
