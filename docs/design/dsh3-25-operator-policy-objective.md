# SLM-400 (DSH3-25): operator-policy objective and DEFER-routing fixture

**Status:** fixture / wiring only.  
**Claim class:** `wiring`.  
**Honest verdict:** `fixture_wiring`.  
**Disposition:** `wiring_underpowered`.

Fixture-scale only: three arms are wired and structurally distinguished (DEFER abstains on PARTIAL_UNKNOWN, the naive CE arm never does), but no claim is made that PU risk or DEFER beats naive CE at production scale in either direction, per the DSH3-25 stop rule.

## What this exercises

- A 12-candidate live domain enumerated by the real `enumerate_operator_legal_set`, truncated via its own `max_combinations_per_operator` at each unknown-fraction level — genuine `PARTIAL` coverage, not a hand-faked one.
- `build_operator_policy_objectives` over the SLM-397 sanitized `OperatorPolicyInputV1`.
- Three matched arms sharing one `CandidateScoringHead`/`OperatorFeatureEncoder` (typed arm) forward pass: `known_supported_ce`, `positive_unlabeled_risk`, `defer_abstention`.

## Route distribution and mean loss by unknown fraction

| Unknown fraction | Routes | known_supported_ce | positive_unlabeled_risk | defer_abstention |
| --- | --- | --- | --- | --- |
| 0.00 | complete_ambiguous=1 | 2.4849 | 0.6935 | 2.4849 |
| 0.25 | partial_witnessed=1 | 2.4849 | 0.6935 | 2.4849 |
| 0.50 | partial_unknown=1 | abstained | abstained | abstained |
| 0.75 | partial_unknown=1 | abstained | abstained | abstained |

## Caveats

- A single synthetic 12-candidate domain and an untrained, randomly-initialized typed encoder/head — logits demonstrate the routing/loss-arm contract, not learned quality.
- `known_supported_ce` and `defer_abstention` are identical whenever there is at least one positive witness (by design — `defer_abstention` only diverges on `PARTIAL_UNKNOWN`), and both correctly abstain (no loss) when there is zero witness at all, since no supervised signal exists for *any* arm in that case. The behavioral difference this ticket cares about is an **inference-time serving policy** one (always-guess versus explicit DEFER on a `PARTIAL_UNKNOWN` route) — this fixture demonstrates that at the route-label level, not via a full serving/decoding harness, which is out of scope here.
- No claim that PU risk or DEFER beats naive CE at production scale in either direction; see the disposition above.
- `TargetUtility` only offers `POSITIVE`/`UNKNOWN` in this version — SLM-399's collapse hard negatives name a whole alternate *operator*, not an alternate candidate within the same slot, so there is no certified per-candidate `NEGATIVE` source yet. Adding one is a follow-up, not approximated here.
- No ship gate is evaluated or weakened.

## Verification commands

```bash
python -m pytest -q tests/test_models/test_operator_policy_objective.py tests/test_dsl/test_operator_legal_set.py tests/test_models/test_legal_edit_batch.py
python -m scripts.verify_version_stamps --check
```

Both commands passed on this branch at the time of writing.