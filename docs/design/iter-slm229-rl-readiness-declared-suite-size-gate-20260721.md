# SLM-229 (RLRG0-02): RL-readiness declared-vs-actual suite-size gate stress test (slm229-rl-readiness-declared-suite-size-gate-20260721)

**Matrix set:** `slm229_rl_readiness_declared_suite_size_gate`
**Version:** `rlrg0-02-v1`
**Status:** fixture
**Claim class:** wiring
**Illustrative candidate:** actual rico_held n>=1500 (declared metadata ignored)
**Gate hash:** `95c605c47c435cb6...`
**Disposition:** gap_confirmed — 3/4 non-control arms (declared_only_ship_gate_floor_n20, declared_only_smoke_scale_n25, declared_far_exceeds_actual_n100) were approved by the real assess_rl_readiness and assert_rl_ready while their actual rico_held record count fell below the illustrative candidate floor (actual n>=1500, ignoring declared metadata). The declared-vs-actual suite-size requirement is confirmed gameable by a self-reported suite_sizes claim decoupled from the evaluated data.

## Hypothesis

The real assess_rl_readiness rico_held>=1500 requirement -- computed as max(actual suites['rico_held']['n'], declared evaluation_snapshot.metadata.suite_sizes['rico_held']) -- is satisfied by a self-reported declared suite_sizes claim alone, decoupled from the actually-evaluated rico_held record count, whenever the actual count independently clears the unrelated, much lower honest-ship-gate n floor (DEFAULT_MIN_SUITE_N=20, no rico_held override).

## Falsifier

Every arm whose actual rico_held n is below 1500 is rejected by assess_rl_readiness / assert_rl_ready regardless of what the declared metadata suite_sizes field claims -- i.e., the check already ties its size floor to the actually-evaluated suite, or the mechanism cannot be exercised at all.

## Honest caveats

- Fixture/wiring evidence only: no checkpoint, GPU run, RL train step, or ship-gate claim is made or implied.
- This exercises the real, unmodified assess_rl_readiness and assert_rl_ready functions against constructed evaluation payloads; the reward/AgentV/frozen-snapshot fields are fixture data built to satisfy every non-suite-size requirement, not a real production evaluation.
- The candidate check (actual suite n alone, ignoring declared metadata, must clear 1500) is an illustrative diagnostic only. It is not implemented in rl_gate.py, not proposed as the correct fix, and passing/failing it makes no gate or promotion claim.
- Whether a smaller-than-1500 rico_held suite actually harms RL training (as opposed to only being a weak proxy in the readiness report) is not measured here; this harness is about the readiness *gate*, not RL training dynamics.
- Arms use hand-picked suite metrics and metadata, not a real evaluation run against a real checkpoint.

## Per-arm results

| arm | actual n | declared n | reported n (max) | approved (real gate) | assert_rl_ready raised | candidate would pass | gameable | control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_actual_1500 | 1500 | 1500 | 1500 | True | False | True | False | False |
| declared_only_ship_gate_floor_n20 | 20 | 1500 | 1500 | True | False | False | True | False |
| declared_only_smoke_scale_n25 | 25 | 1500 | 1500 | True | False | False | True | False |
| declared_far_exceeds_actual_n100 | 100 | 5000 | 5000 | True | False | False | True | False |
| no_declared_field_small_actual_control | 20 | — | 20 | False | True | False | False | True |
| declared_below_floor_control | 20 | 50 | 50 | False | True | False | False | True |

## Arm descriptions

- **matched_actual_1500**: Actual rico_held n=1500, declared metadata matches. The shape a genuine production-scale evaluation should have.
- **declared_only_ship_gate_floor_n20**: Actual rico_held n=20 -- exactly the honest-ship-gate DEFAULT_MIN_SUITE_N floor and nothing more -- with declared metadata claiming 1500. Tests whether the 1500 floor is satisfiable at the smallest actual size that independently clears the (unrelated, much lower) ship-gate n check.
- **declared_only_smoke_scale_n25**: Actual rico_held n=25 -- a smoke/dev-scale suite -- with declared metadata claiming 1500.
- **declared_far_exceeds_actual_n100**: Actual rico_held n=100 with declared metadata claiming 5000 -- an order-of-magnitude over-claim -- to check whether the gap widens without limit.
- **no_declared_field_small_actual_control**: Actual rico_held n=20 (still clears the ship-gate floor) with no declared suite_sizes metadata at all. Negative control -- without a declared-size claim to inflate the reported size, the current gate must still reject this.
- **declared_below_floor_control**: Actual rico_held n=20 with declared metadata honestly claiming 50 (also below 1500, no inflation attempted). Negative control -- the current gate must still reject this.

## No-go for promotion

This report is wiring/fixture evidence only. It does not change `assess_rl_readiness` or `assert_rl_ready`, does not train or run RL, and makes no ship or gate claim. It documents a concrete gap in the existing mechanical declared-vs-actual suite-size check as a candidate for a future, separately reviewed hardening change (never implemented here).

## Reproducibility

```bash
python -m scripts.run_slm229_rl_readiness_declared_suite_size_gate --mode plan-only
python -m scripts.run_slm229_rl_readiness_declared_suite_size_gate --mode fixture
```
