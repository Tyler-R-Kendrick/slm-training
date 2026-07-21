# SLM-228 (RLRG0-01): RL-readiness reward-variance gate stress test (slm228-rl-readiness-variance-gate-20260721)

**Matrix set:** `slm228_rl_readiness_variance_gate`
**Version:** `rlrg0-01-v1`
**Status:** fixture
**Claim class:** wiring
**Illustrative candidate:** n>=8, spread>=0.05
**Gate hash:** `9684e70e35b4e837...`
**Disposition:** gap_confirmed — 3/4 non-control arms (two_sample_wide, two_sample_epsilon, large_n_epsilon_outlier) were approved by the real assess_rl_readiness reward-variance check while failing the illustrative stronger candidate (n>=8, spread>=0.05). The mechanical variance>0 bar is confirmed gameable by degenerate-but-technically-diverse reward samples.

## Hypothesis

The real assess_rl_readiness reward-variance requirement (len(reward_samples) >= 2 and statistics.pvariance(reward_samples) > 0) approves reward-sample arms with degenerate diversity -- vanishing spread and/or only two samples -- whenever every other RL-readiness requirement is independently satisfied, because the check has no minimum sample count beyond 2 and no minimum spread/magnitude floor.

## Falsifier

Every degenerate arm (near-zero spread with n>=2, or n==2 with a wide spread) is rejected by assess_rl_readiness while the healthy diverse arm and the two negative controls (zero variance, single sample) behave as expected -- i.e., some safeguard beyond the read source already closes this gap, or the mechanism cannot be exercised at all.

## Honest caveats

- Fixture/wiring evidence only: no checkpoint, GPU run, RL train step, or ship-gate claim is made or implied.
- This exercises the real, unmodified assess_rl_readiness function against constructed evaluation payloads; the suite/AgentV/frozen-snapshot fields are fixture data built to satisfy every non-reward requirement, not a real production evaluation.
- The CANDIDATE_MIN_SAMPLES / CANDIDATE_MIN_SPREAD thresholds are illustrative diagnostics only. They are not implemented in rl_gate.py, not proposed as the correct values, and passing/failing them makes no gate or promotion claim.
- Whether tiny reward variance actually harms GRPO-lite training (as opposed to only being a weak proxy in the readiness report) is not measured here; this harness is about the readiness *gate*, not RL training dynamics.
- Arms use hand-picked reward_samples values, not rewards sampled from any real policy or environment.

## Per-arm results

| arm | n | spread | reward_variance | approved (real gate) | candidate would pass | gameable | control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| healthy_diverse_n8 | 8 | 9.00e-01 | 9.19e-02 | True | True | False | False |
| two_sample_wide | 2 | 8.00e-01 | 1.60e-01 | True | False | True | False |
| two_sample_epsilon | 2 | 1.00e-09 | 2.50e-19 | True | False | True | False |
| large_n_epsilon_outlier | 100 | 1.00e-07 | 9.90e-17 | True | False | True | False |
| all_identical_control | 10 | 0.00e+00 | 0.00e+00 | False | False | False | True |
| single_sample_control | 1 | 0.00e+00 | 0.00e+00 | False | False | False | True |

## Arm descriptions

- **healthy_diverse_n8**: 8 samples evenly spread across [0.05, 0.95]: the shape a real GRPO-lite reward distribution with a working signal should have.
- **two_sample_wide**: The minimum n=2 the check accepts, but with a wide [0.1, 0.9] spread -- tests whether sample count alone is gameable even with genuinely different rewards.
- **two_sample_epsilon**: n=2 with a ~1e-9 spread: technically nonzero variance, no real reward diversity.
- **large_n_epsilon_outlier**: 100 samples, 99 identical and 1 perturbed by 1e-7: mimics floating-point jitter from an otherwise near-deterministic reward function at a sample count no reasonable minimum-n check would flag.
- **all_identical_control**: 10 identical reward samples: zero variance. Negative control -- the current gate must still reject this.
- **single_sample_control**: A single reward sample: below the len>=2 floor. Negative control -- the current gate must still reject this.

## No-go for promotion

This report is wiring/fixture evidence only. It does not change `assess_rl_readiness`, does not train or run RL, and makes no ship or gate claim. It documents a concrete gap in the existing mechanical reward-variance check as a candidate for a future, separately reviewed hardening change (never implemented here).

## Reproducibility

```bash
python -m scripts.run_slm228_rl_readiness_variance_gate --mode plan-only
python -m scripts.run_slm228_rl_readiness_variance_gate --mode fixture
```
