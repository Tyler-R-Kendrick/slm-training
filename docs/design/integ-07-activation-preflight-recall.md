# INTEG-07 — Activation-preflight recall + false-skip risk (SLM-561)

**Status:** release-blocking recall gate. Certifies that INTEG-02
theorem-backed activation/no-effect preflights do **not** falsely skip
experiments whose mechanism can affect the locked corpus.

Does **not** invent a parallel admission path — every case calls
`admit_mechanism_treatment` (KERN-08 certificates via INTEG-02).

## Artifacts

| Artifact | Path |
| --- | --- |
| Replay set | `src/slm_training/resources/formal/integ07_activation_preflight_replay.v1.json` |
| Benchmark | `src/slm_training/formal/integ07_activation_preflight.py` |
| Verify | `scripts/verify_integ07_activation_preflight.py` |
| Results | [`integ-07-activation-preflight-recall-results.json`](integ-07-activation-preflight-recall-results.json) |
| Tests | `tests/test_formal/test_integ07_activation_preflight.py` |

## Cohorts

| Cohort | Ground truth | Effective expect |
| --- | --- | --- |
| `known_activating` | historical present-trigger + output change | `run` (recall) |
| `provably_inactive` | KERN-08 complete absent fixtures | `skip_no_effect` (specificity) |
| `unknown_fallback` | unknown trigger / HARN-11 negative bind | `run` |
| `adversarial` | incomplete scan, stale trigger, corpus mismatch, mechanism-version mismatch | `run` |

## Metrics

- **Recall** — fraction of known-activating cases with effective `run`. Gate: **100%**.
- **Specificity** — fraction of provably-inactive cases with effective `skip_no_effect`.
- **Unknown/fallback rate** — fraction of unknown + adversarial fallback cases with effective `run`.
- **Compute saved** — sum of `treatment_compute_units` on authorized inactive skips.

Effective disposition = raw INTEG-02 disposition **and** exact identity match on
locked corpus id, mechanism id, trigger theorem ref, and cost-model version.
Raw skip under corpus/theorem/version mismatch does **not** authorize a
beyond-fixture skip.

## Identities persisted

Every report seals:

- live HARN-11 `corpus.rm.hermetic.v1` + `corpus_sha256`
- trigger theorem refs used for authorized skips
- locked corpus ids per case
- kernel + admission module paths

## Run

```bash
export PATH="$HOME/.elan/bin:$PATH"
PYTHONPATH=src uv run python -m scripts.verify_integ07_activation_preflight --check
PYTHONPATH=src uv run python -m scripts.verify_integ07_activation_preflight --write
PYTHONPATH=src uv run pytest tests/test_formal/test_integ07_activation_preflight.py -q
```

## Acceptance

- 100% recall on the frozen known-activating set (false skip = release-blocking).
- Unknown / incomplete / identity-mismatched evidence effective-runs.
- Adversarial kinds covered: `incomplete_scan`, `stale_trigger`, `corpus_mismatch`,
  `mechanism_version_mismatch`.
- Beyond-fixture admission skips require this gate green.

Components: `formal.objects`, `ci.local_merge_gate` (verify wired into
`verify_merge_ready --fast`).

## Related

- [`integ-02-mechanism-no-effect-preflight.md`](integ-02-mechanism-no-effect-preflight.md)
- [`kern-08-mechanism-trigger-no-effect.md`](kern-08-mechanism-trigger-no-effect.md)
- HARN-11 hermetic corpus via `harnesses.reasoning.revmath.corpus`
