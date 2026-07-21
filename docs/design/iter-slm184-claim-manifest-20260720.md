# SLM-184: single-touch confirmation firewall fixture (slm184-claim-manifest-20260720)

Matrix set: `slm184_claim_manifest`

Version: `claim-manifest-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no trainable weights were updated, and no ship-gate claim is made.

## Hypothesis

A preregistered claim manifest can enforce a single confirmation touch on one suite while allowing unlimited logged development touches.

## Falsifier

The firewall allows more than one prediction-materialized confirmation touch.

## Manifest

- experiment_family_id: `slm184-fixture-family`
- confirmation_suite_id: `rico_held`
- confirmation_suite_digest: `sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`
- confirmation_touch_limit: `1`
- mde: `0.05`
- alpha: `0.05`
- power: `0.8`

## Firewall exercise

| touch | allowed | reason |
| --- | --- | --- |
| first confirmation (prediction materialized) | True | confirmation access granted under preregistered manifest |
| second confirmation (prediction materialized) | False | a prediction-materialized confirmation touch already exists |
| dev touch on smoke suite | True | dev access always allowed; touch logged |

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The single-touch confirmation firewall, preregistered manifest, and ledger semantics are exercised, but no real model or eval suite was used. The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_eval`` until it is wired into a real matrix runner.

## Honest caveats

- Digest verification is string equality on a placeholder hash; real suites need a   content-addressed digest from the suite builder.
- The ledger is a local JSON file; production provenance needs an append-only store.
- No ship-gate claim is made; this is wiring evidence only.

## Reproducibility

```bash
python -m scripts.audit_experiment_firewall --mode describe
python -m scripts.audit_experiment_firewall --mode fixture
python -m scripts.audit_experiment_firewall --mode check \
  --manifest <path> --ledger <path> --suite-id <id> --suite-digest <digest>
python -m scripts.audit_experiment_firewall --mode audit-history \
  --iter-dir docs/design --output <json> --output-md <md>
```
