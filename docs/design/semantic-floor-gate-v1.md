# SemanticFloorGateV1 closeout (SLM-213)

**Schema:** `semantic_floor_gate/v1`
**Verdict:** **inconclusive**
**Gate hash:** `6a9bf662bcc3f2a698504f0972a1d1160484343f9f049c77808b435bfe739c0a`
**Evidence cutoff:** `2026-07-21T04:15:30.006965Z`

## Decision

The SDE5 evidence is fixture/wiring only. Strict binding-aware meaning-v2 was
not measured on a preregistered real evaluation, and checkpoint, data,
anti-gaming, paired-statistics, and AgentV identities are unresolved. The
semantic floor therefore remains **inconclusive**, not escaped or rejected.

Constraint-debt, structural, spectral, and recurrent/latent diagnostics remain
usable only as explicitly scoped proxies.

## Allowed claims

- `diagnostic`
- `proxy`
- `constraint_debt`
- `structure`
- `synthetic_wiring`

## Blocked claims

- `semantic_prediction`
- `semantic_causal`
- `learned_latent`
- `floor_escape`
- `promotion`
- `ship`

## Mediators

- Constraint debt: synthetic instrumentation/selection/routing signals exist.
- Protected objectives: not measured in a trained matched campaign.
- Strict meaning-v2: unmeasured (`n=0`).
- Legacy meaning-v1: no SDE5 result was supplied.
- Anti-gaming: scheduled_not_executed; scheduled cells are not pass evidence.
- AgentV: missing (`n=0`).

## Resolving evidence still required

- no durable checkpoint reference for the SDE5 floor-escape family
- no hash-pinned train manifest
- no hash-pinned eval manifest or preregistered strict meaning-v2 sample count
- no executed anti-gaming result bundle (schedule flags are not outcomes)
- no SDE5 AgentV/independent-evaluation bundle
- no paired strict meaning-v2 statistics across the declared seeds
- all SLM-208–212 producer stamps report code_dirty=true

## Source artifacts

| issue | path | SHA-256 | status/claim |
| --- | --- | --- | --- |
| SLM-208 | `docs/design/iter-slm208-constraint-debt-20260720.json` | `11437b4109d39f6815b1305dee148fcb0fa451d60cc7acbbc993d40b73c68cb2` | fixture/wiring |
| SLM-209 | `docs/design/iter-slm209-debt-targeted-curriculum-20260720.json` | `6c4caa766b44dfe44ddbacbc131b46e0e45b110ca2f0477f1471dd181d380c7f` | fixture/wiring |
| SLM-210 | `docs/design/sde5-floor-escape-matrix-results.json` | `0e42f43548e856541e7202e1d55d6dde3e542d7b133d0a4ef5735684bb42151e` | fixture/wiring |
| SLM-211 | `docs/design/iter-slm211-untied-output-head-20260721.json` | `83a21827a4d47010b92762d85b2ecb3ca2001e27501b7e13317c52a63f6affab` | fixture/wiring |
| SLM-212 | `docs/design/iter-slm212-debt-routing-20260721.json` | `fab22a3aa09b9d231020928e7f0f2d49e6fe58e7cb5fde88d414786f74e60da6` | fixture/wiring |

## Reproduction

```bash
python -m scripts.publish_semantic_floor_gate --check
pytest -q tests/test_harnesses/experiments/test_semantic_floor_gate.py tests/test_scripts/test_publish_semantic_floor_gate.py
```

No training, decoder experiment, evaluator change, gate-threshold change, or
checkpoint promotion was performed.
