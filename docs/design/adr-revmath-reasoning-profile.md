# ADR: Reverse mathematics as `reasoning/revmath` profile (EVID-01)

## Status

Accepted — 2026-08-11 (SLM-515 / EVID-01). Authorizes owner boundaries and extension seams only; no production ship claim.

## Context

The repository already implements formal preflights (`FormalPreflightV1`), portable formal objects (`FormalObjectV1`), bounded Lean execution, theorem-name and forbidden-proof auditing, preregistered campaigns (`ExperimentCampaignV1`), decode telemetry (`DecodeStatsRecordV1`, `MechanismActivationV1`), replay bundles (`ReplayBundleV1`), verifier witnesses (`VerifierWitnessV1`), mechanism dispositions (`MechanismDispositionRecordV1`), and the autoresearch `formalize` preflight seam. Reverse-mathematics / computability work was at risk of sprouting parallel trainers, proof stacks, evidence stores, and orchestrators — violating I14 (goals are fixed; approaches are disposable) and the experiment-campaign law.

Audit base: `980aa465223adf7822f82930550c92b9240333ca`.

## Decision

1. **Reverse mathematics is a typed profile `reasoning/revmath`** over the existing G4 reasoning harness and the canonical formal/evidence/campaign owners listed in [reverse-mathematics-computability.md](reverse-mathematics-computability.md).
2. **No parallel authority modules.** Revmath runs bind to `ExperimentCampaignV1`, write formal artifacts through `run_formal_preflight` / `cmd_formalize`, and persist evidence through `DecodeStatsRecordV1`, `ReplayBundleV1`, `VerifierWitnessV1`, and disposition records — never through profile-local stores.
3. **Owner map is frozen in `revmath_owner_map.json`** and certified by `scripts/verify_revmath_owners.py`. Global `ownership_map.json` extensions for formal subsystems remain parent-owned; this ADR does not recreate owners already on main.
4. **EVID-02 is absorbed:** the standalone registry/duplicate-owner ticket is superseded by EVID-01's map + verifier (see design doc).

## Consequences

### Positive

- Agents have one discoverable owner table and CI guard against shadow `revmath_*` authority.
- Downstream EVID-03/04/06 and HARN-01 cite explicit extension seams instead of inventing stacks.
- Formal preflight and campaign governance stay on the promotion-critical path already audited on main.

### Negative / constraints

- Profile task logic (HARN-01+) must route through existing seams even when inconvenient.
- New formal subsystems still require eventual `ownership_map.json` registration in a parent PR when they become code owners.

## Fail-closed falsifiers

This ADR is **violated** if:

- a revmath run can bypass `ExperimentCampaignV1` or canonical evidence envelopes;
- two modules claim the same semantic decision authority without a documented adapter;
- the ADR recreates an owner that already exists on main under a new name;
- timeout/incomplete/skipped checking is treated as semantic refutation.

## References

- Design: [reverse-mathematics-computability.md](reverse-mathematics-computability.md)
- Map: [`src/slm_training/resources/revmath_owner_map.json`](../../src/slm_training/resources/revmath_owner_map.json)
- Verifier: `python -m scripts.verify_revmath_owners`
- Harness parity (HARN-01): [`revmath_harness_parity.json`](../../src/slm_training/resources/revmath_harness_parity.json) · `python -m scripts.verify_revmath_harness_parity`
- Profile registration: [`harnesses/reasoning/profiles.py`](../../src/slm_training/harnesses/reasoning/profiles.py)
- Schemas (HARN-02): [`harnesses/reasoning/revmath/schemas.py`](../../src/slm_training/harnesses/reasoning/revmath/schemas.py)
- Formal autoresearch: [formal-autoresearch.md](formal-autoresearch.md)
- Formal objects: [formal-objects-multi-prover.md](formal-objects-multi-prover.md)
- Campaign law: [experiment-campaign-governance.md](experiment-campaign-governance.md)
