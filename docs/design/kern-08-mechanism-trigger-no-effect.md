# KERN-08 — Mechanism triggers, no-effect implications, dominance certificates (SLM-531)

## Claim

If enabling a mechanism changes an output on a fixture, a declared
**necessary trigger** for that mechanism class must have occurred. Therefore
proven trigger **absence** implies **no output effect** on the locked finite
corpus. **Corpus-local dominance** adds equal outputs plus treatment cost ≥
baseline cost under a named KERN-06 machine model.

Results do **not** generalize to unseen inputs. Unknown activation never
becomes no-effect.

## Definitions

| Concept | Meaning |
| --- | --- |
| Mechanism | Id + whether a safe necessary-trigger theorem exists |
| TriggerEvidence | `absent` / `present` / `unknown` (unknown = incomplete) |
| Observation | Baseline/enabled outputs + costs + trigger on one fixture |
| NecessaryTrigger | `Output(enabled) ≠ Output(base) → trigger = present` |
| NoEffectFromAbsent | Contrapositive: `trigger = absent → outputs equal` |
| dominatesLocally | Equal outputs ∧ `enabledCost ≥ baselineCost` |

## Supported classes (necessary condition justified)

| Mechanism | Necessary trigger | Telemetry owners |
| --- | --- | --- |
| `singleton_bypass` | complete singleton domain | KERN-07 / `forwards_count` bypass |
| `closure_removal` | removable candidates nonempty | goal-support / forest closure |
| `cache_reuse` | cache hit | `*_cache_hits` / branch memo |
| `forced_span` | forced span nonempty | `forced_spans` / common forced run |

## Counterexamples

| Case | Why unsafe |
| --- | --- |
| `unconstrained_rerank` | No well-formedness linking a trigger bit to output equality |
| `unknown` trigger evidence | Incomplete — blocked by certificate evidence obligation |

## Lean API (`LeverProofLean.MechanismTrigger`)

| Symbol | Role |
| --- | --- |
| `no_effect_of_necessary_and_absent` | Discharge contrapositive |
| `noEffectCertificate_outputs_equal` | Certificate ⇒ equal outputs |
| `unknown_not_in_no_effect_corpus` | Unknown never certifies |
| `dominanceCertificate_dominates` | Dominance from no-effect + cost |
| `*_output_change_implies_trigger` | Per-class necessary theorems |
| `unconstrained_rerank_no_safe_trigger_theorem` | Unsafe counterexample |

## Python API (`slm_training.formal.mechanism_trigger`)

Emits `NoEffectCertificateV1` / `DominanceCertificateV1` for autoresearch
admission. Fail-closed on unknown evidence, missing safe theorem, or cost
non-dominance.

Frozen fixtures:
`src/slm_training/resources/formal/mechanism_trigger_fixtures.v1.json`.

## Autoresearch admission (INTEG-02)

[`integ-02-mechanism-no-effect-preflight.md`](integ-02-mechanism-no-effect-preflight.md)
hooks these certificates into the existing preflight seam
(`autoresearch/preflight/mechanism_no_effect.py`). Skip only on complete
trigger-absence; unknown / incomplete always run.

