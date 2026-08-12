# Exportable formal objects and multi-prover loop

**Status:** Implemented. Closes the formal loop for VSS `SupportCertificate`s
and Lean claim modules without relying on a single Lean kernel.

## Problem

Support certificates were replayable only inside the Python oracle path, and
Lean theorems lived only under the Lean 4 kernel. Either path alone is a
**single trust root**. A formal loop is closed only when **independent provers**
agree on exportable objects.

## Formal object

Schema: `formal_object/v1`  
Code: `src/slm_training/formal/objects.py`  
JSON Schema: `src/slm_training/resources/formal_object.schema.json`

| Field | Role |
| --- | --- |
| `kind` | `support_certificate` \| `lean_claim` \| `closure_law` |
| `statement` | Machine-readable claim (verdict, theorem id, laws) |
| `payload` | Certificate body or theorem metadata |
| `content_digest` | SHA-256 of kind+id+statement+payload |
| `required_checkers` | Minimum backends that must run |
| `tier` | `production_core` or `ecosystem_library` |

### Authority envelope (EVID-06)

Schema: `formal_authority/v2`  
Code: `src/slm_training/formal/authority.py`  
JSON Schema: `src/slm_training/resources/formal_authority.schema.json`

Converges portable `FormalObjectV1` and campaign `FormalPreflightV1` into one
claim/evidence authority contract via **adapters** (not a third evidence stack).
The envelope carries KERN-01 `SolverJudgmentV1`, optional EVID-03 four-axis
ledger digest + EVID-04 `bound_ast_ids`, checker results/capabilities,
source/toolchain identity, certificate/replay digests, and an explicit
`authority_class`. Incomplete/skipped/unknown checker or preflight evidence
cannot be represented as checked semantic refutation. Historical v1 artifacts
remain readable; preferred writes emit v2 (`prefer_write_payload`).


### Checked refutation authority (EVID-09)

Code: `src/slm_training/formal/refutation_authority.py` (wired through
`judgment.py`, ExactClosure / GoalSupport Lean mirrors, and support-certificate /
closure adapters).

Destructive `refuted` / UNSUPPORTED authority requires either exact bounded
semantic replay or a checked proof-certificate path, each bound to
state/problem/source/tool identity. Self-attested `exhausted`,
`coverage=complete`, and `replay_ok` bits remain telemetry only and cannot
remove candidates. Skipped replay, timeout, missing tool/context, incomplete
coverage, malformed input, or unsupported features return `unknown`/`invalid`
and strip evidence at the adapter boundary.


### Verified encoding adapters (EVID-10)

Code: `src/slm_training/formal/encoding_adapter.py` (Lean mirror
`LeverProofLean.VerifiedEncoding`; fixtures
`resources/formal/encoding_adapter_fixtures.v1.json`).

A checked solver certificate has **no** semantic authority without a trusted
encoding bridge from the original bounded problem to the SAT/PB/SMT encoding.
The generic adapter exposes a supported-subset predicate, deterministic encode,
decode/witness map, problem+encoding identity, and theorem/checker refs. The
promoted reference family proves
`Satisfiable(encode p) ↔ ∃ x, Satisfies p x`. Unsupported features return
`unsupported`/`unknown` (never silently dropped). v2 evidence records encoder
version/hash, theorem FQN/type binding, certificate format, checker
capability/trust domain, and original problem digest on `FormalAuthorityV2`.
Pilot formats (`lrat_pilot` / `pblean_pilot` / `alethe_pilot`) are named for
future RESEARCH pilots and are not production dependencies.

### RESEARCH-05 — VSS LRAT SAT pilot (SLM-563)

Default-off research adapter: `src/slm_training/formal/vss_lrat_backend.py` +
`harnesses/experiments/research_05_vss_lrat.py`. Control = exhaustive VSS/CNF
replay; treatment = EVID-10 `lrat_pilot` encoding + hermetic RUP certificate
check. Evidence: [`iter-revmath-research-05-preregistered.md`](iter-revmath-research-05-preregistered.md).
Never production decode / ship-gate / serving authority.

### RESEARCH-06 — VSS PBLean PB pilot (SLM-564)

Default-off research adapter: `src/slm_training/formal/vss_pblean_backend.py` +
`harnesses/experiments/research_06_vss_pblean.py`. Control = exhaustive VSS/CNF
replay; treatment = EVID-10 `pblean_pilot` CNF→PB encoding + hermetic VeriPB-style
certificate check (external PBLean optional/unavailable). Evidence:
[`iter-revmath-research-06-preregistered.md`](iter-revmath-research-06-preregistered.md)
(disposition **rejected** — correctness ok, warm/exhaustive ratio ≥ 1.0 on fixture suite).
Still `default_off` / no production decode / ship-gate / serving authority.


### Mutation / red-team acceptance suite (EVID-11)

Code: `src/slm_training/formal/mutation_suite.py`  
Matrix: `resources/formal/evid11_mutation_matrix.v1.json`  
Results: [`formal-evidence-mutation-results.json`](formal-evidence-mutation-results.json)  
CI: `python -m scripts.verify_formal_evidence_mutations --check`

Release-blocking adversarial suite for formal evidence v2. Deterministic
fixtures map each required mutation family to the exact gate/capability that
must reject it (reusing EVID-06/07/09/10 — no third evidence stack). Positive
controls still pass. Shared `python_structural`/`python_reference` trust
domains do **not** count as independent semantic authority (EVID-08
production policy). Structural consistency / provenance alone cannot
confer semantic authority.

| Family | Gate (intended layer) |
| --- | --- |
| proposition tautology / weaken | `theorem_binding.verify_theorem_binding` (EVID-07) |
| declaration / module / source / toolchain / lock drift | `theorem_binding.verify_theorem_binding` (EVID-07) |
| unexpected axiom / forbidden proof escape | binding axiom audit + `FORBIDDEN_SOURCE` (EVID-07) |
| forged completeness / exhaustion / replay metadata | `judgment.classify` / legacy exhaustion (EVID-09) |
| skipped / timeout / missing checker as success | `authority.from_formal_object_v1` + adapters (EVID-06/09) |
| stale problem / state identity | `evidence_authorizes_removal` (EVID-09) |
| certificate mutation / wrong encoding | `encoding_authorizes_semantic_result` (EVID-10) |
| shared parser / checker-family fault | distinct trust-domain gate (EVID-11/08) |
| v1/v2 migration tamper / digest mismatch | `FormalAuthorityV2.from_dict` / `build_formal_authority_v2` (EVID-06) |

### Exact theorem binding (EVID-07)

Schema: `lean.theorem_binding/v1`  
Code: `src/slm_training/formal/theorem_binding.py` (wired through `export_lean_claim`, `check_lean_kernel`, and `FormalAuthorityV2.theorem_binding`).

A Lean claim seals the fully-qualified declaration name, expected proposition fingerprint, Lean toolchain, lake manifest/lock digest, source-tree digest, and transitive axiom footprint. The Lean checker generates a bridge that `#check`s the declaration and proves `example : ExpectedProposition := Fully.Qualified.name`, runs `#print axioms` for that declaration, and fails on proposition mutation, module/catalog redirection, source/toolchain drift, unexpected axioms, missing declarations, or stale digests. The project-wide forbidden-proof audit (`make test`) remains an *additional* check — never a substitute for exact binding.

### Exports

| Source | Exporter |
| --- | --- |
| VSS `SupportCertificate` | `export_support_certificate` |
| Lean theorem catalog | `export_lean_claim` / `lean_claim_catalog` |
| Exact-closure honesty laws | `export_closure_law` |

## Checkers (capability + trust domains)

Registry: `resources/formal/checker_capability_registry.v1.json`  
Code: `src/slm_training/formal/checker_capability.py` (EVID-08)

| Backend | Role | Trust domain | Advertised capabilities |
| --- | --- | --- | --- |
| `python_structural` | redundant conformance | `python_cpython_formal_structural_family` | `structural_consistency` |
| `python_reference` | redundant conformance | *same family* | `structural_consistency` |
| `python_replay` | independent | `python_enumerative_replay` | `semantic_replay`, `certificate_checking` |
| `lean_kernel` | independent | `lean4_kernel` | `exact_proposition`, `axiom_audit`, … |
| `python_encoding_ref` | independent | `python_encoding_bridge` | `encoding_correctness`, … |

**Rule:** Lean is never sole multi-backend loop authority. Distinct backend
*names* are not automatic independent trust roots. Policy reports label
`redundant_conformance` vs `independent_verification`. Skipped checks are not
successful semantic checks. A checker cannot satisfy an unadvertised capability.

Default loop policy is conformance (`formal_object_conformance/v1`):
`min_backends = 2` with structural+reference closes the loop as **redundant
conformance**, not semantic authority. `formal_semantic_authority/v1` requires
an independent trust domain (and ≥2 domains when mixing conformance with
another family).

## Close the loop

```bash
# Catalog Lean claims + VSS closure laws (pure multi-prover, no Lean required)
python -m scripts.verify_formal_objects \
  --catalog-lean --closure-laws \
  --export-dir outputs/formal_objects \
  --out docs/design/formal-loop-report.json

# Optional third backend
python -m scripts.verify_formal_objects --catalog-lean --lean

# Support certificate file(s)
python -m scripts.verify_formal_objects --certificate path/to/cert.json
```

Report schema: `formal_loop_report/v1`  
Code: `src/slm_training/formal/loop.py`

A report has `closed: true` only when every object is accepted by enough
independent backends.

## Trust boundary

| Trusted | Not proved |
| --- | --- |
| Multi-backend agreement on the exported object | That measurements were unbiased |
| Certificate honesty laws + optional full replay | That the expander matches production OpenUI pack semantics without a provided context |
| Structural mirrors of Lean laws | That Lean source text bit-identically matches Python (module build is optional third) |

## Relationship to other formal surfaces

- Lean theories: `src/leverproof_lean/` ([core-formal-claims.md](core-formal-claims.md))
- Metric oracle certificates: LeverProof `metric_certificate/v*` (promotion bands)
- Ecosystem tier split: [ecosystem-tier.md](ecosystem-tier.md)
- VSS contract: [verified-scope-solver.md](verified-scope-solver.md)

## Honesty

This is a **formal-loop / infrastructure** claim, not a ship-quality claim. It
does not replace `--ship-gates`, training, or held-out evals.
