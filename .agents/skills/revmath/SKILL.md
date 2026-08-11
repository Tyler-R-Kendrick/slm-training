---
name: revmath
description: >
  Discover and run the reasoning/revmath profile — reverse-mathematics /
  computability analysis over existing campaign, formal-preflight, evidence,
  and repair owners. Use when editing revmath schemas, fixtures, labeling,
  self-healing, four-axis ledgers, or agent/docs parity for SLM reverse math.
---

# Reverse mathematics (`reasoning/revmath`)

Use this skill for **discoverability and honest fixture runs** of the typed
`reasoning/revmath` profile. Do **not** create a parallel trainer, campaign
store, evidence store, proof stack, or version registry.

## Canonical owners (one per concept)

| Concern | Owner |
| --- | --- |
| Design + ADR | `docs/design/reverse-mathematics-computability.md`, `docs/design/adr-revmath-reasoning-profile.md` |
| Machine owner map | `src/slm_training/resources/revmath_owner_map.json` · `python -m scripts.verify_revmath_owners` |
| Harness parity / self-healing freeze | `src/slm_training/resources/revmath_harness_parity.json` · `python -m scripts.verify_revmath_harness_parity` |
| Profile registration | `src/slm_training/harnesses/reasoning/profiles.py` (`reasoning/revmath`, default stays `reasoning/g4`) |
| Task schemas | `src/slm_training/harnesses/reasoning/revmath/schemas.py` (`RevmathTaskV1`, …) |
| Runner / plugins / report | `harnesses/reasoning/revmath/{runner,plugins,report,replay}.py` |
| Campaign lock | `profile_binding.py` + `ExperimentCampaignV1` |
| Practical computability vocabulary | `formal/computability_classification.py` (KERN-12) |
| Version registry | `src/slm_training/resources/versions.json` only |

Full table: design doc + owner map. Agent-law matrix:
`python -m scripts.verify_agent_surfaces`.

## Practical vs genuine RM

- **`practical_computability_only`** / KERN-12 classes — engineering labels.
- **Genuine Big-Five** (`RCA0` / `WKL0` / … as `reversed_equivalent`) — only with
  an explicit interpretation package (base theory, coding, both evidence digests,
  `explicit_reversal` or `explicit_interpretation`). Ablation alone is
  `rm_inspired_assumption_minimization`, never genuine RM.

## Exact-refutation authority

Refutation requires an independently checked finite model / computable trace
against the **exact** weakened proposition. Timeout, missing tool, incomplete
domain, search failure, and skipped replay stay **unknown** — never semantic
refutation (KERN-01).

## Fixture run (hermetic; no Lean required)

```bash
PYTHONPATH=src uv run python -m scripts.run_revmath_task \
  --task src/slm_training/resources/revmath/fixtures/hermetic_forward_theorem.task.json \
  --hermetic --output-dir outputs/revmath/hermetic

PYTHONPATH=src uv run python -m scripts.run_revmath_profile \
  --task src/slm_training/resources/revmath/fixtures/ablation_necessary.task.json \
  --materialize-fixture --campaign-id camp.rm.hermetic \
  --experiment-id exp.rm.hermetic.profile \
  --store-root outputs/revmath/profile_runs --hermetic
```

Examples index: `src/slm_training/resources/revmath/examples/README.md`.
Current schemas only (`revmath_task/v1`, …). No retired write path.

## Optional external solvers / checkers

Lean, alternate provers, and third-party checkers are **experiments, not prerequisites**.
Default hermetic fixtures certify wiring without them. Enabling an external tool
never upgrades fixture/diagnostic evidence to production ship authority.

## Self-healing + promotion

- May modify only HARN-09 knobs inside the locked budget (`may_modify` in the
  parity resource).
- Must freeze proposition, assumptions, direction, corpus, arms, budgets,
  judges, gates, authority policy.
- Promotion stays on `ExperimentCampaignV1` + `ensure_promote_formal_preflight`
  / honest `--ship-gates`. Revmath reports are decision support, not silent
  promote authority.

## Checks

```bash
PYTHONPATH=src uv run python -m scripts.verify_revmath_owners
PYTHONPATH=src uv run python -m scripts.verify_revmath_harness_parity
PYTHONPATH=src uv run python -m scripts.verify_agent_surfaces --obligation revmath.canonical-owners
```
