# PGS-J01 (SLM-514): Independent red-team disposition — proof-carrying goal support

**Schema:** `proof_carrying_goal_support_disposition/v1`

**Parent:** SLM-493 (Proof-Carrying Goal Support & Domain Adequacy)

**Review commit:** `dfa65fb93d0a26dfc6276587a8893b4d5b95089b` (PGS-I02 / SLM-513 integrated)

**Conclusion:** `sound_within_claimed_bounds`

**Claim class:** independent adversarial security/formal-methods review — not production promotion, not MODEL_CARD, not ship.

**Machine-readable artifact:** [`proof-carrying-goal-support-disposition.json`](proof-carrying-goal-support-disposition.json)

**Disposition digest:** `e904cd15c17b47132d043df444d6d77a0dabfbd9be75f84ea1d762df7a77b851` (canonical JSON excluding `disposition_digest` field)

---

## Executive finding

Independent red-team review (PGS-J01) of the integrated PGS stack (swarms A–I, I02 landed) finds **no blocker, high, or unresolved medium** authority or pruning defects within the documented bounded scope. All 22 mandatory adversarial questions are answered affirmatively for the claimed invariants, with two **low** trackable items (Lean lake unavailable on reviewer host; documented `bound_exhaustion` forward-without-removal under tight bounds — not a false hard prune).

Committed H02 fixture evidence and a deterministic J01 rerun at `dfa65fb` preserve `result_digest=f4c1baac277fff157b9f384cd30a85d6156909e60677f0404408fa4e24945268` with `false_hard_prune_count=0`. Focused adversarial/regression suite: **105/105 pytest passed**. Static merge-ready checks: **green**.

---

## Bounded claim that survived review

Under **default-off** `goal_support_mode`, **`production_exact`** profile, **compiler/verifier-hard prune-eligible sources only** (`pack_contract`, `verification_requirement`), **bounded word-tree fixture evidence**, and **replay-valid** `SupportCertificate` + `GoalSupportResultV1` sidecars:

1. **Unknown, unobserved, advisory, evaluation, and oracle evidence never authorize certified forest removal.**
2. **`false_hard_prune_count=0`** in committed and rerun campaign evidence.
3. **Diagnostic mode preserves forest identity** (`diagnostic_domain_digest_changes=0`).
4. **Certified pruning flows only through** `exact_goal_closure` → `_prune_with_closure` → canonical VSS `exact_closure` / `SupportCertificate` path.
5. **Certified singleton survivors** commit with **zero neural forwards** when exactly one survivor remains (torch CPU spy test + `certified_singleton_zero_forward` fixture).

**Not claimed:** model quality, ship gates, global OpenUI UNSAT, production ONNX latency, or semantic generalization beyond finite fixture domains.

---

## Finding ledger

| ID | Severity | Status | Summary |
| --- | --- | --- | --- |
| J01-L01 | low | accepted limitation | Lean `lake test` not re-executed on reviewer host; Python golden mapping tests passed as correspondence guard |
| J01-L02 | low | documented expected | `bound_exhaustion` certified arm: forwards without removal under `budget:max_verifier_calls` (all actions UNKNOWN); falsifier tracks separately from `false_hard_prune_count` |

No blocker/high/medium findings remain open.

---

## Adversarial questions (Q1–Q22) — summary

| Q | Topic | Verdict |
| --- | --- | --- |
| Q1 | Authority laundering | **No** — closed compile tables, merge non-escalation, profile validation, `exact_goal_closure` guard |
| Q2 | Free-text masquerade | **No** — prompt facts stay advisory; terminal evidence forbids raw NL fields |
| Q3 | SKIP/UNKNOWN → hard prune | **No** — UNAVAILABLE/UNKNOWN/coverage_unknown; replay fail closed |
| Q4 | unobserved conflation | **No** — four disjoint partitions + Lean laws |
| Q5 | Canonical prune path | **Yes** — `goal_support_certified_prune` → `exact_goal_closure` → `exact_closure` |
| Q6 | Sidecar subordination | **Yes** — `base_certificate_digest` binding; no second search engine |
| Q7 | Cache/profile identity | **Yes** — profile-sensitive backend version; bounds/state fingerprints |
| Q8 | Mutable trace isolation | **Yes** — query-local traces; frozen terminal records |
| Q9 | Ordering invariance | **No illicit drift** — canonical sort; permutation fixture stable |
| Q10 | Obstruction honesty | **Yes** — bounded hitting sets; no global UNSAT language |
| Q11 | Stale evidence invalidation | **Yes** — digest binding + stale identity fixtures |
| Q12 | Privacy/leakage | **Heuristic pass** — forbidden fields + redaction fixture; not cryptographic erasure |
| Q13 | Diagnostic parity | **Yes** — forest unchanged; historical defaults off |
| Q14 | Certified guards | **Yes** — rejects non-production profiles; skips incomplete forests |
| Q15 | Singleton zero-forward | **Yes** when singleton; bound_exhaustion documented exception (J01-L02) |
| Q16 | Unconstrained fallback | **No** — not a weakening lever; I6 decode invariants green |
| Q17 | Checkpoint/config drift | **No unexpected drift** — defaults off; legacy round-trip safe |
| Q18 | DecisionEvent trainability | **Yes** — four-way partition mapping; hard-profile gating |
| Q19 | Lean non-vacuity | **Prose bounded**; Python mapping green; Lean rerun delegated to CI (J01-L01) |
| Q20 | Campaign honesty | **Yes** — diagnostic claim; deterministic digest match on rerun |
| Q21 | Gate/fixture tampering | **None detected** — prereg lock; falsifiers hold |
| Q22 | Ownership consistency | **Yes** — singular owners; schema sync; merge-ready ownership_map ok |

Full per-question evidence refs: JSON artifact `adversarial_questions[]`.

---

## Review evidence

| Check | Result |
| --- | --- |
| Focused pytest | 105 passed (`test_goal_support_adversarial`, `decode_g03`, `mapping`, `domain_adequacy`) |
| Campaign deterministic rerun | digest `f4c1baac…` match; `false_hard_prune_count=0` |
| `verify_merge_ready --fast` | green |
| `verify_decode_invariants` | green |
| H02 fixture digest | `f4c1baac277fff157b9f384cd30a85d6156909e60677f0404408fa4e24945268` |
| H02 manifest | `3b8c082a04bab1191048dfda4587612ff3e24b01f6968fe7bf281f93c719e951` |

### Reproducibility

```bash
# Focused adversarial/regression suite
uv run pytest -q \
  tests/test_dsl/test_goal_support_adversarial.py \
  tests/test_dsl/test_goal_support_decode_g03.py \
  tests/test_formal/test_goal_support_mapping.py \
  tests/test_harnesses/experiments/test_goal_support_domain_adequacy.py

# Deterministic fixture campaign rerun (diagnostic)
uv run python -m scripts.run_goal_support_domain_adequacy \
  --mode fixture \
  --out-dir outputs/runs/pgs_j01_rerun \
  --docs-out /tmp/pgs_j01_rerun.json \
  --claim-class diagnostic

# Static closeout
uv run python -m scripts.verify_merge_ready --fast
```

---

## Limitations and unsupported modes

- Evidence class: **15-fixture word-tree diagnostic campaign** (PGS-H02), not trained-checkpoint or ship eval.
- Default production config: **`goal_support_mode=off`** — certified/diagnostic require explicit opt-in.
- ONNX certified singleton path: not independently re-run in J01 (torch CPU spy test covers I2 pattern).
- Lean structural proofs: CI/`make -C src/leverproof_lean test` owns replay; reviewer host lacked `lake`.

---

## Successor hypotheses (documentation only)

1. Frontier-scale domain adequacy under real pack/compiler coverage.
2. Matched production-exact vs structural-only arms on trained checkpoints.
3. Tighter bound-exhaustion certified policy without weakening UNKNOWN law.
4. ONNX backend parity for certified singleton zero-forward beyond torch CPU fixture.

---

## Canonical cross-links

- Design / threat model: [`proof-carrying-goal-support.md`](proof-carrying-goal-support.md)
- Lean/Python proof boundary: [`proof-carrying-goal-support-proofs.md`](proof-carrying-goal-support-proofs.md)
- Measured fixture evidence: [`proof-carrying-goal-support-fixture-results.md`](proof-carrying-goal-support-fixture-results.md)

This disposition **does not** promote the mechanism, change defaults, or update MODEL_CARD.
