# Preflight gates — the refutation-first seam of the continuous loop

Status: shipped (WP-3, 2026-08-09).
Owner package: `src/slm_training/autoresearch/preflight/`.
Motivating review: [harness-evolution-architecture-review-20260809.md](harness-evolution-architecture-review-20260809.md)
(RC3: the identical delta 0.3267→0.3828 was rediscovered and rejected in
**nine separate loops** because nothing consulted the durable evidence record
before spending a screening cycle).

## What it is

A discovery-based gate seam that runs **before** the continuous driver spends
a screening cycle on a candidate arm. Each check is a plugin; any `"block"`
verdict means the candidate is not run and the driver reselects. Verdicts are
persisted into the delivery record so post-hoc review (and the
`skeptic-panel` skill) can audit exactly what was known pre-spend.

This complements — and plugs into — the darkfactory phase-1 machinery
(`evidence_ledger.py` posterior-UCB selection, exact power gate, typed
terminal verdict; see
[darkfactory-hillclimb-optimization.md](darkfactory-hillclimb-optimization.md)):
selection ranks *which open arm is most promising*; preflight decides *whether
the pick deserves any spend at all*.

## Contract (`preflight/__init__.py`)

```python
class PreflightVerdict(BaseModel):
    check_id: str
    verdict: Literal["pass", "warn", "block"]
    reasons: list[str]
    data: dict = {}

class PreflightCheck(Protocol):
    check_id: str
    def run(self, candidate: dict) -> PreflightVerdict: ...

def run_preflight(candidate: dict) -> list[PreflightVerdict]: ...
def has_block(verdicts) -> bool: ...
```

Semantics:

- **pass** — no objection.
- **warn** — information the panel/operator should see; does not stop the run.
- **block** — the candidate must not be run this cycle.
- Any check crash (import failure, `run()` raising, malformed verdict) becomes
  a `warn` verdict with the traceback summary in `reasons`. A preflight bug can
  never take down the loop and can never silently block a candidate.
- `run_preflight` returns all verdicts sorted by `check_id` (deterministic).

## Discovery mechanism

`run_preflight` iterates `pkgutil.iter_modules` over the package path (sorted
by module name, `_`-prefixed modules skipped) and picks up any module exposing
a module-level `CHECK` object. Attachment is purely positional: **a sibling
plugin attaches by existing in the package directory** — no registry, no
imports in `__init__.py`, and absence is never an error.

Sibling plugins (concurrent WPs) attach exactly this way:

- `power_check.py` (`check_id "power_decidability"`) — blocks measurements
  that cannot reject at alpha at all (wraps the exact sign-test arithmetic in
  `evidence_ledger.py`).
- `concluded_family.py` (`check_id "concluded_family"`) — blocks arms whose
  hypothesis family carries a typed terminal verdict.
- `prior_attempts.py` (`check_id "prior_attempts"`, this WP) — see below.

## Candidate dict shape

All keys optional; checks must tolerate missing keys and ignore unknown ones.

| key | type | meaning |
| --- | --- | --- |
| `hypothesis_text` | `str` | arm hypothesis sentence |
| `lever_keys` | `list[str]` | sorted lever/config keys being moved |
| `config_fingerprint` | `str` | canonical config identity hash |
| `n_seeds` | `int` | seeds this attempt would spend |
| `steps` | `int` | train steps per arm |
| `minimum_effect` | `float` | policy minimum effect for the endpoint |
| `endpoint_metric` | `str` | e.g. `smoke.structural_similarity` |

The driver additionally passes `slug` (screening-arm identity) and `levers`
(the materialized lever dict) so checks can fall back to slug matching and
recompute fingerprints.

### Fingerprint algorithm

`prior_attempts.config_fingerprint` mirrors `_knobs_fingerprint` in
`scripts/run_autotrain_continuous.py` (the only pre-existing config-identity
helper in the repo): sha256 over canonical JSON (`sort_keys=True`, compact
separators, `default=str`) of the lever dict with the `steps` cycle-jitter key
excluded, truncated to 16 hex chars. If
`slm_training.evidence_store.records` ships its own fingerprint helper it is
preferred at runtime (probed by name) so both producers stay identical.

## The `prior_attempts` check

Consults `slm_training.evidence_store.client.find_prior_attempts`
(guarded import — the store is concurrent work) and falls back to the
committed arm-level ledger
`src/slm_training/resources/experiments/autotrain_climb/evidence_ledger.v1.json`
(fallback records match on `slug` and carry `config_fingerprint=None`,
because the arm ledger has no per-config fingerprints).

| verdict | condition |
| --- | --- |
| `block` | identical `config_fingerprint` + `endpoint_metric` was `confirm_failed` / `ship_rejected` with adequate power (`n_seeds >= 8`, or a meaningful recorded `p_value` in `(0, 1]`) |
| `warn` | config attempted before but only with inadequate power |
| `pass` | no matching prior attempt (or prior positive) |

## Integration point in `scripts/run_autotrain_continuous.py`

Inside `run_cycle`, immediately after `rec_slug = _select_cycle_slug(...)` and
before the matrix build (`_matrix(...)`):

1. Screening cycles only (`rec_slug is not None and replay is None`;
   confirm/promote carry frozen recipes and are never re-gated).
2. `_preflight_screening_slug` builds the candidate dict from the screening
   arm bank entry (hypothesis + materialized lever extras, fingerprint via
   `_knobs_fingerprint`, endpoint + minimum_effect from
   `primary_for_role(policy, "screening")`) and calls `run_preflight`.
3. Any `block` ⇒ the slug joins the skip set and `_select_cycle_slug`
   reselects; repeated until a candidate passes.
4. **Fail-soft floor:** if every open arm blocks (or the seam itself fails),
   the original pick runs anyway with `override:
   "all_open_arms_blocked_ran_original_pick"` recorded — the preflight never
   exhausts the loop on its own authority (arm *closure* remains the job of
   the multi-seed null evidence path and the typed terminal verdict).
5. The full verdict map is written to `<campaign>/preflight.json`;
   `_phase_a_delivery` folds it into the delivery record as a `preflight`
   field, so it lands in both `sdlc_delivery.json` and the append-only
   `outputs/autoresearch/sdlc_delivery_ledger.jsonl`.
6. At driver closeout (`main`'s `finally`), `scripts/sync_evidence_store.py`
   is invoked when present (subprocess, 120 s timeout, failure logs only) so
   the durable store absorbs the session's outcomes for the next loop's
   preflight.

## Relation to the skeptic panel

Preflight gates **spend**; the [`skeptic-panel`](../../.agents/skills/skeptic-panel/SKILL.md)
skill gates **belief**: after a screening positive, three independent skeptic
lenses (statistical power, prior evidence, mechanism plausibility) attempt
refutation before any confirmation spend, taking `scripts/query_evidence.py`
output and these preflight verdicts as inputs. Majority-refute files the
candidate as rejected with reasons in the delivery record. Gates are never
weakened in either direction.

## Testing

`tests/test_autoresearch/test_preflight_gates.py` covers the contract
(discovery, crash→warn, sorting, block detection) and the `prior_attempts`
verdict matrix (block/warn/pass, fingerprint stability, guarded-import
fallback, never-raise). Unit tests only — no training or eval runs.

## Post-launch reconciliation fixes (2026-08-10)

Two gaps flagged as "known reconciliation items" at launch are now closed:

**Cumulative seeds, not a literal 1.** `_preflight_screening_slug` builds
each candidate's `n_seeds` from the arm's cumulative ledger `n_complete`
(`+1` for the cycle about to run), not a hardcoded `1`. A literal 1 is
undecidable by construction for `power_check` (see
[`power-preflight.md`](power-preflight.md#seeds-policy-reconciliation-post-launch-fix-2026-08-10))
and would have made every screening cycle for every arm block forever,
defeating the very accumulation the loop depends on.

**Arm-fingerprint convergence.** A live candidate's `config_fingerprint` is
lever-derived (hash of its concrete lever dict); `sync_evidence_store.py`'s
climb-ledger-arm records are fingerprinted over a slug-keyed source
descriptor (`evidence_store.records.compute_arm_fingerprint`) — a
deliberately different, cheaper domain that needs no lever
materialization. The two never coincide by construction, so once the
evidence store became the primary lookup path, arm-level records were
invisible to `prior_attempts`'s exact-fingerprint match (worse than the
raw-ledger fallback it was meant to improve on). `prior_attempts.py` now
also computes the candidate's arm fingerprint from its `slug`
(`arm_fingerprint()`) and queries/matches on it alongside the lever-derived
fingerprint, without weakening `_knobs_fingerprint`'s established
(steps-excluded, 16-hex, truncated) identity used elsewhere for
champion-queue dedup.

## The `mechanism_no_effect` check (INTEG-02 / SLM-555)

Optional theorem-backed admission for mechanism treatments. Reuses KERN-08
`emit_no_effect_certificate` — never invents a second no-effect authority.

Candidate keys (all optional; check idles without `mechanism_id`)::

    mechanism_id, corpus_id, observations|locked_corpus, scan_complete,
    campaign_id, experiment_id, manifest_sha256, cost_model, theorem_ref,
    claim_class

| verdict | condition |
| --- | --- |
| `block` (`skip_no_effect`) | complete scan + every trigger proved absent + certificate emits |
| `pass` (`run`) | present trigger, unknown evidence, incomplete scan, unsafe mechanism, or missing corpus |
| `warn` | check crash (package fail-soft law) |

Design notes: [`integ-02-mechanism-no-effect-preflight.md`](integ-02-mechanism-no-effect-preflight.md).

