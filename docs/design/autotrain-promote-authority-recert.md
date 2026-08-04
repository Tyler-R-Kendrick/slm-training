# Autotrain promote authority re-cert

**Status:** continuous driver law (v177+). Not a ship claim.

## Problem

Climb promotions (`promoted` / `climb_accepted`) lived in a durable loop-local
queue. Open/confirmed heads were revalidated under current classifiers, but
**accepted climb promotions were not**. After harness or climb-policy updates
(effect gates, multi-seed, locked expectations), historical “promoted” rows
could remain authoritative without surviving current dispose rules.

One-off `scripts/repair_vacuous_promotes.py` backfilled a known vacuous class;
it is not a substitute for every future harness change.

## Law

1. Successful climb dispose stamps `promote_authority_sha256` =
   `autotrain_promote_authority/v1` over climb-policy file sha, locked promote
   expectations sha, and `harness.autoresearch.experiment_campaign` version.
2. Every continuous cycle startup runs
   `_recertify_promoted_champion_entries` before selecting thrash/confirm/promote
   heads.
3. Stamp mismatch or missing stamp → paper re-dispose under **current** rules:
   - still `climb_accepted` → keep + restamp
   - fail → `promotion_failed` + `historical_reclassification.jsonl`
   - incomplete evidence → `confirmed` + `recert_required` for live promote
4. Policy: `promotion_dispose.recertify_on_authority_change` (default true).

## Non-goals

- Does not weaken ship gates or turn fixture smoke into ship authority.
- Does not auto-ship. Climb re-cert ≠ `--ship-gates` pass.

## Evidence

Unit tests in `tests/test_scripts/test_run_autotrain_continuous.py`
(`test_recertify_promoted_*`, `test_promote_authority_digest_changes_*`).
Live loop applies on next driver cycle after tip includes this harness.
