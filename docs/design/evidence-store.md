# Durable evidence store (committed local mirror + optional Supabase sync)

WP-1 of the harness-evolution mission. Status: shipped (local mirror
committed; Supabase sync is credential-gated and fail-soft).

## Motivation (RC3: the nine-times-rediscovered delta)

The architecture review
([harness-evolution-architecture-review-20260809.md](harness-evolution-architecture-review-20260809.md),
RC3) found the same improvement delta independently rediscovered in nine
separate loops. Root cause: loop memory under `outputs/autoresearch/**` is
gitignored and container-ephemeral, while the durable record — 1,500+ result
JSONs under `docs/design/` — was never machine-consumed, so hypotheses kept
being re-tried. The committed autotrain-climb ledger
(`src/slm_training/autoresearch/evidence_ledger.py`, darkfactory phase 1)
closed this gap for *screening arms*; this store generalizes it to *any
experiment outcome* with a queryable "has this been tried?" surface.

## Components

| Piece | Path |
| --- | --- |
| Record model | `src/slm_training/evidence_store/records.py` (`EvidenceRecordV1`) |
| Committed mirror I/O | `src/slm_training/evidence_store/local_index.py` |
| Fail-soft client | `src/slm_training/evidence_store/client.py` (`find_prior_attempts`) |
| Committed mirror | `src/slm_training/resources/evidence_store/local_index.jsonl` |
| Postgres schema | `src/slm_training/resources/evidence_store/schema.sql` |
| Sync CLI | `scripts/sync_evidence_store.py` |
| Query CLI | `scripts/query_evidence.py` |

## Record schema (`evidence_record/v1`)

One `EvidenceRecordV1` normalizes one experiment outcome:

- `experiment_id`, `campaign_id`, `hypothesis_text`, `lever_keys: list[str]`
- `config_fingerprint` — sha256 of the canonical-JSON-serialized config dict
  (`json.dumps(config, sort_keys=True, default=str)`). This mirrors the exact
  canonicalization already used inline by
  `slm_training.autoresearch.evidence_ledger.build_ledger` for its payload
  digest; that module exposes no importable fingerprint helper, so
  `records.compute_config_fingerprint` pins the same expression and a test
  (`test_fingerprint_matches_evidence_ledger_canonicalization`) guards drift.
  Climb-ledger-arm records use a distinct, deliberately cheaper sub-domain —
  `records.compute_arm_fingerprint(arm)` — fingerprinting a slug-keyed source
  descriptor rather than a concrete lever dict; see
  `docs/design/preflight-gates.md`'s "arm-fingerprint convergence" note for
  why a live candidate needs to reproduce this exact value.
- `endpoint_metric`, `effect_size | None`, `n_seeds`, `steps`, `p_value | None`
- `outcome`: `screen_positive | screen_negative | confirm_failed | confirmed |
  ship_rejected | promoted`
- `blocker | None` (e.g. `fixture_insufficient_n`, `no_complete_measurements`,
  `control_arm`)
- `version_stamp: dict` (pass-through of the source's `version_stamp/v1`)
- `source_path`, `run_date | None`

Dedup identity everywhere (local mirror **and** the Supabase unique index) is
`(config_fingerprint, endpoint_metric)`.

## Committed local mirror

`src/slm_training/resources/evidence_store/local_index.jsonl` — one canonical
JSON object per line (`sort_keys=True`), deduplicated, sorted by
`(config_fingerprint, endpoint_metric)` so regeneration produces clean diffs.
It is committed: a fresh container answers "was this tried?" with zero
credentials and zero network.

## Sync

```
python -m scripts.sync_evidence_store            # parse -> merge -> mirror (+ Supabase iff creds)
python -m scripts.sync_evidence_store --dry-run  # report only
```

Sources (best-effort mapping; unmappable entries are skipped with a counted
warning, missing files likewise):

1. `docs/design/quality-matrix-results.json`, `grammar-matrix-results.json`,
   `perf-matrix-results.json`, `phase-abc-results.json`,
   `baseline-reproduction-results.json` (currently absent → counted warning).
   - quality/grammar arms map pass/fail to `screen_*` (grammar
     `stage: confirmation` maps to `confirmed`/`confirm_failed`); effect is
     `smoke.structural_similarity`.
   - perf arms map to `perf.tokens_per_sec_delta_vs_control` (effect = arm −
     control throughput; the control row carries `blocker: control_arm`).
   - phase-abc maps one aggregate `ship_gates.pass` record
     (`ship_rejected`/`promoted`) plus one record per phase board (champion
     phase → `screen_positive`).
2. `src/slm_training/resources/experiments/autotrain_climb/evidence_ledger.v1.json`
   — one record per arm (`n_positive > 0` → `screen_positive`; arms with zero
   complete measurements carry `blocker: no_complete_measurements`).
3. Any local `outputs/autoresearch/**/*delivery*.json`
   (`autotrain_sdlc_delivery/v1`); arm slugs are extracted by reusing
   `evidence_ledger.slug_from_candidate_token`.

The first local run (no credentials) produced **40 records** in the committed
mirror; re-running is byte-identical (idempotent merge).

When `SUPABASE_URL` **and** `SUPABASE_SERVICE_ROLE_KEY` are both set, the
merged index is upserted to Supabase via PostgREST:
`POST {SUPABASE_URL}/rest/v1/evidence_records?on_conflict=config_fingerprint,endpoint_metric`
with `apikey` / `Authorization: Bearer` headers and
`Prefer: resolution=merge-duplicates`, batched (default 200 rows).

## Query

```
python -m scripts.query_evidence --hypothesis "macro tokens improve parse rate"
python -m scripts.query_evidence --lever bounds --lever binder-topology
python -m scripts.query_evidence --fingerprint <sha256> --json
```

Programmatic surface (frozen signature; never raises):

```python
from slm_training.evidence_store.client import find_prior_attempts

prior = find_prior_attempts(
    hypothesis_text=..., lever_keys=..., config_fingerprint=...
)
```

Match order: exact `config_fingerprint` hits first; then full-text search via
PostgREST (`?search_tsv=wfts.<query>` over a generated tsvector spanning
`hypothesis_text` + flattened lever keys) when credentials exist; otherwise
the committed local index scored by substring + token overlap.

## Fail-soft behavior (repository law)

- Credentials absent → **one** logged notice per process
  (`evidence_store: SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY unset …`); the
  local mirror is still written and queried. No exception, ever.
- Any transport/HTTP/parse failure at query or upsert time logs a warning and
  degrades to the local index (query) or returns the partial pushed count
  (upsert).
- No secrets are committed; the service-role key is read from the environment
  only. Tests exercise the HTTP client through an injected fake transport —
  no network I/O in tests.

## Applying the Postgres schema

`src/slm_training/resources/evidence_store/schema.sql` is idempotent
(`create table if not exists` / `create index if not exists`): paste it into
the Supabase SQL editor, or run
`psql "$DATABASE_URL" -f src/slm_training/resources/evidence_store/schema.sql`.
The table carries the unique dedup index, a stored generated `search_tsv`
tsvector over `hypothesis_text || lever_keys_text` and its GIN index
(`lever_keys_text` is populated by the client because a generated column
cannot reference another generated column).

## Continuous-loop closeout integration (owned by WP-2)

`scripts/run_autotrain_continuous.py` is owned by another work package; its
closeout integration is a guarded one-liner so the loop keeps running when
this package is absent or degraded:

```python
try:  # evidence store is optional at closeout — never blocks the loop
    from slm_training.evidence_store.client import find_prior_attempts
except Exception:
    find_prior_attempts = None
```

At hypothesis-selection/closeout time the loop calls
`find_prior_attempts(hypothesis_text=..., lever_keys=..., config_fingerprint=...)`
(when the import succeeded) and treats returned `screen_negative` /
`confirm_failed` / `ship_rejected` records as prior evidence against
re-spending on the same delta — closing RC3's rediscovery loop. The call is
fail-soft by contract: it returns `[]` rather than raising in every degraded
mode.

## Tests

`tests/test_evidence_store/` — record/fingerprint contract, fixture-JSON →
parse → dedup → local-index roundtrip, local-mode `find_prior_attempts`
matching, and the PostgREST client through a mocked transport
(`python -m pytest tests/test_evidence_store/`).
