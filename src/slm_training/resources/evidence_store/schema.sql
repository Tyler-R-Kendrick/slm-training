-- Durable evidence store: evidence_records (evidence_record/v1).
-- Matches slm_training.evidence_store.records.EvidenceRecordV1.
--
-- How to apply (idempotent — safe to re-run):
--   * Supabase: paste this whole file into the SQL editor
--     (Dashboard -> SQL Editor -> New query) and Run.
--   * psql:     psql "$DATABASE_URL" -f src/slm_training/resources/evidence_store/schema.sql
--
-- Upserts arrive via PostgREST:
--   POST {SUPABASE_URL}/rest/v1/evidence_records?on_conflict=config_fingerprint,endpoint_metric
--   Headers: apikey / Authorization: Bearer <service role key>
--            Prefer: resolution=merge-duplicates
-- The client populates lever_keys_text (space-joined lever_keys) so the
-- generated tsvector below can index hypothesis_text and lever keys together
-- (a generated column may not reference another generated column).

create table if not exists evidence_records (
    id                  bigint generated always as identity primary key,
    experiment_id       text not null,
    campaign_id         text not null default '',
    hypothesis_text     text not null default '',
    lever_keys          jsonb not null default '[]'::jsonb,
    lever_keys_text     text not null default '',
    config_fingerprint  text not null,
    endpoint_metric     text not null,
    effect_size         double precision,
    n_seeds             integer not null default 1,
    steps               integer not null default 0,
    p_value             double precision,
    outcome             text not null check (outcome in (
                            'screen_positive',
                            'screen_negative',
                            'confirm_failed',
                            'confirmed',
                            'ship_rejected',
                            'promoted'
                        )),
    blocker             text,
    version_stamp       jsonb not null default '{}'::jsonb,
    source_path         text not null default '',
    run_date            text,
    inserted_at         timestamptz not null default now(),
    search_tsv          tsvector generated always as (
                            to_tsvector(
                                'english',
                                coalesce(hypothesis_text, '')
                                    || ' '
                                    || coalesce(lever_keys_text, '')
                            )
                        ) stored
);

-- Dedup identity shared with the committed local mirror
-- (src/slm_training/resources/evidence_store/local_index.jsonl).
create unique index if not exists evidence_records_fingerprint_metric_uq
    on evidence_records (config_fingerprint, endpoint_metric);

-- A table created in `public` is exposed through PostgREST to any holder
-- of the project's anon/publishable key; RLS is off by default for a table
-- created this way, so without it anonymous clients could read every
-- evidence row (hypothesis_text, source_path, etc). The sync client
-- authenticates with the service-role key, which bypasses RLS entirely, so
-- enabling it with no policies keeps the sync/query path working and denies
-- everyone else. Add explicit policies here if anon/authenticated read
-- access is ever genuinely needed.
alter table evidence_records enable row level security;
revoke all on evidence_records from anon, authenticated;

-- Full-text search over hypothesis text + flattened lever keys
-- (PostgREST: ?search_tsv=wfts.<query>).
create index if not exists evidence_records_search_tsv_gin
    on evidence_records using gin (search_tsv);

-- Common lookup paths.
create index if not exists evidence_records_campaign_idx
    on evidence_records (campaign_id);
create index if not exists evidence_records_outcome_idx
    on evidence_records (outcome);
