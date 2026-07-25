# SLM-339 / AP-038: idempotent Linear evidence sync (slm339_linear_sync_fixture)

Status: **implemented_dry_run_only** · Live sync performed: **no** (no Linear
credentials in the test environment; live path covered with stub transports)

## What shipped

- `src/slm_training/harnesses/autoresearch/linear_sync.py` —
  `resolve_issue_key` (stable `AP-###` from `campaign_id`; errors when absent),
  `build_evidence_comment` (deterministic, result-object-only, claim-class →
  textual gate summary, no secrets), `comment_marker`
  (`<!-- ap-sync:sha256=<content_sha256> -->`), `SyncTransport` protocol,
  `LinearGraphQLTransport` (env-var-only key, redacted from every error),
  `OfflineTransport` (exact GraphQL payloads for CI review), and
  `sync_result(..., dry_run=True)` returning a resumable `SyncReport`.
- `scripts/sync_abstract_planning_linear.py` — dry-run-first CLI:
  `--result <json> [--live] [--offline-out <file>] [--api-key-env LINEAR_API_KEY]`.
- Contract: `docs/design/linear-evidence-sync.md`; registry component
  `harness.experiments.slm339_linear_sync` v1; `no-bump` history note on the
  SLM-338 component for the shared autoresearch paths.

## Behavior contract (verified by tests)

- Dry run makes **zero** transport calls and prints key / plan / marker.
- Replay with the same content sha is a NO-OP — no duplicate comment; changed
  content updates only the machine comment; markerless human comments are
  never touched.
- Diagnostic/fixture/wiring/screening claim classes are hard-asserted comment-
  only (`_CommentOnlyGuard`); the transport surface has no status/label/
  assignee/priority mutations at all; nothing auto-closes or auto-promotes.
- 429 → honors `Retry-After`; 5xx → bounded exponential backoff; retry-exhausted
  failure returns `action="error"` with a `resume` payload and stays idempotent
  on rerun.
- Canary API key never appears in comment bodies, reports, offline payloads, or
  exception messages.

## Verification

```bash
python -m pytest tests/test_harnesses/autoresearch/test_linear_sync.py \
  tests/test_scripts/test_sync_abstract_planning_linear.py -q   # 20 passed
python -m scripts.sync_abstract_planning_linear --result tests/fixtures/planning_result/ap037_fixture.json
python -m scripts.sync_abstract_planning_linear --result tests/fixtures/planning_result/ap037_fixture.json \
  --offline-out outputs/experiments/slm339/linear_sync_payloads.json
python -m scripts.verify_version_stamps --check
python -m scripts.repo_policy
```
