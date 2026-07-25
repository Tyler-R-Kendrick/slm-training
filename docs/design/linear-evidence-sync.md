# Linear evidence sync contract (AP-038 / SLM-339)

Component: `harness.experiments.slm339_linear_sync` · Module:
`src/slm_training/harnesses/autoresearch/linear_sync.py` · CLI:
`scripts/sync_abstract_planning_linear.py`

Synchronizes one `AbstractPlanningResultV1` (AP-037 / SLM-338) to the matching
Linear issue as **exactly one machine-owned evidence comment** — idempotently,
dry-run-first, with an offline GraphQL review mode for CI.

## Invariants

- **One machine comment per issue.** The comment ends with an idempotency
  marker `<!-- ap-sync:sha256=<content_sha256> -->` where the digest is
  `AbstractPlanningResultV1.content_sha256()`. Replay with the same content is
  a NO-OP; changed content updates the machine comment in place; no machine
  comment creates one. Duplicates are structurally impossible.
- **Human ownership preserved.** Comments without the marker are never read for
  mutation and never modified. The sync never touches labels, status, assignee,
  or priority, and never auto-closes or auto-promotes.
- **Comment-only claim classes.** `wiring`, `fixture`, `diagnostic`, and
  `screening` results produce the evidence comment but are hard-asserted (via
  `_CommentOnlyGuard`) to reach only the three comment transport methods. Gate
  and promotion state are out of scope for every claim class — the
  `SyncTransport` protocol has no issue-field mutation methods at all.
- **Dry-run first.** `sync_result(..., dry_run=True)` (the default) and the CLI
  with no flags make **zero** transport calls and print the resolved issue key,
  the planned comment, and the idempotency marker.
- **No secrets, ever.** `LinearGraphQLTransport` reads the API key from the
  configured env var (default `LINEAR_API_KEY`) only, never logs it, and
  redacts it from every error message. The comment body is built strictly from
  the result object and is deterministic.
- **Offline review.** `OfflineTransport` (CLI `--offline-out <file>`) records
  the exact GraphQL payloads the live path would send — same query/mutation
  constants, same variables — for CI review without network.
- **Resilient.** 429 responses honor `Retry-After`; 5xx responses retry with
  bounded exponential backoff (default 3 retries, base 1s, cap 30s). A failure
  after retries yields a `SyncReport(action="error")` carrying a `resume`
  payload (issue key + comment body); re-running the same sync is safe because
  the marker keeps every retry idempotent.

## Issue resolution

`resolve_issue_key(result)` extracts the stable `AP-###` key from
`campaign_id` (e.g. `ap-037-fixture` → `AP-037`); a campaign id with no AP key
raises `IssueKeyError`. The GraphQL lookup uses Linear's `issue(id:)` query,
which accepts team identifiers.

## Comment body

Deterministic Markdown from the result object only: campaign id, manifest
sha256, claim class / disposition, source commit + dirty flag, data snapshot
sha256, a **textual** gate-state summary mapped from the claim class (the sync
never reads or writes real gate state), the decode-path metrics table, latency
totals, verifier gate pass/FAIL list, audit references, an ownership notice,
and the idempotency marker.

## CLI

```bash
# Dry run (default): plan only, no network.
python -m scripts.sync_abstract_planning_linear --result <result.json>

# Offline GraphQL payloads for CI review (no network).
python -m scripts.sync_abstract_planning_linear --result <result.json> \
  --offline-out outputs/linear_sync_payloads.json

# Live sync (requires LINEAR_API_KEY; missing key → clean error, exit 2).
python -m scripts.sync_abstract_planning_linear --result <result.json> --live \
  [--api-key-env LINEAR_API_KEY]
```

Exit codes: `0` dry-run / offline / successful or NO-OP live sync; `1` live
sync ended in a retry-exhausted transport error (report still printed); `2`
missing API key in live mode.
