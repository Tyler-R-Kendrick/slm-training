# A05 measurement cache publication evidence — 2026-09-08

This narrow change keeps cached evaluation evidence on the same AgentV publication boundary as fresh evaluation.

## Implemented

- Cached measurement paths call the canonical publication helper when publication is requested.
- Publication failures preserve measured rows/counts but set `publication_complete=false`; they do not become a successful published evaluation.
- Stale publication receipts are removed before retry.
- Cache-hit paths do not reload or decode the model.
- Empty suites remain fail-closed.

## Validation

The worker's candidate validation reported 92 measurement/resume tests and 59 evaluation/gate tests passing, plus lint and case-extraction checks. SDK tests used mocks and therefore are not live AgentV evidence.

This is a focused A05 slice. Full exact-tree release verification, a live AgentV SDK publication, and a real six-case model run remain separate obligations.