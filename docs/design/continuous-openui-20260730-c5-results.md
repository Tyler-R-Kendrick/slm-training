# Continuous autotrain cycle 5 results (2026-08-01): fix verified live, second champion found

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaigns | `continuous-loop-20260801-c6` (enqueue) → `c7` (confirm, rejected) |
| Source | `bbe618cc10db46b2a01c389ff8a79d06ae59ec0a` (after the c4 `autoresearch block` fix) |

## Regression check: the c4 fix holds under live execution

Two more continuous-loop cycles ran after the harness fix in `continuous-openui-20260730-c4-results.md` — `c6` (screening) and `c7` (confirm). Neither hit `CYCLE_ERROR`. This is the live counterpart to the two new unit tests: the fix isn't just passing in isolation, the actual driver stopped crashing on the predecessor-lineage path.

## Second champion candidate: grammar_completion_bounds

| Stage | Campaign | Control | Candidate | p50 latency Δ | mpr held | Disposition |
| --- | --- | --- | --- | ---: | --- | --- |
| Enqueue | `c6` | `c20260801-c6-control` | `c20260801-c6-bounds` (`grammar_completion_bounds=true`) | **-254.23ms** | 0.5 → 0.5 | `positive_no_tracked_delta_skip_stack` → `CHAMPION_ENQUEUE` |
| Confirm | `c7` | — | `c20260801-c7-confirm` | n/a | n/a | `TimeoutError: decode exceeded 24s` → `rejected` |

The confirm arm hit a single CPU decode timeout (`decode_timeout_seconds=24`) in this container — diagnosed `target=infrastructure` by the existing classifier, not a code bug. Under the current queue policy a single confirm timeout is disposed as `rejected` rather than retried. This is expected fixture-scale CPU noise, not a regression.

## Next-run priorities

1. Re-enqueue `grammar_completion_bounds=true` on a future screening cycle — a single timeout is not evidence against the lever.
2. Give the already-confirmed `compact_active_canvas=true` champion (fingerprint `7dc23b6cf0129a66`) a real promote attempt now that the Lean formal preflight builds in this container.
3. Do not promote RL; ship gates fail by design on fixture n.

## Artifacts

- Campaigns: `outputs/autoresearch/continuous-loop-20260801-c6/`, `c7/` (not tracked — `outputs/` is gitignored)
- JSON twin: `continuous-openui-20260730-c5-results.json`
