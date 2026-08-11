# INTEG-10 — Bounded regression and release matrix (SLM-576)

**Status:** release-blocking infrastructure matrix. Codifies the bounded,
machine-readable gate catalog that proves formal/revmath/integ infrastructure
contracts without requiring default-off research pilots to succeed.

## Artifacts

| Artifact | Path |
| --- | --- |
| Matrix | `src/slm_training/resources/formal/integ10_release_matrix.v1.json` |
| Orchestrator | `src/slm_training/formal/integ10_release_matrix.py` |
| Verify | `scripts/verify_integ10_release_matrix.py` |
| Release-evidence summary | [`integ-10-bounded-release-matrix-results.json`](integ-10-bounded-release-matrix-results.json) |
| Tests | `tests/test_formal/test_integ10_release_matrix.py` |

## Gate catalog (required)

| Gate | Owner | Mode |
| --- | --- | --- |
| Python formal v2 schema/property | release_matrix (`pytest` formal suite) | invoke |
| Lean contracts / axioms | `ci.lean-formal` (`verify_formal_contracts`) | delegate |
| Cross-language resource-bound parity | release_matrix | invoke |
| Formal mutation / red-team (EVID-11) | `ci.python-static` / merge_ready | delegate |
| INTEG-06 adversarial E2E | `ci.python-static` / merge_ready | delegate |
| Revmath harness parity + owners | release_matrix | invoke |
| INTEG-07 activation-preflight recall | `ci.python-static` / merge_ready | delegate |
| INTEG-08 dominance evidence | release_matrix (`pytest`) | invoke |
| Agent-surface / ownership / versions | `ci.python-static` / merge_ready | delegate |
| Historical v1 read compatibility | release_matrix (`pytest`) | invoke |

## Skip semantics

| Status | Meaning |
| --- | --- |
| `skipped_optional` | Optional tool missing / not invoked in `--check`; **not** success, **not** refutation |
| `skipped_optional_timeout` | Optional probe hit wall budget; same as above |
| `timeout` (required) | Failure; timed-out run is never evidence |
| `research_default_off` | Visible research/pilot arm; not required for infrastructure release |
| `delegated` | Identity-certified to merge_ready / CI owner that already runs the CLI |

Physical performance measurements remain empirical and are **out of scope** for
this abstract-proof matrix (`empirical_performance_separate=true`).

## Run

```bash
export PATH="$HOME/.elan/bin:$PATH"
PYTHONPATH=src uv run python -m scripts.verify_integ10_release_matrix --check
PYTHONPATH=src uv run python -m scripts.verify_integ10_release_matrix --write
# Full live invoke of delegated owners + optional probes when tools exist:
PYTHONPATH=src uv run python -m scripts.verify_integ10_release_matrix --execute --write
PYTHONPATH=src uv run pytest tests/test_formal/test_integ10_release_matrix.py -q
PYTHONPATH=src uv run python -m scripts.verify_merge_ready --fast
```

## Acceptance

- Every required gate is `ok` or `delegated`.
- Optional skips never set `counted_as_success` or `counted_as_refutation`.
- Research/default-off status is reported; `infrastructure_release_needs_research_pilot=false`.
- Release-evidence summary content-addresses the matrix and every gate identity
  (`cas://sha256/...`).

Components: `formal.objects`, `ci.local_merge_gate`, `harness.autoresearch.formal`.
