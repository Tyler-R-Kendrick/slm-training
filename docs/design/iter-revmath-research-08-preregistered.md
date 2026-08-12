# RESEARCH-08 (SLM-540): Myhill–Nerode minimization of the static residual

**Claim class:** `fixture` / research-only (default_off; never production)

**Decision:** `accept_research_only`

**Zero semantic disagreement:** `True`

**Promotion:** `False`

## Gate delta

| Metric | Control | Treatment | Delta |
| --- | --- | --- | --- |
| cold_load_p50_ms | 0.32210100016527576 | 0.10284499967383454 | 0.21925600049144123 |
| state_count | 119 | 28 | 91 |
| artifact_bytes | 61019 | 15656 | 45363 |
| language_equivalence_failures | — | 0 | — |
| complete_domain_parity | — | True | — |

## Scope

Terminal-continuation singleton-stack DFA abstraction of the certified static LALR residual adapter. Request-local overlays excluded. Default-off; no production decode authority.

Command: `PYTHONPATH=src uv run python -m scripts.run_research08_myhill_nerode --mode fixture`

Full detail: `docs/design/iter-revmath-research-08-preregistered.json`.
