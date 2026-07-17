# CAP0-02: Lark/AST arity analyzer

**Issue:** SLM-78  
**Status:** wiring / fixture evidence. No train, eval, benchmark, or ship claim.

## What was added

A repository-owned, Torch-free analyzer package under `src/slm_training/dsl/analysis/arity/`:

- `types.py` — immutable `AnalysisBounds`, `StateAtom`, `StateSignature`, `SupportVerdict`, `SupportQuery`, `SupportResult`, and `SupportOracle` protocol.
- `canonical.py` — Lark typed AST → canonical `StateAtom` tuple. Stable under statement renaming and placeholder surface-string renaming.
- `explorer.py` — bounded program exploration (currently materializes complete-program signatures; partial-prefix BFS to be extended).
- `analyzer.py` — `ArityAnalyzer` implementing `SupportOracle` and emitting `ArityReport`.
- `report.py` — versioned `ArityReport` with deterministic digest.
- `scripts/analyze_grammar_arity.py` — CLI entry point.
- `tests/test_dsl/test_arity_analysis.py` — regression tests for reproducibility, renaming invariance, and conservative support decisions.

## Verified

- `ruff check` passes on new code.
- `python -m compileall` passes.
- `pytest tests/test_dsl/test_arity_analysis.py` passes (8 tests).
- `git diff --check` clean.
- `python -m scripts.repo_policy` ok.

## Caveats

- This is fixture wiring only. The bounded partial-prefix continuation explorer is scaffolded; it currently enumerates complete-program signatures and deduplicates them.
- The prior "86 raw frontier × scope configurations" value has not been reproduced or retired here; that requires a separate CAP0-03/scope-graded reproduction run with the exact frame declared in CAP0-01.
- No model, checkpoint, or ship gate is claimed.
