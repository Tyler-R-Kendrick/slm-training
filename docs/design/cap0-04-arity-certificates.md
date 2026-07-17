# CAP0-04: Exact-vs-estimated arity certificates and provenance-aware reports

**Issue:** SLM-80  
**Status:** wiring / fixture evidence. No train, eval, benchmark, model, checkpoint, or ship claim.

## What was added

Extended the CAP0-02/CAP0-03 arity package with a versioned certificate schema that prevents exact combinatorial results from being silently mixed with estimated or incomplete quantities.

- `src/slm_training/dsl/analysis/arity/certificate.py`
  - `EvidenceKind` enum: `EXACT_LOCAL`, `EXACT_EXTERNAL`, `ESTIMATED`, `INCOMPLETE`.
  - `ConstraintFrame` — grammar/parser/codec versions, bounds, latent role, noise model, packing assumption.
  - `ExactEvidence` — requires `complete=True`, theorem/algorithm, witness/proof hash or source URI.
  - `EstimatedEvidence` — requires dataset/trace/checkpoint provenance, sample count, sampling design, coverage, estimator, seeds, and optional confidence interval/tail metric.
  - `ArityResult` — one metric/value pair with evidence and conservative status (`supported`, `infeasible`, `unknown`, `diagnostic`).
  - `ArityProvenance` — generated-at timestamp, source commit, analyzer/frame version, run/trace IDs, input hashes.
  - `ArityCertificate` — binds a report digest to provenance and results; carries a deterministic JSON digest.
  - `ArityCertificateBundle` — links an `ArityReport` with its certificate and a bundle digest.
  - `exact_certificate_from_report()` — fixture helper that classifies a report as exact-local when a verified construction is present, otherwise as incomplete/unknown.

- `src/slm_training/dsl/analysis/arity/render.py`
  - `to_markdown()` — concise certificate report with summary, claims table, and provenance.
  - `to_csv()` — flattened CSV rows for scoreboards.
  - `one_line_summary()` — compact machine-readable summary.

- `scripts/analyze_grammar_arity.py`
  - `--certificate` emits a CAP0-04 bundle.
  - `--out-md`, `--out-csv`, `--one-line` for alternate renderings.
  - `--source-commit`, `--run-id`, `--trace-id` for provenance.

- Exports added to `src/slm_training/dsl/analysis/arity/__init__.py`.

## Verified

- `ruff check` passes.
- `python -m compileall` passes.
- `pytest tests/test_dsl/test_arity_certificate.py` passes (10 tests).
- `pytest tests/test_dsl/test_arity_analysis.py` passes (11 tests).
- `pytest tests/test_dsl/test_arity_coding.py` passes (11 tests).
- `pytest tests/test_dsl/test_arity_suggest.py` passes (5 tests).
- `.githooks/check-changed` passes (402 passed, 5 skipped).
- `python -m scripts.repo_policy` ok.
- `git diff --check` clean.

## Validation rules enforced

1. `EXACT_LOCAL` requires `complete=True` and a `witness_or_proof_hash`.
2. `EXACT_EXTERNAL` requires a `source_uri`.
3. `ESTIMATED` requires positive `sample_count` and provenance fields.
4. `INCOMPLETE` cannot be used to support an exact/local claim (used only for unknown/diagnostic statuses).
5. A `hankel_rank` metric would require explicit evidence kind; no rank claims are made here.
6. Physical-cost and dynamic-compute claims are rejected unless the required constraint-frame fields are populated (enforced by schema presence, not by silent defaults).

## Example usage

```bash
python -m scripts.analyze_grammar_arity \
  --dsl toy-layout \
  --program 'root = row(title, action)' \
  --max-ast-nodes 4 \
  --include-coding-metadata \
  --certificate \
  --out outputs/runs/arity/toy_certificate.json \
  --out-md docs/design/iter-cap0-04-toy.md
```

## Honesty rules

- Exact claims name the theorem/algorithm or construction that produced them.
- Estimated claims name the data, traces, checkpoints, sampling design, and estimator.
- Incomplete claims are labeled `unknown` or `diagnostic` and never used as pruning authority.
- No model, checkpoint, or ship-quality performance claim is made by this issue.
