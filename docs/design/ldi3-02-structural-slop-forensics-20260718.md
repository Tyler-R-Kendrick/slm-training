# LDI3-02 — Structural-slop forensics for OpenUI generations

**Issue:** SLM-129 (LDI3-02). **Date:** 2026-07-18. **Status:** forensics engine +
robust extractors + statistical ranking + detector classification, implemented and
tested. **No model update, no ban list, no semantic labels** — over-representation
is diagnostic only, not causal preference evidence.

## What this delivers

The OpenUI analogue of Auto-Antislop's model-specific profiling — a
deterministic forensics engine that ranks motifs over-represented in parent
on-policy generations versus the Gold/Silver human/program baseline:

- `src/slm_training/harnesses/quality/slop_forensics.py`
- `scripts/run_slop_forensics.py` — CLI + no-model-update evidence report
- `tests/test_harnesses/quality/test_slop_forensics.py` — 9 tests
- `docs/design/ldi3-02-slop-forensics-report-20260718.json` — committed fixture
  report

## Design

A robust, self-contained **engine** over a clean `ProgramFeatures` abstraction,
plus a defensive `extract_features` that derives the program-derivable families
via the existing canonicalizer — no fragile parse internals:

| Family | Source | Notes |
| --- | --- | --- |
| `surface_ngram` | tokenized source, 1–N grams | per-**program** occurrence, not raw frequency |
| `skeleton` | `canonical_fingerprint` | collapses alpha-equivalent / symbol-renamed programs |
| `placeholder` | `collect_placeholders_from_text` | inventory |
| `component_edge` | optional AST evidence | populates only when supplied |
| `grammar_motif` | optional trace evidence | populates only when supplied |
| verifier `first_failing_gate` | optional verifier evidence | drives `semantic_failure_candidate` |

The trace/verifier-conditional families are the issue's "when traces exist" /
"where verifier evidence permits" cases — accepted as typed optional inputs.

## Statistical ranking (the honesty-critical core)

- per-corpus program-occurrence counts (not token frequency);
- smoothed log-odds vs the Gold/Silver baseline (configurable prior);
- **deterministic group-bootstrap CI** — resamples prompt *groups*, seeded, so a
  motif concentrated in one prompt family gets a wide, honest interval;
- **low-support suppression**: low-support motifs are flagged and sort last —
  they cannot outrank stable, high-support effects regardless of ratio;
- source concentration + held-baseline stability are reported;
- detector-candidate classes: `diagnostic_only`, `counterfactual_probe_candidate`,
  `constraint_distillation_candidate`, `semantic_failure_candidate`,
  `whitelisted_domain_motif` — **no class becomes a training label here**, and a
  semantic-failure candidate still requires same-state replay evidence.

## Commands

```bash
python -m pytest tests/test_harnesses/quality/test_slop_forensics.py -q         # 9 passed
python -m ruff check src/slm_training/harnesses/quality scripts tests
PYTHONPATH=src python -m scripts.run_slop_forensics --fixture \
    --out docs/design/ldi3-02-slop-forensics-report-20260718.json
python -m scripts.repo_policy
```

## Scope

Implements the engine, robust program-derivable extractors, group-bootstrap
statistics, detector classification, deterministic outputs, tests, and a fixture
evidence report. Full AST-subtree and grammar/compiler-trace extraction from live
generation traces, and trace localization to the earliest committing exact state,
consume optional evidence here and are wired for the LDI3 mining issues that
produce that evidence. No inference-time ban/backtracking; no semantic event
labels; no model training.
