# Research citation catalog (RESEARCH-01 / SLM-518)

**Status:** RESEARCH-01 landed — machine-readable catalog + verify gate  
**Base SHA:** `bb9e2dcdd16946e949e1f364a5ca6b301601213f` (`origin/main` at INTEG-10 close)  
**Machine-readable catalog:** [`src/slm_training/resources/research_citation_catalog.json`](../../src/slm_training/resources/research_citation_catalog.json)  
**Loader:** [`src/slm_training/citation_catalog.py`](../../src/slm_training/citation_catalog.py)  
**Human lineage narrative:** [research-lineage.md](research-lineage.md) (paper→code fidelity; not a substitute for this ID index)  
**Revmath policy:** [reverse-mathematics-computability.md](reverse-mathematics-computability.md)  
**Verified by:** `PYTHONPATH=src uv run python -m scripts.verify_research_citation_catalog`

> **Delta baseline law:** every `delta_baseline.path` and every GitHub `blob/main/…` URL is a **repo-head link**, not static truth. Coding agents must re-fetch latest `main` and re-audit before treating a citation as current.

## Purpose

This catalog is the repository-native index that ties each reverse-mathematics /
computability / solver / provenance prior to:

| Field | Meaning |
| --- | --- |
| `claim_supported` | The claim the source actually supports |
| `does_not_prove` | Explicit non-implications (required) |
| `repo_relevance` / `repo_application` | Why it matters here and how to cite it |
| `assumption` / `falsifier` | Load-bearing assumption and how the citation fails |
| `trust_level` | `primary` / `secondary` / `repository_baseline` / `adjacent` |
| `derivative_keys` | Issue / harness / experiment keys that may cite this entry |

**Acceptance:** every research / default-off issue cites stable `source_id`s and
states what the cited work does *not* prove. No derivative experiment may
execute without a grounded catalog entry **or** the explicit
`repo-only-hypothesis` marker.

## Source types

| Type | Use |
| --- | --- |
| `foundational_research` | RM / computability / proof-mining literature |
| `solver_certificate` | ATP / SAT / PB certificate standards and tooling |
| `lean_tooling` | Lean libraries, verified complexity, ML-for-Lean tools |
| `provenance_standard` | PROV-O, SPDX AI, interchange metadata |
| `repository_evidence` | Committed repo docs/policy and repository-only hypotheses |

## Core research set (stable IDs)

| `source_id` | Type | Trust | Title (short) |
| --- | --- | --- | --- |
| `simpson-sosoa` | foundational_research | primary | Simpson, *Subsystems of Second Order Arithmetic* |
| `arxiv-1804.05495` | foundational_research | primary | Constructive reverse mathematics |
| `arxiv-2212.00489` | foundational_research | secondary | Reverse Mathematics Zoo survey |
| `mathlib-partrec` | lean_tooling | secondary | mathlib `Computability.Partrec` |
| `buss-marktoberdorf95` | foundational_research | secondary | Buss, bounded arithmetic / feasible reasoning |
| `arxiv-2601.15571` | lean_tooling | secondary | Lean verified complexity reductions |
| `arxiv-2410.15986` | foundational_research | secondary | Proof mining / quantitative extraction |
| `arxiv-1707.03202` | foundational_research | primary | Weihrauch reducibility survey |
| `arxiv-2607.00815` | solver_certificate | adjacent | LRAT-Catcher |
| `arxiv-2602.08692` | solver_certificate | adjacent | PBLean |
| `szs-ontology` | solver_certificate | primary | SZS ontology |
| `leandojo` | lean_tooling | adjacent | LeanDojo |
| `arxiv-2605.20244` | lean_tooling | adjacent | Lean Refactor |
| `w3c-prov-o` | provenance_standard | primary | W3C PROV-O |
| `spdx-ai-model` | provenance_standard | secondary | SPDX AI model profile |
| `repo-research-lineage` | repository_evidence | repository_baseline | `research-lineage.md` |
| `repo-revmath-policy` | repository_evidence | repository_baseline | revmath policy + owners |
| `repo-only-hypothesis` | repository_evidence | repository_baseline | Explicit no-external-prior marker |

## What cited work does **not** prove (summary)

- **Simpson / Zoo / constructive RM** do not license Big-Five labels from
  `#print axioms`, Lean compile success, or finite fixtures (HARN-08 / KERN-12).
- **Weihrauch** does not imply RCA0/WKL0 production labels.
- **mathlib Partrec / Lean verified complexity / LeanDojo / Lean Refactor** do
  not make LeverProofLean theorems or ship gates.
- **LRAT-Catcher / PBLean / SZS** do not turn Timeout/GaveUp into refutation;
  missing tools stay **unknown**.
- **PROV-O / SPDX AI** do not replace `version_stamp`, MODEL_CARD honesty, or
  ship gates.
- **Repository evidence** rows are delta baselines on `main`, not external
  mathematics.

## Citing from issues and experiment manifests

```text
citations:
  - source_id: simpson-sosoa
    does_not_prove_ack: true
  - source_id: repo-revmath-policy
```

or, when no external prior applies:

```text
citations:
  - source_id: repo-only-hypothesis
```

`RESEARCH-02` (SLM-533) owns the preregistration registry that enforces
this at experiment execution time — see
[research-experiment-preregistry.md](research-experiment-preregistry.md).
This issue owns **catalog + validation only**.

## Validation

```bash
PYTHONPATH=src uv run python -m scripts.verify_research_citation_catalog
PYTHONPATH=src uv run python -m scripts.verify_merge_ready --fast
```

The verifier rejects duplicate `source_id`s, duplicate normalized URLs, missing
required fields, empty `does_not_prove`, unknown `source_type` / `trust_level`,
broken design-doc cross-links, and catalog IDs missing from this document’s
ID table.
