# RESEARCH-03 — Big-Five interpretation Lean package (SLM-549)

**Status:** preregistered evidence (accept)
**Experiment key:** `RESEARCH-03`
**Linear:** [SLM-549](https://linear.app/quickdeploy-ai/issue/SLM-549)
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

An isolated Lean research package can encode one genuine Big-Five bidirectional
classification (WKL ↔ Σ⁰₁ separation over RCA₀) without contaminating the
production finite kernel.

## Contract

| Arm | Role |
| --- | --- |
| Production finite-kernel labeling (no research import; `#print axioms` alone) | matched control |
| `src/revmath_research_lean` + explicit InterpretationPackageV1 | treatment |

| Gate | Result |
| --- | --- |
| Checked bidirectional classification | 1.0 |
| Production import contamination | 0 |
| `#print axioms` alone rejected | True |
| Lake build | ok |
| Decision | **accept** |

Reason: `bidirectional_classification_checked`.

## Explicit coding / interpretation

- coding_id: `coding.wkl_sigma01_sep.v1`
- base: `RCA0`
- target: `rm_research:WKL0`
- Named axioms (insufficient alone — package required):

- `RevMathResearch.Forward.wkl_implies_sigma01_sep`
- `RevMathResearch.Reverse.sigma01_sep_implies_wkl`
- `RevMathResearch.Reverse.not_both_dead_bridge`

## Campaign lock

- Manifest sha256: `4a67246dd3793a45bcd2c979b282462beac0e5f860c882080cc50d3c3a49c19f`
- Lock artifact: `src/slm_training/resources/formal/research_03_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no promotion)

## Four-axis evidence

```json
{
  "assumption_strength": {
    "axis": "assumption_strength",
    "bound_ast_id": null,
    "certificate_sha256": "23650a15e0d553be6fde9ed514c7016a13339d5c40beb4222fc9a0d134a27dd1",
    "notes": "RCA0 research base + named Simpson glue axioms",
    "status": "proved_axis",
    "theorem_ref": "RevMathResearch.Interpretation.forward_from_wkl"
  },
  "computability": {
    "axis": "computability",
    "bound_ast_id": null,
    "certificate_sha256": "5c454d1c2ecaba9af6c34121c1598523c6c094a5aa2ccab99f7d8d075946e9df",
    "notes": "rm_research:WKL0 via explicit interpretation package",
    "status": "proved_axis",
    "theorem_ref": "RevMathResearch.Interpretation.reverse_from_sep"
  },
  "formal_preflight_sha256": null,
  "implementation_refinement": {
    "axis": "implementation_refinement",
    "bound_ast_id": null,
    "certificate_sha256": "a52cb598c88ba94aa4d22170524565ed6195ffae6f72e94c727be1e179cecc0d",
    "notes": "Production Lean trees do not import RevMathResearch",
    "status": "proved_axis",
    "theorem_ref": "package_isolation_scan"
  },
  "resource_bounds": {
    "axis": "resource_bounds",
    "bound_ast_id": null,
    "certificate_sha256": null,
    "notes": "Not the subject of RESEARCH-03",
    "status": "not_claimed",
    "theorem_ref": null
  },
  "schema_version": "revmath_four_axis/v1"
}
```

## Transfer lessons (production assumption-minimization)

- Keep genuine RM encodings in an isolated Lake package; never import into LeverProofLean/OpenUIProofs.
- Big-Five labels require InterpretationPackageV1 (coding_id + both evidence digests); #print axioms is a negative control only.
- Named literature glue axioms are acceptable research scaffolding when listed in the interpretation package — they must not become production authority.
- Assumption-minimization tasks can reuse the four-axis split: assumption_strength vs computability vs isolation refinement.

## Artifacts

| Artifact | Path |
| --- | --- |
| Results JSON | [`iter-revmath-research-03-preregistered.json`](iter-revmath-research-03-preregistered.json) |
| Lean package | `src/revmath_research_lean` |
| Experiment | `src/slm_training/harnesses/experiments/research_03_big_five_interpretation.py` |

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_03_big_five
SLM_ENABLE_RESEARCH_03=1 PYTHONPATH=src uv run python -m scripts.run_research_03_big_five --write
```

## Authority note

Filing or compiling this pilot is not evidence of production readiness.
`rm_research:WKL0` stays research-only; production practical computability
labels remain the KERN-12 finite vocabulary.
