# RESEARCH-04 — Reverse Mathematics Zoo benchmark (SLM-550)

**Status:** preregistered evidence (accept)  
**Experiment key:** `RESEARCH-04`  
**Linear:** [SLM-550](https://linear.app/quickdeploy-ai/issue/SLM-550)  
**Default-off / research-only:** yes — not production decode, ship-gate, or serving authority.

## Hypothesis

A non-Big-Five Reverse Mathematics Zoo benchmark exposes whether the revmath
harness learns assumption/reversal reasoning rather than five-label mapping.

## Contract

| Arm | Role |
| --- | --- |
| Big-Five-only train benchmark | matched control |
| Zoo + Big-Five mixed eval benchmark | treatment |

| Gate | Result |
| --- | --- |
| Eval exact classification/reversal (dependency-disjoint) | 1 |
| Overclaim rate | 0.0 |
| Unknown calibration | 1.0 |
| Library-retrieval-only falsifier | False |
| Zoo discriminates | True |
| Decision | **accept** |

Reason: `zoo_benchmark_pass`.

Decision rule: eval exact rate == 1.0, overclaim == 0, unknown calibration == 1.0,
and zoo corpus discriminates from Big-Five-only retrieval; else reject/retire.

## Corpus

- Frozen corpus: `src/slm_training/resources/formal/rm_zoo_corpus.v1.json`
- Citations: arxiv-2212.00489 (RM Zoo), simpson-sosoa
- Zoo principles: `zoo:DNR`, `zoo:COH`, `zoo:FIP`, `zoo:RT22`, `zoo:CADS`

## Campaign lock

- Manifest sha256: `1051ec7b9bf7c09f3cb6bef5c101be336699fbdd4a55d2b8eb738b5656adc43e`
- Lock artifact: `src/slm_training/resources/formal/research_04_campaign_lock.v1.json`
- Claim class: `fixture` (research pilot; no promotion)

## Toolchain pin

```json
{
  "backend_id": "rm_zoo_benchmark_pilot",
  "corpus_relpath": "src/slm_training/resources/formal/rm_zoo_corpus.v1.json",
  "label_validator": "harn08.validate_label_claim",
  "toolchain_id": "hermetic_python_rm_zoo_v1",
  "trust_domain": "research_only/rm_zoo_benchmark"
}
```

## Artifacts

| Artifact | Path |
| --- | --- |
| Results JSON | [`iter-revmath-research-04-preregistered.json`](iter-revmath-research-04-preregistered.json) |
| Backend | `src/slm_training/formal/rm_zoo_benchmark.py` |
| Experiment | `src/slm_training/harnesses/experiments/research_04_rm_zoo_benchmark.py` |

## Run

```bash
PYTHONPATH=src uv run python -m scripts.run_research_04_rm_zoo
SLM_ENABLE_RESEARCH_04=1 PYTHONPATH=src uv run python -m scripts.run_research_04_rm_zoo --write
```

## Authority note

Filing or compiling this pilot is not evidence of production readiness.
Zoo classifications remain research-only; production RM labels stay HARN-08
conservative labeling over the finite kernel.
