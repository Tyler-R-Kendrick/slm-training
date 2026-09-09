# A05: cached evaluation publication

Base: `943604270784524ccc032b63359a33913d242b74`. Component: `harness.model_build.eval` v106 → v107.

A complete suite-cache hit returned before honoring `publish_agentv=True`. Both the checkpoint preflight and preloaded-model cache branches skipped the SDK boundary. A replay could also inherit a publication receipt from another run.

The existing evaluator now uses one shared suite publisher for fresh and cached results. Replay removes the prior receipt, writes current-run measurements with `publication_complete=False`, invokes the canonical AgentV publisher when requested, and records true only after it returns. SDK failure propagates while preserving completed counts in both smoke output files. Multi-suite replay still leaves publication to the scoreboard owner. No model load or additional decode is required for a cache hit.

## Focused evidence

The five standalone regressions fail on the actual downloaded base owner (5 failed, 1.57s) and pass on that owner with this patch (5 passed, 1.46s). The owner was imported under its canonical module name using the existing Python 3.12 dependency workspace, not a copied helper implementation. Ruff passes; the evaluator's physical line count is unchanged. Details: [machine-readable results](eval-cache-publication-results.json).

The fixture uses one symbolic example, the existing StubModel, actual disk cache and checkpoint serialization, and mocked SDK success/failure. It proves publication routing and failure handling, not live SDK execution, TwoTower learning, or ship eligibility. Full remote-base dependency verification remains outstanding. No remote compute or hosted checks were requested.

Reproduce in a checkout containing this patch:

```bash
timeout -s INT -k 10 170 python -m pytest -q -o addopts= -m '' tests/test_harnesses/model_build/test_cache_publication.py
```

No metric definitions, frozen data, decoding constraints, selection rules, or ship thresholds change. Historical cached measurements are not relabelled as a newly published evaluation.
