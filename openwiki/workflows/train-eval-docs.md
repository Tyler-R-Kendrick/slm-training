# Workflow: train → eval → docs

Iron law (from `AGENTS.md`): **no train / eval / bench / profile / telemetry / matrix / repro without updating docs.**

## Typical CLI path

```bash
# Build data (see scripts/ + docs/design)
python -m scripts.build_train_data ...
python -m scripts.train_model --train-dir outputs/train_data/v1 \
  --context-backend hf --run-id <id> --steps <n>
python -m scripts.evaluate_model ...
python -m scripts.run_quality_matrix ...   # often scratch / local-only
```

Run `npm ci` before evaluation. Shared eval paths automatically publish the
AgentEvals JSONL source and AgentV SDK artifacts under `<run-dir>/agentv/`; see
[`docs/design/agentv-evaluation.md`](../../docs/design/agentv-evaluation.md).

After every run:

1. Update matching `docs/design/*.json` and measured-results markdown.
2. If a checkpoint was created/synced/promoted: update `docs/MODEL_CARD.md` **and** the README model-card summary.
3. Follow skill `documenting-experiment-results`; use `honest-ship-eval` before ship claims.

## Matrix workflows

| Script | Design doc |
| --- | --- |
| `scripts/run_quality_matrix.py` | `docs/design/quality-experiment-matrix.md` |
| `scripts/run_grammar_matrix.py` | grammar design + `grammar-matrix-results.json` |
| `scripts/run_perf_matrix.py` | `docs/design/perf-experiment-matrix.md` |
| `scripts/run_phase_pipeline.py` | phase results under `docs/design/` |

Skill: `running-experiment-matrices`.

## Agent checklist (source-backed)

- Numbers only in `outputs/` or chat = incomplete.
- Fixture / scratch demos ≠ production HF ship.
- Prefer `rtk` for verbose shell output when installed (`RTK.md`).
