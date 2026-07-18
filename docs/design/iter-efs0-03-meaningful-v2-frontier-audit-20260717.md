# EFS0-03 — binding-aware meaningful-v2 frontier audit

Date: 2026-07-17. Status: **complete diagnostic replay**.

This was a CPU replay over a committed, bounded generation envelope. It did not
run training, generate new model outputs, write a checkpoint, or evaluate ship
gates. The machine-readable result is
[iter-efs0-03-meaningful-v2-frontier-audit-20260717.json](iter-efs0-03-meaningful-v2-frontier-audit-20260717.json).

## Recipe and durability

- replay bundle:
  `src/slm_training/resources/evals/meaningful_v2_frontier_replay.json`;
- bundle SHA-256:
  `1c33279636384e1a5eff6a558ae235c060d7157f1293db268f380471aa4fe979`;
- 11 generation sets, three smoke rows each (`n=33`);
- capture verified 11 distinct checkpoint SHA-256 values and persisted raw
  predictions, source records, reconstructed effective `GenerationRequest`
  values, and content digests;
- inputs: E228, E229, E249, E252, E262, E264, E265, E293, E294, E295,
  and E296;
- metric implementation hash:
  `a5604e640b0c9abd0631ed2d697327921466581f1e1eacad1cebe430f39d03ae`;
- AgentEvals/AgentV: 12/12 JSON envelopes validated with zero execution
  errors. This validates envelope execution only; each set's `replayable`
  field is the semantic audit verdict.

The exact replay command was:

```bash
python -m scripts.audit_meaningful_program \
  --replay-bundle src/slm_training/resources/evals/meaningful_v2_frontier_replay.json \
  --gaming-corpus src/slm_training/resources/evals/meaningful_v2_gaming.jsonl \
  --unavailable-set 'e230=no per-case prediction envelope retained' \
  --unavailable-set 'e240-e247=aggregate matrix results only; no per-case prediction envelopes retained' \
  --unavailable-set 'e288-fixed=documented output path absent' \
  --unavailable-set 'e292=stored prediction hits legacy 500-character cap without digest' \
  --unavailable-set 'x22=aggregate grammar results only; no per-case prediction envelopes retained' \
  --minimum-frontier-sets 10 \
  --output docs/design/iter-efs0-03-meaningful-v2-frontier-audit-20260717.json \
  --run-dir outputs/runs/slm105-meaningful-v2-frontier-bundle
```

## Frontier comparison

| Generation set | Checkpoint hash (prefix) | v1 positive | v2 strict | Coverage | Conclusion |
| --- | --- | ---: | ---: | ---: | --- |
| E228 | `7a9be4a665e2` | 1/3 | 0/3 | 3/3 | historical v1 clear does not clear v2 |
| E229 | `23f31fa977cc` | 1/3 | 0/3 | 3/3 | historical v1 clear does not clear v2 |
| E249 | `24285bd44715` | 1/3 | 0/3 | 3/3 | historical v1 clear does not clear v2 |
| E252 | `c01aebc28d8f` | 1/3 | 0/3 | 3/3 | historical v1 clear does not clear v2 |
| E262 | `3f6a2eb2a6b3` | 1/3 | 0/3 | 3/3 | historical v1 clear does not clear v2 |
| E264 | `518d4736571d` | 1/3 | 0/3 | 3/3 | historical v1 clear does not clear v2 |
| E265 | `44079a8ca0f5` | 1/3 | 0/3 | 3/3 | historical v1 clear does not clear v2 |
| E293 | `78b70c81bd16` | 0/3 | 0/3 | 3/3 | unchanged negative |
| E294 | `df30ca03f8f2` | 0/3 | 0/3 | 3/3 | unchanged negative |
| E295 | `5b4c50467454` | 0/3 | 0/3 | 3/3 | unchanged negative |
| E296 | `b3c4df4cca25` | 0/3 | 0/3 | 3/3 | unchanged negative |

Aggregate v1/v2 confusion: 26 true-negative, 7 v1-positive/v2-negative,
0 v2-positive, and 0 joint-positive (`n=33`, agreement 0.7879, Cohen's
kappa 0.0). Treating v1 as the historical reference, v2 recall is 0.0 and
precision is undefined. This is not human-labeled calibration and does not
prove those seven v1 positives were false positives.

The most common v2 reasons were missing required components (32/33),
prompt-component mismatch (19/33), missing required placeholders (15/33), and
no nontrivial content (13/33). Four late cases also showed placeholder spam and
invalid `Stack` value roles. Per-case reports retain typed checks, prompt
provenance, inventories, and AST/path evidence.

No provenance-bearing per-case independent-judge, human, AgentV semantic, or
EFS0-04 labels existed. All four comparison matrices are explicitly `UNKNOWN`
with `n=0`.

## Named availability and gaming audit

E230 retained no per-case prediction envelope; E240–E247 and X22 retained only
aggregate results; E288-fixed's documented output path was absent; E292 had a
legacy 500-character prediction without a digest. These remain explicit
`UNKNOWN`, never inferred.

The deterministic gaming suite matched all 16 expected outcomes: 11 attacks
were rejected and five valid variants were preserved. Coverage was 16/16.
That fixture is wiring and attack-detection evidence only, never a model ship
claim.

## Verdict

The replay plumbing and v2 diagnostic pass their stated audit. Seven historical
v1 smoke positives do not clear v2, but v2 remains uncalibrated and is not the
ship primary. `meaningful_program_v1` remains active, its thresholds are
unchanged, and no v1 threshold was copied to v2. Selecting v2 requires the
follow-on labeled calibration and a separately versioned threshold decision.
