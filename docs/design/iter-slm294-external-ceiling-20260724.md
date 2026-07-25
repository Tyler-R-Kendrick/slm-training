# SLM-294 / LAR1-01: External ceiling campaign — `inconclusive`

Matrix set: `slm294-external-ceiling` · Version: `efs1-01-v2-slm294` · Status: **complete** · Claim class: diagnostic
Decision: **inconclusive** (locked thresholds; all three arms incomplete under the run policy on this host)

## What ran

- **Preregistration locked before execution**:
  `docs/design/slm294-external-ceiling-preregistration.json` (arms, frozen
  requests, thresholds, cost caps) with 6 append-only amendments, all made
  **before any results were visible**.
- **Frozen requests**: `slm294_frozen_requests_v1.jsonl` — 35 leakage-filtered
  rico_held records (sha `8466b402…e2d50e`); evaluated prefix fixed at n=20
  (meets the LAR0 minimum n=20) per amendment.
- **Harness repaired** (SLM-108 line, no parallel evaluator): real frontier
  mode with sha-verified frozen requests, chunked atomic resume, unified
  score pass (official syntax + v1 typed reason report + v2 strict + Wilson
  95%), locked-threshold disposition, EvidenceBundleV1, AgentV emission.
- **Arm A (tiny baseline)**: training **completed within policy** — 600
  steps via chunked `--resume-from` under the 170s command cap, final loss
  1.5187 (checkpoint `outputs/runs/slm294_tiny_baseline`, diagnostic class).

## Why inconclusive — measured cost evidence

| Arm | Measurement | Cap | Outcome |
| --- | --- | --- | --- |
| B (SmolLM2-135M) | 296.3s for one uncontended fp32 request (256 tokens, load ~30s) | 170s/command | not_run |
| C (Qwen2.5-7B) | fp32 load OOMed a 31GB host; bf16 projected at multiple minutes/request | 170s/command | not_run |
| A decode | >280s for 16 tokens with the trained checkpoint (per-token lark lexer rebuild in `build_completion_forest`); fresh untrained model: 8.5s/64 tokens | 170s/command | trained, decode not_run |

Every output produced by an over-cap command was **discarded unscored** —
no timed-out or killed run counts as evidence. Reducing `max_new_tokens`
to fit the cap was rejected: truncating external outputs while arm A gets
full length would bias the scale comparison.

## Disposition

`inconclusive` — arms A/B/C incomplete under the locked evidence bar.
**Resolving evidence**: managed GPU host or HF Jobs (requires `HF_TOKEN`)
for the external arms; a completion-forest decode performance fix for the
tiny baseline on rico-scale layouts. LAR2–LAR4 remain blocked pending that
evidence. The harness, frozen requests, preregistration, and arm-A trained
checkpoint make the campaign directly re-runnable where the constraints clear.

## Artifacts

- `outputs/runs/slm294_external_ceiling/{scoreboard,disposition,evidence_bundle}.json`
- AgentV: `outputs/runs/slm294_external_ceiling/agentv/slm294-external-ceiling.eval.jsonl` (diagnostic claim)
- Tests: `tests/test_scripts/test_run_external_ceiling_frontier.py` (10)
- Registry: `harness.experiments.external_ceiling` v2; `harness.experiments` v77

## Honest caveats

- No semantic conclusions about scale vs task/metric binding are drawn; the
  experiment did not produce scored arm outputs.
- The SmolLM2 revision was pinned at download (`12fd25f7…`); Qwen revision
  was not pinned because no scoring run occurred.
- Arm A's checkpoint is a scratch diagnostic — not promoted, not a ship
  candidate, no bucket sync.
