# Autotrain: c1822 compiler-decision-token fresh-seed confirmation (not reproduced)

**Honesty:** `fixture_screening`. **Not ship.**

[`autotrain-cycle-1822-compiler-decision-token-positive.md`](autotrain-cycle-1822-compiler-decision-token-positive.md)
queued the `compiler-decision-token` candidate (control vs. `compiler_decision_token_loss_weight=1.0`,
1,608,962 params, `wf_smoke_v2`, `steps=20`) for a fresh-seed confirmation before
any promotion or protected cadence could open. This cycle runs that
confirmation on `seed=202603` (c1822 used `seed=101821`), same fixture, same
recipe, against `main` HEAD `e1f5e4f` (already merged, no local patch).

```bash
python -m scripts.train_model --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
  --model twotower --context-backend scratch --steps 20 \
  --run-id autotrain_c1822_freshseed_<control|candidate> --no-sync-checkpoints --device cpu \
  --seed 202603 --compiler-decision-token-loss-weight <0.0|1.0>

python -m scripts.evaluate_model --run-id autotrain_c1822_freshseed_<control|candidate> \
  --train-version wf_smoke_v2 --model twotower --suite smoke --device cpu --seed 202603 \
  --run-class fixture_demo --decode-timeout-seconds 25
```

`--decode-timeout-seconds 25` (default `12.0`) was needed to get a complete
measurement on this container's CPU — at the default, all 3 documents on both
arms hit `decode_timeout_count=3` / `incomplete_document_n=3` (the same
recurring decode-timeout-vs-CPU-speed gap several prior sessions have hit;
see `autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md`). This is a
diagnostic override for `fixture_screening`, not a ship-gate change.

| Arm | Params | Structural | MPR | Component recall | Binder F1 | Fidelity | Reward | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | .2300 | 0 | 0 | 0 | 0 | 0 | 15429.78 |
| compiler-decision token | 1,608,962 | .2642 | 0 | .25 | 0 | 0 | 0 | 4924.95 |

**Result: does not reproduce c1822's win.** At `seed=101821` (c1822), the
candidate improved MPR by `.6667` and reward by `.004` over control. At
`seed=202603`, MPR is `0` for **both** arms — every document in both arms
fails the same check (`failure_breakdown: {"no_placeholders": 3}`,
`required_component_missing` / `prompt_relevant_semantic_content` FAIL) — and
`binder_reference_f1` / `placeholder_fidelity` / `reward_score` are `0` for
both arms too, unlike c1822's `1.0` / `1.0` / `~0.94`. The candidate still
leads on structural similarity (+.0342), component-type recall (+.25), and
p50 latency (-68.1%), but per this repo's positive-classification rule a
latency/efficiency-only win needs `mpr ≥ ~1/3` to count, and this seed's MPR
is `0` on both arms, not just below a third.

**Reading:** `compiler_decision_token_loss_weight=1.0` is not yet shown to
reliably move `meaningful_program_rate` — c1822's win could be seed noise on
an `n=3` smoke fixture (both single-seed measurements are honest and both
stand; they disagree). No promotion, no protected cadence, no Lean claim.

**SDLC Phase A: non-positive.** No stacked layer for this cycle; docs-only,
local commit. Per the `next_hypothesis` in the machine evidence: run 2+ more
seeds before treating either c1822 or this cycle as decisive, or move this
arm off the 3-document smoke fixture where a single seed's parse variance
dominates every rate metric.

Machine evidence:
[`autotrain-c1822-freshseed-confirmation-20260803.json`](autotrain-c1822-freshseed-confirmation-20260803.json).

Environment: fresh `.venv` (`python3.12 -m venv`, `pip install -e ".[dev]"`,
`torch==2.5.1+cu124`) plus `NODE_OPTIONS= npm ci` (a sandbox-global
`NODE_OPTIONS=--import tsx --max-old-space-size=8192` breaks any bare
`node`/`npm` invocation — same class of issue as
`scripts/publish_cap2_operator_policy_rebase.py`'s
`sanitized_node_env`/`bridge_utils` guard; every command here that shells to
node/AgentV was run with `NODE_OPTIONS=` cleared). Both created in this
scheduled session, not committed (`.venv/`, `node_modules/`, `outputs/` are
gitignored).
