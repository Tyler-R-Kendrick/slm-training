# Checkpoint bucket (Hugging Face)

Durable storage for **real full training runs** (HF-context ship track).

| Field | Value |
| --- | --- |
| Bucket | [TKendrick/OpenUI](https://huggingface.co/buckets/TKendrick/OpenUI) |
| URI | `hf://buckets/TKendrick/OpenUI` |
| Layout | `checkpoints/<run_id>/…` |

Autoresearch evidence bundles use a separate prefix,
`autoresearch/<campaign_id>/`. They contain campaign specs, source/evidence
snapshots, decisions, outcomes, telemetry, and checksums—not serving checkpoints.
The checkpoint and model-card rules below still apply to any full training run
launched by a campaign.

## What gets synced

From `outputs/runs/<run_id>/checkpoints/`:

- `last.pt` + `.tokenizer.json` + `.meta.json` (+ optional `.context.tokenizer.json`)
- `best_ship_score.pt` / `best_weighted_nll.pt` (+ sidecars) when present
- `last_full_state.pt` (resume payload) when present
- `promoted.pt` / `promoted.json` when present
- `train_summary.json` (copied from the run dir)

## When sync runs

Auto-enabled when `ModelBuildConfig.context_backend == "hf"` and sync is not
disabled. Scratch / quality-matrix CPU demos stay local-only.

| Entry | Default |
| --- | --- |
| `scripts.train_model` (default `--context-backend hf`) | sync on (CLI sets `sync_checkpoints=True`) |
| `scripts.hf_jobs_train` | sync on (Jobs entrypoint `--sync-checkpoints`) |
| `scripts.remote_train` | sync on (`--sync-checkpoints`) |
| Programmatic `ModelBuildConfig` / pytest | sync off |
| `scripts.run_quality_matrix` (default scratch) | sync off |
| CI / fixture demos | local-only (no CLI sync flags) |

Disable: `--no-sync-checkpoints`, `SLM_DISABLE_CHECKPOINT_BUCKET=1`, or
`checkpoint_bucket=""`.

## Auth

Write access requires a Hub token with bucket permissions:

```bash
export HF_TOKEN=hf_...   # or: hf auth login
```

Also accepted: `HUGGING_FACE_HUB_TOKEN`, `SLM_CHECKPOINT_BUCKET` (override URI).

## Commands

```bash
# Full HF train (auto-sync at end)
python -m scripts.train_model \
  --train-dir outputs/data/train/v1 \
  --run-id twotower_v1 \
  --context-backend hf \
  --steps 200

# Manual / recover a local run
python -m scripts.sync_checkpoints \
  --run-dir outputs/runs/twotower_v1 \
  --ensure-bucket

# Plan only
python -m scripts.sync_checkpoints --run-dir outputs/runs/twotower_v1 --dry-run
```

Artifacts: `outputs/runs/<id>/checkpoint_bucket.json` plus
`train_summary.json` → `checkpoint_bucket` field.

## Checkpoint references (fail-closed provenance)

Every sync hashes each artifact before upload, writes a canonical
`CheckpointReferenceV1` sidecar (`<checkpoint>.ref.json`) plus an aggregate
`checkpoint_references.json` manifest into the uploaded set, and — for a real
sync — re-verifies that the files landed remotely before stamping the reference
`verified`. A verification mismatch raises (the train fails closed); a dry run is
never persistence evidence. Frontier / ship-candidate references must be fully
provenanced and verified to be publishable. Declare the class with
`--claim-class` (default `diagnostic`). Full contract:
[checkpoint-provenance.md](checkpoint-provenance.md); CI audit
`python -m scripts.verify_checkpoint_references --check`.

## Model card (required)

After every successful sync (or fixture bootstrap that writes a checkpoint),
update:

1. [`docs/MODEL_CARD.md`](../MODEL_CARD.md) — roster, eval, history, URI
2. [`README.md`](../../README.md) — “Model card (summary)” table only

Agent process: [`AGENTS.md`](../../AGENTS.md) + skill
`documenting-experiment-results`.

## Measured results

| Date (UTC) | Run | Sync? | Notes |
| --- | --- | --- | --- |
| 2026-07-20 | `e574-e569-slotloss2-r1-48s` | No (`--no-sync-checkpoints`) | 76.23s under `max_wall_minutes=3`, SHA `649cf512ec0f77bfd6d06230d80c06ed14c8cde5425751e183544e073810b7c2`; aggregates exactly match E573 and strict meaning/AgentV fail. |
| 2026-07-20 | `e573-e569-fidelity1-r1-48s` | No (`--no-sync-checkpoints`) | 109.72s under `max_wall_minutes=3`, SHA `ff21fc0c5f1ec8dd4fbb8857f77779dfbe1e663d7a6a161b824f49edcf59070d`; fidelity/reward improve while meaning-v1 holds, but strict meaning/AgentV fail. |
| 2026-07-20 | `e572-e569-fidelity2-r1-48s` | No (`--no-sync-checkpoints`) | 84.26s under `max_wall_minutes=3`, SHA `bb6a58ff4400de90da68c1596ee3ed5b5d64ff1f341dbcd1ac2b4e56cc29efa2`; fidelity/reward improve but semantic coverage regresses and strict meaning/AgentV fail. |
| 2026-07-20 | `e569-e561-matched-cont48-r1-48s` | No (`--no-sync-checkpoints`) | 75.20s under `max_wall_minutes=3`, SHA `8254fcf754591022806a1a87435a5be9a4a4d3706d61754ba59b4a19c6535f73`; matched continuation improves meaning-v1/recall/reward but strict meaning and AgentV fail. |
| 2026-07-20 | `e568-e561-cont48-r1-48s` | No (`--no-sync-checkpoints`) | 116.24s under `max_wall_minutes=3`, SHA `8dcc080449ea945a731c2b206c5a02e7913f3f61b6fd858e555d9d7e0283a12b`; design-context scratch continuation improves reward but regresses fidelity/topology and fails meaning/AgentV. |
| 2026-07-20 | `e561-e544-owner-threshold7-r1-24s` | No (`--no-sync-checkpoints`) | 41.47s under `max_wall_minutes=3`, SHA `35a4fe6dd1b0eb2f59c33cb6d4ae11472c693f43a15fa3e6abc46db323a127f9`; broad non-semantic gains, but meaning and AgentV fail. |
| 2026-07-20 | `e560-e544-owner-threshold4-r1-24s` | No (`--no-sync-checkpoints`) | 42.26s under `max_wall_minutes=3`, SHA `dae11cee1e8fc1a2178b6397e558db9b0e4a723bbfbaf38f63af94557d7686a3`; topology improves but fidelity and semantic gates fail. |
| 2026-07-20 | `e559-e544-owner-coverage2-r1-24s` | No (`--no-sync-checkpoints`) | 31.14s under `max_wall_minutes=3`, SHA `1d11926d6784cac62f6d65249030be9b392d0f185e59b1a4212d9f0ff9aac861`; fidelity/recall improve but reward and semantic gates fail. |
| 2026-07-20 | `e558-e544-owner-coverage-r2-24s` | No (`--no-sync-checkpoints`) | 43.74s under `max_wall_minutes=3`, SHA `a45909dffd103df353bff944aedbafa1e386b2bf657c5dc02f2d956e06381ede`; fidelity improves but structure/reward and semantic gates fail. |
| 2026-07-20 | `e558-e544-owner-coverage-r1-24s` | No (`--no-sync-checkpoints`) | 43.31s under `max_wall_minutes=3`, SHA `8a572738407301a9b27aeba52f88942c5c86c101fee8638034108cd01de85382`; dirty-tree engineering trial excluded from decisions. |
| 2026-07-20 | `e557-e544-slot-pair-balance1-r1-24s` | No (`--no-sync-checkpoints`) | 70.09s under `max_wall_minutes=3`, SHA `438d9871bc8389f6d61d4f3e357d56d280a22aafa26962404f6c47c92b97db05`; metric-identical to E555 and semantic gates fail. |
| 2026-07-20 | `e556-e544-slot-context-combined-r1-24s` | No (`--no-sync-checkpoints`) | 68.42s under `max_wall_minutes=3`, SHA `139c670c7e1d087101111720fbb458f2a0ad1b3284e9d57fa3eff4fa95831f0a`; combined treatment regresses fidelity/reward and fails semantic gates. |
| 2026-07-20 | `e555-e544-slot-pair-interaction-r2-24s` | No (`--no-sync-checkpoints`) | Explicit 24-step local-only scratch diagnostic; 50.29s under `max_wall_minutes=3`, serving SHA `af53e1619e9749eab78379ae7696a929e7409dbd984a5e33481cfa050addf19e`. Pareto topology gain, but semantic gates and AgentV fail; no promotion. |
| 2026-07-20 | `e554-e544-slot-next-context-r2-24s` | No (`--no-sync-checkpoints`) | Explicit 24-step local-only scratch diagnostic; 39.91s under `max_wall_minutes=3`, serving SHA `af3cbce7ca8c2adfbccc8d5ad0550361e2c30f56a6da04f6390615d40c67b579`. Mixed topology gains and fidelity/reward regressions; semantic gates and AgentV fail, so no promotion. |
| 2026-07-20 | `e553-e544-prior-proportional-r3-24s` | No (`--no-sync-checkpoints`) | Explicit 24-step local-only scratch diagnostic; 34.48s under `max_wall_minutes=3`, serving SHA `510e55cf16fe23edd4ac408ed37d2409a895143646a6321c5b491c148e75399d`. Corpus-local proportional priors improve fidelity slightly but sharply regress structure and recall; semantic gates and AgentV fail, so no promotion. |
| 2026-07-19 | `e544-e543-root-identity1-r2-24s` | No (`--no-sync-checkpoints`) | Explicit 24-step local-only scratch diagnostic, not a full train or promoted checkpoint. It completed in 40.96s under `max_wall_minutes=3`, wrote serving SHA `3b6e3c00666b8832187a489d6684ce909fff5b3ccaef57965f9cc1975474f20c`, and is registered in the model card. Rank-only identity decoding improves the matched OOD topology diagnostic, but strict meaning and AST edge F1 remain 0.0 and AgentV is 0/1, so no bucket persistence or promotion was warranted. |
| 2026-07-19 | `e545-e544-root-identity-neg1-control-r1-24s` | No (`--no-sync-checkpoints`) | Explicit matched 24-step local-only scratch control, not a full train or promoted checkpoint. It completed in 30.64s under `max_wall_minutes=3` and wrote serving SHA `9e54d4700938c2e1feececfa3b952d4188c76873281e54d38f19bcea4cc76fa1`. OOD meaningful is 0.0 and AgentV is 0/1; no bucket persistence warranted. |
| 2026-07-19 | `e545-e544-root-identity-neg4-r2-24s` | No (`--no-sync-checkpoints`) | Explicit matched 24-step local-only scratch treatment, not a full train or promoted checkpoint. It completed in 28.64s under `max_wall_minutes=3` and wrote serving SHA `14dd44043887cfb6b5a14b1a99fee3750dc8f72c2d27f205fe3bdc0506de61ae`. Its predictions and OOD metrics exactly match the weight-1 control and both regress from E544; no bucket persistence warranted. |
| 2026-07-19 | `e546-e544-strict-subset1-control-r1-24s` | No (`--no-sync-checkpoints`) | Explicit matched 24-step local-only scratch control; 29.10s under `max_wall_minutes=3`, serving SHA `46aba9048624f766e6052d202a94b689440baca9f1ab94d8d6c8d48adc40fc55`. Meaningful 0.0 and AgentV 0/1; no promotion. |
| 2026-07-19 | `e546-e544-strict-subset5-r2-24s` | No (`--no-sync-checkpoints`) | Explicit matched 24-step local-only scratch treatment; 30.50s under `max_wall_minutes=3`, serving SHA `a1a6bfc94108a8bba9aac18e5570d70e317cdec5bb706f126bf47e67e2b4efe2`. Mixed topology/fidelity gain but recall regression, meaningful 0.0 and AgentV 0/1; no promotion. |
| 2026-07-19 | `e547-e544-strict-subset2-r1-24s` | No (`--no-sync-checkpoints`) | Explicit 24-step local-only scratch diagnostic; 36.48s under `max_wall_minutes=3`, serving SHA `37002bfd3c63d1ac58f5fc505bf034805b57eee2415d9e15ec1acbb81620fc57`. Multiplier 2 leads the ladder on structure/AST node F1 but fidelity and semantic gates fail; no promotion. |
| 2026-07-19 | `e551-e544-strict-subset2-no-lexeme-r1-24s` | No (`--no-sync-checkpoints`) | Explicit 24-step local-only scratch diagnostic; 41.85s under `max_wall_minutes=3`, serving SHA `e7921e66df8d2c76b96c1577c7cfb3b35c97879d69f44da0c41ab787dac32fc6`. Prior removal improves fidelity but regresses topology and recall; semantic gates and AgentV fail, so no promotion. |
| 2026-07-19 | `e552-e544-strict-subset2-lexeme05-r1-24s` | No (`--no-sync-checkpoints`) | Explicit 24-step local-only scratch diagnostic; 34.75s under `max_wall_minutes=3`, serving SHA `49a9c1119d28f95437f86cfca5f8c06467173d56d1e060757cea8af0a151fc04`. Half-strength prior regresses fidelity, recall, and reward; semantic gates and AgentV fail, so no promotion. |
| 2026-07-19 | `e543-e531-root-reference-bounded-r1-24s` | No (`--no-sync-checkpoints`) | Explicit 24-step local-only scratch diagnostic, not a full train or promoted checkpoint. It completed in 37.17s under `max_wall_minutes=3`, wrote serving SHA `c6be3791544def59ad26b8d2b3b605a7efefd93ec83c996371e593a3251d7f90`, and is registered in the model card. Bounded training improves auxiliary calibration, but OOD `n=4` decisions and quality exactly match E542, strict meaning is 0.0, and AgentV is 0/1, so no bucket persistence or promotion was warranted. |
| 2026-07-19 | `e542-e531-root-reference-arity1-r1-24s` | No (`--no-sync-checkpoints`) | Explicit 24-step local-only scratch diagnostic, not a full train or promoted checkpoint. It completed in 52.93s under `max_wall_minutes=3`, wrote serving SHA `2d5cd4b3c8c721e8193e06b5aa231bd9ec5009b4bec9cacfeebe842f6854c5d8`, and is registered in the model card. OOD `n=4` learned weight 1 exactly matches control, strict meaning is 0.0, and AgentV is 0/1, so no bucket persistence or promotion was warranted. |
| 2026-07-20 | `e616-object-property-slot-bias-scratch80-20260720` | No (`--no-sync-checkpoints`) | Fresh CPU scratch **TwoTower** loop from the current checkout, replaying E615 with 80 steps instead of 8; ran on published E530 (244 records) in 12.78s under `max_wall_minutes=3`; loss 26.5243, 1,643,522 trainable params, checkpoint SHA `119dd41a57573fe408ee1208a9159cafca9a45c7050bd00274571938898a8539`, full-state SHA `01605f0e7ceaa35af4dd20ee52eb196955e846c2a5d4913f461ee184ab549466`. The matched OOD `n=4` control (`schema_role_slot_decode_weight=0`) vs treatment (`=8.0`) eval pair (`e616-control-eval-r1` / `e616-treatment-eval-r1`) now produces 4/4 syntactically valid, non-empty predictions (vs E615's 0/4), but remains byte-identical: the Gallery record decodes to `ImageGallery([])` in both arms, so the E615 object-frame lever's precondition (an opened typed-object item) is never reached — reproducing E612's already-rejected empty-array-close finding one step upstream. JSON: [iter-e616-object-frame-slot-bias-scratch80-replay-20260720.json](iter-e616-object-frame-slot-bias-scratch80-replay-20260720.json). Not promotable or ship. |
| 2026-07-20 | `e620-required-slot-coverage-scratch800-20260720` | No (`--no-sync-checkpoints`) | Fresh 800-step CPU scratch **TwoTower** duration diagnostic on published E530; 80.96s under `max_wall_minutes=3`, loss 4.0680, serving SHA `3ce5c9efc70ed69c7a6680129018ec0aa2061f56020ff42301faf144363ecc5f`, full-state SHA `2abcb2f83149e8a60edc44128317ec9ed56ee107bb92004bdeb5188849332659`. Matched OOD treatment improves over its control but strict-v2 remains 0.0 and regresses E619 fidelity/structure despite lower train loss; local checkpoint rejected, not promotable or ship. JSON: [iter-e620-required-slot-coverage-scratch800-20260720.json](iter-e620-required-slot-coverage-scratch800-20260720.json). |
| 2026-07-20 | `e621-margin-sweep-seed0-scratch800-20260720` | No (local-only) | 800-step CPU scratch **TwoTower**, seed 0; 28.65s under `max_wall_minutes=3`, loss 4.0680 (reproduces E620's seed-0 loss to 4 decimals), serving SHA `8ab4f5deeb0e1322064259ca92837e71dfdf662d25b46371172ad8779668a4cd`. Used for E621's margin-sweep + instrumentation trace; not promotable or ship. |
| 2026-07-20 | `e621-margin-sweep-seed1-scratch800-20260720` | No (local-only) | 800-step CPU scratch **TwoTower**, seed 1; 28.89s under `max_wall_minutes=3`, loss 4.5998, serving SHA `86573aa519e86b59f6b64a84ef87899da30cda8e90c9b4ca3698837dc2b7b866`. Used for E621's margin sweep; not promotable or ship. |
| 2026-07-20 | `e621-margin-sweep-seed2-scratch800-20260720` | No (local-only) | 800-step CPU scratch **TwoTower**, seed 2; 31.27s under `max_wall_minutes=3`, loss 3.4020, serving SHA `49a8a64ebca4a549789b8c3df648aeaea11f897425bd73e5ce456d0bf73ebb99`. Used for E621's margin sweep; not promotable or ship. |
| 2026-07-20 | `e548_training_loop_twotower_scratch_20260720` | No (`--no-sync-checkpoints`) | Fresh CPU scratch **TwoTower** loop from the current checkout with CPU-only Torch; ran 8 steps on published E530 (244 records) in 2.236s under `max_wall_minutes=3`; loss 39.4267, 68,514 trainable params, 2,038 prompt tokens and 561 target tokens seen, serving checkpoint SHA `6581e32f34cb261b576c6b36539c6859e6eed75987212adf7e2ad59181386936`, full-state SHA `3752626c53af42de21d9bc4c6f7bb7b5662dd43c19568753adf411aba64fe7ff`. This is local training-loop continuity evidence only; no eval was run, no bucket sync was attempted, and it is **not promotable or ship**. JSON: [e548-training-loop-twotower-scratch-20260720.json](e548-training-loop-twotower-scratch-20260720.json). |
| 2026-07-20 | `e547_training_loop_twotower_scratch_20260720` | No (`--no-sync-checkpoints`) | Fresh CPU scratch **TwoTower** loop from the current checkout with CPU-only Torch; ran 7 steps on published E530 (244 records) in 2.056s under `max_wall_minutes=3`; loss 35.7431, 68,514 trainable params, 1,782 prompt tokens and 467 target tokens seen, serving checkpoint SHA `c67db09f248a0c80984e5d2d117bff2306292b7b311101f4ca20e8db6d17627f`, full-state SHA `d1a3e4f82025a1fe184e53f9d0ee98120d0a029251e5369965100607e0e05241`. This is local training-loop continuity evidence only; no eval was run, no bucket sync was attempted, and it is **not promotable or ship**. JSON: [e547-training-loop-twotower-scratch-20260720.json](e547-training-loop-twotower-scratch-20260720.json). |
| 2026-07-20 | `e546_training_loop_twotower_scratch_20260720` | No (`--no-sync-checkpoints`) | Fresh CPU scratch **TwoTower** loop from the current checkout; ran 6 steps on published E530 (244 records) in 5.042s under `max_wall_minutes=3`; loss 40.3390, 68,514 trainable params, 1,536 prompt tokens and 398 target tokens seen, serving checkpoint SHA `dfc0272ca90171cf58116a707a79d2d4c9c413972b00641a5176ddac351cf9c5`, full-state SHA `a7b0a100eae13d6abe3ff32a53b946b213ee74c44d12bbe06961a4fe27f0f6e0`. This is local training-loop continuity evidence only; no eval was run, no bucket sync was attempted, and it is **not promotable or ship**. JSON: [e546-training-loop-twotower-scratch-20260720.json](e546-training-loop-twotower-scratch-20260720.json). |
| 2026-07-19 | `e545_training_loop_twotower_scratch_20260719` | No (`--no-sync-checkpoints`) | Fresh CPU scratch **TwoTower** loop from the current checkout; ran 5 steps on published E530 (244 records) in 2.509s under `max_wall_minutes=3`; loss 42.1226, 68,514 trainable params, 1,280 prompt tokens and 346 target tokens seen, serving checkpoint SHA `3d08f592832e03fb5cb27602492e318bc34a07d2d2ed616648c2619f1823e1fe`, full-state SHA `c9ac70ce19730f636793cafbd91c61c077de911264151adb3ca89cbd1378203b`. This is local training-loop continuity evidence only; no eval was run, no bucket sync was attempted, and it is **not promotable or ship**. JSON: [e545-training-loop-twotower-scratch-20260719.json](e545-training-loop-twotower-scratch-20260719.json). |
| 2026-07-19 | `e544_training_loop_twotower_scratch_20260719` | No (`--no-sync-checkpoints`) | Fresh CPU scratch **TwoTower** loop restarted from the current checkout because prior gitignored `outputs/runs/e543...` checkpoint was absent; ran 4 steps on published E530 (244 records) in 1.540s under `max_wall_minutes=3`; loss 42.3848, 68,514 trainable params, 1,024 prompt tokens and 296 target tokens seen, serving checkpoint SHA `8531f7d79ad9e50737de99a7a251697f011c5619c96a22b1d8019e3bc2eb8036`, full-state SHA `5ce86d03c3cbe268729d31b2d883812922a7b5f7a373a8b73558497d20492599`. This is local training-loop continuity evidence only; no eval was run, no bucket sync was attempted, and it is **not promotable or ship**. JSON: [e544-training-loop-twotower-scratch-20260719.json](e544-training-loop-twotower-scratch-20260719.json). |
| 2026-07-19 | `e543_training_loop_twotower_resume_scratch_20260719` | No (`--no-sync-checkpoints`) | Resumed CPU scratch **TwoTower** loop from E542 full-state and advanced to step 3 on published E530 (244 records) in 2.597s under `max_wall_minutes=3`; loss 39.7476, 68,514 trainable params, 768 prompt tokens and 199 target tokens seen, serving checkpoint SHA `6219feed267c22fbfa898fb7b89a69192c00bceece1b04b4de81c334130fd1da`, full-state SHA `1e35b47613731beef9176a0db245bca1b678bc0975777704cc4a53c08cdbb320`. This is local resume-loop continuity evidence only; no eval was run, no bucket sync was attempted, and it is **not promotable or ship**. JSON: [e543-training-loop-twotower-resume-scratch-20260719.json](e543-training-loop-twotower-resume-scratch-20260719.json). |
| 2026-07-19 | `e542_training_loop_twotower_resume_scratch_20260719` | No (`--no-sync-checkpoints`) | Resumed CPU scratch **TwoTower** training-loop iteration from E541 full-state and advanced to step 2 on published E530 (244 records) in 2.816s under `max_wall_minutes=3`; loss 43.6742, 68,514 trainable params, 512 prompt tokens and 120 target tokens seen, serving checkpoint SHA `682ab617366480adbcddfd3def55e49292fb591b2d77865fc7c4b57677347c42`, full-state SHA `70081ba399cdda7e8d78a850dcef598607777be7edba5be553ca2e63af19033a`. This verifies local resume-loop continuity only; no eval was run, no bucket sync was attempted, and it is **not promotable or ship**. JSON: [e542-training-loop-twotower-resume-scratch-20260719.json](e542-training-loop-twotower-resume-scratch-20260719.json). |
| 2026-07-19 | `e541_training_loop_twotower_scratch_20260719` | No (`--no-sync-checkpoints`) | One-step CPU scratch **TwoTower** training-loop iteration on published E530 (244 records) completed in 2.595s under `max_wall_minutes=3`; loss 36.9158, 68,514 trainable params, 256 prompt tokens and 63 target tokens seen, checkpoint SHA `ffc8f2e3f01db37e789a0dae61aa37ce9e4bae3f80af94ece942d48cdb3ffbaa`. This is a local scratch wiring/training-continuity run only; no eval was run, no bucket sync was attempted, and it is **not promotable or ship**. JSON: [e541-training-loop-twotower-scratch-20260719.json](e541-training-loop-twotower-scratch-20260719.json). |
| 2026-07-19 | `e540_training_loop_scratch_20260719` | No (`--no-sync-checkpoints`) | One-step CPU scratch stub training-loop sentinel on published E530 (244 records) completed in 0.116s under `max_wall_minutes=3`; checkpoint SHA `46c1d82da2c8c29ce73800224810185de0bd6a2e02db70ad58acbb96589e6014`. This validates local loop wiring only; no eval was run, no bucket sync was attempted, and it is **not promotable or ship**. JSON: [e540-training-loop-scratch-20260719.json](e540-training-loop-scratch-20260719.json). |
| 2026-07-19 | `e531-e396-e530-replay050-slotrole1-honest-context-r1-5k` | Yes | The automatic bucket-create preflight was rejected by the CLI OAuth session, then canonical direct rescue sync to the existing bucket reconciled `train_summary.json`; resync verification and independent nine-file listing passed at `hf://buckets/TKendrick/OpenUI/checkpoints/e531-e396-e530-replay050-slotrole1-honest-context-r1-5k`. Serving checkpoint SHA `6b8c1abc56a36e8aa15acc373b61d5df033a753907330649e379d9ba374a6154`, full-state SHA `3fd97277e1c10fe00cb178158bbcb232fff2a301a7b01ef0dda7b88c453cbcee`. The run completed 99 CPU HF-context steps / 5,059 target tokens in 99.72s under `max_wall_minutes=3`. E532 weakly improves structure but regresses meaning, fidelity, recall, reward, and AgentV; diagnostic checkpoint not promoted. |
| 2026-07-19 | `e528-e396-e527-replay050-slotrole1-honest-context-r1-5k` | Yes | Automatic sync, resync verification, and independent nine-file bucket listing passed at `hf://buckets/TKendrick/OpenUI/checkpoints/e528-e396-e527-replay050-slotrole1-honest-context-r1-5k`; serving checkpoint SHA `6a2180d76c366a282a74d1d27ae2b2fcf4c1b5f2b4d298cf4cef35bc306976d5`, full-state SHA `4a70677f0630a1319003ba839b9b010cbb56e4e220bde2539ada038a13385aa7`. The run completed 99 CPU HF-context steps / 5,059 target tokens in 146.8s under `max_wall_minutes=3`. E529 recovers v1 meaningful rate and reward but regresses hierarchy and fails strict meaning and AgentV; diagnostic checkpoint not promoted. |
| 2026-07-19 | `e525-e396-e524-replay050-slotrole1-honest-context-r2-5k` | Yes | Canonical rescue sync, persisted sync report, resync verification, and independent bucket listing passed at `hf://buckets/TKendrick/OpenUI/checkpoints/e525-e396-e524-replay050-slotrole1-honest-context-r2-5k`; serving checkpoint SHA `dbd11811d826fdf7efd8b22557fb3bd48f879e84ec7484bc0a2680198e55e4b9`, full-state SHA `dd588c80998315e6edf694610ae49f3e9e7115e3125f4a3175125d3aeea9ad63`. The run completed 99 CPU HF-context steps / 5,059 target tokens in 76.7s under `max_wall_minutes=3`. E526 improves recall but regresses fidelity and hierarchy and fails AgentV; diagnostic checkpoint not promoted. |
| 2026-07-19 | `e522-e396-e521-replay050-slotrole1-honest-context-r2-5k` | Yes | Automatic sync, resync verification, and independent bucket listing passed at `hf://buckets/TKendrick/OpenUI/checkpoints/e522-e396-e521-replay050-slotrole1-honest-context-r2-5k`; serving checkpoint SHA `97cb10f43d229b1a15403295f71fa425e844ee4865c31761f3e529b24bf420ce`, full-state SHA `e0eafca2b2ad6c795ae2eaa6860770543cc4ac0b9508a1da993cfab5b5d3b0e8`. The run completed 99 CPU HF-context steps / 5,059 target tokens in 120.7s under `max_wall_minutes=3`. E523 improves fidelity/recall but regresses structure/reward and fails AgentV; diagnostic checkpoint not promoted. |
| 2026-07-19 | `e519-e396-e500-replay050-slotrole1-honest-context-r1-5k` | Yes | Automatic sync and resync verification passed at `hf://buckets/TKendrick/OpenUI/checkpoints/e519-e396-e500-replay050-slotrole1-honest-context-r1-5k`; serving checkpoint SHA `d82155b03531c2d852ec8d497d3fdb0878ac1f678c0c5d247e272bc36c91805f`, full-state SHA `a4ed4300cd43fdf1c8767988f4861630fa69a9347600c1448bf9cc2574fb51f7`. The clean-source harness-v7 run completed 101 CPU HF-context steps / 5,000 target tokens in 103.2s under `max_wall_minutes=3`. E520 exactly matches E518 quality and fails AgentV; honest authority retained, diagnostic checkpoint not promoted. |
| 2026-07-19 | `e517-e396-e500-replay050-slotrole1-context-r1-5k` | Yes | Automatic sync and resync verification passed at `hf://buckets/TKendrick/OpenUI/checkpoints/e517-e396-e500-replay050-slotrole1-context-r1-5k`; serving checkpoint SHA `2b572a04256db14095e813e146079af9e6f6c948963d60f2bd669855e24b60e3`, full-state SHA `37eab2a1641f17860f6bad4271f7f8777d5b13776f39944808c24b5c977d24dd`. The run completed 101 CPU HF-context steps / 5,000 target tokens in 130.7s under `max_wall_minutes=3`. E518 regresses every headline metric versus E515 and fails AgentV; durable diagnostic only. |
| 2026-07-19 | `e515-e396-e500-replay050-slotrole4-focal0-r1-5k` | Yes | Automatic sync and resync verification passed at `hf://buckets/TKendrick/OpenUI/checkpoints/e515-e396-e500-replay050-slotrole4-focal0-r1-5k`; serving checkpoint SHA `97f2e426604e3956f2791398a608b967937ebf548fa7cae0ef59dde324721c1b`, full-state SHA `4f9818f87735df24b3fc21d9e06925d51ca3f2d8fef2fb4fa6737930778c2a06`. The run completed 101 CPU HF-context steps / 5,000 target tokens in 105.8s under `max_wall_minutes=3`. E516 recovers from E513 when focal gamma returns to zero but still fails strict meaning and AgentV; durable diagnostic only. |
| 2026-07-19 | `e513-e396-e500-replay050-slotrole4-focal2-r3-5k` | Yes | Automatic sync and resync verification passed at `hf://buckets/TKendrick/OpenUI/checkpoints/e513-e396-e500-replay050-slotrole4-focal2-r3-5k`; serving checkpoint SHA `59253c679477060694370c5e2d8cd9fce5d7accc7d71df3b6d56edf0a88a9548`, full-state SHA `98b6d71321add3962faa2d717a5963f17d53719f166af17ee0c6120ed7fe5133`. The run completed 101 CPU HF-context steps / 5,000 target tokens in 79.6s under `max_wall_minutes=3`. E514 OOD gates and AgentV fail, so this is durable diagnostic evidence, not a promotion or ship checkpoint. |
| 2026-07-19 | E357 training-data snapshot for E504 replay | Yes (data only) | Exact eight-file, 998-row corpus persisted at `hf://buckets/TKendrick/OpenUI/data/train/e357_card_hierarchy_v1/`. Post-upload sync found all eight files identical; independent download verified semantic manifest SHA `a4f212a3444d0f219fe1b3604f70929fe1a1b91d4fdc11a73167cb74c55b6a51` and records SHA `b1b2c3d0c1965bd9829edfc6ae34b5dce916a68c33bb17497a6392c80d7ea6ef`. E504's five rejected checkpoints were explicitly not synced. |
| 2026-07-18 | `e396-balanced-type-head-continuation-r1` | Yes | Manual recovery sync verified at `hf://buckets/TKendrick/OpenUI/checkpoints/e396-balanced-type-head-continuation-r1`; checkpoint SHA `feefa0564490bd1db42f79ff710143ad8ed07ab9e4e324f2744a30f8c2f2eee0`. E498 restores current-main loading and learned-head application, but smoke semantic gates and AgentV remain red. Persistence and compatibility are verified; this is not a champion or serving promotion. |
| 2026-07-14 | `restructure_cpu_scratch_v0` | No (`--no-sync-checkpoints`) | Post-package-restructure CPU fixture/scratch train on a 4c/15GB host with **no** `HF_TOKEN`. Validates harness wiring after the dsl/harnesses/runtime move. Smoke parse **0.0** @ 80 steps — not a ship claim. JSON: [restructure-cpu-train-results.json](restructure-cpu-train-results.json). |
| 2026-07-14 | `restructure_cpu_scratch_v0_cont` | No | Resume from v0 full-state; +200 CPU scratch steps. Smoke parse still 0.0. HF Jobs blocked: no Cloud Agent HF_TOKEN. JSON: [restructure-cpu-train-cont-results.json](restructure-cpu-train-cont-results.json). |
| 2026-07-14 | `local_directml_adreno_20260714` | No (`--no-sync-checkpoints`) | Five-step local scratch train on Qualcomm Adreno X1-85 through Torch-DirectML. Checkpoint and CPU reload verified; no eval/ship claim. JSON: [local-directml-train-results.json](local-directml-train-results.json). |
