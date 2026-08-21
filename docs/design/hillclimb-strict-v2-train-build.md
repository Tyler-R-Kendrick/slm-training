# hillclimb_strict_v2 train snapshot

Built 2026-08-21 from the frozen `hillclimb_strict_v1` records after closing
the train-data admission gap. Harness prompts are now round-tripped through
`parse_harness_task`, metadata is checked, and symbol-only targets are checked
at admission; violations are retained in `rejected.jsonl`.

Command:

```text
python -m scripts.build_train_data --source existing \
  --derive-from src/slm_training/resources/data/train/hillclimb_strict_v1/records.jsonl \
  --version hillclimb_strict_v2 --profile strict --publish --register-lineage
```

The published snapshot contains 676 admitted records. The two known v1
violations, `1dd2e550c00ebb87_scope` and `aa4c64a9ef0e996d_scope`, are visible
in `rejected.jsonl` with stage `harness_contract`, reason
`symbol_only_contract_violation`, and the non-canonical serialization error.
The published records round-trip cleanly. The producer that emitted the
unsorted `MARKER external_entity` lines remains a follow-up; admission now
fails closed instead of allowing it into training.
