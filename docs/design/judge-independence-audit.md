# Judge-independence audit

Status: **IN PROGRESS / evidence incomplete** for SLM-106 (EFS0-04).

This is the canonical execution contract for testing whether the deterministic
OpenUI quality family shares a blind spot with corpus admission, preference
mining, and evaluation. Hard compiler/schema checks remain authoritative; no
LLM verdict replaces them.

## Evidence contract

`JudgeEvidenceV1` lives in
`src/slm_training/evals/judge_independence.py`. It pins record/generation IDs,
prompt/output/checkpoint hashes, judge provider and model family/version,
rubric ID/version/hash, participation flags, score/verdict/acceptability,
reasons/confidence, retry/refusal/error/cost/latency, and blind ordering.

External evidence is independent only when every participation flag is false,
the rubric was not used for training admission, the judge family differs from
both candidate families, its provider/family is absent from the pinned sets of
prior admission/evaluation judges, and neither identity nor automatic judgments
was shown. `require_independent()` fails closed. AgentV publication validates this
evidence envelope only; it is explicitly not another semantic judge.

The pinned rubric is
`src/slm_training/resources/evals/judge_independence_rubric_v1.md`. A real run
must additionally pin the provider API version, exact model version,
temperature/seed support, retry count, token ceiling, and cost ceiling.

## Frozen human package

`python -m scripts.export_annotations freeze-audit` validates the campaign
sample before writing a redacted `blind_pairs.jsonl` and static `index.html`.
The private unblinding key must be written with `--private-key` outside the
redacted package. The gate requires 90–110 non-training holdout pairs, all seven
specified strata, five checkpoint families, and X22/TwoTower/choice coverage.
`--allow-incomplete-sample` is wiring-only and cannot support an audit claim.

Human imports accept opaque annotator IDs only. Every complete pair needs at
least two raters; disagreement needs exactly one adjudicator. Pair preference,
both acceptability labels, structured reasons, confidence, and duration are
retained without personal details or model identities.

## Intersection retrain

`freeze_matched_training_rows()` will not emit the original, random
size-matched, and intersection arms until external and human aggregate digests
exist. It requires three distinct seeds, identical target-token exposure, and
rejects audit holdouts or nested judge verdict/score/rationale fields. The
external admission judge is never the sole final metric: human holdout,
binding-aware meaningful v2, and AgentV remain separate outcomes.

## Current evidence and blockers

The X22 seed-0 recipe was locally reproduced to restore 19 raw predictions;
see `iter-efs0-04-x22-reproduction-20260717.md`. It is fixture-grade study
material, fails unchanged ship gates, and is not external or human evidence.

No independent external labels, blinded human labels, complete 100-pair source
set, intersection-admission corpus, or three-seed retrain exists yet. Provider
credentials and actual human participants are absent. Therefore agreement,
confidence intervals, admission trust, and the final EFS0-04 verdict remain
**UNKNOWN**, and SLM-106 must not be closed from the scaffolding alone.
