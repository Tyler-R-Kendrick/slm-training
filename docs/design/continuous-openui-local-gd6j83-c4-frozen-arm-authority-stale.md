# Continuous autotrain: 2026-08-04 (session gd6j83) cycle 4 — frozen c2 arm is now permanently unreplayable (authority changed upstream, not a bug)

**Verdict:** infrastructure block, by design — not a harness defect, and not
repairable without weakening a decode-invariant safety check. No stack layer.

## What happened

`retry_measurement` on the identical frozen `c2-control` checkpoint
(`outputs/autoresearch/continuous-loop-20260804-continuous-openui-local-8c0b60dd-c2/runs/c20260804-continuous-openui-local-8c0b60dd-c2-control/checkpoints/last.pt`)
failed immediately at model load, before any decode:

```
ValueError: checkpoint completion artifact does not match the installed
grammar/tokenizer authority
```

raised by
[`require_checkpoint_completion_artifact`](../../src/slm_training/dsl/grammar/fastpath/completion_artifact.py)
in `TwoTowerModel.from_checkpoint`.

## Why this is not a bug

Between cycle 3 (retry) and this cycle 4 (retry-of-retry), this session ran
the loop's own mandatory "get latest" step and merged `origin/main`, which
had advanced with commit `807d4f8` ("HX4 hybrid unmask scheduler ... HX5
... certified LALR adapter ... duration-aware sharding") — a genuinely
unrelated, independently-landed PR. That merge touched
`src/slm_training/resources/decode/openui_completion_v1.manifest.json` and
`.safetensors` (the committed completion-artifact authority) among other
grammar/completion-kernel files. `require_checkpoint_completion_artifact`
exists specifically to reject exactly this situation: a checkpoint whose
embedded `completion_artifact` identity (grammar_sha256 +
tokenizer_authority_sha256, hash-bound at save time) no longer matches the
currently-installed authority. This is the same class of protection as I3/I6
(never silently reuse a stale grammar/decode authority) — the check is
working as designed, not a defect to patch around.

**This means the entire gd6j83-c2/c3 frozen-arm lineage is now permanently
unreplayable at any commit at or after `807d4f8`.** There is no correct
"repair" here: weakening or bypassing
`require_checkpoint_completion_artifact` to force this old checkpoint to
load would be exactly the kind of decode-invariant weakening this repo's
non-negotiable architecture invariants forbid. The only real path forward is
a fresh train under the current authority, not another replay of this
specific frozen manifest.

## Disposition (deviation from the driver's literal recommendation)

The driver's typed handoff again names a `repair_harness` action (frozen
manifest `ce373a5aea9425b95f35844e363d372bcc1cd45c52e9f14dc9ca213f9a77ae03`)
followed by another `retry_measurement`. This session acknowledges
`repair_harness` as **blocked** (not completed) with this doc as evidence:
there is nothing to repair, and forcing a fix would require weakening a
safety invariant, which is forbidden regardless of how many consecutive
cycles have hit a blocker. Per the loop law's self-heal guidance for
"harness-blocked champions after tip change," the correct recovery is to
treat this frozen-arm lineage as exhausted-by-upstream-tip-change and move
the loop's next cycle to a fresh screening hypothesis (a new train+eval,
not a replay of manifest `becbf08d…`/`6953396f…`/`ce373a5…`) rather than
retrying a fourth time.

## SDLC Phase A

**Non-positive** (`measurement_incomplete`, `harness_failure` — but disposed
as a designed fail-closed block, not an open defect). No stack layer.

## Next priorities

1. Start a fresh screening cycle (new train, current authority) rather than
   another retry of the gd6j83-c2/c3 frozen manifest lineage — that lineage
   is closed.
2. If a future session wants the branch-point-cost question
   ([`decode-compiler-tree-branch-point-cost-finding.md`](decode-compiler-tree-branch-point-cost-finding.md))
   answered against the *actual* seed-100002 checkpoint rather than a probe
   model, it must retrain a fresh checkpoint with the same recipe/seed under
   the current authority — the original checkpoint cannot be reused.

Machine evidence:
[`continuous-openui-local-gd6j83-c4-frozen-arm-authority-stale.json`](continuous-openui-local-gd6j83-c4-frozen-arm-authority-stale.json).
