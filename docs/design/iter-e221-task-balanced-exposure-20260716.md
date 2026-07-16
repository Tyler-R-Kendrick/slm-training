# E221 — task-balanced exposure diagnostic

Status: **training and strict five-suite evaluation completed; exposure hypothesis
falsified; ship gates failed; no checkpoint promoted**.

E219 drew 128 examples from the corrected 480-record E218 corpus but exposed
only 29.90 effective records. E221 tests whether the committed task-balanced
mixture reduces that concentration under the otherwise matched E219 recipe.
A five-candidate autoresearch matrix selected the canonical task-balanced policy.

Three preflight failures occurred before model training:

1. The first campaign froze its allowlist before diagnostic checkpoint-policy
   knobs were added, so matrix validation rejected the new knobs.
2. The second campaign compiled bare `python`, unavailable in this environment.
   The shared compiler now uses the active interpreter and persists process-launch
   failures as typed outcomes.
3. The third campaign reached `train_model` but the mixture loader rejected the
   pipeline's canonical `{manifest, diagnostics}` envelope. The loader now accepts
   both that envelope and historical bare manifests.

E221 v4 then completed the matched 32-step CPU train on the committed 480-record
E218 corpus. Last loss was 14.1748 over 21,709 prompt and 6,185 target tokens;
wall time was 129.91 s. Checkpoint SHA is `85f0fb0c…bd7cd`; it remains local and
was not synced.

The task-balanced policy increased unique exposure from 54 to 80 records, but
effective exposure fell slightly from 29.90 to 29.68 records (23.19%). One
renderer row was drawn 18 times. Therefore the exposure hypothesis is falsified:
task balancing alone moves concentration into undersized task/family cells and
does not provide capacity-aware sampling.

The campaign's initial evaluation did not start because the typed compiler
targeted nonexistent `outputs/data/eval/v1`. The complete 19-record, five-suite
`remediated` snapshot is now published as canonical `eval:remediated`, and a
typed `eval_version` knob replaces the hardcoded path. The existing checkpoint
was then evaluated against that snapshot with tree decoding, lexer output,
schema and slot-contract context, local-only weights, and unconstrained fallback
disabled.

| Suite | n | syntax | meaningful parse | structure | component recall | fidelity | reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.2097 | 0.1667 | 0.0000 | 0.4327 |
| held_out | 5 | 1.0000 | 0.0000 | 0.1667 | 0.0500 | 0.0000 | 0.3822 |
| adversarial | 4 | 1.0000 | 0.2500 | 0.3492 | 0.2500 | 0.0000 | 0.1593 |
| ood | 4 | 1.0000 | 0.0000 | 0.2527 | 0.2083 | 0.0000 | 0.1593 |
| rico_held | 3 | 1.0000 | 0.0000 | 0.0901 | 0.0000 | 0.0000 | 0.0000 |

Ship gates failed nine checks; AgentV passed 1/5 suite records with no execution
errors. `fallback_count` was zero for every suite and the persisted policy has
`allow_unconstrained_fallback=false`. The separately reported
`constrained_fallback_rate` therefore remains decoder-path telemetry and must not
be interpreted as permission to emit unconstrained text. An earlier permissive
evaluation was superseded by this strict run. The result confirms that lexical
validity is deterministic here while meaningful structure and fidelity remain
the quality bottlenecks.
