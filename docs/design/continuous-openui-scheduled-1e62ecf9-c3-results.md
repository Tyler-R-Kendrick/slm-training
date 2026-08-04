# Continuous autotrain: 2026-08-04 (scheduled loop `1e62ecf9`) cycle 3 — schema self-heal unblocks the loop, screening ties null

**Loop:** `continuous-openui-scheduled`
**Campaign:** `continuous-loop-20260804-continuous-openui-schedu-1e62ecf9-c3`
**Integration commit:** `1f28f6e` (`fix(autoresearch): allow generate_batch_size in ExperimentKnobs/allowed_knobs`)
**Predecessor:** `continuous-loop-20260804-continuous-openui-schedu-1e62ecf9-c2` (hard-blocked)

## What was blocked (cycles c1, c2)

Both `c1` and `c2` crashed identically during hypothesis formation, before any
experiment manifest could lock: `CampaignSpec` validation rejected **every**
generated screening hypothesis (56/56 in `c2`) with

```
hypotheses.N.experiment.knobs.generate_batch_size
  Extra inputs are not permitted [type=extra_forbidden, input_value=1, input_type=int]
```

Root cause: `scripts/run_autotrain_continuous.py`'s screening-role hypothesis
builder sets `base["generate_batch_size"] = 1` (landed in `01767f1`, the
v180 salvage of PR #1408, to let fair-share decode timeouts work on tiny
screening batches) but `ExperimentKnobs` (a `StrictModel`) never declared
that field and `DEFAULT_ALLOWED_KNOBS` never listed it in
`src/slm_training/autoresearch/schemas.py`. Every screening hypothesis
therefore failed validation, hypothesis count fell to 0/5, and
`campaign_id=...-c1`/`...-c2` both hard-blocked at `campaign_initialized` +
`evidence_captured` + `literature_captured` — never reaching
`hypothesis_matrix_formed`.

## The fix

`1f28f6e` adds `generate_batch_size: int | None = Field(default=None, ge=1,
le=1024)` to `ExperimentKnobs` and `"generate_batch_size"` to
`DEFAULT_ALLOWED_KNOBS`, matching the existing `batch_size` field's bounds.
`harness.autoresearch.experiment_campaign` bumped `v180 -> v181` in
`versions.json` per the version-stamp contract. 352 tests in
`tests/test_autoresearch/test_harness.py` +
`tests/test_scripts/test_run_autotrain_continuous.py` pass unchanged.

## Cycle c3 (post-fix replay)

Same objective/track/knob-bank as the blocked `c1`/`c2` cycles, run
end-to-end for the first time this loop:

| Arm | structural_similarity | meaningful_program_rate | binder_reference_f1 | latency_ms_p50 | trainable_params |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 0.230833 | 0.333333 | 0.733333 | 8694.25 | 1,608,962 |
| both (candidate) | 0.230833 | 0.333333 | 0.733333 | 9178.28 | 1,608,962 |

Checkpoint SHA256: control `b17e6a45af82506304d55adcc84abbcc61d5ee6c98a7c14b4db0e15555f69ae7`,
candidate `9b13d8e26fb62f98c53c50aa14bc680ab4bcbb15d0d4307965cdce90b82c699d`
(both local under `outputs/autoresearch/.../runs/`, explicit no-sync).

Ship gates (honest, fixture-scale smoke suite): **fail** —
`smoke:insufficient_n actual=3 need>=20`,
`smoke:meaningful_program_rate actual=0.333 need>=0.66`,
`smoke:structural_similarity actual=0.231 need>=0.35`,
`smoke:component_type_recall actual=0.167 need>=0.35`,
`smoke:ast_beq_rate actual=0 need>=0.2`,
`smoke:canonical_beq_rate actual=0 need>=0.1`; `held_out` / `adversarial` /
`ood` / `rico_held` all `missing_suite`. Expected at this fixture scale, not
a ship claim.

## SDLC Phase A

**Positive (executable unblocking):** the schema fix removed a
hard, unrecoverable path error (2 consecutive identical crashes, 0/5 valid
hypotheses) and the very next cycle completed with a usable, gate-scored
scoreboard. This qualifies the harness commit for a plain infra-fix PR.

**Non-positive (model):** control and candidate ("both") tie exactly on the
primary metric (`smoke.structural_similarity` 0.230833 both) — a null
delta, not a win — and both arms hit `fixture_insufficient_n` (n=3 need
>=20). Per the driver's own classification
(`SDLC_PHASE_A NON_POSITIVE ... action=no_stack_layer_non_positive`), this
model result does **not** earn a stacked training-win layer. No metric
regression either; the tie is consistent with a screening arm that did not
move the needle versus the size-matched control.

## Delivery

`1f28f6e` lands as a plain harness-fix PR (schema completeness bug, not a
training-win stack layer), following the precedent of prior sessions'
infrastructure-only PRs on this loop family (e.g. `#1403`, `#1406`, `#1410`,
`#1420`, `#1423`, `#1429`, `#1433`). Next-run priority (rank 1, confidence
0.90): the completed non-positive arm is exhausted; the next cycle should
test the distinct size-matched `component-plan` quality hypothesis rather
than repeat this knob bank.
