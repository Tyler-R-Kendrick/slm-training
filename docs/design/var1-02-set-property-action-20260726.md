# VAR1-02 (SLM-425): SET_PROPERTY action -- production reachability delta

- generated_at: `2026-07-26T19:54:14Z`
- gated on: `docs/design/var1-01-set-property-probe-20260725.md` (SLM-424/VAR1-01)
- hypothetical: `false` -- this is the REAL, production action space, checkpoint
  format, and analyzer, not a what-if probe.
- mode: `extended`, max_edits: 8, node_budget: 15 (see "Why node_budget=15" below)
- seed: `root = Stack([], "column")`

> Reachability is space coverage, not model quality: no quality claim follows
> from these proofs alone. No training, evaluation, or promotion ran as part
> of this issue.

## What changed

`ACTION_SET_PROPERTY` (index 11; `N_ACTIONS` 11 -> 12) rebinds an existing
container's declared `rest` (its enum/direction argument) to another value of
the SAME pack-declared domain (`DslPack.component_property_domains`) -- root
included, since (unlike `REMOVE_CONTAINER`) it never removes or re-mints a
node. `CHECKPOINT_FORMAT` bumped 2 -> 3; a format-2 checkpoint fails closed on
load pointing at `checkpoint_migrate.migrate_tree_edit_checkpoint`, which
already upgrades any `format_version < CHECKPOINT_FORMAT` generically (no new
function needed -- see the code changes list below). The tree-edit variant's
`VariantContractV1.kernel_ops` now declares `openui.set_property`. The SLM-299
analyzer's `_check_invariants` now recognizes the real action (pack-domain
bounded, forced on in extended mode) separately from VAR1-01's hypothetical
`set_property_action()` (which may use a wider what-if domain and still
defers to BFS exactly as before) -- every VAR1-01 probe test continues to pass
unchanged.

## Reachable fraction by suite: four points of comparison

| suite | SLM-305 baseline (node_budget=120, older corpora) | VAR1-01 Arm A (before, node_budget=15, current corpora) | VAR1-01 Arm B (`set_property_action()` what-if, node_budget=15) | VAR1-02 production (this run, node_budget=15) | gate: production <= Arm B |
| --- | --- | --- | --- | --- | --- |
| train | 0.0 | corpus_unavailable | corpus_unavailable | corpus_unavailable | n/a |
| smoke | 0.0 | 0.0 | 0.0 | 0.0 | ok (0.0 <= 0.0) |
| held_out | 0.0 | 0.0 | 0.0 | 0.0 | ok (0.0 <= 0.0) |
| adversarial | 0.0 | 0.0 | 0.5 | 0.333333 | ok (0.333333 <= 0.5) |
| ood | 0.0 | 0.0 | 0.0 | 0.0 | ok (0.0 <= 0.0) |
| rico | 0.0 | 0.0 | no_decided_cases | 0.0 | ok (Arm B had no decided cases to exceed) |

The SLM-305 baseline (`docs/design/iter-slm305-edit-language-20260724.md`) is
**not** a same-budget, same-corpus comparison: it ran at `node_budget=120`
against older corpora (rico had 6 records, not 35; adversarial had no
`adv_deep_nest_01` UNKNOWN_BUDGET case; `train`'s corpus existed as a
committed fixture, now a gitignored generated artifact) -- see VAR1-01's own
note on this. VAR1-01's **Arm A_baseline** reproduces the pre-VAR1-02
production action set at `node_budget=15` against the exact same corpora this
run used, so it is the honest apples-to-apples "before"; **Arm B** is the
probe's what-if prediction this issue is gated on.

### Why node_budget=15, not the issue's suggested 120

A single-arm run at `node_budget=120` was attempted first and did not finish
inside a 170s wall-clock cap (`MAX_RUN_MINUTES=3`); a timed-out run is never
evidence per AGENTS.md. `node_budget=15` (VAR1-01's own precedent) completed
in 41.6s. This is a materially weaker probe on the `rico` suite in particular
(see the discrepancy note below); a longer, separately-run job at a larger
budget outside this session's cap would be needed to fully resolve it.

## Per-suite verdict flips vs VAR1-01 Arm A (the same-corpus "before")

- **smoke / held_out / ood**: no flips (0 each) -- these suites' gaps are all
  `unsupported_component` (component-inventory gap, out of this issue's
  scope; VAR0-03/SLM-426 already closed the analogous inventory gap
  elsewhere).
- **adversarial**: 1 flip.
  - `adv_empty_prompt_01`: `PROVEN_UNREACHABLE` (`needs_direction_change`) ->
    `PROVEN_REACHABLE` (`reached`, `edit_lower_bound=2`,
    `path=[ADD, SET_PROPERTY]`) -- **CONFIRMED**, and matches VAR1-01 Arm B's
    confirmed flip on this exact case, same path shape, same 2-edit bound.
  - `adv_many_buttons_01` stays `PROVEN_UNREACHABLE` (`needs_direction_change`)
    -- its target needs a `"row"` rest, which the pack does not declare in
    `component_property_domains` (only `', "column"'` and `""`), so
    `ACTION_SET_PROPERTY` genuinely cannot produce it. Production's
    pack-domain-bounded invariant proves this case decisively; VAR1-01 Arm B
    (unconditional invariant skip, unbounded what-if domain) left it as
    `UNKNOWN_BUDGET` instead. **More decisive, not less correct.**
  - `adv_deep_nest_01` stays `UNKNOWN_BUDGET` in both (budget-bound,
    unaffected by this change).
- **rico**: 34 flips, all `PROVEN_UNREACHABLE` (`needs_direction_change`) ->
  `UNKNOWN_BUDGET` (`budget`). See the discrepancy note below -- this is an
  honest, disclosed side effect, not a reachability regression.

## Honest-verdict

- **Claim class: `capability`** (space coverage only). Not claimed: ship
  readiness, promotion of any checkpoint, a champion-model change, or that
  improved reachability implies improved output quality. No training,
  evaluation, or promotion ran as part of this issue.
- **What is confirmed:** `adv_empty_prompt_01` flips to `PROVEN_REACHABLE`,
  reproducing VAR1-01 Arm B's confirmed positive exactly (same case, same
  2-edit path shape: `ADD` then `SET_PROPERTY`).
- **Gate check (issue requirement: production <= Arm B per suite, else red
  flag):** every suite lands at or below its Arm B counterpart (see the table
  above) -- **no red flag.** `adversarial`'s 0.333333 <= 0.5 is the only
  suite with a numeric Arm B value below 1.0; the apparent "regression" from
  0.5 to 0.333333 is explained by production **deciding** one more case
  (`adv_many_buttons_01`, proven genuinely unreachable) than Arm B's cruder
  unconditional-skip invariant could -- a strictly more decisive result, not
  a worse one.
- **Discrepancy disclosed honestly (not picked for looking favorable):**
  `rico`'s reachable_fraction did **not** improve (stays `0.0`), and its
  decided-case count collapsed: VAR1-01 Arm A decided all 35/35 cases
  (`PROVEN_UNREACHABLE`, `needs_direction_change`); this production run
  decides only 1/35 (`rico_eval_test_56`, whose target needs a `"row"` rest
  outside the pack's domain -- still decisively unreachable), and the other
  34 become `UNKNOWN_BUDGET`. This is **not** a reachability regression: most
  `rico` targets omit a direction argument entirely (`root =
  Stack([...])`, rest `""`), which the pack's own `container_rests` domain
  already permits -- so the OLD hard "root rest must equal seed's" check was
  never actually sound once a real rest-mutation action exists; it was
  masking the true (deeper, structural) reachability question behind a
  cheap-but-now-recognized-as-overly-strict proof. Confirming or refuting the
  REST of each transformation (typically several `ADD_CONTAINER`/
  `INSERT_SUBTREE`/`BIND_PLACEHOLDER` steps to rebuild multiple
  `Card`+leaf subtrees) needs a deeper search than `node_budget=15,
  max_edits=8` can decide. Per AGENTS.md I14, `UNKNOWN_BUDGET` is
  inconclusive and never evidence of unreachability; it is reported
  separately here, not folded into a falsely-precise `0.0`. A longer,
  separately-run job at a larger `node_budget` (outside this session's
  `MAX_RUN_MINUTES=3` cap) is required to resolve those 34 cases either way.

## Files changed

- `src/slm_training/models/tree_edit_diffusion.py` -- `ACTION_SET_PROPERTY`,
  `N_ACTIONS=12`, `CHECKPOINT_FORMAT=3`, `EditDomain.component_property_domains`
  / `.property_names`, `TreeEditSpace.apply` branch (root-inclusive, real
  parser re-validated), `sample_mutation` inverse-edit generation,
  `_enumerate_edits` decode-time scoring.
- `src/slm_training/models/checkpoint_migrate.py` -- docstring only;
  `migrate_tree_edit_checkpoint` already upgrades any
  `source_format < CHECKPOINT_FORMAT` generically (shape-driven, not
  hardcoded to format 1), so format 2 -> 3 needed no new code path.
- `src/slm_training/harnesses/experiments/slm299_edit_reachability.py` --
  `_enumerate_children` gains the real `ACTION_SET_PROPERTY` transitions;
  `_check_invariants` distinguishes the real (pack-domain-bounded) capability
  from VAR1-01's hypothetical (potentially wider-domain) one via
  `hypothetical_set_property_domain`; `analyze_reachability` forces
  `set_property` on in extended mode alongside the existing `container_add`
  forcing.
- `src/slm_training/dsl/variants.py` -- `tree_edit_diffusion` variant's
  `kernel_ops` now includes `openui.set_property`; regenerated
  `src/slm_training/resources/variant_registry.json`.
- `tests/test_models/test_tree_edit_diffusion.py` -- new tests: apply/inverse
  round-trip on root and a non-root container, illegal-value rejection via
  the real parser (not just an index bounds-check), format-2 fail-closed
  load, and format-2->3 migration with bit-identical logits on the
  pre-existing 11 action rows.
- `src/slm_training/resources/versions.json` -- bumped
  `harness.experiments.slm299_edit_reachability` (v5->v6), `dsl.variants`
  (v2->v3), `model.twotower` (v255->v256, watches
  `checkpoint_migrate.py`).
- This doc pair.

## Out of scope (per issue)

- Training, evaluating, promoting, or syncing any checkpoint.
- Widening the component inventory (VAR0-03/SLM-426, already done).
- Any claim that improved reachability implies improved output quality.
