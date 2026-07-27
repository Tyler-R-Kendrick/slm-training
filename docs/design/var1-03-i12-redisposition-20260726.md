# VAR1-03: I12 re-disposition (SLM-427)

**Status:** disposition complete.
**Claim class:** `disposition`.
**Honest verdict:** I12's rejection is **narrowed** to one variant
(`tree_edit_diffusion`); it is neither upheld across all variants nor
reopened. `repl_operators` and `twotower_prompt_ast` are `NOT_MEASURED`.

## Problem

I12's prior text cited a bare `reachable_fraction = 0.0` as if it were a
program-wide status, and named a successor (reachability-aware seeds, macro
actions, certified training pairs) that did not follow from the measured
reason codes. The published histogram
(`iter-slm305-edit-language-20260724.md`) contains exactly two reason codes
across every suite -- `unsupported_component` and `needs_direction_change`
-- both properties of the action alphabet and component inventory, not of
seeds or training pairs. I14 already forbids citing a rejected-approach
label as a reason a goal does not apply; this issue applies the same
discipline to a diagnosis.

## Evidence this disposition rests on

* `docs/design/iter-slm305-edit-language-20260724.md` -- the original
  `tree_edit_diffusion` reachability audit and its full reason histogram
  (`needs_direction_change`: 3+3+6=12 cases across train/adversarial/rico;
  `unsupported_component`: 3+3+5+1+4=16 cases across train/smoke/held_out/adversarial/ood).
* `src/slm_training/dsl/variants.py::build_variant_contracts()` -- the three
  registered variants (`repl_operators`, `tree_edit_diffusion`,
  `twotower_prompt_ast`) and their live `action_alphabet_fingerprint`s
  (SLM-422/VAR0-01, already merged). `tree_edit_diffusion`'s fingerprint as
  of this disposition: `ab2662a497d8359ffaee46ebbd4bee3789f5b0f2accaf8bf46c5dee489622dab`.
* `docs/design/var1-01-set-property-probe-20260725.md` (SLM-424/VAR1-01) --
  a hypothetical `set_property` action flipped one `needs_direction_change`
  case (`adv_empty_prompt_01`) to `PROVEN_REACHABLE`, licensing VAR1-02.
* `src/slm_training/harnesses/experiments/slm299_edit_reachability.py:347-354`
  -- the root's `rest` must match the seed's in every mode, so a different
  seed cannot substitute for a missing action class on root-owned
  properties. This is why seed selection is demoted below the two action/
  inventory successors rather than listed first.

SLM-423 (VAR0-02, per-variant reachability re-scoping) and SLM-425 (VAR1-02,
the real `SET_PROPERTY` action) were both still `In Review` -- not merged --
at the time of this disposition. Neither was required: `VariantContractV1`
(SLM-422, merged) already supplies the variant identity and alphabet
fingerprint I12 needs, and the VAR1-01 probe (merged) already supplies the
property-mutation evidence. This disposition does not depend on unmerged
work and will not need revision when SLM-423/425 land, since it does not
declare patch-as-target viable for any variant.

## Before / after

**Before** (I12, prior to this change): a single `Status: partial` paragraph
citing `reachable_fraction = 0.0` with no variant attribution, and a
successor list (seeds / macro actions / certified pairs) with no
reason-code mapping.

**After**: `Status` names `tree_edit_diffusion` and its alphabet fingerprint
explicitly, marks the other two variants `NOT_MEASURED`, and replaces the
successor list with three items each mapped to its reason code(s):
(1) property-mutation action class → `needs_direction_change`, licensed by
the VAR1-01 probe; (2) pack-derived component inventory →
`unsupported_component`; (3) seed selection and macro actions, demoted,
with the root-`rest` invariant stated as the reason seeds cannot substitute
for a missing action class. The I14 table row for I12 was updated to match.
A new `I14a` subsection states the scoping rule this issue established:
a variant-scoped measurement is never a program-scoped status, and a
successor approach must be traceable to an observed reason code.

## What this disposition does not do

* It does not declare patch-as-target viable for `tree_edit_diffusion` or
  any other variant. I12's rejection for the measured variant stands.
* It does not run a new reachability audit -- it re-attributes and
  re-orders successors against evidence that already exists on `main`.
* It does not depend on, or wait for, SLM-423 or SLM-425.

## Verification

```bash
python -m scripts.verify_decode_invariants
```
