# E299 choice-native minimum content (2026-07-17)

## Question and recipe

E299 asks whether the existing A4 `decode_min_content=-1` policy can stop the
choice checkpoint's empty-root collapse without retraining. The authoritative
run evaluates the unchanged E297 seed-1 checkpoint (SHA-256
`a78193f91ee12d07791cab008a75267e3f6e19cfd223fbc726b3896dd98d14ee`)
on CPU scratch with frozen prompt-only inputs, production choice decoding,
compiler-tree mode, no unconstrained fallback, all five ship suites, and
AgentV publication.

The first run showed that `scripts.evaluate_model` did not expose the existing
minimum-content setting. Runs r2 and r3 then showed that choice decoding
bypassed the compiler-tree A4 implementation; r2 also loaded the shared
checkout through the editable environment and is invalid for patch judgment.
Run r4 proved that merely forcing a bound declaration can preserve a
placeholder while still serializing a content-free Stack. These are superseded
harness-development artifacts. The authoritative result is
`e299-choice-min-content-auto-honest-r5`.

The generalized choice-native policy now:

1. counts completed non-Stack component declarations;
2. enters the v0.5 declaration/root stream when a floor is active;
3. requires a string-bearing component production capable of consuming the
   visible slot inventory; and
4. emits EOS after the first valid content root.

The setting remains off by default. Effective evaluation policy now records
`decode_min_content`, so future boards can distinguish the arm.

## Results

| Suite | n | Meaningful E297 → E299 | Fidelity E297 → E299 | Structure E297 → E299 | Recall E297 → E299 | Reward E297 → E299 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 0.0000 → **0.3333** | 0.0000 → **0.7222** | **0.3094** → 0.1742 | 0.0000 → **0.1667** | 0.0000 → **0.2690** |
| held_out | 5 | 0.0000 → 0.0000 | 0.0000 → **0.4467** | **0.2514** → 0.1088 | 0.0000 → 0.0000 | 0.0000 → 0.0000 |
| adversarial | 4 | 0.0000 → **0.5000** | 0.0000 → **0.6667** | **0.2905** → 0.1926 | 0.0000 → **0.3750** | 0.0000 → **0.4035** |
| ood | 4 | 0.0000 → 0.0000 | 0.0000 → **0.2583** | **0.2369** → 0.1469 | 0.0000 → 0.0000 | 0.0000 → 0.0000 |
| rico_held | 3 | 0.0000 → **0.6667** | 0.0000 → **0.2083** | 0.0901 → **0.1035** | 0.0000 → **0.3333** | 0.0000 → **0.4747** |

Parse remains 1.0 on every suite. Failed ship thresholds fall from 17 to 12,
but AgentV remains 0/5 with zero execution errors. Smoke still misses its
meaningful, structure, recall, and reward gates; held-out and OOD remain
meaningful 0.0. The direct content-root constraint improves semantic density
and placeholder use but sharply reduces layout structure.

## Verdict

The choice-native minimum-content implementation is a useful explicit
diagnostic and closes a silent harness gap, but it is not a ship lever by
itself. Keep it opt-in (`0` remains the default), do not promote the checkpoint,
and do not claim readiness. The next lever should preserve a container root
while making content declarations reachable from it, rather than replacing the
layout with a single content component.

Artifacts:

- `outputs/runs/e299-choice-min-content-auto-honest-r1/` (superseded no-op)
- `outputs/runs/e299-choice-min-content-auto-honest-r2/` (invalid import root)
- `outputs/runs/e299-choice-min-content-auto-honest-r3/` (superseded no-op)
- `outputs/runs/e299-choice-min-content-auto-honest-r4/` (superseded primitive-content arm)
- `outputs/runs/e299-choice-min-content-auto-honest-r5/` (authoritative)
- [machine-readable result](choice-min-content-results-iter-e299-20260717.json)
