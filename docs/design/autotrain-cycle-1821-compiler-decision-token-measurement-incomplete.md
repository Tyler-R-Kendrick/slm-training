# Autotrain c1821: compiler-decision token measurement incomplete

**Verdict:** no model comparison is available. Both size-matched arms trained,
but the smoke evaluator timed out the entire three-document production batch.
Every quality field is therefore unmeasured, not zero, and the exact frozen
control/candidate pair must be replayed after the canonical timeout repair.

| Arm | Params | Loss | Decision rows at final step | Complete docs | Batch timeout | Checkpoint SHA |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| control | 1,608,962 | 15.53068 | 0 | 0 / 3 | 3 / 3 | `e02e53d3...9d029` |
| compiler-decision token | 1,608,962 | 18.64492 | 34 | 0 / 3 | 3 / 3 | `95c5846a...815b1` |

The candidate objective was active and dense: step 20 covered 34 deterministic
compiler decisions with mean CE `9.37192`, versus the two or three rows seen in
the preceding component-edge objectives. It adds no parameters and changes no
decoder score, legal domain, or constrained-decoding authority. Training alone
cannot establish whether that signal improves OpenUI programs.

The shared failure exposed two harness-budget defects. The documented
`decode_timeout_seconds` contract is per record, but the v77 evaluator applied
one 24-second wall to the entire three-record batch. The screening driver also
reserved a third execution lane for a promotion-only Lean preflight even though
formal status was correctly `not_applicable:screening`. The timed-out control
trace reached 167 emitted tokens, 199,702 unique completion states, 208,617
parser forks, and 17.10 seconds of compiler work before the shared wall; the
candidate trace showed the same class of state growth. This is an observed
runtime-capacity signal, not evidence that either checkpoint is better.

Harness eval v78 scales a batch wall by its record count while respecting any
cumulative evaluation deadline. Campaign v119 allocates screening time across
the two arms it actually executes; promotion continues to reserve and run its
Lean lane. The next run must replay these exact seed-101821 arms under those
repairs before testing a new hypothesis.

Both checkpoints are local, explicit no-sync fixture artifacts. They are never
reusable, promotable, syncable, or shippable. Lean is
`not_applicable:screening`; no theorem or promotion claim is made.

Machine evidence:
[`autotrain-cycle-1821-compiler-decision-token-measurement-incomplete.json`](autotrain-cycle-1821-compiler-decision-token-measurement-incomplete.json).
