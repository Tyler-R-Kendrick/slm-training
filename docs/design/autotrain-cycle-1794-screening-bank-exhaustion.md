# Autotrain c1794: screening-bank exhaustion

**Verdict:** harness stop before matrix formation; no train, evaluation,
checkpoint, model metric, or Lean promotion target exists.

c1794 started from merged main `2051bddb` and correctly reclassified the stale
c1793 queue entry from `confirmed` to `rejected`. It then found every registered
screening family closed by complete lineage evidence and failed closed with:

```text
registered screening arm bank exhausted; add a distinct preregistered quality objective instead of recycling a rejected approach
```

This is the intended anti-thrash boundary, but a hands-off loop also needs a
new executable hypothesis after reaching it. Campaign harness v92 adds the
already implemented compiler-decision component-edge alignment loss as a new
size-matched successor family. Both arms instantiate the same component-edge
head; the treatment changes only
`component_edge_alignment_loss_weight: 0→1`. It is train-only and leaves
grammar-constrained decode unchanged.

The next run should test whether supervising legal child-component choices
conditional on active parent components improves structural similarity and
binder-reference F1. It must remain fixture screening, use the same parameter
count in both arms, and require a fresh quality confirmation before any
promotion. Lean is `not_applicable:pre_matrix_harness_stop`.

Machine evidence:
[`autotrain-cycle-1794-screening-bank-exhaustion.json`](autotrain-cycle-1794-screening-bank-exhaustion.json).
