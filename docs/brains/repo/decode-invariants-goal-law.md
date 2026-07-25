---
type: concept
status: active
tags: [decode, grammar, invariants, goal-law, speculation, scheduling]
created: 2026-07-25
design: docs/design/decode-invariants.md
sources: arXiv:2508.10111 (IG-CD), arXiv:2602.00612 (LAVE)
---

# Decode invariants as goal law

## Claim

This repo's architecture invariants are **goals**, not approaches. Constrained
decoding is the product; deterministic completion outranks inference; symbol
tables both rank and schedule; the compute-ops vocabulary is shared
encoder↔decoder; multi-turn is a CRDT event store. A rejected experiment closes
an *approach* and must file its successor — it never closes the goal.

Canonical statement: [`AGENTS.md`](../../../AGENTS.md) § Non-negotiable
architecture invariants. Canonical expansion with file pointers and per-invariant
status: [`docs/design/decode-invariants.md`](../../design/decode-invariants.md).

## Why it might be true

A grammar-constrained symbolic model has a property a general LLM does not: at
many positions the answer is *derivable*, not *predictable*. Every position
where the scope-aware symbol table collapses to a singleton is a position where
inference is pure cost — and the same table that proves legality also tells the
scheduler which rows still need a forward and how far ahead reading is worth
anything. Treating the symbol table as a first-class compute plan (I3, I4) is
what makes this architecture cheaper than a general decoder rather than merely
safer.

The drift risk is specific and observed: two invariants (I12 patch-as-default,
I13 op-token vocabulary) have had an approach *measured and rejected* in-repo.
A repo that reads "approach rejected" as "goal abandoned" quietly becomes a
different project. I14 exists to make that failure mode illegal.

## Falsification boundary

- A production decode path can emit output the grammar rejects → I6 is false in
  practice; find the path and close it, do not relax the claim.
- A scope-proven singleton costs a forward pass in any backend → I2 regressed;
  the `forwards_count == 0` bypass test for that backend is missing or wrong.
- A deterministic ranker over the legal domain measurably loses to the neural
  ranker at equal legality → the *n-gram approach* is falsified, not I3. File
  the successor (trie, learned ranker, campaign-approved successors).
- An encoder-side shared `OPS_VOCAB` experiment measures no benefit → I13's
  current approach is falsified. e803 does **not** falsify it: e803 tested
  decoder targets only.
- A CRDT-converging merge is shown to lose information a conflict-rejecting
  merge preserves → I11's merge approach needs revision, and the divergence
  must be documented as a dated waiver, not as silence.

## Open questions

- What margin does the n-gram ranker need before committing without a forward
  is net-positive on parse rate at fixed legality? (I3 campaign)
- How much of the decode budget is spent reading canvas past the next grammar
  checkpoint? (I4 counters answer this before the lever ships)
- Is `reachable_fraction = 0.0` a seed problem or an edit-language problem?
  (I12 successor: reachability-aware seeds vs macro actions)

## Links

- [[preregistered-experiment-campaigns]] — turning any of the above into a
  confirmatory run requires a pre-start locked manifest.
- [[gate-reachability-and-power]] — measurement integrity is separate from the
  invariants; a rung is never credited by a lower rung's evidence.
