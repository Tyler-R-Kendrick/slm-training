# E181–E194 — semantic mixture and grammar-derived compiler alignment

Status: **diagnostic only; no checkpoint promoted; ship gates not run**.

## Question

Can balanced judge-gated data or training on the same partial states used by the
Lark-derived compiler fix semantic selection after deterministic constraints
already make the stream syntactically legal?

## Recipe and evidence

All three trains used the immutable 496-record `e177_semantic_judge_v2` corpus
(`d59a94d…7803`), CPU, 32 steps, batch 4, seed 0, frozen SmolLM2-135M context,
schema/slot context, no DESIGN.md context, and no checkpoint sync. The committed
E181 online mixture gives human-curated records 0.50 weight, corruption and edit
records 0.15 each, ProgramSpec 0.10, and renderer/web records 0.05 each. Its hash
is `c141a452…acac`.

Every evaluation is a one-example smoke diagnostic with AgentEvals JSONL and an
AgentV SDK bundle under the corresponding `outputs/runs/<run>/agentv/` directory.
The complete machine-readable recipe, checkpoint hashes, run paths, and metrics
are in [the result JSON](iter-e181-e194-compiler-alignment-20260716.json).

| Experiment | Lever | Syntax | Meaningful parse | Structure | Component recall | p50 ms | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| E181 | Balanced online mixture | 1.0 | 0.0 | 0.1542 | 0.25 | 3348.46 | No quality change |
| E182 | Root-choice telemetry | 1.0 | 0.0 | 0.1542 | 0.25 | 3293.09 | Learned root preference isolated |
| E183 | First-edge vs full-path score | 0.0 | 0.0 | 0.1917 | 0.25 | 3366.47 | Path-length hypothesis falsified; recipe mismatch noted below |
| E185–E190 | E184 component alignment + generalized compiler repairs | 0.0–1.0 | 0.0 | 0.0–0.4350 | 0.0–0.25 | 2981–12922 | Root fixed; semantic completion not fixed |
| E192 | E191 all-branch alignment | 1.0 | 0.0 | 0.1542 | 0.25 | 9284.52 | Rejected; root regressed |
| E193–E194 | E184 + grammar/schema scoping | 0.0 | 0.0 | 0.1615–0.3600 | 0.25 | 6311–17916 | Symbol leakage fixed; output still incomplete |

E183 accidentally omitted slot-contract and strict no-fallback flags. Its root
telemetry is still useful because both first-edge and full-path scores ranked
`TextContent` above `Stack`, but its syntax scoreboard is not comparable and is
not used as evidence of a regression.

## Training results

- E181 (`e181-semantic-balanced-32step`) ended at loss 5.5118 in 19.51 s.
  It consumed 128 examples: 72 human-curated, 24 corruption, 13 ProgramSpec,
  8 renderer, 6 edit, and 5 web. Every admitted target and every sampled target
  has `Stack` as its root, so missing root exposure is not the explanation.
- E184 (`e184-compiler-aligned-32step`) added weight-1 component-decision
  alignment. Alignment covered all 128 rows; loss fell from 77.0583 to 3.1030
  (mean 12.6388). Total loss was 10.0153 and wall time 96.04 s. The checkpoint
  recovered `Stack` at the root but did not complete a meaningful program.
- E191 (`e191-full-compiler-aligned-32step`) sampled from every compiler branch
  rather than component branches only. Alignment covered 128 rows and fell from
  67.2121 to 7.7572 (mean 15.4343), but total loss was 14.8498, wall time was
  157.69 s, and E192 regressed the root to `TextContent`. Reject this objective.

## Generalized compiler changes

The decoder now derives decisions from grammar/parser state and generated schema:

- EOS is legal only when the generated AST is complete.
- Typed lexer tokens are converted back to their source surface before advancing
  persistent parser state.
- Active call frames use Lark reduction/value-stack nesting, not comma counting.
- Binder declaration and forward-reference scope comes from typed binder tokens
  and grammar transitions.
- Symbol candidates are admitted only for an active generated-schema string slot,
  with the exact slot contract.
- Repeated newline suppression is a grammar-idempotence rule.

These are reusable grammar/AST invariants. No prompt, component arrangement, or
known failing output is matched by an exact string.

## Conclusion and next hypothesis

Constrained decoding is doing structural work, but it cannot substitute for a
ranker that chooses the right semantic branch when several branches are legal.
Random all-branch alignment dilutes the useful component signal. The next train
should stratify alignment by grammar-derived decision kind, ensuring component
and binder decisions are represented per batch while keeping the implementation
independent of concrete prompt strings or hand-authored output cases.
