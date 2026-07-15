# OpenUI verifier stack and confidence tiers

The corpus verifier is the shared record boundary under
`slm_training.data.verify`. Validity is layered: passing the Lark grammar is
necessary, but it is not sufficient evidence that a program is schema-valid,
resolvable, renderable, behaviorally correct, grounded, or fit for core SFT.

## Gates G0-G12

| Gate | Check | Evidence |
| --- | --- | --- |
| G0 | Lexical | Non-empty text, legal control characters, terminated strings |
| G1 | Grammar | In-process `openui.lark` parse |
| G2 | Schema | Official `@openuidev/lang-core` validation and positional schema |
| G3 | Reference graph | One root, resolved binders, reachability, no cycles |
| G4 | Dataflow | Pinned 0.2.x layout-only surface; later state/query/action syntax fails closed |
| G5 | Runtime | OpenUI preview rendered and emitted no console error |
| G6 | Behavior | Required interaction ran and emitted no page/behavior error |
| G7 | Grounding | Required facts are present and forbidden facts absent |
| G8 | Canonicalization | `C(P) == C(parse(C(P)))` |
| G9 | Patch correctness | `apply(before, patch) == after` when a transition is supplied |
| G10 | Provenance | Source evidence is complete and the row is not ambiguous |
| G11 | Independent judge | Optional separate-family judge evidence; never the generating teacher judging itself |
| G12 | Human audit | Optional sampled human-review evidence |

Each result is `pass`, `fail`, or `skip`. Skip means explicitly not applicable;
it never means a failed required check was ignored. Callers set
`require_runtime` / `require_behavior` when those gates are required. The first
failure is stored as `failing_gate`, while the full ordered report remains in
`meta.verification.gates`.

## Confidence tiers

| Tier | Meaning | Training use |
| --- | --- | --- |
| Gold | Human-audited row; every applicable deterministic gate passed | Core SFT |
| Silver | Program-first or deterministic extraction; every applicable gate passed | Core SFT |
| Bronze | Teacher/web candidate with no observed gate failure but only partial evidence | Candidate pool only |
| Quarantine | Any failed gate, ambiguity, or incomplete provenance | Excluded |

Tier assignment is a pure function of the row and supplied evidence. It does
not use time, randomness, network state, or a model score. `stamp_record()`
copies a row and writes `verification_tier`, `failing_gate`, and the complete
gate report into `meta` without adding a wire field.

## Runtime and behavior evidence

`src/apps/openui_preview/verify.mjs` launches Chromium through Playwright, mounts
the same bundled preview used by the playground, records console/page errors,
and clicks the first rendered button when present. Python callers use
`run_preview_verifier()` and pass the returned evidence into G5/G6. The runner
supports seeded console and behavior exceptions so the negative path is tested,
not merely inferred from a clean render.

## Verification ceiling

This stack is corpus plumbing, **not a model ship gate or readiness claim**.
Deterministic checks can only validate properties they encode. Teacher/web rows
must not be promoted by a single self-judging teacher: use deterministic gates,
a separate-family judge where judgment is needed, and sampled human audit.
Production model claims still require the full multi-suite `--ship-gates`
scoreboard described in `adversarial-review.md` and `honest-ship-eval`.
