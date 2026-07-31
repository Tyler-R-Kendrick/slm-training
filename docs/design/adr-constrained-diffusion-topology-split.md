# ADR: Constrained diffusion × topology — claim-family split

**Status:** accepted (2026-07-30)  
**Kind:** architecture disposition + preregistered experiment contracts  
**Supersedes / tightens:** informal “AOT DPDA + Γ + weighted AND–OR circuit” unit claim  
**Related:** [certified-completion-artifact-and-tps-target.md](certified-completion-artifact-and-tps-target.md),
[grammar-topology-diffusion.md](grammar-topology-diffusion.md),
[decode-invariants.md](decode-invariants.md),
[completion-kernel-perf-results.json](completion-kernel-perf-results.json),
[adversarial-review.md](adversarial-review.md)

## Decision

Keep the exact compiler graph and inject neural scores only as ranking weights
over already-legal symbols (**I6**). Do **not** collapse legality into a manifold
embedding. Split three claim families so falsifying one never reads as
falsifying another (**I14**):

| Family | Job | Near-term status |
| --- | --- | --- |
| **A — Engine executor** | Certified static control + Γ leaf filters; cold/parity | **Primary product track** |
| **B — Diffusion / multi-hole circuit** | Residual multi-hole *predicate*, later semiring reweight | **Gated research** after A |
| **C — Model-facing topology heads** | Binder / mask / scope heads as learned structure | **Frozen** pending anti-E237 |

### Demoted box (what survives)

> **Near-term:** certified AOT **executable static control** + existing **Γ
> (`SemanticState`)** with fail-closed Lark parity.  
> **Separate later:** weighted decision circuit over **choice variables** for
> multi-hole diffusion.  
> **Never:** sell request-local memo warmups, geometric IR, or LAVE recovery as
> the N=1 fix.

Artifact remains a **proof payload**, not the sole execution substrate; only
the tokenizer projection currently accelerates production. Full LALR control
execution is a measured successor (E1→E2), not a rebrand of today’s loader.

### Ordered stack (never put 6 before 1)

1. Executable static control + Γ bitset leaf filters → cold + parity  
2. AOT forced macros + `common_forced_run` scheduling → zero-forward law  
3. Choice-codec as model space → diffuse ambiguity only  
4. Joint multi-hole support on residual (Mündler *predicate*) → MaskGIT cluster safety  
5. Tropical/counting \(h(e)\) reweight inside legal set  
6. Semiring / categorical flow on legal decision simplexes only  

## Claim family A — Engine executor

**Exists today**

- Checked artifact (`openui_completion_artifact/v1`): LALR control arrays +
  tokenizer→terminal projection; checker reconstructs and equates live authority.
- Production consumes **`direct_map`** after checker accept; Lark remains live parser.
- `SemanticState` Γ overlay + `CompletionSession.forced_closure` (no literal /
  incomplete-coverage compression).
- Packed domain path (`_openui_completion_domain`) with differential Lark oracle.

**Not yet a DPDA executor.** Serialized control tables are certificate payload;
calling them a DPDA is literature hygiene until E1 green on an executable adapter.

**Production API (this ADR):**  
`slm_training.dsl.grammar.fastpath.static_control_domain` —

- requires checked artifact (fail closed);
- builds candidate domains via certified projection + request-local forest + Γ;
- exposes multiset/coverage snapshots for E1 parity vs Lark oracle;
- never widens legality; Γ only tightens.

### E1 — Artifact / production domain parity

| | |
| --- | --- |
| **Pass** | Candidate multisets + coverage ≡ Lark oracle on kernel corpus including hard prefix `root = Card([b1,`; no silent UNKNOWN→UNSUPPORTED collapse; Γ filters only tighten |
| **Fail** | Any multiset/coverage mismatch; UNKNOWN collapsed to UNSUPPORTED; Γ widening |

### E2 — Cold p50 (process-warm pack, request-cold)

| | |
| --- | --- |
| **Pass** | Executor ≤ 0.5× V1 cold **and** E1 green |
| **Fail** | Speedup only with primed domain caches; E1 red |

Honest skip allowed if flamegraph shows Γ/arena tax dominates LALR transitions
(P3 risk). **No AOT-cold claim** from warm memo.

### E3 — Load/check tax

| | |
| --- | --- |
| **Pass** | Digest production mode ≥10× faster load vs full reconstruct; still catch corruption |
| **Fail** | Corrupted tensors accepted |

### E4 — Static macro-edges

| | |
| --- | --- |
| **Pass** | Independent static walk (`walk_static_forced_macro` over `outgoing`/`advance_path`, never calls `forced_closure`) matches dynamic `forced_closure`; never across literals / incomplete coverage |
| **Fail** | Compress across literals or incomplete authority; static path is only a double-call of dynamic |

Implementation: `walk_static_forced_macro` (pure) → `extract_static_forced_macro` (compare).

### E8 — Weights-on-graph vs I2

| | |
| --- | --- |
| **Pass** | Real decode entry (`TwoTowerModel._greedy_ltr_decode_batch` under `collect_decode_stats`) records `forwards_count == 0` after complete singleton proof; denoiser hard-fails if invoked |
| **Fail** | Neural ranking / forward before complete singleton proof; theater `DecodeStats()` without a decode path |

### E9 — Reclassify 89.8× warm hard-prefix

| | |
| --- | --- |
| **Pass** | Cite only as `request_local_memo_reuse` (fixture_or_scratch) |
| **Fail** | Market as AOT cold or ship evidence |

Fixture row (`completion-kernel-perf-results.json`): `warm_card_open_comma`
mode `persistent_row_session_no_domain_cache`, V2 median ~8 ms, speedup
~89.8×, `edges_built: 0`, large domain/transition cache hits. Cold
`card_open_comma` V2 is a **regression** (~0.816×). Claim class remains
`fixture_or_scratch`.

## Claim family B — Diffusion / multi-hole circuit

Paper transfer (bounds, not control-loop blueprints):

| Paper | Keep | Reject as production blueprint |
| --- | --- | --- |
| [arXiv:2508.10111](https://arxiv.org/abs/2508.10111) | Multi-hole completability ⇔ CFG ∩ regular emptiness (**predicate**) | “All-hole neural circuit”, one F/B pass for all holes as a theorem of the paper |
| [arXiv:2602.00612](https://arxiv.org/abs/2602.00612) | Unbounded hole spans (Σ*) can accept dead ends under real length bounds | Unbounded LAVE-style invent-then-verify as authority |
| LAVE (lookahead-then-verify slogan) | Verifier-before-commit idea | Sample-N Earley primary; recovery rewrite of prefix; model proposes every step ignoring I1/I2 |

**I3 branding:** production path is **forest-verified speculative completion** —
draft only from the complete legal forest / domain, re-derive domain each step,
oracle-verify before commit. LAVE N-sample recovery is **not** the primary
blueprint (diagnostic only).

**Residual multi-hole:** Boolean support under fixed slots + Γ leaf filters with
**honest over-approx** labels when Γ is external. Never soft legality.
Implementation: `residual_support.py` + `maskgit_constrain.admit_fill`.

Reweight \(h(e)\) and full semiring circuits are stack steps 5–6 only inside
already-legal residual choice sets.

## Claim family C — Model-facing topology heads (frozen)

Measured losing record (not exhaustive):

| ID | Result |
| --- | --- |
| E236/E237 | Topology loss ↑ / acc ↓; **38 applications, 0 choice changes**; detach no-op |
| E729 | Head learned then causally hurt decode (meaningful 0.667→0.333) |
| E1004–E1013 | More topology data/exposure → held_out collapse / homogeny |
| X9–X21 | Mask/scope topology stacks fail ship |
| X22 | Validity-preserving edits beat mask topology at matched tiny budget |
| SLM-384 | Topology-**apply** execution win (−13% node_passes), not representation learning |
| SLM-187 | Solver/runtime parity gap blocks training on topology intermediates |

**Freeze:** no new default production binder/mask/scope topology heads until
anti-E237 passes. Infrastructure topology (legal apply, valid edits) is a
**different claim family** and may ship under executor/edit contracts.

### Anti-E237 experiment contract

Any claim that “topology/hypergraph helps” (model-facing) must preregister:

- **Arms:** control / train-on·decode-off / train-on·decode-on  
- **Kill criteria:** no learning δ; applications>0 & choice_changes=0;
  choice_changes>0 & quality↓; train-on·decode-off damages primary  
- **Report:** stratified head acc by candidate-count; meaningful (not syntax);
  held_out+adversarial; **n≥20** for promotion; SLM-187 parity if solver claims  

Minimum success bar: held_out meaningful ≥ control+0.10 (n≥20) **and** causal
decode not harmful **and** no train interference.

## Non-goals (this ADR)

- Riemannian / simplicial / stack geometry as runtime IR  
- Manifold embeddings of the parse graph  
- LAVE N-sample Earley as primary; production witness-injection / prefix recovery rewrite  
- In-band model-emitted `[REPROCESS]` as authority (verifier-owned outcomes only)  
- SAE as legality validator (optional proof-economics router research only)  
- Full all-hole semiring circuit as first milestone  
- Marketing warm 89.8× as AOT cold or ship gate evidence  

## Research stance (one line)

**Execute the residual you already certify; force the forced; verify multi-hole
commits against residual support with Γ filters; reweight only legal ambiguity.
Do not fund manifolds, LAVE recovery, or another model-facing topology head
until executor parity (E1) and anti-E237 contracts pass.**

## Open risks (self-check)

1. Executable control may not win cold if tax is Γ/arena rebuild — flamegraph
   first (P3); honest E2 skip allowed.  
2. Choice-codec residual circuits may be small enough that step 6 demotion is
   conservative — measure node count vs L before freezing “later forever.”  
3. E729 harm might be weight-scale — still requires causal re-run under anti-E237.  
4. Decode-Time Grammars (arXiv:2607.18357) unvetted for Γ ABI borrow.  

## Implementation pointers

| Concern | Path |
| --- | --- |
| Certified artifact | `dsl/grammar/fastpath/completion_artifact.py` |
| Packed session / forced_closure | `dsl/grammar/fastpath/completion_kernel.py` |
| Production domain + E1 snapshots | `dsl/grammar/fastpath/static_control_domain.py` |
| Residual multi-hole predicate | `dsl/grammar/fastpath/residual_support.py` |
| Forest-verified speculation | `dsl/grammar/fastpath/speculative_rank.py` |
| Γ overlay | `dsl/grammar/fastpath/semantic_state.py` |
| Tests | `tests/test_dsl/test_static_control_domain.py`, `test_residual_support.py` |
| **Lean claim-core proofs** | [`adr-constrained-diffusion-topology-proofs.md`](adr-constrained-diffusion-topology-proofs.md) · `src/leverproof_lean/LeverProofLean/ConstrainedDiffusion.lean` |
