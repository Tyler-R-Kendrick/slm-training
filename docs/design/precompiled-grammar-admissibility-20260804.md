# Precompiled grammar admissibility — preregistered campaign (2026-08-04)

Decode-cost campaign against the completion-forest exploration cost center.
Claim class: fixture_or_scratch / screening. Not a ship claim. Strict
byte-parity posture (R1 option b): every layer must preserve the exact
candidate sets, witness payloads, and generate outputs; the node-budget
re-baseline (total-table oracle change) is explicitly deferred to a separate
campaign.

## Cost center (measured before design)

- `finalize_ms` (repair/certify phase) = **87%** of MaskGIT generate wall
  (5,886 of 6,736 ms/gen, `outputs/runs/profile_maskgit_head_instr.json`;
  telemetry landed in PR #1416).
- One production `_openui_completion_domain` query at the canonical hard
  prefix `root = Card([b1,` (budget 32): `terminal_witness` = **98.8%** of
  2.9 s — 52% candidate-branch construction
  (compiler_draft.py branch loop: `engine.copy()` + feed + forced-suffix
  chase), 38% packed transitions (`CompletionSession.advance`), 2.3%
  `allowed_id_set`, 1.0% memoized `accepts()`.
- **11,272 parser transitions are paid to expand 270 witness nodes**; 155
  forests are built per query with only 16 `_outgoing` cache hits;
  4,785–11,219 unique interned `(control, semantic)` states per query, many
  sharing a control key the state-id-keyed caches cannot exploit.
- Cross-attempt waste: `_ensure_valid_openui` runs up to 3 repair attempts,
  each with a fresh session — the graph is re-explored per attempt.

## Hypothesis

Exactness-preserving memoization and precompute keyed on the LALR control
state — (L2) a control-key expansion memo at the forest-build boundary,
(L3) a certified integer control simulator over the artifact's serialized
LALR tables, and (L4) an offline min-completion lower-bound table — remove
most redundant exploration without changing any candidate set, witness
payload, or generated output. (L1 lands independent micro-fixes first so
attribution stays honest.)

## Preregistered criterion (locked before any timing)

(i) **Equivalence — reject outright on failure, per layer:**
  - `scripts/profile_generate` `sample_outputs` byte-identical to baseline
    under BOTH the default config and `--no-incremental` (the currently
    fastest known configuration);
  - E1 multiset parity green (`tests/test_dsl/test_static_control_domain.py`);
  - completion-kernel fixture payload-digest equality
    (`_v1_domain_is_preserved`) and `replay_witnesses` true over every
    prefix of the canonical corpus.

(ii) **Effect — all three required for the campaign to be supported:**
  - kernel-fixture warm hard-prefix domain-query median **−≥3×**;
  - end-to-end `finalize_ms_mean` **−≥40%**;
  - `sec_per_generate` **−≥25%** on the profile fixture.
  Declared noise band: ±10% wall on this shared WSL2 box; thresholds are
  chosen to sit far outside it.

(iii) **Mechanism engaged — floors:**
  - control-memo hit rate ≥50% on the hard-prefix query (L2 counters);
  - static-simulator transitions > 0 with zero certification mismatches (L3);
  - min-completion table lookups > 0 (L4 counters).

**Failure of (ii)/(iii) ⇒ the failing layer is reverted as pure
complexity** (each layer is measured separately and individually
revertible); successor approach filed in this doc per I14.

**L4 pre-declared decision rule:** budget-neutral pruning is accepted only
if the kernel fixture proves payload-identity over the canonical corpus.
Analysis suggests exact neutrality may be impossible (a pruned subtree's
true node consumption k≥1 is unknowable without doing the work); if the
fixture shows any payload drift, L4 ships table + counters only (pruning
default-off) and pruning moves to the Phase-2 re-baseline campaign.

## Setup

- Code base: main after PRs #1412/#1415/#1416/#1417 merge.
- Checkpoint: `outputs/runs/s1_d64/checkpoints/last.pt` (local, gitignored,
  output-contract v2). The committed playground demo cannot load
  (predates symbol_only/v2; docs/MODEL_CARD.md admits this) — fixture-scale
  evidence, stated plainly.
- Commands: kernel fixture via `python -m scripts.run_perf_matrix
  --completion-kernel --docs-agentv-dir <fresh>`; end-to-end via
  `python -m scripts.profile_generate --checkpoint <ckpt> --maskgit
  --rounds 2` (baseline and per-layer, default + `--no-incremental`).
  Every run ≪ `MAX_RUN_MINUTES`.
- Invariants: masks/tables only skip work; `_ACCEPTS_MEMO` (stack-exact)
  and the packed session remain the sole admissibility authority
  (I1/I2/I6; "static(G,T) ∩ dynamic(S)" — dynamic only tightens). The new
  lever(s) provably cannot widen legality and therefore do NOT enter
  `CONSTRAINT_WEAKENING_LEVERS`.

## Results

(to be filled per layer — no claims before measurement)
