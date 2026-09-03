# Modularity and reproducibility refactor strategy

**Status:** proposal — evidence gathered, no refactor applied.
**Instrument:** `python -m scripts.audit_modularity` (static, `ast`-only, no torch).
**Measured at:** merge of `0478591` into this branch, Python 3.12,
`src/` + `scripts/` (1,242 files).

This document answers three questions with measurements rather than taste:

1. Where is the code actually complex, and how much of that complexity is
   *incidental* (scaffolding) versus *essential* (the model, the grammar, the
   decode invariants)?
2. Which composition patterns retire the incidental part without touching the
   essential part or weakening any invariant in
   [`decode-invariants.md`](decode-invariants.md)?
3. What is the safe order of operations, and how do we know we are winning?

The headline: **42,763 lines — 8.3% of the audited `src/` + `scripts/` surface
(517,166 lines) — are mechanically-derivable scaffolding**, and one class of that scaffolding is
currently producing **provably divergent fingerprints**, which is a
reproducibility defect rather than a style complaint.

---

## 1. Method

`scripts/audit_modularity.py` parses every file under the requested roots and
classifies duplicated definitions into four debt classes. It is static: it
never imports the audited modules, so it runs without torch, without a GPU, and
without network. It reports its own interpreter version, because the project's
`requires-python` floor is 3.12 and PEP 695 / PEP 701 syntax silently fails to
parse on older interpreters — an audit that under-counts is worse than no
audit.

```
python -m scripts.audit_modularity                 # census
python -m scripts.audit_modularity --json          # machine-readable
python -m scripts.audit_modularity --check-divergence   # CI ratchet, exit 1 on fork
```

Every number below is reproducible from that command.

---

## 2. Findings

### 2.1 The census

Scanned 1,242 files declaring **1,685 dataclasses** with **14,134 annotated
fields**.

| Debt class | Definitions | Lines | Files | Note |
| --- | ---: | ---: | ---: | --- |
| `serialization` | 1,644 | 18,120 | 405 | 8,645 lines mechanically derivable |
| `reporting` | 217 | 13,994 | 217 | 7 competing naming conventions |
| `lifecycle` | 212 | 9,264 | 75 | one protocol, ~17–20 re-implementations |
| `primitives` | 393 | 1,385 | 247 | 10 helpers have >1 implementation |
| **Total** | **2,466** | **42,763** | | |

For scale: `src/slm_training/harnesses/experiments/` alone is **181 files /
103,188 lines**. The full Python surface is 732,605 lines — `src` 403,096,
`scripts` 114,070, `tests` 215,439 — of which the audit scans the 517,166
non-test lines.

### 2.2 Serialization — 18,120 lines re-deriving field lists

- `to_dict`: **1,091 definitions / 10,302 lines**. Of these, **750 (68.7%,
  8,309 lines) are a single `return {...}` over constant keys** — bodies that
  carry no information beyond the declaring dataclass' own field annotations.
- `from_dict`: **431 definitions / 7,024 lines**.
- `to_json` (90) and `as_dict` (32) add a further 794 lines.

The repo already declares `pydantic>=2.7,<3` in `pyproject.toml`, but only
**32 of 1,242 files** use it. The base class intended to solve exactly this —
`StrictModel` — is **defined 6 separate times** (`formal/bound_ast.py`,
`harnesses/reasoning/revmath/schemas.py`,
`harnesses/experiments/efs4_04_causal_synthesis.py`, …). The abstraction meant
to remove duplication was itself duplicated.

### 2.3 Reporting — 13,994 lines of hand-assembled markdown

217 emitters across **seven** naming conventions for one job:

| Name | Defs | Lines |
| --- | ---: | ---: |
| `render_markdown` | 93 | 6,783 |
| `_build_markdown` | 37 | 2,006 |
| `_markdown` | 33 | 1,744 |
| `render_evidence_markdown` | 17 | 1,082 |
| `_render_markdown` | 12 | 957 |
| `_write_markdown` | 14 | 833 |
| `_markdown_report` | 11 | 589 |

Each hand-builds tables by appending `"| --- | --- |"` separator strings and
f-string rows. Column alignment, escaping, and `None` handling are re-decided
per file. There is no way to add a column to every evidence report, retarget
reports to HTML for the dashboard, or diff two reports structurally.

### 2.4 Lifecycle — one contract, ~20 implementations

`ExperimentCampaignV1` is defined once in
`autoresearch/experiment_campaign.py` and referenced by 69 files. The
*protocol around it* is re-implemented per experiment:

| Step | Defs | Lines |
| --- | ---: | ---: |
| `run_campaign` | 17 | 2,466 |
| `build_campaign_manifest` | 17 | 2,210 |
| `run_fixture_campaign` | 18 | 1,221 |
| `run_experiment` | 20 | 1,176 |
| `build_manifest` | 14 | 754 |
| `validate_manifest` | 31 | 645 |
| `lock_campaign` / `write_campaign_lock` / `load_campaign_lock` | 52 | 315 |
| `write_evidence` | 17 | 131 |
| `plan_only_preview` / `patch_preregistry` | 26 | 346 |

The *sequence* — preregister, lock, run arms, score, gate, stamp, write
evidence — is identical every time and written out longhand every time. This
is the highest-risk duplication in the repo: the preregistration law says the
locked endpoint, arms, seeds, stopping rule and gates may never be replaced
after outcomes are visible, and that law is currently enforced by 20 hand-typed
copies agreeing with each other.

### 2.5 Primitives — the reproducibility defect

393 copies of 12 micro-helpers, only 1,385 lines, but **10 of them have more
than one implementation**:

| Helper | Copies | Distinct implementations |
| --- | ---: | ---: |
| `fingerprint` | 56 | **24** |
| `_sha` | 51 | **16** |
| `_sha256` | 34 | **12** |
| `_canonical_json` | 36 | **8** |
| `_utc_now` | 12 | **4** |
| `_now` | 73 | 3 |
| `_source_commit` | 19 | 3 |
| `stable_hash` | 4 | 3 |

**359 files build a `version_stamp` on top of these primitives.**

This is not hypothetical. The two most common `_canonical_json` variants are:

```python
# 24 copies
json.dumps(v, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
#  5 copies — ensure_ascii omitted, therefore True
json.dumps(v, allow_nan=False,                    separators=(",", ":"), sort_keys=True)
```

Given a payload containing any non-ASCII character — an em-dash in a prompt, a
curly quote in a design brief, `flex→grid` in an op name — these emit different
bytes and therefore different digests:

```
A: {"op":"flex→grid","prompt":"Design a “hero” card — 20% wider"}
B: {"op":"flex\u2192grid","prompt":"Design a \u201chero\u201d card \u2014 20% wider"}

sha256(A) = 6998954424a0ed0f057f5a47f7dac2cac5fc0cb2c366fe01ab666cea9b28a072
sha256(B) = fc0ff61e5ad4c2404727a54a0bd66223e58539bf0191a06693ba913282f824ed
```

Same payload, same intent, two identities. A further variant omits
`allow_nan`, so a NaN metric serializes to the literal `NaN` — invalid JSON —
where sibling harnesses raise. And `fingerprint`, the function whose entire job
is to produce a comparable identity, has **24 distinct implementations**.

A canonical implementation already exists at
`harness_core/lineage/records.py::canonical_json`. The 36 local copies were
written anyway. **The shared layer exists; nothing routes to it.**

### 2.6 God objects

| Unit | Size | Shape |
| --- | ---: | --- |
| `models/twotower.py::TwoTowerModel` | **15,685 lines / 185 methods** | 10 methods >300 lines = 7,331 lines |
| `scripts/run_autotrain_continuous.py` | 18,752 lines / 321 symbols | `run_cycle` alone is 1,800 lines |
| `harnesses/model_build/eval_runner.py` | 3,596 lines | |

Inside `TwoTowerModel`, the method names describe a design that is already
factored — as private methods rather than as objects:

- **Decode strategies (~4,021 lines):** `_generate_maskgit_one` (1,200),
  `_choice_ltr_decode_batch` (596), `_generate_batch_once` (551),
  `_compiler_ltr_decode_one` (491), `_greedy_ltr_decode_batch` (445),
  `_constrained_ltr_repair` (378), `_compiler_ltr_decode_batch` (360).
- **Logit-bias operators (~1,423 lines):** `_semantic_plan_bias` (266),
  `_semantic_plan_typed_array_nonempty_bias` (244), `_semantic_plan_root_bias`
  (219), `_slot_coverage_close_bias` (208), `_semantic_plan_role_obligations`
  (203), `_schema_role_slot_bias` (157), `_slot_component_bias` (126).
- **Loss terms:** `training_loss` is a single **1,912-line** method.

### 2.7 The invariant that is not structurally checked

`_select_compiler_path` (830 lines, at line 10,834 of a 17,283-line file)
encodes the repo's most important invariant as control flow:

1. singleton / deterministic bypass returns immediately (`_record_exact_bypass`);
2. then `_speculative_ranker()` ranks **only among proven-legal candidates**
   ("Membership is untouched");
3. then, and only then, the learned model score.

That ordering *is* the AGENTS.md law: *deterministic/singleton bypass outranks
any learned score; speculation ranks only over forward-calculated symbol
tables and always verifies before commit.*

`scripts/verify_decode_invariants.py::check_bypass_tests` cannot check that
ordering. It checks a **proxy**: that named test files still contain a
substring such as `"singleton_bypasses"`. The precedence itself — the actual
invariant — is verified by nothing but code review of an 830-line method.

---

## 3. Strategy

Five patterns, each tied to a measured target. The ordering principle: **make
the invariants structural, then delete the scaffolding.**

### 3.1 Adapter — one reproducibility kernel behind a port

*Target: `primitives`, 393 defs / 1,385 lines / 10 forked helpers.*

Promote a single `slm_training/harness_core/canonical.py` exposing
`canonical_json`, `digest`, `fingerprint`, `utc_now`, `source_commit`,
`repo_root`. Every local copy becomes a re-export, then a deleted import.
`harness_core/lineage/records.py` already has the reference implementation;
this makes it the only one.

This is Adapter in the strict sense: harnesses depend on a *port* (the
canonical contract), not on an inlined implementation, so byte-level identity
policy — `ensure_ascii`, `allow_nan`, float repr, timestamp precision — is
decided in one place and versioned as one component.

**Why first:** smallest diff, largest correctness win, and it is a
prerequisite for trusting any before/after comparison in the later stages. A
migration that changes fingerprints must be done *deliberately and once*, with
a component bump, not accidentally 36 times.

**Ratchet:** `audit_modularity --check-divergence` in CI. Once green, a new
forked helper fails the build.

### 3.2 Template Method + Strategy — one campaign runner

*Target: `lifecycle`, 212 defs / 9,264 lines.*

The campaign sequence is fixed; only the arms differ. Invert it:

```python
class CampaignProtocol(Protocol):
    """The parts an experiment actually supplies."""
    def arms(self) -> Sequence[Arm]: ...
    def score(self, arm: Arm, outcome: Outcome) -> Metrics: ...
    def gates(self) -> Sequence[Gate]: ...

def run(spec: ExperimentCampaignV1, impl: CampaignProtocol, *, root: Path) -> Evidence:
    """Preregister -> lock -> run -> score -> gate -> stamp -> write. Once."""
```

An experiment then declares its arms, scorer and gates and inherits locking,
manifest construction, version stamping, evidence writing and markdown
emission. The repo already uses `Protocol` in 43 places and
`runtime_checkable` in 24 — this is the established idiom here, not a new one.

**The safety argument is the point.** Preregistration integrity currently
depends on ~20 hand-written copies staying in agreement. Centralising the
runner means the lock is *enforced by construction*: an experiment physically
cannot mutate its locked endpoint after outcomes are visible, because it no
longer owns that code path.

### 3.3 Visitor — a report IR with pluggable renderers

*Target: `reporting`, 217 defs / 13,994 lines.*

Replace hand-assembled strings with a small document IR
(`Section`, `Table`, `KeyValues`, `Verdict`, `Caveat`) plus renderers that
visit it:

- `MarkdownRenderer` — today's evidence files, byte-compatible.
- `JsonRenderer` — structural diffing of two runs, which is impossible today.
- `HtmlRenderer` — feeds the dashboard, removing a whole class of
  compiled/interpreted parity drift.

Visitor is the right pattern precisely because the node types are stable
(sections, tables, verdicts) while the *operations over them* keep growing.
It also gives the honest-ship-eval rules a single place to enforce the
fixture-demo-vs-ship caveat, instead of trusting 217 authors to remember it.

### 3.4 Generative templates — derive, don't type

*Target: `serialization` (18,120 lines) and CLI (3,004 `add_argument` calls
across 349 files; 367 `main` bodies totalling 29,195 lines).*

Two derivations, both from declarations the code already carries:

**(a) Codec from annotations.** 14,134 annotated fields already describe every
payload. A single `Record` base (dataclass + `dataclasses.fields`, or the
already-declared pydantic 2) derives `to_dict`/`from_dict`/JSON Schema from
those annotations. This retires the 750 mechanical `to_dict` bodies (8,309
lines) outright; the 341 non-mechanical ones keep hand-written hooks for the
genuinely custom cases. Consolidating the 6 `StrictModel` definitions to one is
the first step.

**(b) CLI from spec.** `--seed` is declared 45 times, `--out` 42, `--steps` 35,
`--run-id` 34, `--device` 32. A shared `harness_cli` module contributing
standard argument groups, with entry points declared rather than typed, removes
the bulk of 29,195 lines of `main` and — more importantly — makes flag
semantics uniform, so `--seed` means the same thing everywhere. Today it
demonstrably does not: 141 sites construct a seeded `random.Random`, and the
seeding contract is per-author.

This is the largest raw-line win, but it is deliberately sequenced **after**
§3.1 and §3.2, because derived serialization changes byte output, and that must
land on top of a canonical fingerprint kernel — never underneath one.

### 3.5 Strategy + Composite — decompose `TwoTowerModel`

*Target: `TwoTowerModel`, 15,685 lines.*

Three extractions, in increasing order of risk:

1. **Decode strategies (~4,021 lines) → a declared, ordered chain.** Each
   backend becomes a `DecodeStage` with an explicit rank:

   ```python
   DECODE_CHAIN = (
       DeterministicBypass,   # rank 0 — singleton / complete coverage
       SpeculativeRanker,     # rank 1 — proven-legal candidates only, verifies
       LearnedScore,          # rank 2 — the model
   )
   ```

   This is the highest-value structural change in the document. It converts
   `verify_decode_invariants`' substring proxy into a **direct assertion over
   the registered chain order**: the bypass stage must precede the speculative
   stage, which must precede the learned score, and no stage may widen
   membership. The invariant stops being a comment inside an 830-line method
   and becomes a machine-checkable property. Modularity here *buys*
   verification of the product's core law.

2. **Logit-bias operators (~1,423 lines) → a Composite of `BiasOperator`s.**
   Seven `_*_bias` methods are already a pipeline of additive logit
   adjustments; making them a registered, ordered list makes them individually
   testable and ablatable — which is what the experiment harnesses keep
   reaching into the model to do by hand.

3. **`training_loss` (1,912 lines) → a registry of loss terms**, each with a
   name, weight and contract. Term-level ablation is currently a code edit.

**Guardrail:** this must be pure extraction under characterization tests. There
are 924 test files; the decode paths carry bypass assertions already. Any
extraction that changes a logit, a mask, or a decode decision is a bug, not a
refactor — and `EG_params` is untouched throughout, because this removes lines,
not parameters.

---

## 4. Sequencing

| # | Stage | Target lines | Risk | Unlocks |
| --- | --- | ---: | --- | --- |
| 1 | Adapter: canonical kernel (§3.1) | ~1,385 | **Low** | trustworthy before/after comparison |
| 2 | Ratchet: `--check-divergence` in CI | — | **Low** | prevents regression |
| 3 | Strategy: decode chain (§3.5.1) | ~4,021 | Medium | **structural invariant checking** |
| 4 | Template Method: campaign runner (§3.2) | ~9,264 | Medium | preregistration by construction |
| 5 | Visitor: report IR (§3.3) | ~13,994 | Medium | JSON/HTML reports, dashboard parity |
| 6 | Generative: codec (§3.4a) | ~8,645 | Medium | schema export |
| 7 | Generative: CLI (§3.4b) | ~29,195 (partial) | Medium | uniform flag semantics |
| 8 | Composite: bias ops + loss terms (§3.5.2–3) | ~3,335 | **High** | term-level ablation |

Stages 1–2 are safely separable and worth doing regardless of whether the rest
is approved: stage 1 fixes a live defect, stage 2 stops it recurring.

Each stage lands as its own PR with the audit census in the description, so the
line count is a reviewable number rather than a claim. Per the version-stamp
contract, stages touching harness/metric/gate/matrix paths carry a component
bump; stages 1 and 6 change byte output and **must** bump rather than take a
`no-bump:` note.

---

## 5. What this does not change

This is a scaffolding refactor. It touches no invariant in
[`decode-invariants.md`](decode-invariants.md):

- Constrained decoding stays the product; §3.5.1 makes its precedence *more*
  strictly enforced, never less.
- Deterministic/singleton bypass stays rank 0 and stays ahead of any learned
  score — that becomes a structural assertion instead of a review convention.
- Speculation still ranks only over forward-calculated symbol tables and still
  verifies before commit; extraction preserves the verify step or the
  extraction is wrong.
- Symbol tables still schedule prefills; the ops vocabulary stays shared
  encoder↔decoder; multi-turn stays a CRDT event store.
- **No parameters are added.** Every stage removes lines, not weights;
  `EG_params` is unaffected and no growth claim is implied. Capability is not
  being bought here — maintenance cost is being sold.

Two honest caveats:

- **Fingerprint migration is a real event.** Stage 1 makes divergent digests
  converge, which means some historical fingerprints will not reproduce under
  the unified kernel. That is the defect surfacing, not the fix causing it —
  but it needs a deliberate component bump and a note in the affected evidence,
  not a silent landing.
- **Line-count reduction is not the goal metric.** The goal is that an
  invariant has one implementation and one enforcement point. The 42,763 figure
  measures the opportunity; it should not become a target to optimise.

---

## 6. Tracking

`scripts/audit_modularity.py` is the ratchet. Baseline at `0478591`:

```
serialization   1644 defs   18120 lines   405 files
reporting        217 defs   13994 lines   217 files
lifecycle        212 defs    9264 lines    75 files
primitives       393 defs    1385 lines   247 files
TOTAL                       42763 lines
divergent reproducibility-critical helpers: 8
```

Target after stage 2: **`divergent … helpers: 0`**, enforced by
`--check-divergence` in CI. Subsequent stages report their own delta against
this baseline.

The census totals above were unchanged by the five commits between `0ac2999`
and `0478591`, which is the expected behaviour: this debt accumulates slowly
and is not perturbed by ordinary feature work. The `twotower.py` figures in
§2.6–2.7 did shift over those commits and were re-measured, which is exactly
why the line numbers in this document are provenance-stamped rather than
treated as stable addresses.
