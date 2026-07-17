# E256: immutable semantic counterfactual corpus (2026-07-16)

Status: **admitted for bounded E252 training; not ship evidence**.

E256 expands the repaired E252 prerequisite over every document record in the
E230 judged source. It uses the unchanged E228 checkpoint, strict production
compiler-tree decoding, exact-state replay, and the independent judge plus
meaningful-program/Pareto verifier. The output is persisted under
`src/slm_training/resources/data/preference/e256_counterfactual_semantic_v1/`
so future local preference runs consume source-controlled data rather than
ephemeral telemetry.

## Recipe and identity

- Base commit: `aaca9529824f2e4deddb70b4789733446abfc005`
- Device: CPU; seed: 256
- Source: E230 judged roots V2, fingerprint
  `9f72d85b6cc7118e0f69e010d0debdd2b40ede514e03178dded8e164daaae9bb`
- Checkpoint SHA:
  `7a9be4a665e216d7f7e73883ad74ad972bbf30846896d0c29188d6482f5b093a`
- Decode hash:
  `8c2a2ae5cb5c4ad0ab44de74a172fc2c887fbdc7e5e7e20ede953184a0c3fe5b`
- Policy: `strict_compiler_tree`; two states per record; four candidates per
  state
- Trace: `277a2cb24ba7218a9e545612fb61aa95`

## Measured result

| Measure | Result |
| --- | ---: |
| Document traces accepted | 65 / 65 |
| Exact states replayed | 130 |
| Grammar-legal candidates | 390 |
| Independent-judge pass | 108 / 390 |
| Verified candidates | 23 / 390 |
| Qualified events | 16 |
| Prompt groups | 8 |
| Train / held-out events | 14 / 2 |
| Train / held-out groups | 7 / 1 |
| Set-valued events | 8 |

The held-out group is `train_cta_01`; its split follows the stable group hash
and was not selected or reassigned manually. Every admitted group contributes
one `bind_declaration_root` and one `component_root` event. That is sufficient
for the predefined E252 set-valued/held-out prerequisite, but it is narrow
evidence: it does not establish semantic improvement at later tree decisions.

## Persisted admission evidence

`events.jsonl` contains only counterfactual events and has fingerprint
`0c253f11a9ec096fb81663db43d9aacf3406a01b09a9d39a20e02eaa04686e53`.
`evidence.jsonl` retains the complete qualified probe, candidate completions,
judge decisions, meaningful-program result, and metric vector for every event;
its fingerprint is
`4d4508c29127ef49ea93599ec4ab976e8560ebc4c83bd7354777b025102fd98f`.
The manifest binds both files to checkpoint, tokenizer, decode configuration,
source records, and trace identity.

## Decision

Admit this immutable corpus for a bounded E252 `ftpo_set` run with unchanged
quality gates. Do not claim ship readiness from corpus admission. Before the
training command, fetch and reconcile latest `origin/main`, require a clean
worktree, and prove zero commits behind. Promotion still requires held-out event
movement plus the full quality scoreboard without structural or reward
regression.

Machine-readable evidence:
[`quality-matrix-v10-e256-corpus-results.json`](quality-matrix-v10-e256-corpus-results.json).
