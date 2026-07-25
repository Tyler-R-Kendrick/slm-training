# I2 — close the short-horizon repair bypass gap

Closes the "Known gap — short-horizon repair loses the bypass" section of
[`decode-invariants.md` §I2](decode-invariants.md#i2--forced-bypass-on-singletons).
Measured evidence: [`i2-horizon-limited-bypass-20260725.json`](i2-horizon-limited-bypass-20260725.json).

## The gap

`test_repair_exact_token_skips_forward_and_records_authority` was red on
`main` (confirmed at `d77dfa0`, still red after `995813d`, and reproduced
again here at `993a70c` before this change). In `_constrained_ltr_repair`,
`st.remaining_tokens = length - len(prefix)` shrinks every repair step. For a
DSL-native (lexer) tokenizer, `exact_forced_token_id` required
`build_completion_forest` to prove a *complete* terminal witness reaching EOS
within that shrinking budget. When the budget was too short to enumerate a
witness, `CompletionForest.coverage` came back `"none"` — even when the
budget-independent structural forest had already proven exactly one legal
next action (the DFA had proven `=` was the sole legal lexeme after `root`).
The proof was correct; it was discarded, and an avoidable forward ran.

The doc's own framing: this was never a legality violation, only an I1/I2
efficiency loss, and the named successor was to distinguish "no witness
within horizon" from "witness disagrees" in `CompletionForest.coverage".

## The fix

`CompletionDomainStatus` gains a `"partial"` value alongside `"complete"` /
`"incomplete"` / `"unsupported"`. In `_openui_completion_domain`
(`src/slm_training/dsl/pack.py`), when the terminal-witness search finds zero
witnesses within the decode budget, the structural forest (computed
independent of budget) is checked: if it has exactly one legal action, that
action is returned as a `"partial"` domain (sole candidate, no full witness)
instead of `"incomplete"`. Zero or multiple structural actions still return
`"incomplete"` unchanged — that is the genuinely contradictory/ambiguous case
and it keeps failing closed. `build_completion_forest`
(`compiler_draft.py`) carries `"partial"` through to `CompletionForest.coverage`
verbatim. `exact_forced_token_id` (`src/slm_training/models/grammar.py`, the
DSL-native branch) now accepts `coverage in ("complete", "partial")` when
exactly one candidate id survives — the same singleton discipline as before,
just no longer discarded purely because the horizon was short.

Every other reader of `CompletionForest.coverage` (`speculative_rank.py`,
`solver/decode.py`, `onnx_inference.py`, `twotower.py`'s forward-admission
checks, `grammar.py`'s `pick_constrained_token._compiler_admits`) still
requires `coverage == "complete"` literally, so they keep failing closed on
`"partial"` exactly as they did on `"none"` before. Only the one function this
gap named — the exact-bypass proof — was widened.

## Measured evidence

Recipe: CPU, fixture-scale `TwoTowerModel.from_records`, `output_tokenizer="lexer"`,
`d_model=32`, single-row `SAMPLE = 'root = Card(":t.x")\n'`. Honesty mode:
wiring/regression evidence for a code-level invariant fix, not a ship-gate
readiness claim.

| Check | Before | After |
| --- | --- | --- |
| `test_repair_exact_token_skips_forward_and_records_authority` | **FAILED** — `AssertionError: forwarded` (forward ran when none was legal to run) | **PASSED** — `forwards_count == 0`, `decision_source == "dfa_singleton"` |
| `tests/test_models/test_inference_speed.py` (full file) | — | 22 passed in 95.56s |

Singleton probe (`root` prefix, exactly one legal next lexeme `=`):

| `remaining_tokens` | coverage | `n_paths` |
| --- | --- | --- |
| 1 | `partial` | 1 |
| 8 | `partial` | 1 |

`coverage` stays `partial` (not `complete`) even at budget 8 for this grammar
— the full `Card(":t.x")` witness needs more tokens than that to reach EOS.
The fix forces only the single legal next token; it never claims full-witness
completeness.

Ambiguity control probe (`root =` prefix, 19–40 legal component choices) —
proves the fix did not widen the bypass to genuinely ambiguous positions:

| `remaining_tokens` | coverage | `n_paths` |
| --- | --- | --- |
| 1 | `none` | 0 |
| 2 | `none` | 0 |
| 4 | `none` | 0 |
| 8 | `complete` | 19 |
| 20 | `complete` | 40 |

Unchanged from pre-fix behavior: small budgets here still return `none`
(0 or >1 structural paths), so `exact_forced_token_id` still returns `None`
and a forward still runs — the "witness disagrees"/contradictory case keeps
failing closed, exactly as I2 requires.

## Invariant compliance

Not a legality violation: the forced token was already the unique legal
continuation under the structural (DFA-forest) proof before this change; only
the over-strict requirement to *also* prove full termination-reachability
within the current truncated budget is relaxed. Certainty is never downgraded
into a soft preference — the ambiguity control probe shows genuinely
non-singleton positions still fail closed and still forward.

## Version stamps

`dsl.grammar_capabilities` v3→v4, `dsl.operators.registry` v3→v4,
`model.twotower` v241→v242 — see
`src/slm_training/resources/versions.json`.
