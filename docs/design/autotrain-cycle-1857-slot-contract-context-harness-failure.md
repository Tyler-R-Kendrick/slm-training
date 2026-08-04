# Autotrain c1857: slot-contract context harness failure

**Verdict:** measurement incomplete; no model attribution or ship claim.

The size-matched control trained and completed smoke evaluation, but the
`slot_contract_in_context` candidate failed before producing a checkpoint or
scoreboard. The failure was in shared prompt inventory extraction: the training
DESIGN.md contains an instructional `:slot_4` example, which was incorrectly
merged with the prompt inventory and violated the contiguous opaque-ordinal
contract. The exact frozen comparison is replayable after the extractor repair;
the control metrics are not evidence for or against the candidate.

| Arm | Params | Loss | Struct | MPR | Recall | Binder F1 | Fidelity | p50 ms | Tokens | Forwards | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 1,608,962 | 7.9206 | .1742 | .333 | .25 | .633 | .528 | 853 | 21 | 4 | complete, gates fail |
| slot-contract-context | — | — | — | — | — | — | — | — | — | — | failed before scoreboard |

The canonical repair keeps incidental DESIGN.md examples out of inventory
extraction while preserving explicit `Placeholders:` lines. It was committed
as `3bf2f2f164f32b1e70be99abcd7c2eac09722cf2`, tested against all 101
`wf_smoke_v2` records, and is bound to the handoff's `repair_harness` receipt.

Machine evidence:
[`autotrain-cycle-1857-slot-contract-context-harness-failure.json`](autotrain-cycle-1857-slot-contract-context-harness-failure.json).
