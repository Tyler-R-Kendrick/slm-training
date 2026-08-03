# Autotrain c1848: binder-slot-ownership harness repair

**Verdict:** incomplete infrastructure measurement, not a model failure.

The matched control trained and evaluated on the smoke fixture, but the new
binder-slot-ownership candidate failed before training. The campaign schema and
command compiler accepted the new training/decode signal, while the model
capability gate correctly rejected its decode weight because the arm did not
request the required tree compiler mode. Treating that as a model null would be
wrong; the frozen arm must be replayed after the canonical repair.

| Arm | Status | Loss | Structure | MPR | Binder F1 | Fidelity | Reward | p50 ms | Checkpoint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| capacity-aware control | complete, fixture | 15.6932 | .0575 | 0 | 0 | 0 | 0 | 911.1 | `a9a2f810...d269f` |
| binder-slot-ownership | harness failed before training | — | — | — | — | — | — | — | none |

The repair is committed in `d876037b5`: the registered successor now requests
`compiler_decode_mode=tree`, and the selector test covers the full recipe. The
supervised loop must replay this exact control/candidate manifest before any new
hypothesis. This remains a fixture-only result (`n=3`); no ship or promotion
claim is possible.

Machine evidence:
[`autotrain-cycle-1848-binder-slot-ownership-harness-repair.json`](autotrain-cycle-1848-binder-slot-ownership-harness-repair.json).
