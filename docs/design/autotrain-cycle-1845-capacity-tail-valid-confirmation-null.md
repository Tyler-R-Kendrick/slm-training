# Autotrain c1845: valid capacity-aware tail confirmation null

**Verdict:** reject the c1830 capacity-aware tail winner. After repairing
champion recipe projection, the first lever-complete fresh-seed confirmation is
an exact quality and decode-work null.

| Arm | Params | Effective / draws | Unique | Structure | MPR | Recall | Binder F1 | Fidelity | Reward | AST / canonical | Tokens | Forwards | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| capacity-aware control | 1,608,962 | 22.86 / 40 | 28 | .46417 | .3333 | .25 | .6333 | .5278 | .8073 | 0 / 0 | 54 | 10 | 2638.10 |
| + tail supervision | 1,608,962 | 22.86 / 40 | 28 | .46417 | .3333 | .25 | .6333 | .5278 | .8073 | 0 / 0 | 54 | 10 | 2680.05 |

Both arms used CPU scratch TwoTower, 20 steps, batch size 2, one thread, seed
101845, capacity-aware sampling, all-family compiler alignment, and 1,608,962
trainable parameters. The control manifest is
`c20260803-continuous-openui-202607-98199209-c1845-control.json`
(`ab8f32f4…afd89`) with `ltr_tail_loss_weight=0.0`; the candidate manifest is
`c20260803-continuous-openui-202607-98199209-c1845-confirm.json`
(`c57e6f3e…5553e`) with `ltr_tail_loss_weight=1.0`. Maximum repeat was 3.
Candidate loss was `30.6868` in
11.76 seconds; control loss was `26.2332` in 11.93 seconds. Candidate SHA is
`2d9c0294...4baa`; control SHA is `360ba581...cbac`. Both are local explicit
no-sync artifacts and are never reusable, promotable, syncable, or shippable.

The treatment changes no quality or work metric and is 1.59% slower at p50.
The original c1830 positive is therefore seed-sensitive rather than a robust
learned OpenUI improvement. Honest ship gates still fail on smoke `n=3<20`,
MPR, component recall, exact AST/canonical metrics, and all missing production
suites. Lean is `not_applicable:confirmation_rejected`; no promotion is open.

The next screening hypothesis targets rare decision exposure at the same model
size and loss: `exposure_targeted` versus `capacity_aware` sampling. This tests
the leading model-side blocker—insufficient coverage—without recycling the
rejected tail objective or increasing capacity.

Machine evidence:
[`autotrain-cycle-1845-capacity-tail-valid-confirmation-null.json`](autotrain-cycle-1845-capacity-tail-valid-confirmation-null.json).
