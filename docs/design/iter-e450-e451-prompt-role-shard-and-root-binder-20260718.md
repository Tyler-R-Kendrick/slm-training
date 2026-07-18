# E450–E451 prompt-role shard and root-binder repair — 2026-07-18

E450 broadens E449's prompt-role constraint from one row to E441's matched
96-row RICO shard (rows 1344–1439). It uses the unchanged E396 checkpoint and
the same CPU/HF-local/grammar-LTR policy as E441, adding only the opt-in
prompt-role constraint and disabling an unused unconstrained fallback.

| Run | n | Meaningful | Fidelity | Structure | Type recall | Reward | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E441 shard 14 | 96 | 0.9792 | 0.9896 | 0.6426 | 0.8681 | 0.9775 | 2 low recall |
| E450 | 96 | 1.0000 | 1.0000 | 0.8537 | 1.0000 | 0.9959 | none |

E450 completes normally in about 217 seconds under the external 290-second
cap. It has no fallback or decode timeout and emits AgentEvals JSONL plus an
AgentV bundle. AgentV is 0/5 with zero execution errors because the diagnostic
contains one partial suite, not the five complete ship-gate suites.

Ninety-four of 96 predictions change. Structure improves on 87 rows and
regresses on one; type recall improves on 26 and regresses on none. The sole
structure regression, `rico_hf_test_3283`, exposes a corpus defect:

- prompt/meta: six Cards;
- E334 gold: one root Card;
- E450: six Cards, honoring the visible prompt.

The canonical RICO converter allowed an element resource slug `root` to collide
with the reserved program binder. It emitted both `root = Stack(...)` and a
child `root = Card(...)`; canonicalization retained the child and discarded
the intended six-card root graph.

E451 reserves `root` during child-name allocation, adds a regression test, and
rebuilds the same cached-HF evaluation corpus as
`outputs/data/eval/e451_root_binder_v1`. The capped build completes normally
in 4.6 seconds with 1,500 RICO rows, 158 leakage rejections, and zero conversion
errors.

| Artifact | SHA-256 |
| --- | --- |
| E451 manifest | `edd78b481b923666b232b9ef8defa111b08f0c5c209fd628a3904ddb96f178f8` |
| E451 RICO records | `35ecf5596933cb06e04184c99733fa56954355e9534ad9a24d060412d8f46432` |

A full prompt/gold count audit on E451 finds no overclaiming prompt contracts:
Card 1365/1365, Button 177/177, DatePicker 15/15, SwitchItem 46/46, Input
21/21, ImageBlock 212/212, Slider 3/3, and CheckBoxItem 1/1 are exact.
TextContent is exact on 119 rows and a lower bound on 902 because Cards
introduce title/body text children.

**Verdict:** E450 is strong regression evidence for the prompt-role lever, but
its one loss correctly identified stale invalid gold. E451 repairs the
canonical data path rather than weakening the visible prompt contract. Re-run
both baseline and candidate on the repaired 96-row shard before expanding.
