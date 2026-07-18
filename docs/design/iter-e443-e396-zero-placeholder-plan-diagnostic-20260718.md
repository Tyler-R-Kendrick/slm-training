# E443 E396 zero-placeholder plan diagnostic — 2026-07-18

E443 localizes E396's only full-RICO placeholder-fidelity miss. The gold
record at row 1408 requests two `DatePicker` controls, has no placeholders,
and scored zero type recall in E441. This diagnostic changes only
component-plan decode weight from 2 to 0.

| Row | n | Plan weight | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1408 | 1/1500 | 0 | 1.0 | 0.0 | 0.0 | 0.4000 | 0.0 | 0.0 |

The prediction exactly matches E441:

```openui
root = Stack([v0])
v0 = TextContent(":hero.title")
```

The run used the unchanged E396 SHA, CPU, local HF context, 320-token grammar
LTR, automatic content floor, slot-component weight 8, honest constrained
slot contract, eight generation steps, and three attempts. It decoded in
3.25 seconds with no fallback or decode timeout. The external process cap was
290 seconds with a ten-second forced kill. AgentV is 0/5 with zero execution
errors because this one-row diagnostic omits four bounded suites and the full
RICO suite.

**Verdict:** plan weight zero does not change the failure, so the trained
component-plan head is not causal. Do not expand the scalar sweep. The next
lever should address explicit prompt-role coverage, especially components
without placeholder slots, without using hidden gold structure.
