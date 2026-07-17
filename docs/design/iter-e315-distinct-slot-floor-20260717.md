# E315 distinct-slot auto content floor — 2026-07-17

E315 fixes a generalized decoder bug found in E314 error analysis.
`decode_min_content=-1` promised one content-bearing component per distinct
declared slot, but split every slot at its first dot. Namespaced contracts such
as `:held.form.title`, `:held.form.body`, and `:held.form.submit` therefore
collapsed to one `:held` obligation and allowed completion after one child.

The correction counts distinct declared slots. A regression test covers
same-namespace slots and duplicate slots. E315 re-evaluates the unchanged E314
checkpoint under the otherwise identical honest policy.

| Suite | Fidelity E314→E315 | Structure E314→E315 | Meaningful E314→E315 | Recall E314→E315 | Reward E314→E315 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Smoke | 0.5278→**1.0000** | 0.4642→0.4492 | 0.3333→0.3333 | 0.1667→0.1667 | 0.2497→**0.3243** |
| Held-out | 0.2800→**1.0000** | 0.3369→**0.3891** | 0→0 | 0→0 | 0→0 |
| Adversarial | 0.5417→**1.0000** | 0.4744→**0.5970** | 0.5→0.5 | 0.375→0.375 | 0.4245→**0.4805** |
| OOD | 0.2583→**1.0000** | 0.3750→**0.4206** | 0→**0.25** | 0→**0.125** | 0→**0.25** |
| Limited RICO | 0.5417→**1.0000** | 0.3104→**0.3322** | 0.6667→0.6667 | 0.3333→0.3333 | 0.5567→**0.6667** |

Parse remains 1.0 across all 19 records. Failures fall 7→5: smoke reward and
OOD meaningful rate now pass. AgentV remains 2/5. The generated programs use
every declared slot, but mostly assign them to generic `TextContent` plus a
fixed tail of other components, so held-out component recall remains zero and
OOD recall 0.125 remains below its 0.20 gate.

**Verdict:** accept the generalized decoder correction. Do not promote the
unchanged checkpoint: component-type selection and held-out composition still
fail. The next lever should condition each forced slot-bearing component on the
slot's semantic name rather than only its ordinal position.
