# Autotrain c4: component-plan efficiency win, quality held, pending confirmation

**Verdict:** positive (efficiency, quality held) — screening only, not yet
confirmed or stacked. Both arms (1,755,760 params, 21 steps) are exactly tied
on quality: parse `1.0`, meaningful-program rate `.3333`, structure `.4167`,
binder F1 `.9524`, recall `.25`, reward `.936` — identical to four decimal
places. The `component-plan` candidate is `712.51` ms faster p50 (`6496.64`
vs `7209.15`), an `mpr_per_ms` efficiency gain of `10.97%` against the
`5%` minimum, with quality held (`parse=1.0`, `mpr=.3333` at the `~1/3`
floor). SDLC Phase A classifies this **POSITIVE**.

The driver intentionally does **not** open a stack layer for this cycle
(`stack_layer=false`, `action=positive_no_tracked_delta_skip_stack`): a
single-seed screening positive queues a champion candidate
(`champ-continuous-openui-local-4-2694d77fc99953e4`) but promotion formal
preflight stays locked until a fresh-seed confirmation run reproduces the
same result with the exact size-matched recipe. AgentV bundles are complete;
ship gates fail honestly on fixture volume (`n=3`) and quality floors
(`meaningful_program_rate .3333 < .66`, others below floor) — expected on a
3-sample smoke fixture, not a failure.

Checkpoints (`4c3b3f18...c24bc` control, `b317fe46...a8429` component-plan)
are local, explicit no-sync, and not reusable, promotable, or ship until
confirmed. Lean is `not_applicable:screening`.

Next: run the fresh-seed confirmation
(`c20260803-continuous-openui-local-8c0b60dd-c4-component-plan-fresh-confirmation`)
with the identical size-matched treatment/control recipe. Only a reproduced
positive there earns a stack layer.

Machine evidence:
[`autotrain-cycle-c4-component-plan-efficiency-win.json`](autotrain-cycle-c4-component-plan-efficiency-win.json).
