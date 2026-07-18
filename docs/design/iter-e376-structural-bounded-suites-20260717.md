# E376 structural bounded suites — 2026-07-17

E376 applies the retained E375 structural decoder policy to the four complete
bounded suites on the unchanged local E368 checkpoint:
`decode_min_content=-1`, component-plan weight 2, slot-component weight 8, and
a 320-token choice canvas.

All 16 records parse and preserve every visible placeholder. Smoke,
held-out, adversarial, and OOD score structure
0.5600/0.5136/0.5546/0.5114 and component recall
0.5000/0.4333/0.6667/0.6042. AgentV passes 4/4 with zero execution errors.
The overall ship gate remains false because `rico_held` is absent.

The command completed in under 27 seconds with an external 290-second
interrupt and forced kill ten seconds later. No checkpoint was created.

**Verdict:** retain the E375 policy for broader RICO evaluation. This is a
bounded-suite pass, not a production ship result.
