# E363–E364 Card topology diagnosis — 2026-07-17

E363 tests whether E360's changed component-plan head contains Card evidence
that the retained plan-off policy hides. On RICO rows 0–15, plan weight 2
applies 32 times and changes 19 internal component choices relative to the
unbiased logits. Nevertheless, all 16 final programs are byte-for-byte
identical to E362 and contain zero Cards. The numeric scoreboard is therefore
unchanged. The diagnostic completes in 26.4 seconds and AgentV correctly
reports 0/1 for 16/1500 coverage.

E364 removes the slot-component bias while retaining plan weight 2. Two final
programs change, both by swapping direct `TextContent`/`Button` choices.
Meaningful rate falls from 1.0 to 0.9375 and component recall from 0.5208 to
0.4792; zero programs contain a Card. The run completes in 17.0 seconds and
AgentV remains 0/1.

These controls falsify the simple “auxiliary bias ordering hides Card” theory.
The constrained skeleton creates direct placeholder-bearing content children
under the root Stack. It has no train-conditioned operation that inserts an
intermediary container binder, so neither component-plan nor slot-component
weights can express the Card hierarchy learned by E360.

Both commands used an external interrupt at 290 seconds and hard kill by 300
seconds.

**Verdict:** reject both decode-weight variants. The next correction must
generalize topology creation for learned intermediary containers; another
Card-frequency or component-bias sweep is not justified.
