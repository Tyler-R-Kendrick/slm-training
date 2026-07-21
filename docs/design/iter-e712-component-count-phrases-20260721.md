# E712 — component counts across descriptive phrases

Date: 2026-07-21  
Status: completed five-suite retained scratch improvement; not ship

E712 generalizes prompt component-count parsing. The shared semantic-plan parser
now carries the closest explicit count through any number of descriptive modifiers,
while stopping at conjunctions, prepositions, or another component family. Thus
`three equally important action buttons` requires three Buttons, but `three cards
and a button` still requires one Button and three Cards.

On the frozen E620 checkpoint, the targeted adversarial record changes from one
Button with one of three required placeholders to three Buttons with complete
placeholder coverage. Across the matched five-suite `n=19` replay, adversarial
binding-aware strict meaningfulness rises 0.25→0.50, normalized placeholder
fidelity 0.8333→1.0, structure 0.76575→0.88325, node F1 0.8318→0.9152, and edge
F1 0.7169→0.8419. Smoke, held-out, OOD, and Rico quality values match E711.

Retain model v184 and meaningfulness metric 2.10.0. This remains scratch-matrix
evidence, not a ship evaluation: AgentV is 0/5 and Rico's p95 target length is
190 against the 160-token canvas. No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e712-component-count-phrases-20260721.json).
