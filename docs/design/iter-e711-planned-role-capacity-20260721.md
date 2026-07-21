# E711 — planned role capacity before joint carriers

Date: 2026-07-21  
Status: completed five-suite retained scratch improvement; not ship

E711 fixes semantic-plan ordering upstream of decode. Before adding an unprompted
joint role carrier, the planner now checks whether the prompt-planned component
families can cover the entire role group with distinct schema capacity. When they
can, it preserves those planned families and lets the existing per-role assignment
bind them instead of inventing another component.

The first OOD diagnostic that screened required content properties merely replaced
an unnecessary `CheckBoxItem` with `Slider` and did not improve strictness. A second
attempt failed before evaluation after an adjacent `@staticmethod` decorator was
accidentally removed; it produced no metrics and is not evidence. The corrected
planned-capacity rule removes both extra carriers. The final five-suite `n=19`
replay leaves every non-OOD quality metric unchanged from E709. OOD binding-aware
strict meaningfulness rises 0.5→1.0, structure 0.7073→0.7823, node F1
0.8056→0.85, and edge F1 0.6951→0.7486; reward shifts 0.979→0.973. All OOD
semantic reason-code lists are empty.

Retain v183. This remains scratch-matrix evidence, not a ship evaluation: AgentV
is 0/5 and Rico's p95 target length is 190 against the 160-token canvas. No
checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e711-planned-role-capacity-20260721.json).
