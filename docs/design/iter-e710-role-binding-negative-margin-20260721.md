# E710 — role-binding negative margin

Date: 2026-07-21  
Status: rejected after full-suite regression

E710 applies typed semantic-role binding protection at the final score boundary.
An unused slot assigned to the active component/property retains its positive
margin; wrong-owner and already-used slots are floored below the best legal
non-slot value. This is a general score-ordering constraint, not a fixture or
surface-name special case.

The first OOD diagnostic (`n=4`) raised binding-aware strict meaningfulness from
0.50 to 0.75 by preventing `ImageGallery` from stealing already-owned Callout and
Image slots. The required five-suite replay rejected that broad constraint:
held-out strict fell 1.0→0.8, fidelity fell 1.0→0.96, and reward fell
0.9658→0.9538. A narrower component-only diagnostic did not recover the loss and
introduced duplicate-slot spam; held-out stayed 0.8 and OOD returned to 0.5.

The code lever is fully reverted. The durable result is negative: downstream
score suppression cannot safely compensate for a component whose required string
properties exceed its assigned role capacity. The next experiment must fix that
semantic-plan selection boundary before decode begins.

No checkpoint was created, synced, or promoted. The full replay's AgentV result is
0/5 and the Rico length budget still fails, so this is not ship evidence.

Evidence: [JSON](iter-e710-role-binding-negative-margin-20260721.json).
