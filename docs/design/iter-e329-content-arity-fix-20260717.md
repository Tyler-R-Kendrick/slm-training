# E329 corrected content-arity ablation — 2026-07-17

E329 evaluates the unchanged E326 checkpoint with schema-derived multi-slot
score averaging and no ordered span prior. It exactly reproduces E328:
smoke remains at recall 0.3333, while held-out recall/structure regress from
0.40/0.5458 to 0.30/0.4758. All other headline suite metrics are unchanged;
AgentV remains 4/5.

This isolates the regression to content-arity averaging and shows that E328's
span prior had no measurable effect.

**Verdict:** retain the truthful `ChoiceTokenizer.slot_content_count` API, but
keep multi-slot score averaging behind explicit configuration. Existing
checkpoints retain one-slot scoring by default, preserving E326 as strongest
scratch. Do not promote or claim ship.
