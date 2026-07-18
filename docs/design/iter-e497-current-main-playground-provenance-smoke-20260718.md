# E497 current-main playground provenance smoke — 2026-07-18

E497 validates the new evaluation provenance envelope and establishes a
loadable current-main baseline with the committed `playground_demo` fixture.
The envelope records exact code revision
`bccf2355db8fc4487375ad68a95a7f5220dc770a`, `code_dirty: false`, checkpoint
SHA `df517cf9b0071deda66e53ddc082159f9c2f74f909f24baf07807c58348f8504`,
and the pinned clone import root.

Recipe: CPU scratch context, complete smoke n=3, honest constrained slot
contracts, eight generation steps, three attempts, no fallback, 45-second
per-record decode timeout, and a 180-second total cap. The run completed
normally in 113.9 seconds.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 0.0 | 0.0 | 0.0 | 0.2203 | 0.1667 | 0.0 |

AgentV passes 0/5 with zero execution errors. One record reaches its 45-second
decode timeout; no fallback runs. This is expected fixture-grade negative
evidence, not a regression from a ship candidate.

**Verdict:** accept the provenance mechanism and reject the checkpoint as a
quality candidate. E496 separately shows that the durable E396 diagnostic
cannot load on current `main`.
