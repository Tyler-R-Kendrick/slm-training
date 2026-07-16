# E48 persistent-DFA resynchronization — 2026-07-15

The first RHS transition failed because the persistent grammar engine disagreed
with a fresh DFA at the decoded prefix. The repair loop now resets the engine
to that exact prefix and retries the same logits before padding.

On the unchanged E48 checkpoint, this moved generation substantially forward:

| metric | result |
| --- | ---: |
| parse | 0/3 |
| structural similarity | 0.3744 |
| dead ends | 0 |
| p50 latency | 35,583 ms |

Predictions now reach `root = Stack([...])` and subsequent bindings instead of
stopping at `root =`. Later native-symbol and newline commits produce noisy,
invalid programs, so the checkpoint is still rejected. The resynchronization
is retained as a real decoder fix; the next intervention targets repeated
newline commits in incomplete list/literal states.

This is scratch smoke evidence, not a ship claim.
