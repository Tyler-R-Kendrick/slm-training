# E899: focused typed-role corpus

E899 is a strict two-row corpus containing only the canonical human-curated
`train_tabs_01` and `train_card_stack_01` fixtures already admitted by E897.
It exists to guarantee typed-role exposure in a short continuation while the
broader E851 corpus is supplied through replay; it does not duplicate a new
producer or weaken any gate.

The build admitted 2/2 rows at Silver verification and quality 1.0. There were
zero source, normalization, verification, quality, deduplication,
decontamination, exposure-cap, or sanitizer-fallback failures. The rejection
ledger is empty, and synthesis feedback has no warnings, recommendations, or
experiment candidates. Content fingerprint: `d037a43b…2e31d6`.

The version stamp is dirty solely because unrelated untracked SLM189 result
files remain in the shared worktree; they were not read, modified, staged, or
included. Retain E899 only as a focused training input. Data admission alone is
not model or ship evidence.
