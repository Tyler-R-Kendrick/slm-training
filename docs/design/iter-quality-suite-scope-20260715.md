# Suite-scoped AgentV publication (2026-07-15)

The quality runner accepted `--suites smoke`, but AgentV publication still
constructed cases for every canonical suite. Full matrices must retain that
fail-closed behavior; subset evaluations now publish only the suites actually
requested, avoiding misleading missing-suite cases and unnecessary AgentV
work.

The bounded smoke rerun remains incomplete because model decoding terminated
before a final summary. Its progress marker is durable, and no quality claim is
made from it.
