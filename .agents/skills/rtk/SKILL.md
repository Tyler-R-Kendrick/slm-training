---
name: rtk
description: >
  Rust Token Killer (rtk-ai/rtk) — compress shell/tool command output by 60–90%
  before it enters the LLM context. Use for git status/diff/log, pytest, ruff,
  ls, docker, kubectl, and other verbose CLI output. Prefer `rtk <cmd>` over raw
  commands when rtk is installed. Triggers: "use rtk", "token killer", "compress
  shell output", "rtk gain", or any long CLI dump about to be read.
---

# RTK — Rust Token Killer

Read and follow project [`RTK.md`](../../../RTK.md).

1. Check `rtk --version` and `rtk gain` (savings stats = correct binary).
2. Prefix verbose shell commands with `rtk`.
3. Do **not** use the wrong crates.io “Rust Type Kit” package.
4. Built-in Read/Grep/Glob tools may bypass hooks — for huge logs/files prefer
   `rtk read`, `rtk grep`, `rtk ls`, or shell + `rtk`.
