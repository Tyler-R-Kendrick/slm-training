# RTK — Rust Token Killer

CLI proxy that compresses shell output before it reaches the agent context
(typically 60–90% fewer tokens). Source: https://github.com/rtk-ai/rtk

## Install (once per machine)

```bash
curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
rtk gain   # must show savings stats (not "command not found")
```

Verify it is **Rust Token Killer** (`rtk-ai/rtk`), not Rust Type Kit.

## Rule

Prefer `rtk` for verbose shell commands:

```bash
rtk git status
rtk git log -n 20
rtk git diff
rtk pytest -q
rtk ruff check .
rtk ls .
```

Meta:

```bash
rtk gain              # savings dashboard
rtk gain --history
rtk proxy <cmd>       # raw passthrough (still tracked)
```

## Agent hooks (optional, per machine)

```bash
rtk init -g --auto-patch                 # Claude Code
rtk init -g --agent cursor --auto-patch  # Cursor (global hooks)
rtk init --codex                         # project: RTK.md + AGENTS.md ref
rtk init --copilot                       # project: .github/copilot-instructions.md + hooks
```

Restart the agent after hook install.
