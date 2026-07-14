<!-- rtk-instructions v2 -->
# RTK — Token-Optimized CLI

**rtk** is a CLI proxy that filters and compresses command outputs, saving 60-90% tokens.

## Rule

Always prefix shell commands with `rtk`:

```bash
# Instead of:              Use:
git status                 rtk git status
git log -10                rtk git log -10
cargo test                 rtk cargo test
docker ps                  rtk docker ps
kubectl get pods           rtk kubectl pods
```

## Meta commands (use directly)

```bash
rtk gain              # Token savings dashboard
rtk gain --history    # Per-command savings history
rtk discover          # Find missed rtk opportunities
rtk proxy <cmd>       # Run raw (no filtering) but track usage
```
<!-- /rtk-instructions -->


<!-- token-efficiency-stack -->
# Token-efficiency stack (project)

Canonical skills: `.agents/skills/` (Claude / Codex / Cursor / GitHub Copilot).

| Tool | Role |
| --- | --- |
| **ponytail** | Minimal code — YAGNI ladder before writing |
| **caveman** | Opt-in terse chat (`/caveman`); code/commits stay normal |
| **headroom** | Compress large tool outputs in-context |
| **rtk** | Compress shell command output (`rtk <cmd>`) |

Always-on for Copilot Chat: prefer minimal diffs (ponytail ladder) and `rtk` for verbose CLI. Activate caveman/headroom skills when needed.
<!-- /token-efficiency-stack -->
