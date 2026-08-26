# Continuous cycle `continuous-loop-20260826-continuous-openui-local-8c0b60dd-c569`

- loop/cycle: `continuous-openui-local` / `569`
- role/intent: `screening` / `screening`
- recipe: local CPU fixture TwoTower, 20 steps, two frozen 70-second arms
- primary metric: `smoke.eval_nll`
- measurement complete: **False**
- result: pre-arm infrastructure failure; no arm launched and no scoreboard or AgentV bundle exists
- failure: symmetric decision-arm preparation had `151.4366983029995s` remaining but required `155.25s` (`3.813301697000495s` short)

Import profiling identified command-specific researcher/hypothesizer evaluation,
mixture materialization, and verified-metric modules on every autoresearch
subprocess startup. Deferring those imports at their existing command handlers
reduced `python -m scripts.autoresearch --help` from 4.46s to 0.25s. The cycle
starts three autoresearch subprocesses before its arm check, so the measured
reduction projects 12.63s of recovered pre-arm budget. The profile ran on the
documented dirty repair tree; a clean exact-head live cycle is the acceptance
check. No gate, arm budget, model capacity, or constrained-decode invariant
changed.

This is failed fixture infrastructure and a local profile, not an evaluation or ship claim.
