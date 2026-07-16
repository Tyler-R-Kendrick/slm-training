# E154 — One-attempt constrained decode (2026-07-15)

Changing only `max_attempts` from 3 to 1 removed the timeout on the E147 checkpoint and improved one-record structural similarity from `0.0` to `0.1917` and placeholder validity from `0.0` to `0.4`. Parse remained `0.0`; this is a bounded diagnostic improvement, not a ship result.
