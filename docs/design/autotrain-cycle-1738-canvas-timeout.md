# Autotrain c1738: exact canvas replay remains incomplete

**Verdict:** the frozen c1737 replay finalized both three-record smoke suites,
but each arm timed out on `smoke_hero_01`. The two completed records have
scoreable partial-suite metrics; the comparison does not. Canvas is neither
promoted nor rejected because the authoritative suite measurement is
incomplete.

## Result matrix

| Arm | Params | Complete | Timeout | Parse | Meaning | Structure | Binder F1 | p50 incl. incomplete | Forwards | States | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 22,379.94 ms | 107 | 297,345 | incomplete |
| compact canvas | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 22,530.74 ms | 107 | 297,158 | incomplete |

Rates above exclude the timed-out record and describe only the two completed
documents. They must not be read as full-suite quality. Both AgentV bundles
fail the runtime timeout criterion; ship gates are blocked.

## Signals and next run

- The candidate saved 187 completion states and 332 parser forks, but the
  observed delta is too small and the authoritative endpoint is incomplete.
- `smoke_hero_01` is the shared blocker, so the leading action remains a
  canonical model-build runtime repair followed by the identical frozen replay.
- Harness SDLC classification now requires every authoritative suite
  scoreboard to report zero incomplete/timeout records before a positive
  delivery can be emitted.
- Lean is integrated as a required validation surface for the repair. Formal
  promotion evidence is not applicable to this incomplete screening result.

Machine-readable evidence is in
[`autotrain-cycle-1738-canvas-timeout.json`](autotrain-cycle-1738-canvas-timeout.json).

