# SLM-232 latent-state rank and causal-use audit

Verdict: **unstable**

Report hash: `b745f23241f294213d8c9b3baff6d40bf1a4d0c8d91be7917d78b57360b617e4`

This bounded CPU audit reuses the rejected SLM-230 scratch checkpoint. It changes no source weights, training default, generation default, or promotion state.

## Pathway and intervention points

```text
z_latent[position] ----[zero_z_latent]---\
ctx_proj(pool(context))-[zero_ctx_proj]----+--> z0 --> F(norm(y+z)) --> z'
position[position] ----[remove_z_position]-/      |                  |
                                                       detach_y_to_z    |
y0 -----------------------------------------------> G(norm(y+z')) --> y'
                                                       detach_z_to_y
```

All cells are evaluation-only functional overrides. The checkpoint state hash is identical before and after the matrix.

## Representation

| depth | effective rank | participation | centered energy | z/y CKA |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2.105443 | 1.776128 | 0.695884 | 0.325026 |
| 1 | 1.976732 | 1.677717 | 5.919628 | 0.945122 |
| 2 | 1.821334 | 1.521484 | 19.445855 | 0.977853 |
| 3 | 1.808104 | 1.512703 | 41.630090 | 0.983756 |
| 4 | 1.828133 | 1.536134 | 72.821166 | 0.985599 |

The z0 rank after removing the pooled-context and position terms is `0.000000`. This is a four-record descriptive estimate; token positions are not treated as independent groups.

## Causal cells

| cell | full-vocab KL | top-1 change | target accuracy |
| --- | ---: | ---: | ---: |
| `none` | 0.000000 | 0.000000 | 0.000000 |
| `zero_z0` | 1.315897 | 0.189222 | 0.000000 |
| `mean_z0` | 1.243782 | 0.189222 | 0.000000 |
| `shuffle_z_across_examples` | 0.007266 | 0.016393 | 0.000000 |
| `swap_z_matched` | 0.007266 | 0.016393 | 0.000000 |
| `zero_ctx_proj` | 0.015385 | 0.035228 | 0.000000 |
| `zero_z_latent` | 0.000001 | 0.000000 | 0.000000 |
| `remove_z_position` | 1.178461 | 0.189222 | 0.000000 |
| `detach_z_to_y` | 1.727525 | 0.189222 | 0.000000 |
| `detach_y_to_z` | 1.214687 | 0.404604 | 0.000000 |
| `y_only_repeated_control` | 0.977756 | 0.189222 | 0.000000 |
| `random_norm_matched` | 1.147272 | 0.170387 | 0.000000 |

The largest targeted full-vocabulary effect is `detach_z_to_y`. This is sensitivity, not evidence of useful or legal reasoning: exact legal/protected candidate sets are unavailable and remain censored.

## Outcome join and disposition

- SLM-230: `stagnant` (`7e9534057fa22bd041366f62cd1ba24e02c97b3b3095b4d726601f17063a8cbc`).
- SLM-231: `expansive_unstable` (`06bfd4ecd68f936b4c548b2a7a5fad39bbe56e0bdf76dc7308477635bb2e6913`).
- AgentV: `{"durationMs": 28, "executionErrors": 0, "failed": 0, "meanScore": 1, "passed": 5, "total": 5}`.

The current z is measurably variable and its removal can alter full-vocabulary logits, but the authoritative disposition is **unstable**: joined recurrence dynamics are expansive, bounded outputs remain vacuous, and no provenance-compatible legal/protected outcome artifact exists. RSC2/RSC3 must not treat this checkpoint as evidence for a causally useful reasoning workspace. Diagnostic replication and architecture repair remain allowed.

## Reproduction

```bash
timeout 170s env PYTHONPATH=src .venv/bin/python -m scripts.run_slm232_latent_state_use --check
```
