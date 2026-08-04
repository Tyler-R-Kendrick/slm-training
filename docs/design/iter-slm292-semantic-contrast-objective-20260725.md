# SLM-292 semantic-contrast objective — local wiring control

Durable payload: [`iter-slm292-semantic-contrast-objective-20260725.json`](iter-slm292-semantic-contrast-objective-20260725.json).

On 2026-07-25, the canonical TwoTower training loop was exercised locally on
two immutable, declared-`train` hard-valid pairs. The default-off configuration
produced zero contrast objective and bit-exact legacy loss. With weight `0.25`
and margin `1.0`, the pair branch emitted complete-pair, family, positive/negative
NLL, score-distance, and violation telemetry.

| Mode | Pairs | Result |
| --- | ---: | --- |
| disabled | 0 | objective `0.0`; legacy loss bit-exact |
| enabled | 2 | margin loss `0.7715`; score distance `0.2285` |

Recipe: CPU, scratch context/denoiser, one seed, no optimizer step, no checkpoint,
no decode/evaluation suite, and no human-rating gate. Declared held-out/OOD corpus
rows were skipped rather than relabeled into training.

This is wiring evidence only—not a quality, ship, or promotion claim. The
preregistered matched-token, three-seed meaningful-parse/binder-reference study
remains required before promotion; this change neither weakens its gates nor uses
external replay or Hugging Face compute.

## Continuous-harness compatibility (2026-08-02)

The canonical continuous harness can now run SLM-292 as a size-matched fixture
screen. Control and treatment load the same immutable train-split pairs with
the same margin and pair fraction; only `semantic_contrast_loss_weight` changes
from `0` to `0.25`. This removes the prior pair-exposure confound.

The immutable published corpus contains historical external marker spellings.
The read-only pair loader now projects both sides through one shared request-local
opaque ordinal codec before model use. Dropped or duplicated negative-side
symbols retain their positive-side ordinal, so projection enforces output
contract v2 without erasing the semantic contrast. The source corpus bytes and
its data/evaluation claim remain unchanged.
