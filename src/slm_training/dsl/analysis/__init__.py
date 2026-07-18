"""Static DSL analysis passes (Torch-free).

Owner package for offline, deterministic analyses over the frozen DSL grammars —
starting with :mod:`slm_training.dsl.analysis.arity` (CAP0-02 exact arity /
adaptive-precision capacity). Nothing here imports torch or model runtime code.
"""

from __future__ import annotations
