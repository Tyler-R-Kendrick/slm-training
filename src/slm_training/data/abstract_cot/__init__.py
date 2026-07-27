"""AP-018: bottleneck warm-up records bridging SemanticPlanV1 to AbstractPlanV1.

See :mod:`slm_training.data.abstract_cot.warmup` for the record schema and
builder. This package is data-build only: it derives training-only warm-up
records from already-verified ``ExampleRecord`` rows and the existing
``OpenUISemanticPlanExtractor`` / ``AbstractPlanV1`` contracts. No production
decode/serving path imports this package.
"""

from __future__ import annotations

from slm_training.data.abstract_cot.warmup import (
    AbstractCotWarmupConfig,
    AbstractCotWarmupCorpusV1,
    AbstractCotWarmupRecordV1,
    SegmentLayoutV1,
    WarmupExclusionV1,
    build_abstract_cot_warmup_corpus,
    verbalize_semantic_plan,
)

__all__ = [
    "AbstractCotWarmupConfig",
    "AbstractCotWarmupCorpusV1",
    "AbstractCotWarmupRecordV1",
    "SegmentLayoutV1",
    "WarmupExclusionV1",
    "build_abstract_cot_warmup_corpus",
    "verbalize_semantic_plan",
]
