"""G4 reasoning harness: checkable-answer sketch-of-thought bench."""

from slm_training.harnesses.reasoning.abstract_cot_warmup import (
    AbstractWarmupCampaignV1,
    IterationRecord,
    WarmupResult,
    run_abstract_cot_warmup,
)
from slm_training.harnesses.reasoning.bench import (
    ANSWER_TOLERANCE,
    ReasoningBenchConfig,
    run_reasoning_bench,
    score_direct_output,
    score_sketch_output,
)
from slm_training.harnesses.reasoning.profiles import (
    DEFAULT_PROFILE_ID,
    REVMATH_PROFILE_ID,
    ReasoningProfileError,
    ReasoningProfileV1,
    assert_default_unchanged,
    get_profile,
    list_profile_ids,
    resolve_profile,
)

__all__ = [
    "ANSWER_TOLERANCE",
    "AbstractWarmupCampaignV1",
    "DEFAULT_PROFILE_ID",
    "IterationRecord",
    "REVMATH_PROFILE_ID",
    "ReasoningBenchConfig",
    "ReasoningProfileError",
    "ReasoningProfileV1",
    "WarmupResult",
    "assert_default_unchanged",
    "get_profile",
    "list_profile_ids",
    "resolve_profile",
    "run_abstract_cot_warmup",
    "run_reasoning_bench",
    "score_direct_output",
    "score_sketch_output",
]
