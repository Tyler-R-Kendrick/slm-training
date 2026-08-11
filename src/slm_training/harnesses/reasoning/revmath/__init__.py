"""reasoning/revmath typed contracts + deterministic runner (HARN-02/03).

Schemas: HARN-02. Runner / replay / report: HARN-03.
"""

from __future__ import annotations

from slm_training.harnesses.reasoning.revmath.plugins import (
    HermeticForwardPlugin,
    PluginCheckEvidence,
    RevmathCheckPlan,
    RevmathTaskPlugin,
    default_plugin_registry,
    resolve_plugin,
)
from slm_training.harnesses.reasoning.revmath.report import build_revmath_report
from slm_training.harnesses.reasoning.revmath.replay import (
    assert_revmath_replay_integrity,
    build_revmath_replay_bundle,
)
from slm_training.harnesses.reasoning.revmath.runner import RevmathRunRecord, run_revmath_task
from slm_training.harnesses.reasoning.revmath.schemas import (
    REVMATH_MUTABLE_REPAIR_KNOBS,
    REVMATH_SCHEMA_VERSIONS,
    AxisCertificateRefV1,
    RevmathBaseTheoryV1,
    RevmathBudgetV1,
    RevmathCampaignBindingV1,
    RevmathCorpusEntryV1,
    RevmathCorpusIdentityV1,
    RevmathFourAxisAnalysisV1,
    RevmathProofArtifactRefV1,
    RevmathPropositionIdentityV1,
    RevmathRepairKnobChangeV1,
    RevmathRepairRecordV1,
    RevmathReportV1,
    RevmathResultV1,
    RevmathSchemaError,
    RevmathTaskKind,
    RevmathTaskV1,
    RevmathVerifierJudgeIdentityV1,
    TheoremDirection,
    assert_repair_preserves_identity,
    canonical_hash,
    export_json_schemas,
    parse_revmath_corpus_entry,
    parse_revmath_repair,
    parse_revmath_report,
    parse_revmath_result,
    parse_revmath_task,
)

__all__ = [
    "REVMATH_MUTABLE_REPAIR_KNOBS",
    "REVMATH_SCHEMA_VERSIONS",
    "AxisCertificateRefV1",
    "HermeticForwardPlugin",
    "PluginCheckEvidence",
    "RevmathBaseTheoryV1",
    "RevmathBudgetV1",
    "RevmathCampaignBindingV1",
    "RevmathCheckPlan",
    "RevmathCorpusEntryV1",
    "RevmathCorpusIdentityV1",
    "RevmathFourAxisAnalysisV1",
    "RevmathProofArtifactRefV1",
    "RevmathPropositionIdentityV1",
    "RevmathRepairKnobChangeV1",
    "RevmathRepairRecordV1",
    "RevmathReportV1",
    "RevmathResultV1",
    "RevmathRunRecord",
    "RevmathSchemaError",
    "RevmathTaskKind",
    "RevmathTaskPlugin",
    "RevmathTaskV1",
    "RevmathVerifierJudgeIdentityV1",
    "TheoremDirection",
    "assert_repair_preserves_identity",
    "assert_revmath_replay_integrity",
    "build_revmath_replay_bundle",
    "build_revmath_report",
    "canonical_hash",
    "default_plugin_registry",
    "export_json_schemas",
    "parse_revmath_corpus_entry",
    "parse_revmath_repair",
    "parse_revmath_report",
    "parse_revmath_result",
    "parse_revmath_task",
    "resolve_plugin",
    "run_revmath_task",
]
