"""Bounded arity analyzer implementing a SupportOracle protocol."""

from __future__ import annotations

from typing import Any

from slm_training.dsl.analysis.arity.explorer import explore
from slm_training.dsl.analysis.arity.report import ArityReport, CodingMetadata, ContinuationSummary
from slm_training.dsl.analysis.arity.suggest import smallest_feasible_alphabet
from slm_training.dsl.analysis.arity.types import (
    AnalysisBounds,
    StateSignature,
    SupportOracle,
    SupportQuery,
    SupportResult,
    SupportVerdict,
)
from slm_training.dsl.grammar.backends import get_backend


class ArityAnalyzer(SupportOracle):
    """Torch-free analyzer for bounded grammar arity."""

    def __init__(self, dsl: str, bounds: AnalysisBounds | None = None) -> None:
        self.dsl = dsl
        self.bounds = bounds or AnalysisBounds(max_ast_nodes=128)
        self._backend = get_backend(dsl)
        self._summaries: tuple[ContinuationSummary, ...] | None = None

    def analyze(
        self,
        seed_sources: list[str] | None = None,
        *,
        include_coding_metadata: bool = False,
    ) -> ArityReport:
        """Run bounded exploration and emit a versioned arity report.

        When ``include_coding_metadata`` is true, attach a CAP0-03 coding-theory
        frame derived from the minimized state count. This makes the report
        self-contained for downstream exact/estimated evidence classification.
        """
        summaries = explore(self._backend, self.bounds, seed_sources)
        self._summaries = tuple(summaries)
        coding_metadata = None
        if include_coding_metadata:
            coding_metadata = _coding_metadata_for_state_count(
                state_count=len({s.state_signature for s in self._summaries}),
                frame_id=f"{self.dsl}/{self.bounds.max_ast_nodes}",
            )
        return ArityReport.from_summaries(
            frame_id=f"{self.dsl}/{self.bounds.max_ast_nodes}",
            bounds=self.bounds,
            summaries=self._summaries,
            coding_metadata=coding_metadata,
        )

    def check(self, state: StateSignature, query: SupportQuery) -> SupportResult:
        """Answer whether a candidate atom is supported by the state.

        This is a conservative local membership check over the state's atoms.
        It never invents legality beyond what has been observed.
        """
        certificate: dict[str, Any] = {
            "state_version": state.version,
            "state_fingerprint": state.fingerprint(),
            "hole_id": query.hole_id,
            "candidate": repr(query.candidate),
        }
        for atom in state.atoms:
            if _contains_atom(atom, query.candidate):
                return SupportResult(
                    verdict=SupportVerdict.SUPPORTED,
                    certificate={**certificate, "matched_atom": repr(atom)},
                )
        return SupportResult(
            verdict=SupportVerdict.UNKNOWN,
            certificate=certificate,
        )


def _coding_metadata_for_state_count(
    *,
    state_count: int,
    frame_id: str,
    dimensions: int = 4,
    min_distance: int = 3,
) -> CodingMetadata:
    """Build CAP0-03 coding metadata for an exact state count.

    Uses the smallest feasible alphabet under the Singleton bound and records
    whether a locally verified construction is available.
    """
    from slm_training.dsl.analysis.arity.coding import (
        build_mds_7_4_2_3,
        build_shortened_ternary_hamming_7_4_3,
        singleton_upper_bound,
        verify_code,
    )

    bound_value = singleton_upper_bound(7, dimensions, min_distance)
    construction: str | None = None
    proof_status = "external_exact_bound"
    source_citation: str | None = None
    utilization: float | None = None

    if state_count <= build_mds_7_4_2_3().__len__():
        code = build_mds_7_4_2_3()
        result = verify_code(code, q=7, n=dimensions, required_size=state_count, required_distance=min_distance)
        if result.ok:
            construction = "mds_7_4_2_3"
            proof_status = "local_verified_construction"
            utilization = state_count / len(code)
    elif state_count <= build_shortened_ternary_hamming_7_4_3().__len__():
        code = build_shortened_ternary_hamming_7_4_3()
        result = verify_code(code, q=3, n=7, required_size=state_count, required_distance=min_distance)
        if result.ok:
            construction = "shortened_ternary_hamming_7_4_3"
            proof_status = "local_verified_construction"
            utilization = state_count / len(code)
    else:
        # A_3(6,3)=38 is the cited external bound for the toy robust argument.
        source_citation = "A_3(6,3)=38 (external exact code table)"

    alphabet_size = smallest_feasible_alphabet(state_count, dimensions)
    feasible = alphabet_size <= 7 and bound_value >= state_count

    return CodingMetadata(
        state_count=state_count,
        dimensions=dimensions,
        alphabet_size=alphabet_size,
        min_distance=min_distance,
        feasible=feasible,
        status="feasible" if feasible else "infeasible",
        bound_name="singleton_upper_bound",
        bound_value=bound_value,
        construction=construction,
        proof_status=proof_status,
        source_citation=source_citation,
        utilization=utilization,
        scale_mode=None,
        ecoc_width=None,
        margin_planes=None,
    )


def _contains_atom(haystack: StateSignature | object, needle: object) -> bool:
    """Recursive atom membership check used by the conservative oracle."""
    from slm_training.dsl.analysis.arity.types import StateAtom

    if haystack == needle:
        return True
    if isinstance(haystack, StateAtom):
        if haystack.kind == "component":
            _, props = haystack.payload
            for _key, value in props:
                if _contains_atom(value, needle):
                    return True
        if haystack.kind == "list":
            for item in haystack.payload:
                if _contains_atom(item, needle):
                    return True
    return False
