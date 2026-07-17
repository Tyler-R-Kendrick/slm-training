"""Bounded arity analyzer implementing a SupportOracle protocol."""

from __future__ import annotations

from typing import Any

from slm_training.dsl.analysis.arity.explorer import explore
from slm_training.dsl.analysis.arity.report import ArityReport, ContinuationSummary
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

    def analyze(self, seed_sources: list[str] | None = None) -> ArityReport:
        """Run bounded exploration and emit a versioned arity report."""
        summaries = explore(self._backend, self.bounds, seed_sources)
        self._summaries = tuple(summaries)
        return ArityReport.from_summaries(
            frame_id=f"{self.dsl}/{self.bounds.max_ast_nodes}",
            bounds=self.bounds,
            summaries=self._summaries,
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
