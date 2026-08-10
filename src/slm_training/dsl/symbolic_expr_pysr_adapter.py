"""SRP-011 (SLM-475): optional PySR/SRBench adapter with an isolated,
pinned environment manifest.

Never imports ``pysr``/Julia and is never imported by any core path -- this
module only defines the *translation contract* between the pack's typed
symbolic-expression IR and a plain infix text format external symbolic-
regression tools consume/emit. The actual external call is injected as
``raw_output_provider`` (a plain ``str -> str`` callable supplied by the
caller); core repo tests supply a mocked/golden callable, never a real
subprocess or ``import pysr``. Live execution against a real, separately
installed PySR/SRBench environment is optional and out of scope for this
sandbox (no internet/Julia toolchain available here to install or pin
against honestly) -- the contract below is complete and independently
testable without it, per this issue's own scope line: "Core repo tests use
mocked/golden adapter output; live external execution is optional."

Translation is genuinely bidirectional and reuses every existing owner
rather than re-deriving anything:

* *To* the external tool: :func:`build_adapter_run_input` translates a
  frozen ``SymbolicRegressionProblemV1`` (variable names, its declared
  operator allowlist mapped through :data:`PYSR_OPERATOR_MAP`, row count,
  seed) into a plain, JSON-serializable input record -- no coefficients are
  sent, since PySR performs its own search from ``(X, y)`` data, not from
  one of the pack's own candidate expressions.
* *From* the external tool: :func:`parse_external_expression` parses the
  tool's returned infix text back into the typed
  ``VarRef``/``CoefficientHole``/``OpApply`` IR. The IR's symbol-only
  contract (no free numeric literal ever appears in the structural codec)
  is preserved by mapping every numeric literal in the external text to a
  fresh ``CoefficientHole``, with its value carried alongside in a parallel
  ``coefficients`` tuple -- the same externally-bound-coefficient shape
  ``symbolic_expr_evaluator.evaluate_vectorized`` already accepts. Any
  operator or identifier the pack cannot represent (e.g. PySR's ``^``/
  ``square``) is reported as ``unsupported_operators``, never silently
  approximated (AC: "Record unsupported operators/semantics rather than
  silently approximating them").
* Once parsed, :func:`run_adapter` re-scores the recovered expression
  through the pack's *own* evaluator/fitter/reporter
  (``symbolic_expr_report.build_evidence_record``) -- an external tool's
  self-reported loss is never trusted or persisted directly (AC: "External
  output is reparsed and rescored by the pack's own evaluator/metrics").

Every non-``"ok"`` outcome (the external call raising, a timeout, an
unsupported operator, or a syntax error in the returned text) is reported
as a typed, non-``"ok"`` :class:`AdapterRunResultV1` with ``evidence=None``
-- never as a benchmark loss (AC: "Adapter failure produces
external-blocked/incomplete evidence, not a benchmark loss").
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

from slm_training.dsl.symbolic_expr_ir import (
    BINARY_OPERATORS,
    UNARY_OPERATORS,
    CoefficientHole,
    OpApply,
    SymbolicExprNode,
    VarRef,
    serialize_expr,
    validate_expr_budget,
)
from slm_training.dsl.symbolic_expr_report import ExpressionEvidenceV1, build_evidence_record

if TYPE_CHECKING:
    from slm_training.dsl.symbolic_regression_pack import SymbolicRegressionProblemV1

ADAPTER_SCHEMA_VERSION = "v1"
MANIFEST_SCHEMA_VERSION = "v1"
SUPPORTED_TOOLS = frozenset({"pysr", "srbench"})
AdapterStatus = Literal["ok", "external_blocked", "timeout", "operator_mismatch", "parse_error"]
ADAPTER_STATUSES: frozenset[str] = frozenset(
    {"ok", "external_blocked", "timeout", "operator_mismatch", "parse_error"}
)

# Our internal operator name -> the external tool's operator/function name.
# Covers every operator symbolic_expr_ir.ALLOWED_OPERATORS declares; a 12th
# operator added to the pack without a corresponding entry here must fail
# closed (see build_adapter_run_input), never silently drop support.
PYSR_OPERATOR_MAP: dict[str, str] = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "sin": "sin",
    "cos": "cos",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "abs": "abs",
    # PySR has no built-in named unary "neg"; a real run must register it
    # via extra_sympy_mappings/unary_operators=["neg(x) = -x"]. Recorded
    # here as the pinned mapping name so build_adapter_run_input's output
    # is complete, not because PySR supports it out of the box.
    "neg": "neg",
}

_SUPPORTED_FUNCTIONS = frozenset(UNARY_OPERATORS) - {"neg"}
_SUPPORTED_BINARY_TOKENS = frozenset({"+", "-", "*", "/"})


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class AdapterEnvironmentManifestV1:
    """Pinned, self-fingerprinting identity for an isolated external
    environment. Purely descriptive data -- constructing this never
    installs or imports anything."""

    schema_version: str
    tool_name: str
    tool_version: str
    python_version: str
    seed: int
    cpu_budget: int
    memory_budget_mb: int
    time_budget_s: float
    iteration_budget: int

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported adapter environment manifest schema: {self.schema_version!r}")
        if self.tool_name not in SUPPORTED_TOOLS:
            raise ValueError(f"unsupported tool_name: {self.tool_name!r}")
        if self.cpu_budget < 1:
            raise ValueError("cpu_budget must be >= 1")
        if self.memory_budget_mb < 1:
            raise ValueError("memory_budget_mb must be >= 1")
        if self.time_budget_s <= 0:
            raise ValueError("time_budget_s must be > 0")
        if self.iteration_budget < 1:
            raise ValueError("iteration_budget must be >= 1")

    def identity(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        return _digest(self.identity())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdapterEnvironmentManifestV1":
        return cls(**payload)


# Pinned, isolated environment identities. Descriptive only: importing this
# module never installs, imports, or shells out to either tool.
DEFAULT_PYSR_MANIFEST = AdapterEnvironmentManifestV1(
    schema_version=MANIFEST_SCHEMA_VERSION,
    tool_name="pysr",
    tool_version="0.19.4",
    python_version="3.11",
    seed=0,
    cpu_budget=4,
    memory_budget_mb=4096,
    time_budget_s=600.0,
    iteration_budget=100,
)
DEFAULT_SRBENCH_MANIFEST = AdapterEnvironmentManifestV1(
    schema_version=MANIFEST_SCHEMA_VERSION,
    tool_name="srbench",
    tool_version="2.0",
    python_version="3.11",
    seed=0,
    cpu_budget=4,
    memory_budget_mb=4096,
    time_budget_s=600.0,
    iteration_budget=100,
)


@dataclass(frozen=True)
class AdapterRunInputV1:
    """Pure, JSON-serializable translation of a frozen problem into external
    runner configuration -- never includes training data itself, only the
    identity/shape the external tool needs to be pointed at it."""

    problem_fingerprint: str
    pack_id: str
    variable_names: tuple[str, ...]
    binary_operators: tuple[str, ...]
    unary_operators: tuple[str, ...]
    n_train_rows: int
    seed: int
    manifest_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_adapter_run_input(
    problem: "SymbolicRegressionProblemV1", manifest: AdapterEnvironmentManifestV1
) -> AdapterRunInputV1:
    """Translate ``problem`` into the external tool's input configuration.

    Fails closed if the pack's declared operator set ever grows an operator
    with no entry in :data:`PYSR_OPERATOR_MAP`, or if a declared variable
    name collides with a supported function name / is not representable as
    a plain identifier -- both would make the translation silently lossy.
    """
    unmapped = sorted(set(problem.allowed_operators) - set(PYSR_OPERATOR_MAP))
    if unmapped:
        raise ValueError(f"operator(s) without an external mapping: {unmapped}")
    for name in problem.variables:
        if not _IDENT_RE.match(name):
            raise ValueError(f"variable name {name!r} is not a valid external identifier")
        if name in _SUPPORTED_FUNCTIONS:
            raise ValueError(f"variable name {name!r} collides with a supported function name")

    binary = tuple(
        sorted({PYSR_OPERATOR_MAP[op] for op in problem.allowed_operators if op in BINARY_OPERATORS})
    )
    unary = tuple(
        sorted({PYSR_OPERATOR_MAP[op] for op in problem.allowed_operators if op in UNARY_OPERATORS})
    )
    return AdapterRunInputV1(
        problem_fingerprint=problem.fingerprint,
        pack_id=problem.pack_id,
        variable_names=tuple(problem.variables),
        binary_operators=binary,
        unary_operators=unary,
        n_train_rows=len(problem.train_indices),
        seed=manifest.seed,
        manifest_fingerprint=manifest.fingerprint(),
    )


# ---------------------------------------------------------------------------
# External infix text -> typed IR. No eval/exec: a hand-rolled tokenizer and
# recursive-descent parser admitting only numbers, known variable names,
# known unary function names, and +-*/() -- everything else is reported as
# an unsupported operator/identifier, never guessed at.
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z0-9_]*|\S")


class ExternalParseError(ValueError):
    """A syntactically malformed external expression (unbalanced parens,
    trailing tokens, etc.) -- distinct from an unsupported-operator report."""


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(text))


def _find_unsupported(tokens: tuple[str, ...], variable_names: frozenset[str]) -> tuple[str, ...]:
    unsupported: list[str] = []
    for index, token in enumerate(tokens):
        if _NUMBER_RE.match(token) or token in _SUPPORTED_BINARY_TOKENS or token in "()":
            continue
        if _IDENT_RE.match(token):
            if token in variable_names:
                continue
            if token in _SUPPORTED_FUNCTIONS and index + 1 < len(tokens) and tokens[index + 1] == "(":
                continue
            unsupported.append(token)
            continue
        unsupported.append(token)
    seen: set[str] = set()
    ordered: list[str] = []
    for item in unsupported:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


class _InfixParser:
    """Standard precedence: unary ``-`` binds tighter than ``* /``, which
    bind tighter than ``+ -``; function calls and parens bind tightest."""

    def __init__(self, tokens: tuple[str, ...], variable_names: tuple[str, ...]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._variable_index = {name: index for index, name in enumerate(variable_names)}
        self.coefficients: list[float] = []

    def _peek(self) -> str | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _advance(self) -> str:
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def parse(self) -> SymbolicExprNode:
        node = self._expr()
        if self._pos != len(self._tokens):
            raise ExternalParseError(f"unexpected trailing token {self._peek()!r}")
        return node

    def _expr(self) -> SymbolicExprNode:
        node = self._term()
        while self._peek() in ("+", "-"):
            operator = self._advance()
            node = OpApply(operator=operator, args=(node, self._term()))
        return node

    def _term(self) -> SymbolicExprNode:
        node = self._unary()
        while self._peek() in ("*", "/"):
            operator = self._advance()
            node = OpApply(operator=operator, args=(node, self._unary()))
        return node

    def _unary(self) -> SymbolicExprNode:
        if self._peek() == "-":
            self._advance()
            return OpApply(operator="neg", args=(self._unary(),))
        return self._atom()

    def _atom(self) -> SymbolicExprNode:
        token = self._peek()
        if token is None:
            raise ExternalParseError("unexpected end of expression")
        if _NUMBER_RE.match(token):
            self._advance()
            index = len(self.coefficients)
            self.coefficients.append(float(token))
            return CoefficientHole(index=index)
        if token == "(":
            self._advance()
            node = self._expr()
            if self._peek() != ")":
                raise ExternalParseError("missing closing parenthesis")
            self._advance()
            return node
        if _IDENT_RE.match(token):
            self._advance()
            if token in self._variable_index:
                return VarRef(index=self._variable_index[token])
            if self._peek() == "(":
                self._advance()
                argument = self._expr()
                if self._peek() != ")":
                    raise ExternalParseError(f"missing closing parenthesis for {token}(")
                self._advance()
                return OpApply(operator=token, args=(argument,))
            raise ExternalParseError(f"unknown identifier {token!r}")
        raise ExternalParseError(f"unexpected token {token!r}")


@dataclass(frozen=True)
class ExternalTranslationResultV1:
    """Internal, non-persisted translation outcome (carries the live node,
    unlike :class:`AdapterRunResultV1` which persists only its S-expression
    serialization)."""

    status: str
    node: SymbolicExprNode | None
    coefficients: tuple[float, ...]
    unsupported_operators: tuple[str, ...]
    error_detail: str | None


def parse_external_expression(
    text: str, *, problem: "SymbolicRegressionProblemV1"
) -> ExternalTranslationResultV1:
    """Parse external infix ``text`` back into the typed IR, or report
    exactly why it could not be represented -- never an approximation."""
    tokens = _tokenize(text)
    unsupported = _find_unsupported(tokens, frozenset(problem.variables))
    if unsupported:
        return ExternalTranslationResultV1(
            status="operator_mismatch",
            node=None,
            coefficients=(),
            unsupported_operators=unsupported,
            error_detail=None,
        )
    parser = _InfixParser(tokens, tuple(problem.variables))
    try:
        node = parser.parse()
        validate_expr_budget(node)
    except (ExternalParseError, ValueError) as exc:
        return ExternalTranslationResultV1(
            status="parse_error", node=None, coefficients=(), unsupported_operators=(), error_detail=str(exc)
        )
    return ExternalTranslationResultV1(
        status="ok",
        node=node,
        coefficients=tuple(parser.coefficients),
        unsupported_operators=(),
        error_detail=None,
    )


# ---------------------------------------------------------------------------
# Orchestration: translate -> call the injected external runner -> parse ->
# rescore through the pack's own evaluator/fitter/reporter.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterRunResultV1:
    """Typed, JSON-serializable outcome of one external adapter run.

    ``evidence`` is populated only when ``status == "ok"`` -- every other
    status is external-blocked/incomplete evidence, never a fabricated
    benchmark loss.
    """

    schema_version: str
    problem_fingerprint: str
    pack_id: str
    manifest: dict[str, Any]
    status: str
    raw_output: str | None
    node_sexpr: str | None
    coefficients: tuple[float, ...]
    unsupported_operators: tuple[str, ...]
    error_detail: str | None
    evidence: dict[str, Any] | None
    wall_ms: float

    def __post_init__(self) -> None:
        if self.schema_version != ADAPTER_SCHEMA_VERSION:
            raise ValueError(f"unsupported adapter run result schema: {self.schema_version!r}")
        if self.status not in ADAPTER_STATUSES:
            raise ValueError(f"unsupported adapter status: {self.status!r}")
        if self.status == "ok" and self.evidence is None:
            raise ValueError("status='ok' requires evidence")
        if self.status != "ok" and self.evidence is not None:
            raise ValueError(f"status={self.status!r} must not carry evidence (not a benchmark loss)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _blocked_result(
    problem: "SymbolicRegressionProblemV1",
    manifest: AdapterEnvironmentManifestV1,
    *,
    status: str,
    raw_output: str | None,
    error_detail: str | None,
    unsupported_operators: tuple[str, ...] = (),
    wall_ms: float,
) -> AdapterRunResultV1:
    return AdapterRunResultV1(
        schema_version=ADAPTER_SCHEMA_VERSION,
        problem_fingerprint=problem.fingerprint,
        pack_id=problem.pack_id,
        manifest=manifest.to_dict(),
        status=status,
        raw_output=raw_output,
        node_sexpr=None,
        coefficients=(),
        unsupported_operators=unsupported_operators,
        error_detail=error_detail,
        evidence=None,
        wall_ms=wall_ms,
    )


def run_adapter(
    problem: "SymbolicRegressionProblemV1",
    manifest: AdapterEnvironmentManifestV1,
    *,
    raw_output_provider: Callable[[AdapterRunInputV1], str],
) -> AdapterRunResultV1:
    """Run one external-adapter round trip.

    ``raw_output_provider`` is the caller-injected external call (a real
    live run shells out to an isolated PySR/SRBench process; core tests
    pass a mocked/golden callable). Raising ``TimeoutError`` is reported as
    ``status="timeout"``; any other exception is reported as
    ``status="external_blocked"`` -- the external tool's failure never
    propagates as an uncaught exception or a fabricated loss.
    """
    start = time.perf_counter()
    adapter_input = build_adapter_run_input(problem, manifest)
    try:
        raw_output = raw_output_provider(adapter_input)
    except TimeoutError as exc:
        return _blocked_result(
            problem, manifest, status="timeout", raw_output=None, error_detail=str(exc),
            wall_ms=round((time.perf_counter() - start) * 1000.0, 4),
        )
    except Exception as exc:  # noqa: BLE001 -- an external tool's failure must never crash the caller
        return _blocked_result(
            problem, manifest, status="external_blocked", raw_output=None, error_detail=str(exc),
            wall_ms=round((time.perf_counter() - start) * 1000.0, 4),
        )

    translation = parse_external_expression(raw_output, problem=problem)
    wall_ms = round((time.perf_counter() - start) * 1000.0, 4)
    if translation.status != "ok" or translation.node is None:
        return AdapterRunResultV1(
            schema_version=ADAPTER_SCHEMA_VERSION,
            problem_fingerprint=problem.fingerprint,
            pack_id=problem.pack_id,
            manifest=manifest.to_dict(),
            status=translation.status,
            raw_output=raw_output,
            node_sexpr=None,
            coefficients=(),
            unsupported_operators=translation.unsupported_operators,
            error_detail=translation.error_detail,
            evidence=None,
            wall_ms=wall_ms,
        )

    evidence: ExpressionEvidenceV1 = build_evidence_record(
        translation.node, problem, domain_policy=problem.numeric_domain_policy
    )
    return AdapterRunResultV1(
        schema_version=ADAPTER_SCHEMA_VERSION,
        problem_fingerprint=problem.fingerprint,
        pack_id=problem.pack_id,
        manifest=manifest.to_dict(),
        status="ok",
        raw_output=raw_output,
        node_sexpr=serialize_expr(translation.node),
        coefficients=translation.coefficients,
        unsupported_operators=(),
        error_detail=None,
        evidence=evidence.to_dict(),
        wall_ms=wall_ms,
    )


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "ADAPTER_STATUSES",
    "DEFAULT_PYSR_MANIFEST",
    "DEFAULT_SRBENCH_MANIFEST",
    "MANIFEST_SCHEMA_VERSION",
    "PYSR_OPERATOR_MAP",
    "SUPPORTED_TOOLS",
    "AdapterEnvironmentManifestV1",
    "AdapterRunInputV1",
    "AdapterRunResultV1",
    "ExternalParseError",
    "ExternalTranslationResultV1",
    "build_adapter_run_input",
    "parse_external_expression",
    "run_adapter",
]
