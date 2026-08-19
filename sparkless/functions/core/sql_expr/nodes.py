"""Abstract syntax tree for the SQL expression grammar accepted by ``F.expr``.

Parsing and binding are separate steps on purpose. The parser's only job is to
get the *shape* right -- precedence, associativity, which tokens are arguments
of which call -- and it does that without importing a single sparkless
function. The binder then turns that shape into columns.

Keeping them apart is what makes the precedence bug testable: an AST can be
asserted on directly, without a DataFrame and without evaluating anything.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple


class Node:
    """Base class for every AST node."""


@dataclass(frozen=True)
class Literal(Node):
    """A constant: number, string, boolean or NULL."""

    value: Any


@dataclass(frozen=True)
class ColumnReference(Node):
    """A column reference, possibly qualified (``a.b.c``)."""

    parts: Sequence[str]

    @property
    def name(self) -> str:
        """The dotted name as written."""
        return ".".join(self.parts)


@dataclass(frozen=True)
class Star(Node):
    """The ``*`` of ``count(*)``."""


@dataclass(frozen=True)
class FunctionCall(Node):
    """A function call: ``concat(a, b)``, ``count(DISTINCT x)``."""

    name: str
    arguments: List[Node] = field(default_factory=list)
    distinct: bool = False


@dataclass(frozen=True)
class UnaryOperation(Node):
    """A prefix operator: ``-x``, ``NOT x``, ``~x``."""

    operator: str
    operand: Node


@dataclass(frozen=True)
class BinaryOperation(Node):
    """An infix operator: arithmetic, comparison, ``AND``/``OR``, ``||``."""

    operator: str
    left: Node
    right: Node


@dataclass(frozen=True)
class IsNull(Node):
    """``x IS NULL`` / ``x IS NOT NULL``."""

    operand: Node
    negated: bool = False


@dataclass(frozen=True)
class IsBoolean(Node):
    """``x IS [NOT] TRUE`` / ``x IS [NOT] FALSE``."""

    operand: Node
    expected: bool
    negated: bool = False


@dataclass(frozen=True)
class Between(Node):
    """``x [NOT] BETWEEN low AND high``."""

    operand: Node
    lower: Node
    upper: Node
    negated: bool = False


@dataclass(frozen=True)
class InList(Node):
    """``x [NOT] IN (a, b, c)``."""

    operand: Node
    items: List[Node]
    negated: bool = False


@dataclass(frozen=True)
class PatternMatch(Node):
    """``x [NOT] LIKE/ILIKE/RLIKE/REGEXP pattern``."""

    kind: str
    operand: Node
    pattern: Node
    negated: bool = False


@dataclass(frozen=True)
class Cast(Node):
    """``CAST(x AS type)`` / ``TRY_CAST(x AS type)``."""

    operand: Node
    type_name: str
    try_cast: bool = False


@dataclass(frozen=True)
class CaseWhen(Node):
    """``CASE [operand] WHEN c THEN v ... [ELSE v] END``."""

    branches: List[Any]
    else_value: Optional[Node] = None
    operand: Optional[Node] = None


@dataclass(frozen=True)
class Lambda(Node):
    """A higher-order function argument: ``x -> x IS NOT NULL``."""

    parameters: Sequence[str]
    body: Node


@dataclass(frozen=True)
class Interval(Node):
    """An ``INTERVAL`` literal, e.g. ``INTERVAL 90 DAYS``.

    Attributes:
        parts: The ``(quantity, unit)`` pairs as written, units upper-cased.
            Spark allows several (``INTERVAL 1 YEAR 2 MONTHS``); sparkless can
            only evaluate a single day-based interval, and the binder says so
            for anything else rather than guessing.
        text: The literal as written, quoted in error messages.
    """

    parts: Sequence[Tuple[int, str]]
    text: str
