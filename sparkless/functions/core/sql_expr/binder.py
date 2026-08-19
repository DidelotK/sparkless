"""Binds a parsed SQL expression AST to sparkless columns.

The binder resolves every function call against the **real** ``F`` namespace
and calls it with the arguments the SQL wrote. That is the whole fix for the
dropped-argument half of the defect: the previous parser built
``ColumnOperation(None, "concat", args)`` by hand, a shape no evaluator
implements, so ``concat(sku, dept)`` rendered as ``concat()`` and evaluated to
NULL -- while ``F.concat(F.col("sku"), F.col("dept"))`` had been correct all
along. Binding through ``F`` means ``F.expr`` can never again be more wrong
than the programmatic API it mirrors.

Literal arguments are wrapped in ``F.lit`` or passed raw according to the
target function's own type annotations: a parameter that accepts a ``Column``
gets a literal column, a parameter annotated ``int``/``str`` gets the raw
Python value. sparkless is typed strictly enough for this to be read off the
signature, which keeps the rule in one place instead of a per-function table
that would rot. ``F.ifnull(col, "X")`` and ``F.ifnull(col, F.lit("X"))``
return different answers -- ``None`` and ``"X"`` -- so this distinction is not
cosmetic.

Anything the binder cannot express *raises*. It never falls back to a value.
A dropped argument that yields NULL is the failure mode this module exists to
remove; replacing it with a different silent answer would not be a fix.
"""

import inspect
import operator
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from ....core.exceptions.analysis import ParseException
from ..column import Column, ColumnOperation
from . import nodes

_ARITHMETIC = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "%": operator.mod,
}

_COMPARISONS = {
    "=": operator.eq,
    "==": operator.eq,
    "<>": operator.ne,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}

# Higher-order functions take a lambda. sparkless evaluates all of them to NULL
# for every row (Solya-app/solya-data-platform#2419), so binding a SQL lambda
# to them would turn today's loud ParseException into a silent wrong answer.
_HIGHER_ORDER = frozenset(
    {
        "aggregate",
        "exists",
        "filter",
        "forall",
        "map_filter",
        "map_zip_with",
        "reduce",
        "transform",
        "transform_keys",
        "transform_values",
        "zip_with",
    }
)

# SQL spells DISTINCT as part of the call; sparkless spells it as a function.
_DISTINCT_EQUIVALENTS = {
    "count": "count_distinct",
    "sum": "sum_distinct",
}

_registry: Optional[Dict[str, Callable[..., Any]]] = None


def _function_registry() -> Dict[str, Callable[..., Any]]:
    """The SQL-callable functions, keyed by lower-cased name.

    Built once from the ``F`` namespace itself, so a function added to
    sparkless becomes callable from ``F.expr`` with no change here.
    """
    global _registry
    if _registry is None:
        from ...functions import Functions

        registry: Dict[str, Callable[..., Any]] = {}
        for name in dir(Functions):
            if name.startswith("_"):
                continue
            attribute = getattr(Functions, name, None)
            if inspect.isfunction(attribute) or inspect.isbuiltin(attribute):
                registry.setdefault(name.lower(), attribute)
        _registry = registry
    return _registry


def _expression_types() -> tuple:
    """Every type a sparkless function may return as an expression.

    They share no base class -- ``Column`` and ``Literal`` are unrelated, and
    ``AggregateFunction`` and ``WindowFunction`` inherit from ``object`` -- so
    the set has to be enumerated. Leaving one out makes the binder wrap it in
    ``F.lit``: that is how ``count(DISTINCT dept)``, which returns an
    ``AggregateFunction``, first came back as a literal.
    """
    from ...base import AggregateFunction
    from ...conditional import CaseWhen
    from ...window_execution import WindowFunction
    from ..literals import Literal

    return (
        Column,
        ColumnOperation,
        CaseWhen,
        Literal,
        AggregateFunction,
        WindowFunction,
    )


def _is_column(value: Any) -> bool:
    """Whether ``value`` is already a column-like expression.

    ``Column.__eq__`` returns a ``ColumnOperation`` rather than a bool, so the
    binder must never compare columns for equality -- only ``isinstance``.
    """
    return isinstance(value, _expression_types())


def _as_column(value: Any) -> Any:
    """Coerce a bound value to a column, wrapping literals in ``F.lit``."""
    if _is_column(value):
        return value
    from ...functions import Functions

    return Functions.lit(value)


def _expects_column(annotation: Any) -> bool:
    """Whether a parameter wants a column rather than a raw Python value.

    Read off the target function's own annotation:

    * ``Union[Column, str]`` -- column-like, so a SQL literal becomes
      ``F.lit(value)``. Passing the raw string here would be read as a
      *column name*, which is how ``coalesce(name, 'FALLBACK')`` lost its
      fallback.
    * ``Union[Column, float, int]`` -- accepts a bare number (``F.pow``,
      ``F.log``), so the raw value is passed. ``F.log(F.lit(10), col)``
      evaluates to NULL where ``F.log(10, col)`` is correct.
    * ``int`` / ``str`` -- a scalar parameter (``F.substring``'s ``start``,
      ``F.regexp_extract``'s ``idx``), passed raw.
    """
    if annotation is inspect.Parameter.empty:
        # Unannotated: a SQL literal means a literal, so prefer F.lit over a
        # raw string that would be read as a column name.
        return True
    if annotation is Any:
        return True

    origin = getattr(annotation, "__origin__", None)
    if origin is Union:
        arguments = getattr(annotation, "__args__", ())
        has_column = any(_is_column_type(argument) for argument in arguments)
        if has_column:
            return not any(argument in (int, float) for argument in arguments)
        return any(argument is Any for argument in arguments)

    return _is_column_type(annotation)


def _is_column_type(annotation: Any) -> bool:
    """Whether an annotation is the ``Column`` class or a subclass."""
    return isinstance(annotation, type) and issubclass(annotation, Column)


class SQLExpressionBinder:
    """Turns an AST into a sparkless column expression."""

    def __init__(self, source: str) -> None:
        """Initialise the binder.

        Args:
            source: The original SQL text, quoted in error messages.
        """
        self.source = source

    # -- entry point ------------------------------------------------------

    def bind_root(self, node: nodes.Node) -> Any:
        """Bind the root of an expression, always yielding a column.

        Args:
            node: The root AST node.

        Returns:
            A ``Column``, ``ColumnOperation`` or ``CaseWhen``.
        """
        return _as_column(self.bind(node))

    def bind(self, node: nodes.Node) -> Any:
        """Bind one AST node.

        Literals bind to raw Python values; the caller decides whether they
        need wrapping. Everything else binds to a column expression.

        Args:
            node: The AST node to bind.

        Returns:
            A column expression, or a raw Python value for a literal.

        Raises:
            ParseException: If the construct has no sparkless equivalent.
        """
        binder = self._DISPATCH.get(type(node))
        if binder is None:
            raise ParseException(
                f"Unsupported SQL expression {self.source!r}: "
                f"{type(node).__name__} has no sparkless equivalent"
            )
        return binder(self, node)

    # -- leaves -----------------------------------------------------------

    def _bind_literal(self, node: nodes.Literal) -> Any:
        """Bind a constant to its Python value."""
        return node.value

    def _bind_column_reference(self, node: nodes.ColumnReference) -> Any:
        """Bind a column reference, keeping any qualifier in the name."""
        return Column(node.name)

    def _bind_star(self, node: nodes.Star) -> Any:
        """Reject ``*`` outside ``count(*)``."""
        raise ParseException(
            f"Invalid SQL expression {self.source!r}: '*' is only supported "
            f"as the argument of count(*)"
        )

    def _bind_interval(self, node: nodes.Interval) -> Any:
        """Reject INTERVAL literals, which sparkless cannot evaluate."""
        raise ParseException(
            f"Unsupported SQL expression {self.source!r}: sparkless has no "
            f"INTERVAL literal ({node.text!r}). Use the date functions "
            f"instead, e.g. F.expr('date_sub(current_date(), 30)')"
        )

    def _bind_lambda(self, node: nodes.Lambda) -> Any:
        """Reject a lambda outside a higher-order call."""
        raise ParseException(
            f"Unsupported SQL expression {self.source!r}: lambda expressions "
            f"are only valid as arguments of higher-order functions"
        )

    # -- operators --------------------------------------------------------

    def _bind_unary(self, node: nodes.UnaryOperation) -> Any:
        """Bind ``-x``, ``NOT x`` and ``~x``."""
        operand = self.bind(node.operand)

        if node.operator == "NOT":
            return ~_as_column(operand)

        if node.operator == "~":
            # SQL's unary ``~`` is *bitwise* NOT, and every sparkless spelling
            # of it (F.bitwise_not, F.bitwiseNOT, Column.bitwise_not)
            # evaluates to NULL. Binding it to Column's ``__invert__`` would
            # quietly substitute logical NOT, so refuse it instead.
            raise ParseException(
                f"Unsupported SQL expression {self.source!r}: sparkless's "
                f"bitwise NOT evaluates to NULL for every row"
            )

        if not _is_column(operand):
            return -operand

        # Lowered to ``0 - x`` rather than ``-x``: ``Column.__neg__`` builds a
        # subtraction whose right operand is None, so it evaluates to NULL for
        # every row (a separate sparkless defect, on the programmatic API).
        # The two agree on every numeric value except the sign of zero.
        return 0 - operand

    def _bind_binary(self, node: nodes.BinaryOperation) -> Any:
        """Bind an infix operator at the precedence the parser decided."""
        symbol = node.operator

        if symbol == "AND":
            return _as_column(self.bind(node.left)) & _as_column(self.bind(node.right))
        if symbol == "OR":
            return _as_column(self.bind(node.left)) | _as_column(self.bind(node.right))

        left = self.bind(node.left)
        right = self.bind(node.right)

        if symbol == "||":
            from ...functions import Functions

            return Functions.concat(_as_column(left), _as_column(right))
        if symbol == "<=>":
            return _as_column(left).eqNullSafe(right)
        if symbol == "DIV":
            raise ParseException(
                f"Unsupported SQL expression {self.source!r}: the DIV operator "
                f"has no sparkless equivalent"
            )

        arithmetic = _ARITHMETIC.get(symbol)
        if arithmetic is not None:
            return arithmetic(left, right)

        comparison = _COMPARISONS.get(symbol)
        if comparison is not None:
            return comparison(left, right)

        raise ParseException(
            f"Unsupported SQL expression {self.source!r}: unknown operator {symbol!r}"
        )

    # -- predicates -------------------------------------------------------

    def _bind_is_null(self, node: nodes.IsNull) -> Any:
        """Bind ``x IS [NOT] NULL``."""
        operand = _as_column(self.bind(node.operand))
        return operand.isNotNull() if node.negated else operand.isNull()

    def _bind_is_boolean(self, node: nodes.IsBoolean) -> Any:
        """Bind ``x IS [NOT] TRUE`` / ``x IS [NOT] FALSE``."""
        operand = _as_column(self.bind(node.operand))
        result = operand == node.expected
        return ~result if node.negated else result

    def _bind_between(self, node: nodes.Between) -> Any:
        """Bind ``x [NOT] BETWEEN low AND high``."""
        operand = _as_column(self.bind(node.operand))
        result = operand.between(self.bind(node.lower), self.bind(node.upper))
        return ~result if node.negated else result

    def _bind_in_list(self, node: nodes.InList) -> Any:
        """Bind ``x [NOT] IN (...)``."""
        operand = _as_column(self.bind(node.operand))
        items = [self.bind(item) for item in node.items]
        result = operand.isin(items)
        return ~result if node.negated else result

    def _bind_pattern_match(self, node: nodes.PatternMatch) -> Any:
        """Bind ``LIKE`` / ``ILIKE`` / ``RLIKE`` / ``REGEXP``."""
        operand = _as_column(self.bind(node.operand))
        pattern = self.bind(node.pattern)
        if not isinstance(pattern, str):
            raise ParseException(
                f"Unsupported SQL expression {self.source!r}: {node.kind} "
                f"requires a string literal pattern in sparkless"
            )

        if node.kind == "LIKE":
            result = operand.like(pattern)
        elif node.kind == "ILIKE":
            from ...functions import Functions

            result = Functions.ilike(operand, pattern)
        else:
            result = operand.rlike(pattern)

        return ~result if node.negated else result

    def _bind_cast(self, node: nodes.Cast) -> Any:
        """Bind ``CAST``/``TRY_CAST``.

        sparkless's cast already returns NULL for a value it cannot convert
        rather than raising, so ``TRY_CAST`` and ``CAST`` bind identically.
        """
        return _as_column(self.bind(node.operand)).cast(node.type_name)

    def _bind_case_when(self, node: nodes.CaseWhen) -> Any:
        """Bind a ``CASE`` expression to an ``F.when`` chain."""
        from ...functions import Functions

        operand = self.bind(node.operand) if node.operand is not None else None

        result = None
        for condition_node, value_node in node.branches:
            condition = self.bind(condition_node)
            if operand is not None:
                condition = _as_column(operand) == condition
            value = self.bind(value_node)
            if result is None:
                result = Functions.when(_as_column(condition), value)
            else:
                result = result.when(_as_column(condition), value)

        if result is None:  # pragma: no cover - the parser requires a WHEN
            raise ParseException(
                f"Invalid SQL expression {self.source!r}: CASE without WHEN"
            )

        if node.else_value is not None:
            return result.otherwise(self.bind(node.else_value))
        return result

    # -- calls ------------------------------------------------------------

    def _bind_function_call(self, node: nodes.FunctionCall) -> Any:
        """Bind a function call by dispatching to the real ``F`` function."""
        name = node.name.lower()

        if any(isinstance(argument, nodes.Lambda) for argument in node.arguments):
            raise ParseException(
                f"Unsupported SQL expression {self.source!r}: sparkless's "
                f"higher-order functions evaluate to NULL for every row "
                f"(solya-data-platform#2419), so F.expr refuses to bind "
                f"{node.name}(...) with a lambda rather than return a wrong "
                f"answer"
            )

        if node.distinct:
            equivalent = _DISTINCT_EQUIVALENTS.get(name)
            if equivalent is None:
                raise ParseException(
                    f"Unsupported SQL expression {self.source!r}: sparkless has "
                    f"no DISTINCT form of {node.name}"
                )
            name = equivalent

        if any(isinstance(argument, nodes.Star) for argument in node.arguments):
            if name != "count" or len(node.arguments) != 1:
                raise ParseException(
                    f"Invalid SQL expression {self.source!r}: '*' is only "
                    f"supported as the argument of count(*)"
                )
            arguments: List[nodes.Node] = []
        else:
            arguments = list(node.arguments)

        function = _function_registry().get(name)
        if function is None:
            raise ParseException(
                f"Undefined function {node.name!r} in SQL expression "
                f"{self.source!r}: sparkless's F namespace has no such "
                f"function, so F.expr cannot bind it"
            )
        if name in _HIGHER_ORDER:
            raise ParseException(
                f"Unsupported SQL expression {self.source!r}: sparkless's "
                f"{node.name} evaluates to NULL for every row "
                f"(solya-data-platform#2419)"
            )

        bound = self._bind_arguments(node.name, function, arguments)
        try:
            return function(*bound)
        except TypeError as error:
            raise ParseException(
                f"Invalid SQL expression {self.source!r}: {node.name} was "
                f"called with {len(bound)} argument(s), which sparkless's "
                f"F.{name} does not accept ({error})"
            )

    def _bind_arguments(
        self,
        display_name: str,
        function: Callable[..., Any],
        arguments: Sequence[nodes.Node],
    ) -> List[Any]:
        """Bind call arguments against the target function's signature."""
        parameters = self._positional_parameters(function)
        bound: List[Any] = []

        for position, argument in enumerate(arguments):
            parameter = self._parameter_at(parameters, position)
            annotation = (
                inspect.Parameter.empty if parameter is None else parameter.annotation
            )
            value = self.bind(argument)

            if _expects_column(annotation):
                bound.append(_as_column(value))
                continue

            if _is_column(value):
                parameter_name = "?" if parameter is None else parameter.name
                raise ParseException(
                    f"Unsupported SQL expression {self.source!r}: parameter "
                    f"{parameter_name!r} of {display_name} takes a literal in "
                    f"sparkless, not a column expression"
                )
            bound.append(value)

        return bound

    @staticmethod
    def _positional_parameters(
        function: Callable[..., Any],
    ) -> List[inspect.Parameter]:
        """The parameters a SQL call can fill positionally."""
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):  # pragma: no cover - builtins only
            return []
        return [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
        ]

    @staticmethod
    def _parameter_at(
        parameters: List[inspect.Parameter], position: int
    ) -> Optional[inspect.Parameter]:
        """The parameter filled by the argument at ``position``."""
        if not parameters:
            return None
        if position < len(parameters):
            return parameters[position]
        last = parameters[-1]
        if last.kind is inspect.Parameter.VAR_POSITIONAL:
            return last
        return None

    _DISPATCH: Dict[type, Callable[..., Any]] = {}


SQLExpressionBinder._DISPATCH = {
    nodes.Literal: SQLExpressionBinder._bind_literal,
    nodes.ColumnReference: SQLExpressionBinder._bind_column_reference,
    nodes.Star: SQLExpressionBinder._bind_star,
    nodes.FunctionCall: SQLExpressionBinder._bind_function_call,
    nodes.UnaryOperation: SQLExpressionBinder._bind_unary,
    nodes.BinaryOperation: SQLExpressionBinder._bind_binary,
    nodes.IsNull: SQLExpressionBinder._bind_is_null,
    nodes.IsBoolean: SQLExpressionBinder._bind_is_boolean,
    nodes.Between: SQLExpressionBinder._bind_between,
    nodes.InList: SQLExpressionBinder._bind_in_list,
    nodes.PatternMatch: SQLExpressionBinder._bind_pattern_match,
    nodes.Cast: SQLExpressionBinder._bind_cast,
    nodes.CaseWhen: SQLExpressionBinder._bind_case_when,
    nodes.Lambda: SQLExpressionBinder._bind_lambda,
    nodes.Interval: SQLExpressionBinder._bind_interval,
}


def bind(node: nodes.Node, source: str) -> Any:
    """Bind a parsed SQL expression to a sparkless column.

    Args:
        node: The AST produced by the parser.
        source: The original SQL text, quoted in error messages.

    Returns:
        A ``Column``, ``ColumnOperation`` or ``CaseWhen``.

    Raises:
        ParseException: If the expression has no sparkless equivalent.
    """
    return SQLExpressionBinder(source).bind_root(node)
