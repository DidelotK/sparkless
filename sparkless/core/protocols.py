"""
Type protocols for Sparkless.

This module defines structural typing protocols (PEP 544) for better
type safety and clearer contracts without tight coupling.
"""

from typing import Any, Protocol, Union, runtime_checkable


@runtime_checkable
class ColumnLike(Protocol):
    """Protocol for column-like objects."""

    @property
    def name(self) -> str:
        """Column name."""
        ...


@runtime_checkable
class OperationLike(Protocol):
    """Protocol for column operation objects."""

    @property
    def column(self) -> Any:
        """The column being operated on."""
        ...

    @property
    def operation(self) -> str:
        """The operation type."""
        ...

    @property
    def value(self) -> Any:
        """The operation value/operand."""
        ...

    @property
    def name(self) -> str:
        """Operation name."""
        ...


@runtime_checkable
class LiteralLike(Protocol):
    """Protocol for literal value objects."""

    @property
    def value(self) -> Any:
        """The literal value."""
        ...

    @property
    def name(self) -> str:
        """Literal name."""
        ...


@runtime_checkable
class CaseWhenLike(Protocol):
    """Protocol for CASE WHEN expression objects."""

    @property
    def conditions(self) -> Any:
        """List of (condition, value) tuples."""
        ...

    @property
    def default_value(self) -> Any:
        """Default value for ELSE clause."""
        ...


def is_row_evaluatable_expression(column: Any) -> bool:
    """Whether an aggregate/window target must be computed per row.

    ``F.sum(F.col("x"))`` targets a plain column and can be read straight out
    of each row by name. ``F.sum(F.col("x") * 2)`` and
    ``F.sum(F.when(cond, x))`` target an *expression* that has to be evaluated
    for every row first -- there is no column of that name to look up.

    ``ColumnOperation`` advertises itself with ``.operation``, but ``CaseWhen``
    does not: it is identified by its ``conditions``/``default_value`` pair
    (see :class:`CaseWhenLike`). Gating only on ``.operation`` sent every
    CASE WHEN down the plain-column path, where the lookup of a column
    literally named ``"CASE WHEN"`` missed on every row and the aggregate
    collapsed to its empty default -- a constant ``0`` for ``sum``.
    """
    if column is None:
        return False
    if hasattr(column, "operation"):
        return True
    return hasattr(column, "conditions") and hasattr(column, "default_value")


@runtime_checkable
class DataFrameLike(Protocol):
    """Protocol for DataFrame-like objects."""

    @property
    def data(self) -> Any:
        """DataFrame data."""
        ...

    @property
    def schema(self) -> Any:
        """DataFrame schema."""
        ...

    def collect(self) -> Any:
        """Collect DataFrame rows."""
        ...


@runtime_checkable
class SchemaLike(Protocol):
    """Protocol for schema-like objects."""

    @property
    def fields(self) -> Any:
        """Schema fields."""
        ...

    def fieldNames(self) -> Any:
        """Get field names."""
        ...


# Type aliases for common unions (improved type safety)
# Use string literals for forward references to avoid import cycles
ColumnExpression = Union[
    ColumnLike,
    OperationLike,
    LiteralLike,
    str,
]  # Can also include Column, ColumnOperation, Literal at runtime
AggregateExpression = Union[
    str, OperationLike, ColumnLike
]  # Can be string name or column operation
WindowExpression = Any  # WindowFunction is complex - keep as Any for now
