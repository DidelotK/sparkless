"""Reading the target of an aggregate, whatever shape that target has.

``F.countDistinct(F.col("dept"))`` names a column that can be read out of each
row. ``F.countDistinct(F.upper(F.col("dept")))``, ``F.countDistinct(F.struct(...))``
and ``F.countDistinct(F.when(...))`` name an *expression*, which has to be
evaluated per row first -- there is no column of that name to look up, so
reading one by name misses on every row and the aggregate collapses to its
empty default. For the counting aggregates that default is **0**, a legitimate
answer, which is why the failure was silent.

The composite values such targets produce (a ``struct`` is a ``dict``) are not
hashable, so distinctness needs a key rather than the value itself.

Verified against PySpark 4.0.0 (``local[1]``):

===============================================  ===========================
expression over ``(A,eng) (B,ops) (C,eng) (D,NULL)``  result
===============================================  ===========================
``countDistinct(dept)``                          2 -- NULL targets skipped
``countDistinct(upper(dept))``                   2
``countDistinct(struct(sku, dept))``             4 -- a struct is never NULL,
                                                 so no row is skipped
``countDistinct(struct(dept))``                  3 -- the NULL is *inside* the
                                                 struct, and counts
===============================================  ===========================
"""

from typing import Any, Iterable, List

from .protocols import is_row_evaluatable_expression

__all__ = ["aggregate_target_values", "distinct_count"]

#: Evaluating one row of an expression target can fail on malformed data; the
#: row is skipped rather than taking the whole aggregate down. Mirrors the
#: exceptions the sum/avg/max branches already tolerate.
_ROW_EVALUATION_FAILURES = (ValueError, TypeError, AttributeError)


def aggregate_target_values(
    df: Any,
    expr: Any,
    col_name: str,
    group_rows: List[Any],
) -> List[Any]:
    """Non-NULL values of an aggregate's target, one per row of the group.

    Args:
        df: The DataFrame owning the rows; supplies the row-wise evaluator.
        expr: The aggregate function expression, whose ``column`` is the target.
        col_name: The target's rendered name, used for the plain-column case.
        group_rows: The rows of the group being aggregated.

    Returns:
        The evaluated target values, with NULLs dropped -- Spark's counting and
        summing aggregates all ignore NULL inputs.
    """
    from ..spark_types import get_row_value

    target = getattr(expr, "column", None)
    if is_row_evaluatable_expression(target):
        values: List[Any] = []
        for row in group_rows:
            try:
                value = df._evaluate_column_expression(row, target)
            except _ROW_EVALUATION_FAILURES:
                continue
            if value is not None:
                values.append(value)
        return values

    return [
        value
        for value in (get_row_value(row, col_name) for row in group_rows)
        if value is not None
    ]


def _distinct_key(value: Any) -> Any:
    """A hashable identity for an aggregate target value.

    A ``struct`` evaluates to a ``dict`` and an array to a ``list``; neither is
    hashable, and ``set()`` would raise on them. Field order is part of a
    struct's identity and is preserved by keying on the items in order.
    """
    if isinstance(value, dict):
        return ("__struct__", tuple((k, _distinct_key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return ("__array__", tuple(_distinct_key(item) for item in value))
    try:
        hash(value)
    except TypeError:
        return ("__repr__", repr(value))
    return value


def distinct_count(values: Iterable[Any]) -> int:
    """Number of distinct values, tolerating unhashable composite values."""
    return len({_distinct_key(value) for value in values})
