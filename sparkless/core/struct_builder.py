"""Canonical construction of ``struct`` / ``named_struct`` values.

``F.struct(...)`` produces a composite value: an ordered mapping of field
name to field value. Two independent evaluators need it -- the lazy
``select`` path goes through
:class:`sparkless.core.condition_evaluator.ConditionEvaluator` and the
``withColumn`` path through
:class:`sparkless.dataframe.evaluation.expression_evaluator.ExpressionEvaluator`.
Keeping the logic here means a single definition of PySpark's field-naming
rules instead of two that drift.

Field-naming rules, verified against PySpark 4.0.0:

===========================================  =========================
Argument at 1-based position *i*             Resulting field name
===========================================  =========================
``F.col("a").alias("x")`` / any alias        ``x``
``"a"`` (bare string)                        ``a`` -- a *column reference*,
                                             never a string literal
``F.col("a")``                               ``a``
``(F.col("a") + F.col("b"))``                the expression's name
``F.lit(1)`` (unaliased literal)             ``col1`` -- the *position*,
                                             not a count of anonymous fields
===========================================  =========================

So ``F.struct("a", F.lit("k"))`` yields ``{"a": ..., "col2": "k"}`` and
``F.struct(F.lit(1), F.col("a"))`` yields ``{"col1": 1, "a": ...}``.
"""

from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "struct_argument_columns",
    "struct_argument_source",
    "field_name_for",
    "build_struct_value",
]


def _is_literal(item: Any) -> bool:
    """Whether ``item`` is a :class:`~sparkless.functions.core.literals.Literal`."""
    from ..functions.core.literals import Literal

    return isinstance(item, Literal)


def _explicit_alias(item: Any) -> Optional[str]:
    """Return the user-supplied alias of ``item``, if any."""
    alias = getattr(item, "_alias_name", None)
    return str(alias) if alias else None


def struct_argument_columns(operation: Any) -> List[Any]:
    """Recover the ordered argument list of a struct/named_struct operation.

    ``StructFunctions.struct`` stores the first argument in
    ``operation.column`` and the rest in ``operation.value`` -- unless the
    first argument is a ``Literal``, in which case a ``__struct_dummy__``
    placeholder column is used and *every* argument lives in
    ``operation.value``. This flattens both encodings back into one list.
    """
    base = getattr(operation, "column", None)
    rest = getattr(operation, "value", None)

    items: List[Any] = []
    if isinstance(rest, (list, tuple)):
        items = list(rest)
    elif rest is not None:
        items = [rest]

    base_name = getattr(base, "name", None)
    is_placeholder = base is None or base_name == "__struct_dummy__"
    if not is_placeholder:
        items.insert(0, base)
    return items


def struct_argument_source(item: Any) -> Any:
    """Return the expression whose *value* a struct argument contributes.

    ``Column.alias("x")`` builds a fresh ``Column`` named ``x`` and keeps the
    original on ``_original_column``. Resolving the alias against the row
    would look up a column named ``x``, which does not exist -- the value has
    to come from the original expression. ``ColumnOperation.alias`` keeps the
    operation intact, so only the ``Column`` wrapper needs unwrapping.
    """
    original = getattr(item, "_original_column", None)
    return original if original is not None else item


def field_name_for(item: Any, position: int) -> str:
    """Name the struct field produced by ``item`` at 1-based ``position``."""
    alias = _explicit_alias(item)
    if alias:
        return alias

    if isinstance(item, str):
        # A bare string is a column reference in PySpark, so the field takes
        # the column's name.
        return item

    if _is_literal(item):
        # Unaliased literals are named after their position.
        return f"col{position}"

    name = getattr(item, "name", None)
    if name:
        return str(name)

    return f"col{position}"


def build_struct_value(
    operation: Any,
    resolve: Callable[[Any], Any],
) -> Dict[str, Any]:
    """Build the dict value of a ``struct`` / ``named_struct`` operation.

    Args:
        operation: The ``struct`` or ``named_struct`` ``ColumnOperation``.
        resolve: Callable evaluating one argument against the current row.
            Callers pass their own evaluator so this module stays free of
            any dependency on a particular evaluation strategy.

    Returns:
        Ordered mapping of field name to evaluated field value.
    """
    if getattr(operation, "operation", None) == "named_struct":
        return _build_named_struct_value(operation, resolve)

    result: Dict[str, Any] = {}
    for position, item in enumerate(struct_argument_columns(operation), start=1):
        result[field_name_for(item, position)] = resolve(struct_argument_source(item))
    return result


def _build_named_struct_value(
    operation: Any,
    resolve: Callable[[Any], Any],
) -> Dict[str, Any]:
    """Build a ``named_struct`` value from alternating name/value arguments."""
    args = getattr(operation, "value", None) or ()
    if not isinstance(args, (list, tuple)):
        args = (args,)

    result: Dict[str, Any] = {}
    for index in range(0, len(args) - 1, 2):  # noqa: PLR2004 - name/value pairs
        raw_name = args[index]
        # The field name is itself usually an F.lit("name"); unwrap it.
        if _is_literal(raw_name):
            field_name = str(raw_name.value)
        elif isinstance(raw_name, str):
            field_name = raw_name
        else:
            field_name = str(getattr(raw_name, "name", raw_name))
        result[field_name] = resolve(struct_argument_source(args[index + 1]))
    return result
