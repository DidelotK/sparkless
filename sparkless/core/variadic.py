"""Variadic comparison functions -- one implementation, shared by every path.

``greatest`` and ``least`` were previously implemented three times: correctly in
``ConditionEvaluator._evaluate_function_operation_value`` (the lazy ``select``
path), as a stub in ``ConditionEvaluator._evaluate_function_operation`` that
returned the first operand, and not at all in
``ExpressionEvaluator._evaluate_function_call`` -- where an unregistered name
falls through to ``return value``, which is *also* the first operand.

Two of the three therefore answered ``greatest(a, b)`` with ``a``. That is right
whenever ``a`` happens to be the larger operand, which is most of the time, so
the defect survived casual testing (BUG-038). The module exists so the
semantics live in exactly one place and the evaluators differ only in how they
resolve operands to values.

Reference behaviour captured from PySpark 4.0.0 on OpenJDK 21 (the DBR 17.3
pairing); ``tests/unit/functions/test_least_greatest_operand_shapes.py`` runs
against both engines so the two cannot drift apart silently.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def spark_greatest(values: List[Any]) -> Any:
    """Largest of ``values``, skipping NULLs.

    Spark's ``greatest``/``least`` **skip** NULL operands rather than
    propagating them -- the opposite of nearly every other function, and the
    detail most easily got wrong. ``greatest(NULL, 2, 3)`` is ``3``, not NULL;
    only an all-NULL argument list yields NULL. Verified on PySpark 4.0.0.

    Args:
        values: Already-resolved operand values, NULLs included.

    Returns:
        The largest non-NULL value, or ``None`` when every operand is NULL.
    """
    return _reduce(values, max)


def spark_least(values: List[Any]) -> Any:
    """Smallest of ``values``, skipping NULLs. See :func:`spark_greatest`."""
    return _reduce(values, min)


def _reduce(values: List[Any], chooser: Callable[[List[Any]], Any]) -> Any:
    """Apply ``chooser`` to the non-NULL operands."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    try:
        return chooser(present)
    except TypeError:
        # Operands of incomparable types. Spark rejects this at analysis time
        # with DATATYPE_MISMATCH.DATA_DIFF_TYPES; sparkless has no analysis
        # phase, so NULL is the closest available answer. See BUG-049.
        return None


#: Functions taking an arbitrary number of operands, keyed by operation name.
#: An evaluator resolves every operand to a value and applies the entry -- so
#: adding a variadic function is a table entry, not a new hand-written branch
#: in each of the evaluators.
VARIADIC_FUNCTIONS: Dict[str, Callable[[List[Any]], Any]] = {
    "greatest": spark_greatest,
    "least": spark_least,
}


def variadic_reducer(name: str) -> Optional[Callable[[List[Any]], Any]]:
    """Return the variadic reducer registered for ``name``, if any."""
    return VARIADIC_FUNCTIONS.get(name)
