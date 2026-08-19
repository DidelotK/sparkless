"""Canonical evaluation of the element-wise array functions.

Two independent evaluators need these: the lazy ``select`` path goes through
:class:`sparkless.core.condition_evaluator.ConditionEvaluator` and the
``withColumn`` path through
:class:`sparkless.dataframe.evaluation.expression_evaluator.ExpressionEvaluator`.
They had already drifted -- ``flatten`` was implemented in the second and
absent from the first, so ``df.select(F.flatten(x))`` answered NULL while
``df.withColumn("f", F.flatten(x))`` answered correctly on the same data. This
module is the single definition both call, in the same spirit as
:mod:`sparkless.core.struct_builder`.

Semantics verified against PySpark 4.0.0 (``local[1]``):

==========================================  ====================================
expression                                  result
==========================================  ====================================
``flatten([[1, 2], [3]])``                  ``[1, 2, 3]`` -- **one** level only,
                                            so ``[[[1]], [[3]]]`` gives
                                            ``[[1], [3]]``
``flatten([[1, 2], NULL])``                 ``NULL`` -- one NULL inner array
                                            poisons the whole result
``flatten([[]])``                           ``[]``
``array_min([5, NULL, 2])``                 ``2`` -- NULLs are skipped, not
                                            propagated
``array_min([])`` / ``array_min([NULL])``   ``NULL``
``slice([3, 1, 2], 1, 2)``                  ``[3, 1]`` -- ``start`` is 1-based
``slice([3, 1, 2], -2, 2)``                 ``[1, 2]`` -- negative counts back
                                            from the end
``slice([3, 1, 2], 3, 5)``                  ``[3]`` -- a length past the end
                                            truncates
``slice([3, 1, 2], 10, 2)``                 ``[]`` -- a start past the end is
                                            empty, not NULL
``slice(a, 0, n)`` / ``slice(a, n, -1)``    raises -- Spark rejects both
``array_distinct(["a", NULL, "a", NULL])``  ``["a", NULL]`` -- first-seen order
                                            kept, and NULL is a value like any
                                            other
==========================================  ====================================

Every function returns ``None`` for a NULL input and for an input that is not
an array, because that is what a NULL array column yields in Spark.
"""

from typing import Any, List, Optional, Sequence

__all__ = [
    "array_distinct_value",
    "flatten_value",
    "array_min_value",
    "array_max_value",
    "slice_value",
    "validate_slice_arguments",
]


def _as_sequence(value: Any) -> Optional[Sequence[Any]]:
    """Return ``value`` as a sequence, or ``None`` if it is not an array."""
    if isinstance(value, (list, tuple)):
        return value
    return None


def flatten_value(value: Any) -> Optional[List[Any]]:
    """Concatenate one level of nesting, PySpark ``flatten`` semantics.

    A NULL element makes the whole result NULL: Spark cannot concatenate an
    unknown array, so it does not silently drop it. The previous
    ``ExpressionEvaluator`` implementation appended the ``None`` as an element
    instead, which is a different answer on the same data.
    """
    items = _as_sequence(value)
    if items is None:
        return None

    result: List[Any] = []
    for item in items:
        if item is None:
            return None
        inner = _as_sequence(item)
        if inner is None:
            # The declared type is array<array<T>>, so this is unreachable for
            # well-typed input; keep the element rather than losing data.
            result.append(item)
        else:
            result.extend(inner)
    return result


def _extremum(value: Any, largest: bool) -> Any:
    """Shared body of :func:`array_min_value` and :func:`array_max_value`."""
    items = _as_sequence(value)
    if items is None:
        return None

    present = [item for item in items if item is not None]
    if not present:
        # Both an empty array and an all-NULL array give NULL in Spark.
        return None
    try:
        return max(present) if largest else min(present)
    except TypeError:
        # Mixed, non-comparable element types. Spark would have rejected this
        # at analysis time on a typed array; NULL is the honest answer here.
        return None


def array_min_value(value: Any) -> Any:
    """Smallest non-NULL element, or NULL for an empty/NULL/all-NULL array."""
    return _extremum(value, largest=False)


def array_max_value(value: Any) -> Any:
    """Largest non-NULL element, or NULL for an empty/NULL/all-NULL array."""
    return _extremum(value, largest=True)


def validate_slice_arguments(start: int, length: int) -> None:
    """Reject the two ``slice`` argument values Spark refuses.

    Raised when the expression is *built* rather than when a row is evaluated.
    A row evaluator that raises risks being swallowed by an enclosing
    ``except Exception`` and degrading back into the silent NULL this module
    exists to remove; failing at construction cannot be swallowed.
    """
    from .exceptions.validation import PySparkValueError

    if start == 0:
        raise PySparkValueError(
            "The value of parameter `start` in `slice` is invalid: expects a "
            "positive or a negative value for `start`, but got 0."
        )
    if length < 0:
        raise PySparkValueError(
            "The value of parameter `length` in `slice` is invalid: expects "
            f"`length` greater than or equal to 0, but got {length}."
        )


def slice_value(value: Any, start: int, length: int) -> Optional[List[Any]]:
    """Sub-array of ``length`` elements from 1-based ``start``.

    ``start`` counts from the end when negative. A ``start`` or a ``length``
    that runs past the end truncates instead of erroring, and a ``start``
    entirely past the end gives an empty array rather than NULL.
    """
    validate_slice_arguments(start, length)
    items = _as_sequence(value)
    if items is None:
        return None

    if start > 0:
        begin = start - 1
    else:
        begin = len(items) + start
        if begin < 0:
            begin = 0
    if begin >= len(items):
        return []
    return list(items[begin : begin + length])


def _element_key(item: Any) -> Any:
    """A hashable identity for an array element.

    Lists are unhashable but are legitimate elements of ``array<array<T>>``,
    so they are keyed by their tuple form; anything still unhashable falls
    back to ``repr``.
    """
    if isinstance(item, list):
        return ("__list__", tuple(_element_key(sub) for sub in item))
    try:
        hash(item)
    except TypeError:
        return ("__repr__", repr(item))
    return item


def array_distinct_value(value: Any) -> Optional[List[Any]]:
    """Deduplicate an array, keeping first-seen order.

    NULL is an ordinary value here: ``["a", NULL, "a", NULL]`` gives
    ``["a", NULL]``, not ``["a"]``.
    """
    items = _as_sequence(value)
    if items is None:
        return None

    seen = set()
    result: List[Any] = []
    for item in items:
        key = _element_key(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
