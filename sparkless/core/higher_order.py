"""Canonical evaluation of the higher-order array functions.

``F.exists`` / ``F.forall`` / ``F.filter`` / ``F.transform`` take a Python
lambda that builds a Spark *expression* out of its argument. sparkless has an
AST-based ``LambdaParser`` for translating such a lambda into DuckDB syntax,
but nothing needs a translation to answer these four during row evaluation:
the array elements are already concrete values, so the lambda can simply be
applied to each one and the expression it returns evaluated by the same
evaluator that handles every other expression.

That is what this module does. It is deliberately independent of any
particular evaluator -- callers pass an ``evaluate`` callable -- in the same
spirit as :mod:`sparkless.core.struct_builder` and
:mod:`sparkless.core.array_values`.

Three-valued semantics, verified against PySpark 4.0.0 (``local[1]``):

=================================  ============================================
call                               answer
=================================  ============================================
``exists(NULL, f)``                NULL
``exists([], f)``                  FALSE
``exists`` -- some TRUE            TRUE, even if another element is NULL
``exists`` -- no TRUE, some NULL   NULL, *not* FALSE
``forall(NULL, f)``                NULL
``forall([], f)``                  TRUE
``forall`` -- some FALSE           FALSE, even if another element is NULL
``forall`` -- no FALSE, some NULL  NULL
``filter``                         keeps TRUE only; NULL predicate drops the
                                   element; NULL array stays NULL
``transform``                      maps element-wise, NULL array stays NULL
=================================  ============================================
"""

import inspect
from typing import Any, Callable, List, Optional, Sequence

__all__ = [
    "ElementApplier",
    "element_applier",
    "exists_value",
    "forall_value",
    "filter_value",
    "transform_value",
]

#: Applies the lambda to one element and returns the evaluated result.
#: Takes the element and its 0-based index.
ElementApplier = Callable[[Any, int], Any]

#: Arity of the ``(element, index)`` lambda form Spark also accepts.
_INDEXED_ARITY = 2


def _as_sequence(value: Any) -> Optional[Sequence[Any]]:
    """Return ``value`` as a sequence, or ``None`` if it is not an array."""
    if isinstance(value, (list, tuple)):
        return value
    return None


def _lambda_arity(lambda_func: Any) -> int:
    """Number of parameters the lambda declares.

    Spark's higher-order functions accept ``(element)`` or ``(element, index)``.
    ``inspect.signature`` answers this without needing the lambda's *source*,
    which matters: the source is unavailable for a lambda defined in a REPL or
    an ``exec``, and requiring it is what made ``F.transform`` raise
    ``LambdaTranslationError`` instead of computing.
    """
    try:
        return len(inspect.signature(lambda_func).parameters)
    except (TypeError, ValueError):
        return 1


def element_applier(
    lambda_expression: Any,
    evaluate: Callable[[Any], Any],
) -> ElementApplier:
    """Build the per-element applier the four functions below consume.

    Args:
        lambda_expression: What the ``F.*`` builder stored on the operation --
            a ``MockLambdaExpression`` wrapping the user's callable, or the
            callable itself.
        evaluate: Evaluates one expression against the current row. The
            caller supplies its own evaluator so this module does not depend
            on a particular evaluation strategy.

    Returns:
        A callable taking ``(element, index)`` and returning the evaluated
        result of the lambda for that element.
    """
    from ..functions.core.literals import Literal

    lambda_func = getattr(lambda_expression, "lambda_func", lambda_expression)
    if not callable(lambda_func):
        raise TypeError(
            "higher-order array functions need a callable lambda, got "
            f"{type(lambda_expression).__name__}"
        )
    arity = _lambda_arity(lambda_func)

    def apply(element: Any, index: int) -> Any:
        # The element is a concrete value, so it enters the expression as a
        # literal; whatever the lambda builds on top of it is then an ordinary
        # expression the caller's evaluator already knows how to evaluate.
        argument = Literal(element)
        built = (
            lambda_func(argument, Literal(index))
            if arity >= _INDEXED_ARITY
            else lambda_func(argument)
        )
        return evaluate(built)

    return apply


def _as_sql_boolean(value: Any) -> Optional[bool]:
    """Read an evaluated predicate as a SQL boolean, keeping NULL as NULL."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)


def exists_value(value: Any, apply: ElementApplier) -> Optional[bool]:
    """TRUE if any element satisfies the predicate.

    NULL when no element satisfies it but at least one answer was unknown:
    that element could have been the match, so FALSE would be an assertion
    Spark is not entitled to make.
    """
    items = _as_sequence(value)
    if items is None:
        return None

    unknown = False
    for index, element in enumerate(items):
        answer = _as_sql_boolean(apply(element, index))
        if answer is True:
            return True
        if answer is None:
            unknown = True
    return None if unknown else False


def forall_value(value: Any, apply: ElementApplier) -> Optional[bool]:
    """TRUE if every element satisfies the predicate.

    A single FALSE settles it even when another element is unknown; NULL only
    when nothing contradicts the predicate but something is unknown.
    """
    items = _as_sequence(value)
    if items is None:
        return None

    unknown = False
    for index, element in enumerate(items):
        answer = _as_sql_boolean(apply(element, index))
        if answer is False:
            return False
        if answer is None:
            unknown = True
    return None if unknown else True


def filter_value(value: Any, apply: ElementApplier) -> Optional[List[Any]]:
    """Elements whose predicate is TRUE. A NULL predicate drops the element."""
    items = _as_sequence(value)
    if items is None:
        return None

    return [
        element
        for index, element in enumerate(items)
        if _as_sql_boolean(apply(element, index)) is True
    ]


def transform_value(value: Any, apply: ElementApplier) -> Optional[List[Any]]:
    """Every element mapped through the lambda, preserving length and order."""
    items = _as_sequence(value)
    if items is None:
        return None

    return [apply(element, index) for index, element in enumerate(items)]
