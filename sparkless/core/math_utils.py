"""Math helpers that reproduce Spark's numeric semantics.

Python's built-in :func:`round` is *not* a drop-in for Spark's ``round``: it
uses banker's rounding (round-half-to-even), so ``round(2.5) == 2``. Spark's
``round`` rounds halves **away from zero** (``HALF_UP``), giving ``3.0``.
Spark's ``bround`` is the banker's-rounding variant.
"""

from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, ROUND_HALF_UP
from typing import Any, Optional


def spark_round(value: Any, scale: int = 0) -> Optional[float]:
    """Round ``value`` to ``scale`` decimal places the way Spark does.

    Two behaviours distinguish this from Python's :func:`round`:

    * **Ties go away from zero** (``HALF_UP``), not to the nearest even digit.
      ``spark_round(2.5) == 3.0`` where ``round(2.5) == 2``.
    * **Rounding happens on the decimal representation.** Spark converts the
      double via its shortest round-tripping string (Java's
      ``BigDecimal.valueOf``), so ``round(2.675, 2)`` is ``2.68`` -- not the
      ``2.67`` you get from the exact binary value ``2.67499999...``.

    A negative ``scale`` rounds to the left of the decimal point:
    ``spark_round(1234.5678, -2) == 1200.0``.

    Args:
        value: Number to round. ``None`` and non-numeric input yield ``None``.
        scale: Decimal places to keep; may be negative.

    Returns:
        The rounded value as a float, or ``None`` when it is not a number.
    """
    return _quantize(value, scale, ROUND_HALF_UP)


def spark_bround(value: Any, scale: int = 0) -> Optional[float]:
    """Round ``value`` to ``scale`` places with Spark's ``bround`` semantics.

    ``bround`` is ``round``'s banker's-rounding sibling: ties go to the nearest
    **even** digit (``HALF_EVEN``) rather than away from zero, so
    ``spark_bround(2.5) == 2.0`` and ``spark_bround(3.5) == 4.0`` where
    ``spark_round`` gives ``3.0`` and ``4.0``.

    It is *not* Python's built-in :func:`round`, despite both being HALF_EVEN.
    Python rounds the exact binary expansion of the double, Spark rounds its
    shortest round-tripping decimal string. They disagree wherever the two
    differ: ``round(2.675, 2) == 2.67`` but Spark's
    ``bround(2.675, 2) == 2.68`` -- confirmed against PySpark 4.0.0 on
    OpenJDK 21. Sharing :func:`_quantize` with :func:`spark_round` is what
    keeps the two functions from drifting apart on that detail.

    Args:
        value: Number to round. ``None`` and non-numeric input yield ``None``.
        scale: Decimal places to keep; may be negative.

    Returns:
        The rounded value as a float, or ``None`` when it is not a number.
    """
    return _quantize(value, scale, ROUND_HALF_EVEN)


def _quantize(value: Any, scale: int, rounding: str) -> Optional[float]:
    """Quantize ``value`` to ``scale`` decimal places under ``rounding``.

    Shared by :func:`spark_round` and :func:`spark_bround` so that the two
    differ *only* in their rounding mode -- the decimal-representation detail
    that both depend on lives here, once.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        # str(float(...)) reproduces Java's BigDecimal.valueOf(double), which
        # is what Spark rounds -- not the exact binary expansion of the double.
        decimal_value = Decimal(str(float(value)))
    except (TypeError, ValueError, ArithmeticError):
        return None

    try:
        quantum = Decimal(1).scaleb(-scale)
        return float(decimal_value.quantize(quantum, rounding=rounding))
    except (InvalidOperation, OverflowError, ValueError):
        # Scale too large/small to represent: fall back to the unrounded value.
        return float(decimal_value)
