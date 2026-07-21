"""Math helpers that reproduce Spark's numeric semantics.

Python's built-in :func:`round` is *not* a drop-in for Spark's ``round``: it
uses banker's rounding (round-half-to-even), so ``round(2.5) == 2``. Spark's
``round`` rounds halves **away from zero** (``HALF_UP``), giving ``3.0``.
Spark's ``bround`` is the banker's-rounding variant.
"""

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
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
        return float(decimal_value.quantize(quantum, rounding=ROUND_HALF_UP))
    except (InvalidOperation, OverflowError, ValueError):
        # Scale too large/small to represent: fall back to the unrounded value.
        return float(decimal_value)
