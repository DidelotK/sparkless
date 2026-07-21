"""Date helpers that reproduce Spark's ``last_day`` / ``trunc`` semantics.

Both functions are *date*-valued even when handed a timestamp, and both are
total: an unparseable input, a NULL, or an unrecognised truncation unit yields
``None`` rather than raising. That matches Spark, which answers NULL for
``trunc(d, 'bogus')`` instead of failing the query.

Shared by both evaluators (``core.condition_evaluator`` and
``dataframe.evaluation.expression_evaluator``) so the filter path and the
``withColumn`` path cannot drift apart -- the same split that let BUG-051's
sibling hide.
"""

import datetime as dt_module
from typing import Any, Optional, Tuple

# Spark's `trunc` accepts these unit spellings, case-insensitively.
_YEAR_UNITS = frozenset({"year", "yyyy", "yy"})
_MONTH_UNITS = frozenset({"month", "mon", "mm"})
_WEEK_UNITS = frozenset({"week"})
_QUARTER_UNITS = frozenset({"quarter"})


def parse_temporal(value: Any) -> Optional[dt_module.date]:
    """Coerce ``value`` to a ``date``/``datetime``, or ``None``.

    Accepts ``date``, ``datetime`` and ISO-8601 strings (Spark implicitly casts
    a STRING operand to DATE). Anything else -- including a bool, which is not
    a temporal value despite ``isinstance(True, int)`` -- yields ``None``.

    Args:
        value: The value to coerce.

    Returns:
        A ``date`` or ``datetime``, or ``None`` when it is not temporal.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, dt_module.datetime):
        return value
    if isinstance(value, dt_module.date):
        return value
    if isinstance(value, str):
        try:
            return dt_module.datetime.fromisoformat(value.replace(" ", "T"))
        except (ValueError, TypeError):
            return None
    return None


def _as_date(value: dt_module.date) -> dt_module.date:
    """Narrow a ``datetime`` to its ``date``; both functions return DATE."""
    if isinstance(value, dt_module.datetime):
        return value.date()
    return value


def coerce_temporal_pair(
    left: Any, right: Any
) -> Optional[Tuple[dt_module.date, dt_module.date]]:
    """Reconcile a comparison pair when either side is a date/datetime.

    Spark implicitly casts across the temporal boundary, so ``date_col >=
    '2026-01-01'`` compares two DATEs and ``date_col >= timestamp_col``
    promotes the DATE to a TIMESTAMP at midnight. Python does neither:
    ``date >= str`` and ``date >= datetime`` both raise ``TypeError``, which
    this evaluator converts to NULL -- silently dropping every row of the
    enclosing filter (BUG-053).

    Args:
        left: Left-hand comparison operand.
        right: Right-hand comparison operand.

    Returns:
        The reconciled ``(left, right)`` pair, or ``None`` when the pair is not
        temporal or the string side is unparseable (leave it to the caller's
        existing handling).
    """
    left_temporal = isinstance(left, dt_module.date)
    right_temporal = isinstance(right, dt_module.date)
    if not (left_temporal or right_temporal):
        return None

    # A string beside a temporal is implicitly cast, as Spark does.
    if left_temporal and isinstance(right, str):
        parsed = parse_temporal(right)
        if parsed is None:
            return None
        right, right_temporal = parsed, True
    elif right_temporal and isinstance(left, str):
        parsed = parse_temporal(left)
        if parsed is None:
            return None
        left, left_temporal = parsed, True

    if not (left_temporal and right_temporal):
        return None

    # DATE beside TIMESTAMP: promote the DATE to midnight, as Spark does.
    left_is_dt = isinstance(left, dt_module.datetime)
    right_is_dt = isinstance(right, dt_module.datetime)
    if left_is_dt != right_is_dt:
        if not left_is_dt:
            left = dt_module.datetime(left.year, left.month, left.day)
        else:
            right = dt_module.datetime(right.year, right.month, right.day)

    return left, right


def spark_last_day(value: Any) -> Optional[dt_module.date]:
    """Return the last day of ``value``'s month, as Spark's ``last_day`` does.

    The result is always a DATE, even for a TIMESTAMP operand, and the month
    length is calendar-correct: February 2024 gives the 29th.

    Args:
        value: A date, datetime or ISO date string.

    Returns:
        The month's final date, or ``None`` for NULL / non-temporal input.
    """
    parsed = parse_temporal(value)
    if parsed is None:
        return None
    day = _as_date(parsed)
    if day.month == 12:
        first_of_next = dt_module.date(day.year + 1, 1, 1)
    else:
        first_of_next = dt_module.date(day.year, day.month + 1, 1)
    return first_of_next - dt_module.timedelta(days=1)


def spark_trunc(value: Any, unit: Any) -> Optional[dt_module.date]:
    """Truncate ``value`` down to ``unit``, as Spark's ``trunc`` does.

    Supported units (case-insensitive): ``year``/``yyyy``/``yy``,
    ``month``/``mon``/``mm``, ``week`` and ``quarter``. ``week`` truncates to
    the **Monday** of that week, matching Spark. The result is always a DATE.

    An unrecognised unit returns ``None`` -- Spark answers NULL rather than
    raising, so a typo silently nulls the column there too.

    Args:
        value: A date, datetime or ISO date string.
        unit: The truncation unit.

    Returns:
        The truncated date, or ``None`` for NULL input or an unknown unit.
    """
    parsed = parse_temporal(value)
    if parsed is None or unit is None:
        return None
    day = _as_date(parsed)
    key = str(unit).lower()

    if key in _YEAR_UNITS:
        return dt_module.date(day.year, 1, 1)
    if key in _MONTH_UNITS:
        return dt_module.date(day.year, day.month, 1)
    if key in _WEEK_UNITS:
        return day - dt_module.timedelta(days=day.weekday())
    if key in _QUARTER_UNITS:
        first_month_of_quarter = 3 * ((day.month - 1) // 3) + 1
        return dt_module.date(day.year, first_month_of_quarter, 1)
    # Unknown unit: Spark yields NULL rather than raising.
    return None
