"""PySpark parity for ``last_day`` / ``trunc`` and date predicates (BUG-052/053).

Two independent defects made ``df.filter(F.last_day(d) >= lit(s).cast("date"))``
return an **empty frame** where PySpark returns the matching rows -- silently,
with no error, which is the worst shape for a mock to fail in.

**BUG-052 --** ``last_day`` and ``trunc`` are exported from the public API and
marked supported, but neither had an evaluator implementation. Both built a
``ColumnOperation`` that matched no dispatch branch, so both answered NULL for
every row::

    df.select(F.last_day(F.col("d")))       # sparkless None;  PySpark 2026-01-31
    df.select(F.trunc(F.col("d"), "month")) # sparkless None;  PySpark 2026-01-01

``date_add`` on the same column worked, so this was not value parsing -- it was
a missing branch.

**BUG-053 --** two gaps in date *predicates*, which kept the filter empty even
once the values above were correct:

* ``F.lit("2026-01-01").cast("date")`` kept its cast on the projection path but
  **dropped it** on the predicate path, resolving to the raw ``str``.
* The comparison kernel coerced string/numeric pairs but not temporal ones, so
  ``date >= str`` raised ``TypeError`` and was swallowed into NULL.

Every expectation is the value captured from **PySpark 4.0.0 on OpenJDK 21**,
so this file is written against the backend-agnostic ``spark`` fixture and
passes under ``MOCK_SPARK_TEST_BACKEND=pyspark`` too.
"""

import datetime
from typing import Any, List

from tests.fixtures.parity_base import ParityTestBase
from tests.fixtures.spark_imports import get_spark_imports


def _values(rows: List[Any], key: str) -> List[Any]:
    """Extract one column from collected rows, in row order."""
    return [row[key] for row in rows]


def _dates(spark: Any) -> Any:
    """Dates spanning a month end, a leap February and a year end, plus NULL."""
    T = get_spark_imports()
    schema = T.StructType([T.StructField("d", T.DateType())])
    return spark.createDataFrame(
        [
            (datetime.date(2026, 1, 15),),
            (datetime.date(2026, 2, 3),),
            (datetime.date(2024, 2, 10),),
            (datetime.date(2026, 12, 31),),
            (None,),
        ],
        schema=schema,
    )


class TestLastDayParity(ParityTestBase):
    """``last_day`` returns the month's final date, not NULL."""

    def test_last_day_values(self, spark: Any) -> None:
        """PySpark 4.0.0 values; the bug returned NULL for every row."""
        F = get_spark_imports().F

        result = _dates(spark).select(F.last_day(F.col("d")).alias("v")).collect()

        assert _values(result, "v") == [
            datetime.date(2026, 1, 31),
            datetime.date(2026, 2, 28),
            datetime.date(2024, 2, 29),  # leap year
            datetime.date(2026, 12, 31),  # year end
            None,  # NULL in, NULL out
        ]


class TestTruncParity(ParityTestBase):
    """``trunc`` truncates to the requested unit, not to NULL."""

    def test_trunc_month(self, spark: Any) -> None:
        """PySpark 4.0.0: the 1st of each row's month."""
        F = get_spark_imports().F

        result = _dates(spark).select(F.trunc(F.col("d"), "month").alias("v")).collect()

        assert _values(result, "v") == [
            datetime.date(2026, 1, 1),
            datetime.date(2026, 2, 1),
            datetime.date(2024, 2, 1),
            datetime.date(2026, 12, 1),
            None,
        ]

    def test_trunc_year(self, spark: Any) -> None:
        """PySpark 4.0.0: January 1st of each row's year."""
        F = get_spark_imports().F

        result = _dates(spark).select(F.trunc(F.col("d"), "year").alias("v")).collect()

        assert _values(result, "v") == [
            datetime.date(2026, 1, 1),
            datetime.date(2026, 1, 1),
            datetime.date(2024, 1, 1),
            datetime.date(2026, 1, 1),
            None,
        ]

    def test_trunc_week_truncates_to_monday(self, spark: Any) -> None:
        """PySpark 4.0.0 truncates a week to its **Monday**."""
        F = get_spark_imports().F

        result = _dates(spark).select(F.trunc(F.col("d"), "week").alias("v")).collect()

        assert _values(result, "v") == [
            datetime.date(2026, 1, 12),
            datetime.date(2026, 2, 2),
            datetime.date(2024, 2, 5),
            datetime.date(2026, 12, 28),
            None,
        ]

    def test_trunc_quarter(self, spark: Any) -> None:
        """PySpark 4.0.0: the first day of the row's quarter."""
        F = get_spark_imports().F

        result = (
            _dates(spark).select(F.trunc(F.col("d"), "quarter").alias("v")).collect()
        )

        assert _values(result, "v") == [
            datetime.date(2026, 1, 1),
            datetime.date(2026, 1, 1),
            datetime.date(2024, 1, 1),
            datetime.date(2026, 10, 1),  # Q4
            None,
        ]

    def test_trunc_unit_is_case_insensitive(self, spark: Any) -> None:
        """``MONTH``/``mon``/``mm`` are the same unit as ``month``."""
        F = get_spark_imports().F
        df = _dates(spark)

        canonical = _values(
            df.select(F.trunc(F.col("d"), "month").alias("v")).collect(), "v"
        )
        for spelling in ("MONTH", "Mon", "mm"):
            assert (
                _values(
                    df.select(F.trunc(F.col("d"), spelling).alias("v")).collect(), "v"
                )
                == canonical
            ), spelling

    def test_trunc_unknown_unit_is_null_not_an_error(self, spark: Any) -> None:
        """PySpark answers NULL for an unrecognised unit rather than raising."""
        F = get_spark_imports().F

        result = _dates(spark).select(F.trunc(F.col("d"), "bogus").alias("v")).collect()

        assert _values(result, "v") == [None, None, None, None, None]


class TestDatePredicateParity(ParityTestBase):
    """Date comparisons must evaluate, not collapse to NULL (BUG-053)."""

    def _two_dates(self, spark: Any) -> Any:
        """One row on either side of 2026-01-01."""
        T = get_spark_imports()
        schema = T.StructType(
            [T.StructField("d", T.DateType()), T.StructField("s", T.StringType())]
        )
        return spark.createDataFrame(
            [
                (datetime.date(2026, 1, 15), "2026-01-15"),
                (datetime.date(2025, 6, 1), "2025-06-01"),
            ],
            schema=schema,
        )

    def test_cast_literal_to_date_keeps_the_cast(self, spark: Any) -> None:
        """``lit(str).cast("date")`` is a DATE on every path, not a string."""
        F = get_spark_imports().F

        result = (
            self._two_dates(spark)
            .select(F.lit("2026-01-01").cast("date").alias("v"))
            .collect()
        )

        assert _values(result, "v") == [
            datetime.date(2026, 1, 1),
            datetime.date(2026, 1, 1),
        ]

    def test_date_column_compared_to_cast_literal(self, spark: Any) -> None:
        """PySpark 4.0.0: ``[True, False]``. The bug returned ``[None, None]``."""
        F = get_spark_imports().F
        cast_lit = F.lit("2026-01-01").cast("date")

        result = (
            self._two_dates(spark).select((F.col("d") >= cast_lit).alias("v")).collect()
        )

        assert _values(result, "v") == [True, False]

    def test_date_column_compared_to_bare_string_literal(self, spark: Any) -> None:
        """Spark implicitly casts the string operand; so must we."""
        F = get_spark_imports().F

        result = (
            self._two_dates(spark)
            .select((F.col("d") >= F.lit("2026-01-01")).alias("v"))
            .collect()
        )

        assert _values(result, "v") == [True, False]

    def test_string_column_compared_to_date_literal(self, spark: Any) -> None:
        """The coercion is symmetric -- string on the left, date on the right."""
        F = get_spark_imports().F
        cast_lit = F.lit("2026-01-01").cast("date")

        result = (
            self._two_dates(spark).select((F.col("s") >= cast_lit).alias("v")).collect()
        )

        assert _values(result, "v") == [True, False]

    def test_date_equality_against_string(self, spark: Any) -> None:
        """Equality coerces the same way ordering does."""
        F = get_spark_imports().F

        result = (
            self._two_dates(spark)
            .select((F.col("d") == F.lit("2026-01-15")).alias("v"))
            .collect()
        )

        assert _values(result, "v") == [True, False]


class TestDateFilterKeepsRows(ParityTestBase):
    """The end-to-end shape: a filter over ``last_day``/``trunc`` keeps rows."""

    def _two_dates(self, spark: Any) -> Any:
        T = get_spark_imports()
        schema = T.StructType([T.StructField("d", T.DateType())])
        return spark.createDataFrame(
            [(datetime.date(2026, 1, 15),), (datetime.date(2025, 6, 1),)],
            schema=schema,
        )

    def test_filter_on_last_day_keeps_matching_rows(self, spark: Any) -> None:
        """PySpark 4.0.0 keeps 1 row. The bug kept 0 -- silently."""
        F = get_spark_imports().F
        cast_lit = F.lit("2026-01-01").cast("date")

        assert (
            self._two_dates(spark).filter(F.last_day(F.col("d")) >= cast_lit).count()
            == 1
        )

    def test_filter_on_trunc_keeps_matching_rows(self, spark: Any) -> None:
        """Same for ``trunc`` -- the other half of the production expression."""
        F = get_spark_imports().F
        cast_lit = F.lit("2026-01-01").cast("date")

        assert (
            self._two_dates(spark)
            .filter(F.trunc(F.col("d"), "month") >= cast_lit)
            .count()
            == 1
        )

    def test_filter_and_projection_agree(self, spark: Any) -> None:
        """The predicate path and the projection path must not diverge."""
        F = get_spark_imports().F
        cast_lit = F.lit("2026-01-01").cast("date")
        df = self._two_dates(spark)
        predicate = F.last_day(F.col("d")) >= cast_lit

        projected = _values(df.select(predicate.alias("v")).collect(), "v")

        assert projected == [True, False]
        assert df.filter(predicate).count() == projected.count(True)
