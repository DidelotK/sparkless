"""Regression tests for the scale argument and rounding mode of ``F.round``.

Two independent defects made ``F.round(x, 2)`` wrong:

1. **The scale was dropped.** Of the three round implementations, two ignored
   the argument outright and the third read it from an ``operation.precision``
   attribute that has never existed on ``ColumnOperation`` (the scale is
   carried on ``operation.value``), so ``getattr(..., "precision", 0)``
   silently rounded everything to zero decimal places.
2. **The rounding mode was wrong.** All three used Python's :func:`round`,
   which rounds halves to even -- ``round(2.5) == 2``. Spark's ``round``
   rounds halves *away from zero*: ``2.5 -> 3.0``. (Spark's banker's-rounding
   function is ``bround``.)

Spark also rounds the *decimal* representation of the double rather than its
exact binary expansion, so ``round(2.675, 2)`` is ``2.68``, not the ``2.67``
that the binary value ``2.67499999...`` would give.

Verified against PySpark 4.0.0 (the DBR 17.3 runtime). These tests are
backend-agnostic and also pass under ``MOCK_SPARK_TEST_BACKEND=pyspark``.
"""

import pytest

from tests.fixtures.spark_imports import get_spark_imports


def _scalar(spark, column):
    """Evaluate a single column expression against a one-row frame."""
    df = spark.createDataFrame([("x",)], ["dummy"])
    return df.select(column.alias("r")).collect()[0]["r"]


class TestRoundScaleArgument:
    """The scale argument must actually be applied."""

    @pytest.mark.parametrize(
        "value,scale,expected",
        [
            (3.14159, 2, 3.14),
            (3.14159, 3, 3.142),
            (3.14159, 0, 3.0),
            (2.71828, 4, 2.7183),
            (1234.5678, -2, 1200.0),
            (1234.5678, -3, 1000.0),
        ],
    )
    def test_round_applies_scale(self, spark, value, scale, expected) -> None:
        """round(v, n) keeps n decimal places, including negative n."""
        imports = get_spark_imports()
        F = imports.F
        assert _scalar(spark, F.round(F.lit(value), scale)) == expected

    def test_round_of_computed_column(self, spark) -> None:
        """The scale survives when the target is an arithmetic expression."""
        imports = get_spark_imports()
        F = imports.F
        df = spark.createDataFrame([(10.0,), (20.0,), (50.0,)], ["x"])

        rows = df.select(F.round(F.col("x") / 3, 2).alias("r")).collect()

        assert [r["r"] for r in rows] == [3.33, 6.67, 16.67]

    def test_round_without_scale_defaults_to_zero(self, spark) -> None:
        """round(v) with no scale rounds to a whole number."""
        imports = get_spark_imports()
        F = imports.F
        assert _scalar(spark, F.round(F.lit(3.7))) == 4.0

    def test_round_of_null_is_null(self, spark) -> None:
        """NULL in, NULL out."""
        imports = get_spark_imports()
        F = imports.F
        assert _scalar(spark, F.round(F.lit(None).cast("double"), 2)) is None


class TestRoundHalfUpSemantics:
    """Spark rounds halves away from zero, not to even."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (2.5, 3.0),  # Python's round() gives 2 here
            (3.5, 4.0),
            (-2.5, -3.0),  # away from zero, not toward it
            (0.5, 1.0),
            (1.5, 2.0),
        ],
    )
    def test_halves_round_away_from_zero(self, spark, value, expected) -> None:
        """round(2.5) is 3.0 -- banker's rounding would give 2.0."""
        imports = get_spark_imports()
        F = imports.F
        assert _scalar(spark, F.round(F.lit(value))) == expected

    @pytest.mark.parametrize(
        "value,scale,expected",
        [
            (0.125, 2, 0.13),
            (2.675, 2, 2.68),
            (1.005, 2, 1.01),
        ],
    )
    def test_halves_at_scale_round_away_from_zero(
        self, spark, value, scale, expected
    ) -> None:
        """Spark rounds the decimal representation, so 2.675 -> 2.68."""
        imports = get_spark_imports()
        F = imports.F
        assert _scalar(spark, F.round(F.lit(value), scale)) == expected
