"""Regression tests for the constant functions ``F.pi`` and ``F.e``.

Both built ``ColumnOperation(Literal(value), "lit")``. ``ExpressionEvaluator``
implements ``pi`` and ``e`` as *operations* and has no handler for that shape,
so the programmatic API returned NULL for every row while ``F.expr("pi()")``
-- which used to construct the operation by hand -- returned the constant. The
two paths disagreed, and the wrong one was the documented API.

Values are PySpark 4.0.0's: ``math.pi`` and ``math.e`` to double precision.
"""

import math
from typing import Any

from tests.fixtures.spark_imports import get_spark_imports


def _single_value(spark: Any, column: Any) -> Any:
    """Evaluate one column expression against a one-row frame."""
    return spark.createDataFrame([{"dummy": 1}]).select(column).collect()[0][0]


class TestMathConstants:
    """A constant function must return its constant, not NULL."""

    def test_pi_through_the_function_api(self, spark: Any) -> None:
        """``F.pi()`` returns pi."""
        F = get_spark_imports().F
        assert _single_value(spark, F.pi()) == math.pi

    def test_e_through_the_function_api(self, spark: Any) -> None:
        """``F.e()`` returns Euler's number."""
        F = get_spark_imports().F
        assert _single_value(spark, F.e()) == math.e

    def test_pi_through_expr(self, spark: Any) -> None:
        """``F.expr("pi()")`` agrees with ``F.pi()``."""
        F = get_spark_imports().F
        assert _single_value(spark, F.expr("pi()")) == math.pi

    def test_e_through_expr(self, spark: Any) -> None:
        """``F.expr("e()")`` agrees with ``F.e()``."""
        F = get_spark_imports().F
        assert _single_value(spark, F.expr("e()")) == math.e

    def test_pi_keeps_its_pyspark_column_name(self, spark: Any) -> None:
        """PySpark names the column ``PI()``."""
        F = get_spark_imports().F
        df = spark.createDataFrame([{"dummy": 1}]).select(F.pi())
        assert df.columns == ["PI()"]
