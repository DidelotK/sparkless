"""PySpark parity tests for nested CASE WHEN expressions (BUG-051).

A ``when`` branch whose value is *itself* a ``when(...).otherwise(...)`` was
never evaluated. Neither of the two evaluation paths recognised a nested
``CaseWhen``, and each failed differently::

    inner = F.when(F.col("a") <= F.col("ub"), F.lit(1)).otherwise(F.lit(0))
    nested = F.when(F.col("lb").isNotNull(), inner).otherwise(F.lit(0))

    df.select(nested)   # sparkless -> [None, None, 0]   PySpark -> [1, 0, 0]
    df.agg(F.sum(nested))
    # sparkless -> ColumnOperation('(CASE WHEN ... END + 0) + <CaseWhen>')
    #              PySpark -> 1

``CaseWhen`` is neither a ``Column`` nor a ``ColumnOperation``, so:

* ``ConditionalEvaluator.evaluate_case_when`` fell through to its terminal
  ``return value`` and handed back the **unevaluated object**. ``F.sum`` then
  folded that object into its accumulator (``acc + CaseWhen``), producing a
  ``ColumnOperation`` where a number was expected -- which surfaced downstream
  as ``TypeError: int() argument must be ... not 'ColumnOperation'``.
* ``CaseWhen._evaluate_value`` matched the ``hasattr(value, "name")`` fallback,
  because a ``CaseWhen`` carries a generated ``.name`` ("CASE WHEN ... END").
  It was therefore looked up as though it were a *column of that name*. No such
  column exists, so the branch silently evaluated to NULL.

Every expectation in this module is the value captured from **PySpark 4.0.0 on
OpenJDK 21**, so the file is written against the backend-agnostic ``spark``
fixture and passes under ``MOCK_SPARK_TEST_BACKEND=pyspark`` too.
"""

from typing import Any, List

from tests.fixtures.parity_base import ParityTestBase
from tests.fixtures.spark_imports import get_spark_imports


def _values(rows: List[Any], key: str) -> List[Any]:
    """Extract one column from collected rows, in row order."""
    return [row[key] for row in rows]


def _frame(spark: Any) -> Any:
    """Four rows exercising both branches plus a NULL guard.

    ``lb`` is NULL on row 3, so the outer condition is false there and the
    nested branch is never reached -- that row pins the ELSE path.
    """
    return spark.createDataFrame(
        [
            {"a": 5.0, "lb": 1.0, "ub": 10.0, "g": "x"},
            {"a": 50.0, "lb": 1.0, "ub": 10.0, "g": "x"},
            {"a": 7.0, "lb": None, "ub": 10.0, "g": "y"},
            {"a": 2.0, "lb": 1.0, "ub": 10.0, "g": "y"},
        ]
    )


def _nested_then(F: Any) -> Any:
    """The shape from the original report: a nested CASE in the THEN branch."""
    return F.when(
        F.col("lb").isNotNull() & F.col("ub").isNotNull(),
        F.when(
            (F.col("a") >= F.col("lb")) & (F.col("a") <= F.col("ub")),
            F.lit(1),
        ).otherwise(F.lit(0)),
    ).otherwise(F.lit(0))


class TestNestedCaseWhenProjectionParity(ParityTestBase):
    """A nested CASE WHEN must evaluate, not collapse to NULL."""

    def test_nested_case_in_then_branch_projects_values(self, spark: Any) -> None:
        """PySpark 4.0.0: ``[1, 0, 0, 1]``. The bug returned ``[None, None, 0, None]``."""
        F = get_spark_imports().F

        result = _frame(spark).select(_nested_then(F).alias("v")).collect()

        assert _values(result, "v") == [1, 0, 0, 1]

    def test_nested_case_in_else_branch_projects_values(self, spark: Any) -> None:
        """The nesting is in ``otherwise``, which took the same fallthrough."""
        F = get_spark_imports().F
        nested_else = F.when(F.col("lb").isNull(), F.lit(-1)).otherwise(
            F.when(F.col("a") > F.lit(10), F.lit(100)).otherwise(F.lit(7))
        )

        result = _frame(spark).select(nested_else.alias("v")).collect()

        # PySpark 4.0.0: [7, 100, -1, 7]
        assert _values(result, "v") == [7, 100, -1, 7]

    def test_nested_case_returning_a_column_not_a_literal(self, spark: Any) -> None:
        """The inner branches resolve columns, not just literals."""
        F = get_spark_imports().F
        nested_col = F.when(
            F.col("lb").isNotNull(),
            F.when(F.col("a") > F.lit(10), F.col("a")).otherwise(F.col("ub")),
        ).otherwise(F.lit(0.0))

        result = _frame(spark).select(nested_col.alias("v")).collect()

        # PySpark 4.0.0: [10.0, 50.0, 0.0, 10.0]
        assert _values(result, "v") == [10.0, 50.0, 0.0, 10.0]

    def test_nested_case_containing_arithmetic(self, spark: Any) -> None:
        """A ColumnOperation inside the nested branch still evaluates."""
        F = get_spark_imports().F
        nested_arith = F.when(
            F.col("lb").isNotNull(),
            F.when(F.col("a") > F.lit(3), F.col("a") * F.lit(2)).otherwise(F.lit(0.0)),
        ).otherwise(F.lit(0.0))

        result = _frame(spark).select(nested_arith.alias("v")).collect()

        # PySpark 4.0.0: [10.0, 100.0, 0.0, 0.0]
        assert _values(result, "v") == [10.0, 100.0, 0.0, 0.0]

    def test_three_levels_of_nesting(self, spark: Any) -> None:
        """Recursion must hold past a single level."""
        F = get_spark_imports().F
        deep = F.when(
            F.col("lb").isNotNull(),
            F.when(
                F.col("a") > F.lit(3),
                F.when(F.col("a") > F.lit(20), F.lit(3)).otherwise(F.lit(2)),
            ).otherwise(F.lit(1)),
        ).otherwise(F.lit(0))

        result = _frame(spark).select(deep.alias("v")).collect()

        # PySpark 4.0.0: [2, 3, 0, 1]
        assert _values(result, "v") == [2, 3, 0, 1]


class TestNestedCaseWhenAggregateParity(ParityTestBase):
    """``F.sum`` over a nested CASE WHEN returns a number, not an expression."""

    def test_sum_over_nested_case_returns_a_number(self, spark: Any) -> None:
        """The reported failure: the sum used to be a ``ColumnOperation``."""
        F = get_spark_imports().F

        row = _frame(spark).agg(F.sum(_nested_then(F)).alias("s")).collect()[0]

        # PySpark 4.0.0: 2. The bug produced an unevaluated ColumnOperation,
        # so int() on it raised TypeError.
        assert row["s"] == 2
        assert isinstance(row["s"], (int, float))

    def test_sum_over_nested_case_in_else_branch(self, spark: Any) -> None:
        """Same, with the nesting in ``otherwise``."""
        F = get_spark_imports().F
        nested_else = F.when(F.col("lb").isNull(), F.lit(-1)).otherwise(
            F.when(F.col("a") > F.lit(10), F.lit(100)).otherwise(F.lit(7))
        )

        row = _frame(spark).agg(F.sum(nested_else).alias("s")).collect()[0]

        # PySpark 4.0.0: 113
        assert row["s"] == 113

    def test_sum_over_three_level_nesting(self, spark: Any) -> None:
        """PySpark 4.0.0: 6."""
        F = get_spark_imports().F
        deep = F.when(
            F.col("lb").isNotNull(),
            F.when(
                F.col("a") > F.lit(3),
                F.when(F.col("a") > F.lit(20), F.lit(3)).otherwise(F.lit(2)),
            ).otherwise(F.lit(1)),
        ).otherwise(F.lit(0))

        row = _frame(spark).agg(F.sum(deep).alias("s")).collect()[0]

        assert row["s"] == 6

    def test_grouped_sum_over_nested_case(self, spark: Any) -> None:
        """The nested branch resolves per group, not just per whole frame."""
        F = get_spark_imports().F

        rows = (
            _frame(spark).groupBy("g").agg(F.sum(_nested_then(F)).alias("s")).collect()
        )

        # PySpark 4.0.0: [('x', 1), ('y', 1)]
        assert sorted((row["g"], row["s"]) for row in rows) == [("x", 1), ("y", 1)]


class TestNestedCaseWhenProjectionAndFilterAgree(ParityTestBase):
    """A projected nested CASE must be filterable -- it used to be all NULL."""

    def test_with_column_then_filter(self, spark: Any) -> None:
        """PySpark 4.0.0: 2 rows match. The bug matched 0 (NULL != 1)."""
        F = get_spark_imports().F

        projected = _frame(spark).withColumn("v", _nested_then(F))

        assert projected.filter(F.col("v") == F.lit(1)).count() == 2

    def test_select_and_with_column_agree(self, spark: Any) -> None:
        """The two projection paths evaluate the nesting identically."""
        F = get_spark_imports().F
        nested = _nested_then(F)

        via_select = _values(_frame(spark).select(nested.alias("v")).collect(), "v")
        via_with_column = _values(_frame(spark).withColumn("v", nested).collect(), "v")

        assert via_select == via_with_column == [1, 0, 0, 1]
