"""Regression tests for ``least``/``greatest`` and ``bround`` (BUG-038, BUG-045).

Both are the same defect wearing different clothes.
``ExpressionEvaluator._evaluate_function_call`` ends in ``return value`` --
"the function's first operand" -- for any name it does not recognise, and
neither ``greatest``/``least`` nor ``bround`` was registered. So
``greatest(a, b)`` answered ``a`` and ``bround(v, 2)`` answered the unrounded
``v``.

That is the worst possible failure mode, because the identity is *often* the
right answer: ``greatest`` is correct whenever its first operand happens to be
the largest, and ``bround`` is correct on every already-round value. The defect
was found only because a ``greatest`` assertion passed while its ``least``
twin failed.

The operand *shape* matters more than the operand values here, and is what the
parametrisation below varies. Three shapes reach three different evaluators:

======================  ===============================================
Call shape              Evaluator
======================  ===============================================
``df.select(...)``      ``ConditionEvaluator`` (correct before the fix)
``df.withColumn(...)``  ``ExpressionEvaluator`` (returned operand 1)
``df.groupBy().agg()``  ``ExpressionEvaluator`` via BUG-037's resolver
======================  ===============================================

Holding the shape fixed at "bare column in a select" -- the obvious way to test
this -- exercises only the one path that was already right.

Every expectation below was captured from real PySpark 4.0.0 on OpenJDK 21 (the
DBR 17.3 pairing). These tests use the backend-agnostic ``spark`` fixture, so
the same file runs against real PySpark with
``MOCK_SPARK_TEST_BACKEND=pyspark``.
"""

from typing import Any, List, Optional

import pytest

from tests.fixtures.spark_imports import get_spark_imports


def _frame(spark: Any) -> Any:
    """Frame whose first operand is *smaller* than the second on row 1.

    Row 1 is ``(30.0, 60.0, 45.0)``: any implementation returning operand 1
    gives ``30.0`` where ``greatest`` must give ``60.0``. Row 2 is
    ``(5.0, 1.0, 3.0)`` -- operand 1 *is* the greatest, so a broken
    implementation is right on that row. Both rows are needed: a single-row
    fixture built the other way round would pass against the bug.
    """
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("a", imports.DoubleType()),
            imports.StructField("b", imports.DoubleType()),
            imports.StructField("c", imports.DoubleType()),
            imports.StructField("grp", imports.StringType()),
        ]
    )
    return spark.createDataFrame(
        [(30.0, 60.0, 45.0, "g"), (5.0, 1.0, 3.0, "g")], schema
    )


def _nullable_frame(spark: Any) -> Any:
    """Frame covering every NULL arrangement, including all-NULL."""
    imports = get_spark_imports()
    schema = imports.StructType(
        [
            imports.StructField("a", imports.DoubleType()),
            imports.StructField("b", imports.DoubleType()),
            imports.StructField("c", imports.DoubleType()),
        ]
    )
    return spark.createDataFrame(
        [
            (None, 2.0, 3.0),
            (1.0, None, 3.0),
            (None, None, None),
            (5.0, None, None),
        ],
        schema,
    )


def _values(rows: List[Any], key: str) -> List[Any]:
    """Column ``key`` from ``rows``, floats normalised across backends."""
    out = []
    for row in rows:
        value = row[key]
        out.append(float(value) if isinstance(value, (int, float)) else value)
    return out


class TestGreatestLeastOperandShapes:
    """The same expression through select / withColumn / agg must agree."""

    def test_select_bare_columns(self, spark: Any) -> None:
        """The path that was already correct -- the control for the others."""
        imports = get_spark_imports()
        F = imports.F
        df = _frame(spark)
        assert _values(df.select(F.greatest("a", "b").alias("r")).collect(), "r") == [
            60.0,
            5.0,
        ]
        assert _values(df.select(F.least("a", "b").alias("r")).collect(), "r") == [
            30.0,
            1.0,
        ]

    def test_with_column_bare_columns(self, spark: Any) -> None:
        """withColumn used to answer 30.0 -- operand 1, not the greatest."""
        imports = get_spark_imports()
        F = imports.F
        df = _frame(spark)
        assert _values(df.withColumn("r", F.greatest("a", "b")).collect(), "r") == [
            60.0,
            5.0,
        ]
        assert _values(df.withColumn("r", F.least("a", "b")).collect(), "r") == [
            30.0,
            1.0,
        ]

    def test_with_column_string_and_column_operands_agree(self, spark: Any) -> None:
        """``greatest("a", "b")`` must equal ``greatest(col(a), col(b))``.

        ``F.greatest`` promotes only its *first* argument to a ``Column`` and
        leaves the rest as ``str``, so the two spellings take different
        resolution paths. An early version of this fix handled the ``Column``
        form and returned NULL for the bare-string form.
        """
        imports = get_spark_imports()
        F = imports.F
        df = _frame(spark)
        strings = _values(df.withColumn("r", F.greatest("a", "b")).collect(), "r")
        columns = _values(
            df.withColumn("r", F.greatest(F.col("a"), F.col("b"))).collect(), "r"
        )
        assert strings == columns == [60.0, 5.0]

    def test_three_operands(self, spark: Any) -> None:
        """A third operand must not be dropped."""
        imports = get_spark_imports()
        F = imports.F
        df = _frame(spark)
        assert _values(
            df.withColumn("r", F.greatest("a", "b", "c")).collect(), "r"
        ) == [60.0, 5.0]
        assert _values(df.withColumn("r", F.least("a", "b", "c")).collect(), "r") == [
            30.0,
            1.0,
        ]

    def test_expression_operand(self, spark: Any) -> None:
        """An arithmetic expression as an operand, not a bare column."""
        imports = get_spark_imports()
        F = imports.F
        df = _frame(spark)
        assert _values(
            df.withColumn("r", F.greatest(F.col("a") * 2, F.col("b"))).collect(), "r"
        ) == [60.0, 10.0]

    def test_nested_function_operand(self, spark: Any) -> None:
        """A function call as an operand."""
        imports = get_spark_imports()
        F = imports.F
        df = _frame(spark)
        assert _values(
            df.withColumn("r", F.greatest(F.abs(F.col("a")), F.col("b"))).collect(), "r"
        ) == [60.0, 5.0]

    def test_literal_operand(self, spark: Any) -> None:
        """A literal operand must participate in the comparison."""
        imports = get_spark_imports()
        F = imports.F
        df = _frame(spark)
        assert _values(
            df.withColumn("r", F.greatest(F.col("a"), F.lit(100.0))).collect(), "r"
        ) == [100.0, 100.0]

    def test_aggregate_operands(self, spark: Any) -> None:
        """``greatest(sum, max)`` -- the shape that hid the bug in the wild.

        ``least`` returns ``35.0`` here *either way*, because the sum happens
        to be the smaller of the two. Only ``greatest`` distinguishes a correct
        implementation from one returning operand 1.
        """
        imports = get_spark_imports()
        F = imports.F
        df = _frame(spark)
        greatest = df.groupBy("grp").agg(F.greatest(F.sum("a"), F.max("b")).alias("r"))
        assert _values(greatest.collect(), "r") == [60.0]
        least = df.groupBy("grp").agg(F.least(F.sum("a"), F.max("b")).alias("r"))
        assert _values(least.collect(), "r") == [35.0]

    def test_select_level_aggregate_operands(self, spark: Any) -> None:
        """The same through a bare ``select`` -- one row, not one per input."""
        imports = get_spark_imports()
        F = imports.F
        df = _frame(spark)
        rows = df.select(F.greatest(F.sum("a"), F.max("b")).alias("r")).collect()
        assert len(rows) == 1
        assert _values(rows, "r") == [60.0]

    def test_in_a_filter_predicate(self, spark: Any) -> None:
        """As a predicate, where a wrong value changes the row count."""
        imports = get_spark_imports()
        F = imports.F
        df = _frame(spark)
        assert df.filter(F.greatest("a", "b") > 50).count() == 1
        assert df.filter(F.least("a", "b") < 2).count() == 1

    def test_string_operands(self, spark: Any) -> None:
        """``greatest`` orders strings too, not only numbers."""
        imports = get_spark_imports()
        F = imports.F
        schema = imports.StructType([imports.StructField("s", imports.StringType())])
        df = spark.createDataFrame([("b",), ("z",)], schema)
        rows = df.withColumn("r", F.greatest(F.col("s"), F.lit("m"))).collect()
        assert [row["r"] for row in rows] == ["m", "z"]

    def test_single_row_frame(self, spark: Any) -> None:
        """A one-row frame, where an off-by-one over rows would not show."""
        imports = get_spark_imports()
        F = imports.F
        schema = imports.StructType(
            [
                imports.StructField("a", imports.DoubleType()),
                imports.StructField("b", imports.DoubleType()),
            ]
        )
        df = spark.createDataFrame([(7.0, 8.0)], schema)
        assert _values(df.withColumn("r", F.greatest("a", "b")).collect(), "r") == [8.0]


class TestGreatestLeastNullHandling:
    """Spark *skips* NULLs here -- the opposite of most functions."""

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "func,expected",
        [
            ("greatest", [3.0, 3.0, None, 5.0]),
            ("least", [2.0, 1.0, None, 5.0]),
        ],
    )
    def test_nulls_are_skipped_not_propagated(
        self, spark: Any, func: str, expected: List[Optional[float]]
    ) -> None:
        """A NULL operand is ignored; only an all-NULL row yields NULL."""
        imports = get_spark_imports()
        F = imports.F
        df = _nullable_frame(spark)
        column = getattr(F, func)("a", "b", "c")
        assert _values(df.withColumn("r", column).collect(), "r") == expected

    def test_null_column_against_literal(self, spark: Any) -> None:
        """A literal keeps the result non-NULL even where the column is NULL."""
        imports = get_spark_imports()
        F = imports.F
        df = _nullable_frame(spark)
        rows = df.withColumn("r", F.greatest(F.col("a"), F.lit(0.0))).collect()
        assert _values(rows, "r") == [0.0, 1.0, 0.0, 5.0]

    def test_select_and_with_column_agree_on_nulls(self, spark: Any) -> None:
        """The two evaluators must not disagree about NULL handling."""
        imports = get_spark_imports()
        F = imports.F
        df = _nullable_frame(spark)
        via_select = _values(
            df.select(F.least("a", "b", "c").alias("r")).collect(), "r"
        )
        via_with_column = _values(
            df.withColumn("r", F.least("a", "b", "c")).collect(), "r"
        )
        assert via_select == via_with_column == [2.0, 1.0, None, 5.0]


class TestGreatestLeastArity:
    """Spark rejects a single-argument call; accepting it made it identity."""

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "func", ["greatest", "least"]
    )
    def test_single_argument_is_rejected(self, spark: Any, func: str) -> None:
        """PySpark raises ``[WRONG_NUM_COLUMNS]``; sparkless used to return ``a``."""
        imports = get_spark_imports()
        F = imports.F
        with pytest.raises(Exception, match="(?i)at least 2 columns"):
            getattr(F, func)("a")


class TestBround:
    """``bround`` is HALF_EVEN, and is *not* Python's built-in ``round``."""

    @pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]
        "value,scale,expected",
        [
            # Ties go to the nearest even digit, unlike F.round's HALF_UP.
            (2.5, 0, 2.0),
            (3.5, 0, 4.0),
            (-2.5, 0, -2.0),
            (1.2345, 0, 1.0),
            # Spark rounds the shortest round-tripping decimal string, so this
            # is 2.68. Python's round(2.675, 2) is 2.67 -- it rounds the exact
            # binary value 2.67499999... The BUG_LOG's note that Python's round
            # "is precisely the correct semantics for bround" is wrong here.
            (2.675, 2, 2.68),
            (1234.5678, -2, 1200.0),
            (None, 0, None),
        ],
    )
    def test_scalar_values(
        self, spark: Any, value: Optional[float], scale: int, expected: Optional[float]
    ) -> None:
        """Values captured from PySpark 4.0.0."""
        imports = get_spark_imports()
        F = imports.F
        schema = imports.StructType([imports.StructField("v", imports.DoubleType())])
        df = spark.createDataFrame([(value,)], schema)
        assert _values(
            df.select(F.bround(F.col("v"), scale).alias("r")).collect(), "r"
        )[0] == pytest.approx(expected)

    def test_bround_differs_from_round_on_ties(self, spark: Any) -> None:
        """The two rounding modes must not collapse into each other."""
        imports = get_spark_imports()
        F = imports.F
        schema = imports.StructType([imports.StructField("v", imports.DoubleType())])
        df = spark.createDataFrame([(2.5,), (3.5,)], schema)
        rounded = _values(df.select(F.round(F.col("v"), 0).alias("r")).collect(), "r")
        broundded = _values(
            df.select(F.bround(F.col("v"), 0).alias("r")).collect(), "r"
        )
        assert rounded == [3.0, 4.0]
        assert broundded == [2.0, 4.0]

    def test_with_column_shape(self, spark: Any) -> None:
        """withColumn used to return the *unrounded* operand."""
        imports = get_spark_imports()
        F = imports.F
        schema = imports.StructType([imports.StructField("v", imports.DoubleType())])
        df = spark.createDataFrame([(2.5,), (3.5,), (1.2345,)], schema)
        assert _values(df.withColumn("r", F.bround(F.col("v"), 0)).collect(), "r") == [
            2.0,
            4.0,
            1.0,
        ]

    def test_expression_operand(self, spark: Any) -> None:
        """An expression operand, not a bare column."""
        imports = get_spark_imports()
        F = imports.F
        schema = imports.StructType([imports.StructField("v", imports.DoubleType())])
        df = spark.createDataFrame([(2.5,), (3.5,)], schema)
        assert _values(
            df.select(F.bround(F.col("v") * 1, 0).alias("r")).collect(), "r"
        ) == [2.0, 4.0]

    def test_nested_function_operand(self, spark: Any) -> None:
        """A function call as the operand."""
        imports = get_spark_imports()
        F = imports.F
        schema = imports.StructType([imports.StructField("v", imports.DoubleType())])
        df = spark.createDataFrame([(2.5,), (3.5,)], schema)
        assert _values(
            df.select(F.bround(F.abs(F.col("v")), 0).alias("r")).collect(), "r"
        ) == [2.0, 4.0]

    def test_over_an_aggregate(self, spark: Any) -> None:
        """``bround(sum(v))`` -- the identity was accidentally right when the
        sum was already round, so the operand values here are chosen so it is
        not: ``2.5 + 3.7 == 6.2`` rounds to ``6.0``."""
        imports = get_spark_imports()
        F = imports.F
        schema = imports.StructType([imports.StructField("v", imports.DoubleType())])
        df = spark.createDataFrame([(2.5,), (3.7,)], schema)
        assert _values(df.agg(F.bround(F.sum("v"), 0).alias("r")).collect(), "r") == [
            6.0
        ]

    def test_in_a_filter_predicate(self, spark: Any) -> None:
        """As a predicate, where the unrounded operand changes the row count."""
        imports = get_spark_imports()
        F = imports.F
        schema = imports.StructType([imports.StructField("v", imports.DoubleType())])
        df = spark.createDataFrame([(2.5,), (3.5,)], schema)
        assert df.filter(F.bround(F.col("v"), 0) == 2).count() == 1
